import hashlib
import json
import math
import numpy as np
import torch
from PIL import Image
import folder_paths
import os
from server import PromptServer
from aiohttp import web

# The official MOSS compiler, vendored unchanged (moss_compile.py sits beside this file).
try:
    from .moss_compile import compile_prompt as moss_compile_prompt
except ImportError:  # loaded outside a package
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "boyo_moss_compile", os.path.join(os.path.dirname(os.path.abspath(__file__)), "moss_compile.py"))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    moss_compile_prompt = _mod.compile_prompt


def _clean_list(value):
    """Comma-separated string or list -> list of non-empty stripped strings."""
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, (list, tuple)):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def _floor16(v):
    """Round down to a multiple of 16 (never below 16). Same rule as the editor."""
    return max(16, (int(v) // 16) * 16)


def _announce_size(unique_id, width, height, raw=None):
    """
    Tell the browser editor the size actually used (a linked Resolution Selector is invisible
    to JS). raw = the size that was linked, when it differs from the size used.
    """
    if unique_id is None:
        return
    payload = {"node": str(unique_id), "width": int(width), "height": int(height)}
    if raw:
        payload["raw_width"], payload["raw_height"] = int(raw[0]), int(raw[1])
    try:
        PromptServer.instance.send_sync("boyo_bbox_size", payload)
    except Exception:
        pass


def _half_up(v):
    # same rounding as the editor's Math.round, so JS and Python agree on .5 cases
    return int(math.floor(v + 0.5))


def _fit_boxes(elements, old_size, new_size, scale):
    """
    Make every box safe for the canvas it will be compiled against, instead of letting
    one stray coordinate kill the run:
      1. scale: boxes are proportional to the canvas they were drawn on, so when the
         canvas is a different size (a linked Resolution Selector changed, say) they are
         scaled by the width and height ratios, i.e. the layout follows the aspect;
      2. clip: anything still overhanging is clamped to the canvas (logged).
    Only a box with nothing left on the canvas is an error. Malformed boxes are passed
    through untouched so the compiler can report them in its own words.
    """
    ow, oh = old_size
    nw, nh = new_size
    do_scale = bool(scale) and ow > 0 and oh > 0 and (ow, oh) != (nw, nh)
    out = []
    for i, el in enumerate(elements):
        b = el.get("bbox") if isinstance(el, dict) else None
        if not (isinstance(b, (list, tuple)) and len(b) == 4
                and all(type(v) in (int, float) and math.isfinite(v) for v in b)):
            out.append(el)
            continue
        x1, y1, x2, y2 = b
        if not (x1 < x2 and y1 < y2):
            out.append(el)          # zero/negative area: the compiler's message is clearer
            continue
        if do_scale:
            rx, ry = nw / ow, nh / oh
            x1, x2 = _half_up(x1 * rx), _half_up(x2 * rx)
            y1, y2 = _half_up(y1 * ry), _half_up(y2 * ry)
            if x2 <= x1:
                x2 = x1 + 1         # a very thin box must not vanish when scaled down
            if y2 <= y1:
                y2 = y1 + 1
        c = [min(max(x1, 0), nw), min(max(y1, 0), nh), min(max(x2, 0), nw), min(max(y2, 0), nh)]
        if not (c[0] < c[2] and c[1] < c[3]):
            raise ValueError(
                f"MOSS layout: items[{i}].bbox lies entirely outside the {nw}x{nh} canvas, "
                f"so there is nothing left to place (box was {list(b)})")
        if c != [x1, y1, x2, y2]:
            print(f"[BoyoBBOX] items[{i}].bbox {[x1, y1, x2, y2]} clipped to the {nw}x{nh} canvas -> {c}")
        out.append({**el, "bbox": c})
    return out


def build_moss_draft(moss, width, height):
    """
    Translate the editor's MOSS state into the public Layout draft schema.
    Editor-only fields (id, colour) never leave this function, empty optional
    fields are omitted (the compiler rejects empty strings/arrays), and key
    order matches the spec: label, caption, text, bbox, tags, palette.
    """
    sm = moss.get("summary") or {}
    summary = {"caption": str(sm.get("caption", ""))}
    meta = _clean_list(sm.get("meta"))
    if meta:
        summary["meta"] = meta
    if sm.get("canvas"):
        summary["canvas"] = sm["canvas"]
    palette = _clean_list(sm.get("palette"))
    if palette:
        summary["palette"] = palette

    items = []
    for el in moss.get("elements") or []:
        label = el.get("label", "object")
        item = {"label": label, "caption": str(el.get("desc", ""))}
        # Only text items carry literal text; stale text on a relabelled item is dropped.
        if label == "text" and str(el.get("text", "")).strip():
            item["text"] = str(el["text"])
        item["bbox"] = el.get("bbox", [])
        tags = _clean_list(el.get("tags"))
        if tags:
            item["tags"] = tags
        pal = _clean_list(el.get("palette"))
        if pal:
            item["palette"] = pal
        items.append(item)

    return {
        "width": width,
        "height": height,
        "bbox_coord_space": "abs",
        "bbox_format": "xyxy",
        "summary": summary,
        "items": items,
    }


class BoyoBBOX:
    """
    Boyo Bounding Box Spatial Prompt Editor — Flux 3 / MOSS.

    Flux 3 format: dual mode Generate (blank canvas, manual resolution) or Edit
    (image input, auto dims). Outputs a BFL prompt string + bbox JSON.

    MOSS format (generation only): flat pixel Layout draft compiled by the official
    MOSS compiler. Outputs the compiled one-line prompt, the editable draft JSON,
    and the canvas width/height. The image input is ignored.

    width_in / height_in (optional INT sockets, e.g. from a Resolution Selector) override
    the node's own size in either format when connected, so the size is chosen once.
    In Flux edit mode a connected image still wins, as before.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox_data": ("STRING", {"default": "{}"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "width_in": ("INT", {"forceInput": True}),
                "height_in": ("INT", {"forceInput": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("prompt", "bbox_json", "width", "height")
    FUNCTION = "process"
    CATEGORY = "BoyoNodes/Flux3"
    OUTPUT_NODE = False

    @classmethod
    def IS_CHANGED(cls, bbox_data="{}", width_in=None, height_in=None, **kwargs):
        try:
            data = json.loads(bbox_data)
        except Exception:
            data = {}
        if isinstance(data, dict) and data.get("format") == "moss":
            # Deterministic compile: only re-run (and invalidate downstream) when the draft changes.
            return hashlib.sha256(f"{bbox_data}|{width_in}|{height_in}".encode("utf-8")).hexdigest()
        return float("nan")

    def _process_moss(self, data, width_in=None, height_in=None, unique_id=None):
        moss = data.get("moss") or {}
        try:
            width = int(moss.get("width", 1024))
            height = int(moss.get("height", 1024))
        except (TypeError, ValueError):
            raise ValueError("MOSS layout: width and height must be whole numbers") from None

        editor_size = (width, height)   # the canvas the editor's boxes were drawn on

        # Connected size sockets win over the node's own selection
        if width_in is not None or height_in is not None:
            if width_in is not None:
                width = int(width_in)
            if height_in is not None:
                height = int(height_in)
            # People round their sizes to 8, 16, 32, 64...; MOSS needs multiples of 16, so round DOWN
            # (never more pixels than were asked for). The width/height outputs carry the rounded
            # size, so a latent fed from this node always matches the compiled canvas.
            raw = (width, height)
            width, height = _floor16(width), _floor16(height)
            if (width, height) != raw:
                print(f"[BoyoBBOX] linked size {raw[0]}x{raw[1]} rounded down to {width}x{height} "
                      f"(MOSS needs multiples of 16)")
            _announce_size(unique_id, width, height, raw)  # before compiling, so the editor syncs even on failure

        # Boxes follow the canvas (and are clipped to it) rather than killing the run
        scale = moss.get("scale_boxes", True) is not False
        moss = {**moss, "elements": _fit_boxes(moss.get("elements") or [], editor_size, (width, height), scale)}

        draft = build_moss_draft(moss, width, height)
        try:
            prompt = moss_compile_prompt(draft)
        except ValueError as e:
            raise ValueError(f"MOSS layout: {e}") from None

        draft_json = json.dumps(draft, ensure_ascii=False, indent=2)
        return (prompt, draft_json, width, height)

    def process(self, bbox_data, image=None, width_in=None, height_in=None, unique_id=None):
        try:
            data = json.loads(bbox_data)
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}

        # MOSS is generation-only: no image handling, no preview.
        if data.get("format") == "moss":
            return self._process_moss(data, width_in, height_in, unique_id)

        mode = data.get("mode", "generate")
        caption = data.get("caption", "")
        elements = data.get("elements", [])
        width = data.get("width", 1024)
        height = data.get("height", 1024)

        # Connected size sockets override the node's own size (an image in edit mode still wins, below)
        if width_in is not None:
            width = int(width_in)
        if height_in is not None:
            height = int(height_in)
        if width_in is not None or height_in is not None:
            _announce_size(unique_id, width, height)  # the linked size, before any image override

        # If image connected, save preview and use its dims
        if image is not None:
            height = image.shape[1]
            width = image.shape[2]
            # Save temp preview for the load button to fetch
            img_np = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            temp_dir = folder_paths.get_temp_directory()
            fpath = os.path.join(temp_dir, "boyo_bbox_preview.png")
            pil_img.save(fpath)

        # Build bbox rows
        rows = []
        for el in elements:
            rows.append({
                "id": el.get("id", "element_1"),
                "bbox": el.get("bbox", [0, 0, 1000, 1000]),
                "desc": el.get("desc", ""),
            })

        bbox_json_str = json.dumps(rows)
        prompt = f"{caption} {bbox_json_str}".strip()

        return (prompt, bbox_json_str, width, height)


NODE_CLASS_MAPPINGS = {
    "BoyoBBOX": BoyoBBOX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoBBOX": "Boyo BBOX Spatial Editor (Flux 3 / MOSS)",
}


# ── Custom API route — serves the cached preview image to JS ──────────────────
@PromptServer.instance.routes.get("/boyo_bbox/preview")
async def boyo_bbox_preview(request):
    temp_dir = folder_paths.get_temp_directory()
    fpath = os.path.join(temp_dir, "boyo_bbox_preview.png")
    if not os.path.exists(fpath):
        return web.Response(status=404, text="No preview available — run the node once with an image connected.")
    return web.FileResponse(fpath)


@PromptServer.instance.routes.get("/boyo_bbox/preview_info")
async def boyo_bbox_preview_info(request):
    """Returns image dimensions so JS can set canvas aspect before loading."""
    temp_dir = folder_paths.get_temp_directory()
    fpath = os.path.join(temp_dir, "boyo_bbox_preview.png")
    if not os.path.exists(fpath):
        return web.json_response({"exists": False})
    with Image.open(fpath) as img:
        w, h = img.size
    return web.json_response({"exists": True, "width": w, "height": h})
