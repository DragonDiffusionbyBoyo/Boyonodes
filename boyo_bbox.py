import json
import numpy as np
import torch
from PIL import Image
import folder_paths
import os
from server import PromptServer
from aiohttp import web


class BoyoBBOX:
    """
    Boyo Bounding Box Spatial Prompt Editor for Flux 3.
    Dual mode: Generate (blank canvas, manual resolution) or Edit (image input, auto dims).
    Outputs a correctly formatted BFL prompt string + bbox JSON.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox_data": ("STRING", {"default": "{}"}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("prompt", "bbox_json", "width", "height")
    FUNCTION = "process"
    CATEGORY = "BoyoNodes/Flux3"
    OUTPUT_NODE = False

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def process(self, bbox_data, image=None):
        try:
            data = json.loads(bbox_data)
        except Exception:
            data = {}

        mode = data.get("mode", "generate")
        caption = data.get("caption", "")
        elements = data.get("elements", [])
        width = data.get("width", 1024)
        height = data.get("height", 1024)

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
    "BoyoBBOX": "Boyo BBOX Spatial Editor (Flux 3)",
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
