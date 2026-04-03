import glob
import hashlib
import json
import os
import random

import folder_paths
import numpy as np
import torch
from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_EXT = (
    ".jpg", ".jpeg", ".png", ".webp",
    ".bmp", ".gif", ".tiff", ".tif",
)

# ---------------------------------------------------------------------------
# Simple persistent counter store
# Saves a JSON file alongside this module so incremental mode survives
# ComfyUI restarts without needing WAS's database.
# ---------------------------------------------------------------------------

_COUNTER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "boyo_batch_counters.json")


def _load_counters() -> dict:
    try:
        with open(_COUNTER_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_counters(data: dict) -> None:
    try:
        with open(_COUNTER_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a ComfyUI-style [1, H, W, C] float32 tensor."""
    arr = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_input_subfolders() -> list[str]:
    """Return sorted subfolder names inside ComfyUI's input directory."""
    input_dir = folder_paths.get_input_directory()
    try:
        folders = sorted(
            name for name in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, name))
        )
    except Exception:
        folders = []
    return folders if folders else ["[ no subfolders found ]"]


# ---------------------------------------------------------------------------
# BatchImageLoader
# ---------------------------------------------------------------------------

class _BatchImageLoader:
    def __init__(self, directory_path: str, label: str, pattern: str):
        self._counters = _load_counters()
        self.label = label
        self.image_paths: list[str] = []

        self._scan(directory_path, pattern)
        self.image_paths.sort()

        stored_path = self._counters.get(f"{label}__path")
        stored_pattern = self._counters.get(f"{label}__pattern")

        if stored_path != directory_path or stored_pattern != pattern:
            self.index = 0
            self._counters[f"{label}__path"] = directory_path
            self._counters[f"{label}__pattern"] = pattern
            self._counters[f"{label}__index"] = 0
            _save_counters(self._counters)
        else:
            self.index = int(self._counters.get(f"{label}__index", 0))

    def _scan(self, directory_path: str, pattern: str) -> None:
        for file_name in glob.glob(
            os.path.join(glob.escape(directory_path), pattern), recursive=True
        ):
            if file_name.lower().endswith(ALLOWED_EXT):
                self.image_paths.append(os.path.abspath(file_name))

    # ------------------------------------------------------------------

    def get_image_by_id(self, image_id: int):
        if not self.image_paths or image_id < 0 or image_id >= len(self.image_paths):
            print(f"[BoyoLoadImageBatch] Invalid index {image_id} "
                  f"(total images: {len(self.image_paths)})")
            return None, None
        img = Image.open(self.image_paths[image_id])
        img = ImageOps.exif_transpose(img)
        return img, os.path.basename(self.image_paths[image_id])

    def get_next_image(self):
        if not self.image_paths:
            return None, None
        if self.index >= len(self.image_paths):
            self.index = 0
        path = self.image_paths[self.index]
        self.index = (self.index + 1) % len(self.image_paths)
        self._counters[f"{self.label}__index"] = self.index
        _save_counters(self._counters)
        print(f"[BoyoLoadImageBatch] label={self.label!r} next index → {self.index}")
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        return img, os.path.basename(path)

    def get_current_image_path(self) -> str:
        if not self.image_paths:
            return ""
        idx = min(self.index, len(self.image_paths) - 1)
        return self.image_paths[idx]


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class BoyoLoadImageBatch:
    """
    Load images one-at-a-time from a subfolder inside ComfyUI's input
    directory.  The subfolder is chosen from a dropdown — no typing paths —
    which keeps things reliable on Linux / RunPod deployments.

    Behaviour is otherwise identical to WAS Load Image Batch:
      • single_image  – load by explicit index
      • incremental   – advance through the folder each execution
      • random        – pick a random image each execution
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["single_image", "incremental_image", "random"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "index": ("INT", {"default": 0, "min": 0, "max": 150000, "step": 1}),
                "label": ("STRING", {"default": "Batch 001", "multiline": False}),
                "subfolder": (_get_input_subfolders(),),
                "pattern": ("STRING", {"default": "*", "multiline": False}),
                "allow_RGBA_output": (["false", "true"],),
            },
            "optional": {
                "filename_text_extension": (["true", "false"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "filename_text")
    FUNCTION = "load_batch_images"
    CATEGORY = "Boyo Nodes/IO"

    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_path(subfolder: str) -> str:
        return os.path.join(folder_paths.get_input_directory(), subfolder)

    # ------------------------------------------------------------------

    def load_batch_images(
        self,
        subfolder: str,
        mode: str = "single_image",
        seed: int = 0,
        index: int = 0,
        label: str = "Batch 001",
        pattern: str = "*",
        allow_RGBA_output: str = "false",
        filename_text_extension: str = "true",
    ):
        path = self._resolve_path(subfolder)

        if not os.path.isdir(path):
            print(f"[BoyoLoadImageBatch] Subfolder does not exist: {path!r}")
            return (None, "")

        loader = _BatchImageLoader(path, label, pattern)

        if not loader.image_paths:
            print(f"[BoyoLoadImageBatch] No images matched pattern {pattern!r} in {path!r}")
            return (None, "")

        if mode == "single_image":
            image, filename = loader.get_image_by_id(index)
        elif mode == "incremental_image":
            image, filename = loader.get_next_image()
        else:  # random
            random.seed(seed)
            image, filename = loader.get_image_by_id(
                int(random.random() * len(loader.image_paths))
            )

        if image is None:
            return (None, "")

        if allow_RGBA_output != "true":
            image = image.convert("RGB")

        if filename_text_extension == "false":
            filename = os.path.splitext(filename)[0]

        return (_pil_to_tensor(image), filename)

    # ------------------------------------------------------------------

    @classmethod
    def IS_CHANGED(cls, subfolder, mode, label, pattern, index, **kwargs):
        path = cls._resolve_path(subfolder)
        if mode != "single_image":
            return float("NaN")
        loader = _BatchImageLoader(path, label, pattern)
        current = loader.get_current_image_path()
        if not current or not os.path.isfile(current):
            return ""
        return _sha256(current)


# ---------------------------------------------------------------------------
# Registrations
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "BoyoLoadImageBatch": BoyoLoadImageBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoLoadImageBatch": "Boyo Load Image Batch (Input Folder)",
}
