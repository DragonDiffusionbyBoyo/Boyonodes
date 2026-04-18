"""
BoyoWatermarks - Quick watermark node for ComfyUI
Stamps a watermark image onto every frame of a video batch.
Dragon Diffusion UK Ltd
"""

import torch
import torch.nn.functional as F


class BoyoWatermarks:
    """
    Stamps a watermark image onto a video frame batch.
    Watermark is resized to the specified dimensions and placed in the chosen corner.
    """

    CORNER_OPTIONS = ["bottom-right", "bottom-left", "top-right", "top-left"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames":     ("IMAGE",  {"tooltip": "Video frames batch (B, H, W, C)"}),
                "watermark":  ("IMAGE",  {"tooltip": "Watermark image (single frame)"}),
                "wm_width":   ("INT",    {"default": 150, "min": 8, "max": 2048, "step": 1}),
                "wm_height":  ("INT",    {"default": 150, "min": 8, "max": 2048, "step": 1}),
                "corner":     (cls.CORNER_OPTIONS, {"default": "bottom-right"}),
                "opacity":    ("FLOAT",  {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05}),
                "padding":    ("INT",    {"default": 10, "min": 0, "max": 256, "step": 1,
                                          "tooltip": "Pixel gap between watermark and frame edge"}),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("frames",)
    FUNCTION      = "apply_watermark"
    CATEGORY      = "BoyoNodes/video"

    # ------------------------------------------------------------------ #

    def apply_watermark(self, frames, watermark, wm_width, wm_height,
                        corner, opacity, padding):
        """
        frames    : (B, H, W, C)  float32 0-1
        watermark : (1, H, W, C)  float32 0-1  – first frame used if batch passed
        returns   : (B, H, W, C)  float32 0-1
        """

        # --- grab first WM frame, normalise to (1, C, H, W) for F.interpolate ---
        wm = watermark[0:1]                          # (1, H, W, C)
        wm = wm.permute(0, 3, 1, 2)                 # (1, C, H, W)

        # --- resize watermark ---
        wm_resized = F.interpolate(
            wm,
            size=(wm_height, wm_width),
            mode="bilinear",
            align_corners=False,
        )                                            # (1, C, wm_h, wm_w)

        wm_resized = wm_resized.squeeze(0)           # (C, wm_h, wm_w)

        # --- handle alpha channel if present ---
        if wm_resized.shape[0] == 4:
            wm_alpha = wm_resized[3:4] * opacity     # (1, wm_h, wm_w)
            wm_rgb   = wm_resized[:3]                # (3, wm_h, wm_w)
        else:
            wm_alpha = torch.ones(
                1, wm_height, wm_width,
                device=wm_resized.device,
                dtype=wm_resized.dtype
            ) * opacity
            wm_rgb   = wm_resized[:3]

        # --- compute corner position ---
        frame_h = frames.shape[1]
        frame_w = frames.shape[2]

        if "top" in corner:
            y0 = padding
        else:
            y0 = frame_h - wm_height - padding

        if "left" in corner:
            x0 = padding
        else:
            x0 = frame_w - wm_width - padding

        y1 = y0 + wm_height
        x1 = x0 + wm_width

        # --- clamp to frame bounds (safety net for tiny frames / large WMs) ---
        y0 = max(0, y0)
        x0 = max(0, x0)
        y1 = min(frame_h, y1)
        x1 = min(frame_w, x1)

        actual_h = y1 - y0
        actual_w = x1 - x0

        # Re-crop WM if we had to clamp
        wm_rgb   = wm_rgb[:, :actual_h, :actual_w]
        wm_alpha = wm_alpha[:, :actual_h, :actual_w]

        # --- stamp onto every frame ---
        # Work on a copy so we don't trash the input tensor
        out = frames.clone()                         # (B, H, W, C)

        # Slice is (B, actual_h, actual_w, C)
        region = out[:, y0:y1, x0:x1, :3]           # only RGB channels

        # wm_rgb/alpha are (C, h, w) → need (1, h, w, C) to broadcast
        wm_rgb_b   = wm_rgb.permute(1, 2, 0).unsqueeze(0)    # (1, h, w, 3)
        wm_alpha_b = wm_alpha.permute(1, 2, 0).unsqueeze(0)  # (1, h, w, 1)

        blended = region * (1.0 - wm_alpha_b) + wm_rgb_b * wm_alpha_b

        out[:, y0:y1, x0:x1, :3] = blended

        return (out,)


# ------------------------------------------------------------------ #
# Registration
# ------------------------------------------------------------------ #

NODE_CLASS_MAPPINGS = {
    "BoyoWatermarks": BoyoWatermarks,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoWatermarks": "Boyo Watermarks 🐉",
}
