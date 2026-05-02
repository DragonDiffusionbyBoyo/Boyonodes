import os
import torch
import numpy as np
import colour

# Find the luts folder relative to this node file
NODE_DIR = os.path.dirname(os.path.abspath(__file__))
LUT_DIR = os.path.join(NODE_DIR, "luts")
SUPPORTED_EXTS = (".cube", ".3dl", ".spi3d")


def scan_luts():
    """Scan the luts/ folder and return a sorted list of filenames."""
    if not os.path.isdir(LUT_DIR):
        os.makedirs(LUT_DIR, exist_ok=True)
        return ["[No LUTs found — add .cube/.3dl files to luts/ folder]"]
    
    found = sorted([
        f for f in os.listdir(LUT_DIR)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    ])
    
    return found if found else ["[No LUTs found — add .cube/.3dl files to luts/ folder]"]


class BoyoApplyLUT:
    
    CATEGORY = "BoyoNodes/Colour"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_lut"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "lut_file": (scan_luts(),),          # dropdown, rescans on each load
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "interpolation": (["tetrahedral", "trilinear", "linear"],),
                "clamp": ("BOOLEAN", {"default": True}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force rescan when the luts folder modification time changes
        if os.path.isdir(LUT_DIR):
            return os.path.getmtime(LUT_DIR)
        return 0

    def apply_lut(self, image, lut_file, strength, interpolation, clamp):
        if lut_file.startswith("[No LUTs"):
            raise ValueError("BoyoLUT: No LUT files found in luts/ folder.")
    
        lut_path = os.path.join(LUT_DIR, lut_file)
        lut = colour.io.read_LUT(lut_path)
    
        # Map string selection to actual colour-science interpolator functions
        interpolator_map = {
            "tetrahedral": colour.algebra.table_interpolation_tetrahedral,
            "trilinear":   colour.algebra.table_interpolation_trilinear,
            "linear":      colour.algebra.table_interpolation_trilinear,  # fallback
        }
        interpolator = interpolator_map.get(interpolation, colour.algebra.table_interpolation_tetrahedral)
    
        batch = image.cpu().numpy()
        results = []
    
        for frame in batch:
            graded = lut.apply(frame, interpolator=interpolator)
            if clamp:
                graded = np.clip(graded, 0.0, 1.0)
            blended = frame * (1.0 - strength) + graded * strength
            results.append(blended)
    
        out = np.stack(results, axis=0).astype(np.float32)
        return (torch.from_numpy(out),)


NODE_CLASS_MAPPINGS = {
    "BoyoApplyLUT": BoyoApplyLUT,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoApplyLUT": "Boyo Apply LUT",
}