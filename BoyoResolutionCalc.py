import torch
import comfy.utils

class BoyoResolutionCalc:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "aspect_ratio": (["1:1 Square", "4:3", "3:2", "16:9 HD", "21:9", "2:3", "3:4", "9:16 Tiktok-Shorts"], {"default": "1:1"}),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "calculate"
    CATEGORY = "Boyonodes"

    def calculate(self, width, aspect_ratio):
        aspect_ratios = {
            "1:1 Square": (1, 1),
            "4:3": (4, 3),
            "3:2": (3, 2),
            "16:9 HD": (16, 9),
            "21:9": (21, 9),
            "2:3": (2, 3),
            "3:4": (3, 4),
            "9:16 Tiktok-Shorts": (9, 16)
        }

        ratio_w, ratio_h = aspect_ratios[aspect_ratio]
        
        # Calculate height and ensure both width and height are divisible by 8
        width = (width + 7) // 8 * 8
        height = ((width * ratio_h) // ratio_w + 7) // 8 * 8

        # Recalculate width to maintain exact aspect ratio
        width = (height * ratio_w) // ratio_h
        width = (width + 7) // 8 * 8

        return (width, height)

NODE_CLASS_MAPPINGS = {
    "BoyoResolutionCalc": BoyoResolutionCalc
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoResolutionCalc": "Boyo Resolution Calculator"
}