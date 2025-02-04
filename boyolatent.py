import torch
import comfy.utils

class Boyolatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "aspect_ratio": (["1:1 Square", "4:3", "3:2", "16:9 HD", "21:9", "2:3", "3:4", "9:16 Tiktok-Shorts"], {"default": "1:1"}),
                "init_type": (["zeros", "normal"], {"default": "normal"}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "Boyonodes"  # Changed from "latent" to "Boyonodes"

    def generate(self, width, batch_size, aspect_ratio, init_type):
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

        if init_type == "zeros":
            latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        else:  # normal distribution
            latent = torch.randn([batch_size, 4, height // 8, width // 8])

        return ({"samples": latent},)

NODE_CLASS_MAPPINGS = {
    "Boyolatent": Boyolatent
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Boyolatent": "Boyolatent (Empty Latent)"
}
