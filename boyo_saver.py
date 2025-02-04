import os
import torch
import numpy as np
from PIL import Image

class BoyoSaver:
    def __init__(self):
        self.counter = 0

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "upscaled_image": ("IMAGE",),
                "prefix": ("STRING", {"default": "boyo"}),
                "directory": ("STRING", {"default": "output"})
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Boyonodes"

    def save_images(self, original_image, upscaled_image, prefix, directory):
        self.counter += 1
        os.makedirs(directory, exist_ok=True)

        def process_and_save_image(image, filename):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img.save(os.path.join(directory, filename))

        # Save original image
        original_filename = f"{prefix}{self.counter:03d}A.png"
        process_and_save_image(original_image[0], original_filename)

        # Save upscaled image
        upscaled_filename = f"{prefix}{self.counter:03d}B.png"
        process_and_save_image(upscaled_image[0], upscaled_filename)

        print(f"Saved images: {original_filename} and {upscaled_filename}")
        return ()

NODE_CLASS_MAPPINGS = {
    "BoyoSaver": BoyoSaver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoSaver": "Boyo Saver"
}
