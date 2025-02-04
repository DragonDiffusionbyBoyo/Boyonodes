import os
import torch
import numpy as np
from PIL import Image

class BoyoSaver:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_path": ("STRING", {"default": "outputs"}),
                "filename_prefix": ("STRING", {"default": "Boyo"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "Boyonodes"

    def save_images(self, images, output_path, filename_prefix):
        results = list()

        # Ensure output_path is an absolute path
        output_path = os.path.abspath(output_path)

        # Create the output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        # Get the highest existing number suffix
        existing_files = [f for f in os.listdir(output_path) if f.startswith(filename_prefix) and f.endswith('.png')]
        highest_suffix = 0
        for file in existing_files:
            try:
                # Extract the numeric part of the filename and convert to int
                suffix = int(''.join(filter(str.isdigit, file.split('_')[-1])))
                highest_suffix = max(highest_suffix, suffix)
            except ValueError:
                pass

        for idx, image in enumerate(images):
            try:
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                
                # Generate a unique filename
                suffix = highest_suffix + idx + 1
                file = f"{filename_prefix}_{suffix:04}.png"
                save_path = os.path.join(output_path, file)

                # Save the image
                img.save(save_path)
                results.append({
                    "filename": file,
                    "subfolder": "",
                    "type": "output"
                })
                print(f"Successfully saved image: {save_path}")
            except Exception as e:
                print(f"Error processing or saving image {idx}: {str(e)}")

        return { "ui": { "images": results } }

NODE_CLASS_MAPPINGS = {
    "BoyoSaver": BoyoSaver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoSaver": "Boyo Saver"
}
