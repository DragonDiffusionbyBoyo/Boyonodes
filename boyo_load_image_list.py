import os
import torch
from PIL import Image
import numpy as np

class BoyoLoadImageList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": "input_images"}),
                "target_width": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
                "target_height": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
                "resize_method": (["resize", "crop", "pad"], {"default": "resize"}),
                "pad_color": ("STRING", {"default": "black"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "filenames")
    FUNCTION = "load_images"

    CATEGORY = "Boyonodes"

    def load_images(self, directory, target_width, target_height, resize_method, pad_color):
        images = []
        filenames = []

        if not os.path.exists(directory):
            raise ValueError(f"Directory does not exist: {directory}")

        for filename in sorted(os.listdir(directory)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                img_path = os.path.join(directory, filename)
                try:
                    img = Image.open(img_path)
                    img = img.convert('RGB')

                    if resize_method == "resize":
                        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                    elif resize_method == "crop":
                        img_ratio = img.width / img.height
                        target_ratio = target_width / target_height
                        if img_ratio > target_ratio:
                            # Image is wider, crop the sides
                            new_width = int(img.height * target_ratio)
                            left = (img.width - new_width) // 2
                            img = img.crop((left, 0, left + new_width, img.height))
                        else:
                            # Image is taller, crop the top and bottom
                            new_height = int(img.width / target_ratio)
                            top = (img.height - new_height) // 2
                            img = img.crop((0, top, img.width, top + new_height))
                        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                    elif resize_method == "pad":
                        img_ratio = img.width / img.height
                        target_ratio = target_width / target_height
                        if img_ratio > target_ratio:
                            # Image is wider, pad the top and bottom
                            new_height = int(img.width / target_ratio)
                            pad_color_rgb = Image.new('RGB', (1, 1), pad_color).getpixel((0, 0))
                            padded_img = Image.new('RGB', (img.width, new_height), pad_color_rgb)
                            paste_y = (new_height - img.height) // 2
                            padded_img.paste(img, (0, paste_y))
                            img = padded_img
                        else:
                            # Image is taller, pad the sides
                            new_width = int(img.height * target_ratio)
                            pad_color_rgb = Image.new('RGB', (1, 1), pad_color).getpixel((0, 0))
                            padded_img = Image.new('RGB', (new_width, img.height), pad_color_rgb)
                            paste_x = (new_width - img.width) // 2
                            padded_img.paste(img, (paste_x, 0))
                            img = padded_img
                        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

                    # Convert the image to a tensor
                    img = np.array(img).astype(np.float32) / 255.0
                    img = torch.from_numpy(img)[None,]
                    images.append(img)
                    filenames.append(filename)
                except Exception as e:
                    print(f"Error loading image {filename}: {str(e)}")

        if not images:
            raise ValueError(f"No valid images found in directory: {directory}")

        # Concatenate all images into a single tensor
        images_tensor = torch.cat(images, dim=0)
        filenames_str = ",".join(filenames)

        return (images_tensor, filenames_str)

NODE_CLASS_MAPPINGS = {
    "BoyoLoadImageList": BoyoLoadImageList
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoLoadImageList": "Boyo Load Image List"
}
