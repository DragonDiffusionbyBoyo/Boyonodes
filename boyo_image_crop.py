import os
import torch
import numpy as np
from PIL import Image
import math

class BoyoImageCrop:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64}),
                "crop_height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64}),
                "overlap": ("INT", {"default": 128, "min": 0, "max": 512, "step": 32}),
                "output_path": ("STRING", {"default": "crops", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "crop_image"
    CATEGORY = "Boyo"

    def crop_image(self, image, crop_width, crop_height, overlap, output_path):
        # Convert tensor to PIL Image
        if len(image.shape) == 4:
            # Batch of images - take the first one
            image_tensor = image[0]
        else:
            image_tensor = image
            
        # Convert from torch tensor (0-1 range) to PIL Image
        image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        
        # Get image dimensions
        img_width, img_height = pil_image.size
        
        # Calculate step size (crop size minus overlap)
        step_x = crop_width - overlap
        step_y = crop_height - overlap
        
        # Calculate how many complete crops fit
        crops_x = math.floor((img_width - crop_width) / step_x) + 1 if img_width >= crop_width else 0
        crops_y = math.floor((img_height - crop_height) / step_y) + 1 if img_height >= crop_height else 0
        
        total_crops = crops_x * crops_y
        
        if total_crops == 0:
            return (f"Error: Image ({img_width}x{img_height}) is smaller than crop size ({crop_width}x{crop_height})",)
        
        # Create output directory in ComfyUI's output folder
        full_output_path = os.path.join("output", output_path)
        os.makedirs(full_output_path, exist_ok=True)
        
        # Generate crops
        crop_count = 0
        for y in range(crops_y):
            for x in range(crops_x):
                # Calculate crop coordinates
                start_x = x * step_x
                start_y = y * step_y
                end_x = start_x + crop_width
                end_y = start_y + crop_height
                
                # Ensure we don't exceed image bounds (shouldn't happen with our calculation, but safety first)
                if end_x <= img_width and end_y <= img_height:
                    # Crop the image
                    cropped = pil_image.crop((start_x, start_y, end_x, end_y))
                    
                    # Save with sequential numbering
                    crop_count += 1
                    filename = f"crop_{crop_count:03d}.png"
                    filepath = os.path.join(full_output_path, filename)
                    cropped.save(filepath, "PNG")
        
        status_message = f"Target image = {total_crops} crops"
        print(status_message)  # Console output
        
        return (status_message,)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "BoyoImageCrop": BoyoImageCrop
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoImageCrop": "Boyo Image Crop"
}
