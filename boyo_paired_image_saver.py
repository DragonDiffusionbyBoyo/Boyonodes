import os
import torch
import numpy as np
from PIL import Image
import folder_paths

class BoyoPairedImageSaver:
    def __init__(self):
        self.counters = {}  # Track counters per folder/prefix combination

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "controlnet_image": ("IMAGE",),
                "folder_name": ("STRING", {"default": "batch_output"}),
                "filename_prefix": ("STRING", {"default": "generated"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_paired_files"
    OUTPUT_NODE = True
    CATEGORY = "Boyonodes"

    def save_paired_files(self, original_image, controlnet_image, folder_name, filename_prefix):
        # Get the output directory from ComfyUI
        output_dir = folder_paths.get_output_directory()
        
        # Create the subfolder path
        save_dir = os.path.join(output_dir, folder_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Create a unique key for this folder/prefix combination
        counter_key = f"{folder_name}_{filename_prefix}"
        
        # Get or initialise the counter for this combination
        if counter_key not in self.counters:
            # Find the highest existing number in the directory
            existing_files = [f for f in os.listdir(save_dir) if f.startswith(filename_prefix) and f.endswith('.png')]
            if existing_files:
                # Extract numbers from existing files
                numbers = []
                for f in existing_files:
                    try:
                        # Remove prefix and .png, convert to int
                        num_str = f.split('.')[0][len(filename_prefix):]
                        numbers.append(int(num_str))
                    except ValueError:
                        continue
                self.counters[counter_key] = max(numbers) + 1 if numbers else 1
            else:
                self.counters[counter_key] = 1
        else:
            self.counters[counter_key] += 1
        
        # Generate the filename with zero-padded number
        file_number = f"{self.counters[counter_key]:03d}"
        base_filename = f"{filename_prefix}{file_number}"
        
        # Full paths for both images
        original_image_path = os.path.join(save_dir, f"{base_filename}_original.png")
        controlnet_image_path = os.path.join(save_dir, f"{base_filename}_controlnet.png")
        
        # Helper function to save a single image tensor
        def save_tensor_to_file(image_tensor, path):
            # Convert tensor to PIL Image
            i = 255. * image_tensor[0].cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img.save(path)

        # Save the original image
        save_tensor_to_file(original_image, original_image_path)
        
        # Save the ControlNet image
        save_tensor_to_file(controlnet_image, controlnet_image_path)
        
        print(f"Saved paired images: {os.path.basename(original_image_path)} and {os.path.basename(controlnet_image_path)} in {folder_name}/")
        
        return ()

NODE_CLASS_MAPPINGS = {
    "BoyoPairedImageSaver": BoyoPairedImageSaver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoPairedImageSaver": "Boyo Paired Image Saver"
}