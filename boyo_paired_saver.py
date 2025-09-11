import os
import torch
import numpy as np
from PIL import Image
import folder_paths

class BoyoPairedSaver:
    def __init__(self):
        self.counters = {}  # Track counters per folder/prefix combination

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "enhanced_prompt": ("STRING", {"forceInput": True}),
                "folder_name": ("STRING", {"default": "batch_output"}),
                "filename_prefix": ("STRING", {"default": "generated"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_paired_files"
    OUTPUT_NODE = True
    CATEGORY = "Boyonodes"

    def save_paired_files(self, image, enhanced_prompt, folder_name, filename_prefix):
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
                        num_str = f[len(filename_prefix):-4]
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
        image_filename = f"{filename_prefix}{file_number}.png"
        text_filename = f"{filename_prefix}{file_number}.txt"
        
        # Full paths
        image_path = os.path.join(save_dir, image_filename)
        text_path = os.path.join(save_dir, text_filename)
        
        # Save the image
        # Convert tensor to PIL Image
        i = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        img.save(image_path)
        
        # Save the enhanced prompt text
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(enhanced_prompt)
        
        print(f"Saved paired files: {image_filename} and {text_filename} in {folder_name}/")
        
        return ()

NODE_CLASS_MAPPINGS = {
    "BoyoPairedSaver": BoyoPairedSaver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoPairedSaver": "Boyo Paired Saver"
}