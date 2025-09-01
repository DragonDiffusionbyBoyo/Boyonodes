import os
import glob
import time
from PIL import Image, ImageOps
import torch
import numpy as np

class BoyoImageGrab:
    """
    A node that automatically grabs the most recently added image from a specified directory
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "placeholder": "Enter directory path..."
                }),
            },
            "optional": {
                "file_extensions": ("STRING", {
                    "default": "jpg,jpeg,png,bmp,tiff,webp",
                    "multiline": False,
                    "tooltip": "Comma-separated list of file extensions to search for"
                }),
                "auto_refresh": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically check for new files on each execution"
                }),
                "refresh_trigger": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999,
                    "tooltip": "Change this value to manually force refresh"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("image", "filename", "full_path", "timestamp")
    FUNCTION = "grab_latest_image"
    CATEGORY = "Boyo/loaders"
    
    # This prevents ComfyUI from caching the result
    @classmethod
    def IS_CHANGED(cls, directory_path, file_extensions="jpg,jpeg,png,bmp,tiff,webp", auto_refresh=True, refresh_trigger=0):
        """
        This method tells ComfyUI when the node should be re-executed.
        It returns a hash that changes when we want the node to refresh.
        """
        print(f"[BoyoImageGrab] IS_CHANGED called - auto_refresh: {auto_refresh}, refresh_trigger: {refresh_trigger}")
        
        if not auto_refresh and refresh_trigger == 0:
            print(f"[BoyoImageGrab] Auto-refresh disabled and no trigger, returning: {refresh_trigger}")
            return refresh_trigger
            
        if not directory_path or not os.path.exists(directory_path):
            print(f"[BoyoImageGrab] Directory missing or empty: '{directory_path}'")
            return f"missing_dir_{time.time()}"
        
        print(f"[BoyoImageGrab] Scanning directory: '{directory_path}'")
        
        # Parse file extensions
        extensions = [ext.strip().lower() for ext in file_extensions.split(',')]
        print(f"[BoyoImageGrab] Looking for extensions: {extensions}")
        
        # Build search patterns
        search_patterns = []
        for ext in extensions:
            if not ext.startswith('.'):
                ext = '.' + ext
            search_patterns.extend([
                os.path.join(directory_path, f"*{ext}"),
                os.path.join(directory_path, f"*{ext.upper()}")
            ])
        
        # Find all matching files
        all_files = []
        for pattern in search_patterns:
            found_files = glob.glob(pattern)
            all_files.extend(found_files)
            if found_files:
                print(f"[BoyoImageGrab] Pattern '{pattern}' found {len(found_files)} files")
        
        print(f"[BoyoImageGrab] Total files found: {len(all_files)}")
        
        if not all_files:
            print(f"[BoyoImageGrab] No files found, returning time-based hash")
            return f"no_files_{time.time()}"
        
        # Find the newest file and its modification time
        latest_file = max(all_files, key=os.path.getmtime)
        latest_mtime = max(os.path.getmtime(f) for f in all_files)
        
        print(f"[BoyoImageGrab] Latest file: '{latest_file}' (mtime: {latest_mtime})")
        
        # Return the modification time of the newest file as the hash
        hash_value = f"{latest_mtime}_{refresh_trigger}"
        print(f"[BoyoImageGrab] Returning hash: {hash_value}")
        return hash_value
    
    def grab_latest_image(self, directory_path, file_extensions="jpg,jpeg,png,bmp,tiff,webp", auto_refresh=True, refresh_trigger=0):
        """
        Find and load the most recently modified image file from the specified directory
        """
        
        if not directory_path or not os.path.exists(directory_path):
            raise ValueError(f"Directory path '{directory_path}' does not exist or is empty")
        
        # Parse file extensions
        extensions = [ext.strip().lower() for ext in file_extensions.split(',')]
        
        # Build search patterns
        search_patterns = []
        for ext in extensions:
            if not ext.startswith('.'):
                ext = '.' + ext
            search_patterns.extend([
                os.path.join(directory_path, f"*{ext}"),
                os.path.join(directory_path, f"*{ext.upper()}")
            ])
        
        # Find all matching files
        all_files = []
        for pattern in search_patterns:
            all_files.extend(glob.glob(pattern))
        
        if not all_files:
            raise ValueError(f"No image files found in '{directory_path}' with extensions: {extensions}")
        
        # Find the most recently modified file
        latest_file = max(all_files, key=os.path.getmtime)
        latest_mtime = os.path.getmtime(latest_file)
        
        # Load the image
        try:
            image = Image.open(latest_file)
            
            # Handle EXIF rotation
            image = ImageOps.exif_transpose(image)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to tensor format expected by ComfyUI
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]  # Add batch dimension
            
            filename = os.path.basename(latest_file)
            
            return (image_tensor, filename, latest_file, latest_mtime)
            
        except Exception as e:
            raise ValueError(f"Failed to load image '{latest_file}': {str(e)}")

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "BoyoImageGrab": BoyoImageGrab
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoImageGrab": "Boyo Image Grab"
}