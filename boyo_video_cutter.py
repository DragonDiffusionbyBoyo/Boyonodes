import os
import subprocess
import tempfile
import torch
import numpy as np
from PIL import Image

class BoyoVideoCutter:
    """
    Cuts specific frame ranges from video to remove overlap sections.
    Takes frame list from BoyoVideoLengthCalculator and removes those frames.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "trim_positions": ("STRING", {"default": ""}),
            },
            "optional": {
                "debug_mode": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("trimmed_images", "cut_info", "frames_removed")
    FUNCTION = "cut_video_frames"
    CATEGORY = "Boyo/Video"
    
    def cut_video_frames(self, images, trim_positions, debug_mode=True):
        """
        Remove specific frames from image sequence.
        trim_positions format: "94,95,96|188,189,190" (ranges separated by |)
        Cuts in reverse order to avoid moving target issues.
        """
        
        if not trim_positions or trim_positions.strip() == "":
            # No trimming needed
            return (images, "No frames to trim", 0)
        
        # Parse trim positions
        frames_to_remove = set()
        trim_sections = trim_positions.split("|")
        
        for section in trim_sections:
            if section.strip():
                frame_numbers = [int(x.strip()) for x in section.split(",") if x.strip()]
                frames_to_remove.update(frame_numbers)
        
        if not frames_to_remove:
            return (images, "No valid frame numbers to trim", 0)
        
        # Convert to list and sort in DESCENDING order for backwards cutting
        frames_to_remove = sorted(list(frames_to_remove), reverse=True)
        
        # Get image dimensions
        batch_size, height, width, channels = images.shape
        total_frames = batch_size
        
        # Validate frame numbers
        valid_frames_to_remove = [f for f in frames_to_remove if 0 <= f < total_frames]
        invalid_frames = [f for f in frames_to_remove if f not in valid_frames_to_remove]
        
        if debug_mode:
            print(f"[Boyo] BoyoVideoCutter: Total input frames: {total_frames}")
            print(f"[Boyo] BoyoVideoCutter: Cutting in reverse order: {valid_frames_to_remove}")
            if invalid_frames:
                print(f"[Boyo] BoyoVideoCutter: Invalid frame numbers (skipped): {invalid_frames}")
        
        if not valid_frames_to_remove:
            return (images, "No valid frames to remove within video range", 0)
        
        # Start with original images and cut backwards one frame at a time
        current_images = images
        frames_removed_count = 0
        
        for frame_idx in valid_frames_to_remove:
            # Adjust frame index for the current tensor size
            current_frame_count = current_images.shape[0]
            if frame_idx < current_frame_count:
                # Remove this specific frame
                keep_mask = torch.ones(current_frame_count, dtype=torch.bool)
                keep_mask[frame_idx] = False
                current_images = current_images[keep_mask]
                frames_removed_count += 1
                
                if debug_mode:
                    print(f"[Boyo] BoyoVideoCutter: Removed frame {frame_idx}, tensor now {current_images.shape[0]} frames")
        
        final_frame_count = current_images.shape[0]
        cut_info = f"Removed {frames_removed_count} frames. Input: {total_frames} → Output: {final_frame_count}"
        
        if debug_mode:
            print(f"[Boyo] BoyoVideoCutter: {cut_info}")
        
        return (current_images, cut_info, frames_removed_count)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "BoyoVideoCutter": BoyoVideoCutter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoVideoCutter": "Boyo Video Cutter"
}
