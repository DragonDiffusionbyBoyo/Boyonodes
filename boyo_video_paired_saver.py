import os
import torch
import numpy as np
from PIL import Image
import folder_paths
import subprocess
import tempfile
import shutil

class BoyoVideoPairedSaver:
    def __init__(self):
        self.counters = {}  # Track counters per folder/prefix combination

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),  # Batch of images that form the video
                "enhanced_prompt": ("STRING", {"forceInput": True}),
                "folder_name": ("STRING", {"default": "video_batch_output"}),
                "filename_prefix": ("STRING", {"default": "video_gen"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 0.1}),
                "codec": (["libx264", "libx265", "av1"], {"default": "libx264"}),
                "quality": (["high", "medium", "low"], {"default": "high"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_video_and_prompt"
    OUTPUT_NODE = True
    CATEGORY = "Boyonodes"

    def save_video_and_prompt(self, images, enhanced_prompt, folder_name, filename_prefix, fps, codec, quality):
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
            existing_files = [f for f in os.listdir(save_dir) if f.startswith(filename_prefix) and f.endswith('.mp4')]
            if existing_files:
                # Extract numbers from existing files
                numbers = []
                for f in existing_files:
                    try:
                        # Remove prefix and .mp4, convert to int
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
        video_filename = f"{filename_prefix}{file_number}.mp4"
        text_filename = f"{filename_prefix}{file_number}.txt"
        
        # Full paths
        video_path = os.path.join(save_dir, video_filename)
        text_path = os.path.join(save_dir, text_filename)
        
        # Create temporary directory for frames
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Save frames to temporary directory
            batch_size = images.shape[0]
            print(f"Processing {batch_size} frames for video...")
            
            for i in range(batch_size):
                # Convert tensor to PIL Image
                frame_data = 255. * images[i].cpu().numpy()
                frame_img = Image.fromarray(np.clip(frame_data, 0, 255).astype(np.uint8))
                
                # Save frame with zero-padded filename
                frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                frame_img.save(frame_path)
            
            # Set quality parameters based on quality setting
            if quality == "high":
                crf = "18"
                preset = "slow"
            elif quality == "medium":
                crf = "23"
                preset = "medium"
            else:  # low
                crf = "28"
                preset = "fast"
            
            # Build ffmpeg command
            input_pattern = os.path.join(temp_dir, "frame_%06d.png")
            
            cmd = [
                "ffmpeg", "-y",  # Overwrite output file
                "-framerate", str(fps),
                "-i", input_pattern,
                "-c:v", codec,
                "-preset", preset,
                "-crf", crf,
                "-pix_fmt", "yuv420p",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # Ensure even dimensions
                video_path
            ]
            
            # Run ffmpeg
            print(f"Creating video with ffmpeg: {codec} codec, {fps} fps, {quality} quality...")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Save the enhanced prompt text
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(enhanced_prompt)
            
            print(f"Saved video and prompt: {video_filename} and {text_filename} in {folder_name}/")
            print(f"Video details: {batch_size} frames, {fps} fps, {codec} codec")
            
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e}")
            print(f"FFmpeg stderr: {e.stderr}")
            raise Exception(f"Failed to create video: {e.stderr}")
            
        except Exception as e:
            print(f"Error creating video: {e}")
            raise
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return ()


class BoyoVideoSaver:
    """
    Simple video saver without prompt pairing - just saves the video
    """
    def __init__(self):
        self.counters = {}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "folder_name": ("STRING", {"default": "video_output"}),
                "filename_prefix": ("STRING", {"default": "video"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 0.1}),
                "codec": (["libx264", "libx265", "av1"], {"default": "libx264"}),
                "quality": (["high", "medium", "low"], {"default": "high"}),
            }
        }

    RETURN_TYPES = ("STRING",)  # Return the video path
    RETURN_NAMES = ("video_path",)
    FUNCTION = "save_video"
    OUTPUT_NODE = True
    CATEGORY = "Boyonodes"

    def save_video(self, images, folder_name, filename_prefix, fps, codec, quality):
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
            existing_files = [f for f in os.listdir(save_dir) if f.startswith(filename_prefix) and f.endswith('.mp4')]
            if existing_files:
                # Extract numbers from existing files
                numbers = []
                for f in existing_files:
                    try:
                        # Remove prefix and .mp4, convert to int
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
        video_filename = f"{filename_prefix}{file_number}.mp4"
        video_path = os.path.join(save_dir, video_filename)
        
        # Create temporary directory for frames
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Save frames to temporary directory
            batch_size = images.shape[0]
            print(f"Processing {batch_size} frames for video...")
            
            for i in range(batch_size):
                # Convert tensor to PIL Image
                frame_data = 255. * images[i].cpu().numpy()
                frame_img = Image.fromarray(np.clip(frame_data, 0, 255).astype(np.uint8))
                
                # Save frame with zero-padded filename
                frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                frame_img.save(frame_path)
            
            # Set quality parameters based on quality setting
            if quality == "high":
                crf = "18"
                preset = "slow"
            elif quality == "medium":
                crf = "23"
                preset = "medium"
            else:  # low
                crf = "28"
                preset = "fast"
            
            # Build ffmpeg command
            input_pattern = os.path.join(temp_dir, "frame_%06d.png")
            
            cmd = [
                "ffmpeg", "-y",  # Overwrite output file
                "-framerate", str(fps),
                "-i", input_pattern,
                "-c:v", codec,
                "-preset", preset,
                "-crf", crf,
                "-pix_fmt", "yuv420p",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # Ensure even dimensions
                video_path
            ]
            
            # Run ffmpeg
            print(f"Creating video with ffmpeg: {codec} codec, {fps} fps, {quality} quality...")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            print(f"Saved video: {video_filename} in {folder_name}/")
            print(f"Video details: {batch_size} frames, {fps} fps, {codec} codec")
            
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e}")
            print(f"FFmpeg stderr: {e.stderr}")
            raise Exception(f"Failed to create video: {e.stderr}")
            
        except Exception as e:
            print(f"Error creating video: {e}")
            raise
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return (video_path,)


NODE_CLASS_MAPPINGS = {
    "BoyoVideoPairedSaver": BoyoVideoPairedSaver,
    "BoyoVideoSaver": BoyoVideoSaver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoVideoPairedSaver": "Boyo Video Paired Saver",
    "BoyoVideoSaver": "Boyo Video Saver"
}