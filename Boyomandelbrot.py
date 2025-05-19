import os
import subprocess
import folder_paths
import random
from PIL import Image
import numpy as np
import torch

class MandelbrotVideoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resolution": (["square", "widescreen", "vertical"], {"default": "square"}),
                "output_type": (["video", "frames"], {"default": "video"}),
                "grayscale": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "video_path")
    FUNCTION = "generate"
    CATEGORY = "video"

    @classmethod
    def IS_CHANGED(cls, resolution, output_type, grayscale):
        return True

    def load_images_as_tensors(self, frame_dir, grayscale=False):
        tensors = []
        expected_size = (512, 512)  # Adjust based on your resolution
        for fn in sorted(os.listdir(frame_dir)):
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                img_path = os.path.join(frame_dir, fn)
                try:
                    img = Image.open(img_path)
                    # Ensure the image is in the correct mode
                    if grayscale:
                        img = img.convert("L")
                        arr = np.array(img, dtype=np.float32) / 255.0
                        if arr.ndim == 1 or arr.shape != expected_size:
                            print(f"Warning: Reshaping image {fn} from {arr.shape} to {expected_size}")
                            arr = np.resize(arr, expected_size)
                        arr = np.stack([arr, arr, arr], axis=-1)  # Shape: (H, W, 3)
                    else:
                        img = img.convert("RGB")
                        arr = np.array(img, dtype=np.float32) / 255.0
                        if arr.shape[:2] != expected_size:
                            print(f"Warning: Resizing image {fn} from {arr.shape} to {expected_size+(3,)}")
                            img = img.resize(expected_size, Image.Resampling.LANCZOS)
                            arr = np.array(img, dtype=np.float32) / 255.0
                    t = torch.from_numpy(arr)  # Shape: (H, W, 3)
                    print(f"Loaded {fn} with shape: {t.shape}")
                    tensors.append(t)
                except Exception as e:
                    print(f"Error loading {fn}: {e}")
        if not tensors:
            raise ValueError(f"No valid images found in {frame_dir}")
        tensor_stack = torch.stack(tensors, dim=0)  # Shape: (B, H, W, 3)
        print(f"Final tensor shape: {tensor_stack.shape}")
        return tensor_stack

    def generate(self, resolution, output_type, grayscale):
        script = os.path.join(os.path.dirname(__file__), "mandelbrot_video.py")
        out_dir = folder_paths.get_output_directory()
        video_path = os.path.join(out_dir, f"mandelbrot_{random.randint(1000,9999)}.mp4")
        frames_dir = os.path.join(out_dir, "mandelbrot_frames")

        cmd = ["python", script, "--resolution", resolution, "--output", video_path]
        if output_type == "frames":
            cmd.append("--output_frames")
        if grayscale:
            cmd.append("--grayscale")

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"Script output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            raise Exception(f"Mandelbrot script failed:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")

        if output_type == "frames":
            if not os.path.isdir(frames_dir):
                raise Exception(f"Frame output directory not found: {frames_dir}")
            img_tensors = self.load_images_as_tensors(frames_dir, grayscale)
            return (img_tensors, video_path)

        return (torch.empty(0), video_path)

NODE_CLASS_MAPPINGS = {
    "MandelbrotVideo": MandelbrotVideoNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MandelbrotVideo": "Boyo Mandelbrot Generator"
}