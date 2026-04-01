import os
import glob
import random
import cv2
import torch
import numpy as np
import psutil
import itertools
from typing import Optional, Tuple, Dict, Any, Generator

# BoyoLoadVideoDirectory - ported from DJZ LoadVideoDirectoryV2
# Original by Drift Johnson (opensourced), audio fix + BoyoNodes integration by Broken Boyo

ALLOWED_VIDEO_EXT = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv')

def boyo_load_audio(video_path: str, start_time: float, duration: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Load audio data from video file"""
    try:
        import ffmpeg
        import numpy as np
        
        # Probe for audio stream info
        try:
            probe = ffmpeg.probe(video_path)
            audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
            
            if not audio_stream:
                print(f"[BoyoLoadVideoDirectory] No audio stream found in {video_path}")
                return None
                
        except ffmpeg.Error as e:
            print(f"[BoyoLoadVideoDirectory] FFmpeg probe error: {e.stderr.decode()}")
            return None
        
        try:
            # Setup ffmpeg input with seeking
            stream = ffmpeg.input(video_path, ss=start_time)

            # Extract .audio FIRST, then apply atrim to the audio stream only
            audio = stream.audio
            if duration:
                audio = audio.filter('atrim', duration=duration)
                audio = audio.filter('asetpts', 'PTS-STARTPTS')  # Reset timestamps after trim
            
            # Get audio as 32-bit float PCM
            process = (audio
                      .output('pipe:', format='f32le', acodec='pcm_f32le', ac=2, ar=44100)
                      .overwrite_output()
                      .run_async(pipe_stdout=True, pipe_stderr=True))
            
            out, err = process.communicate()
            
            if process.returncode != 0:
                print(f"[BoyoLoadVideoDirectory] FFmpeg extraction error: {err.decode()}")
                return None
            
            # Convert to numpy array then to tensor
            audio_data = np.frombuffer(out, np.float32).reshape(-1, 2)
            audio_tensor = torch.from_numpy(audio_data).t()  # Transpose to [channels, samples]
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension [1, channels, samples]
            
            return {
                "path": video_path,
                "waveform": audio_tensor,
                "sample_rate": 44100,
                "start_time": start_time,
                "duration": duration,
                "stream_info": audio_stream
            }
            
        except ffmpeg.Error as e:
            print(f"[BoyoLoadVideoDirectory] FFmpeg extraction error: {err.decode()}")
            return None
            
    except Exception as e:
        print(f"[BoyoLoadVideoDirectory] Audio extraction failed: {str(e)}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"[BoyoLoadVideoDirectory] FFmpeg stderr: {e.stderr.decode()}")
    return None


def boyo_frame_generator(video_path: str, start_frame: int, frames_to_read: int, fps: float) -> Generator[torch.Tensor, None, None]:
    """Generator function that yields frames from a video"""
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Get video info first
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        target_frame_time = 1/fps if fps > 0 else 0
        
        # Initial yield of video information
        yield (width, height, fps, duration, total_frames, target_frame_time, frames_to_read)
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames_read = 0
        
        while frames_read < frames_to_read:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_float = frame_rgb.astype(np.float32) / 255.0
            frame_tensor = torch.from_numpy(frame_float)
            yield frame_tensor
            frames_read += 1
            
    finally:
        cap.release()


class BoyoLoadVideoDirectory:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["single_video", "incremental_video", "random"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "index": ("INT", {"default": 0, "min": 0, "max": 150000, "step": 1}),
                "skip_frames": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1}),
                "memory_limit_mb": ("INT", {"default": 0, "min": 0, "max": 128000, "step": 64}),
                "force_rate": ("INT", {"default": 0, "min": 0, "max": 60, "step": 1}),
                "label": ("STRING", {"default": 'Boyo Video Batch 001', "multiline": False}),
                "path": ("STRING", {"default": '', "multiline": False}),
                "pattern": ("STRING", {"default": '*', "multiline": False}),
            },
            "optional": {
                "meta_batch": ("VHS_BatchManager",),
                "vae": ("VAE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "AUDIO", "VHS_VIDEOINFO", "STRING")
    RETURN_NAMES = ("frames", "frame_count", "audio", "video_info", "filename_text")
    FUNCTION = "load_video_directory"
    CATEGORY = "BoyoNodes/video"

    def reset(self):
        """Reset the node state"""
        if hasattr(self, 'inputs'):
            self.close_inputs()
        self.inputs = {}
        
    def close_inputs(self):
        """Close any open inputs"""
        if not hasattr(self, 'inputs'):
            return
        for key in list(self.inputs.keys()):
            if isinstance(self.inputs[key], (list, tuple)) and len(self.inputs[key]) > 0:
                if hasattr(self.inputs[key][0], 'close'):
                    self.inputs[key][0].close()
            self.inputs.pop(key, None)

    def load_video_directory(self, path: str, pattern: str = '*', index: int = 0,
                             skip_frames: int = 0, max_frames: int = 0, mode: str = "single_video",
                             seed: int = 0, label: str = 'Boyo Video Batch 001', force_rate: int = 0,
                             meta_batch=None, unique_id=None, memory_limit_mb: int = 0,
                             vae=None) -> Tuple[torch.Tensor, int, Optional[Dict], Dict, str]:

        if not hasattr(self, 'inputs'):
            self.inputs = {}

        if not os.path.exists(path):
            raise ValueError(f"[BoyoLoadVideoDirectory] Path does not exist: {path}")

        vl = self.BoyoVideoDirectoryLoader(path, pattern)

        if len(vl.video_paths) == 0:
            raise ValueError(f"[BoyoLoadVideoDirectory] No videos found at {path} with pattern '{pattern}'")

        # Handle video selection based on mode
        if mode == 'single_video':
            video_id = index
        elif mode == 'incremental_video':
            video_id = index % len(vl.video_paths)
        else:  # random
            random.seed(seed)
            video_id = int(random.random() * len(vl.video_paths))

        if video_id < 0 or video_id >= len(vl.video_paths):
            raise ValueError(f"[BoyoLoadVideoDirectory] Invalid video index: {video_id} (found {len(vl.video_paths)} videos)")

        video_path = vl.video_paths[video_id]
        filename = os.path.basename(video_path)

        # Initialize or retrieve meta batch state
        if meta_batch is None or unique_id not in meta_batch.inputs:
            self.reset()
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"[BoyoLoadVideoDirectory] Could not open video: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS) if force_rate == 0 else force_rate
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            cap.release()

            start_frame = min(skip_frames, total_frames - 1) if skip_frames > 0 else 0
            remaining_frames = total_frames - start_frame
            frames_to_read = remaining_frames if max_frames == 0 else min(remaining_frames, max_frames)

            meta_state = {
                "video_path": video_path,
                "current_frame": start_frame,
                "frames_to_read": frames_to_read,
                "total_frames_read": 0,
                "width": width,
                "height": height,
                "fps": fps,
                "duration": duration,
                "total_frames": total_frames,
                "target_frame_time": 1/fps if fps > 0 else 0,
            }

            if meta_batch is not None:
                meta_batch.inputs[unique_id] = meta_state
                if frames_to_read:
                    meta_batch.total_frames = min(meta_batch.total_frames, frames_to_read)
                self.inputs[unique_id] = [boyo_frame_generator(video_path, start_frame, frames_to_read, fps)]
        else:
            meta_state = meta_batch.inputs[unique_id]

        # Memory limit calculation
        if memory_limit_mb > 0:
            memory_limit = memory_limit_mb * 2**20
        else:
            try:
                memory_limit = (psutil.virtual_memory().available + psutil.swap_memory().free) - 2**27
            except:
                print("[BoyoLoadVideoDirectory] Failed to calculate available memory. Limit disabled.")
                memory_limit = None

        if memory_limit is not None:
            bytes_per_frame = meta_state["width"] * meta_state["height"] * 3 * 4
            max_loadable_frames = memory_limit // bytes_per_frame

            if meta_batch is not None:
                if meta_batch.frames_per_batch > max_loadable_frames:
                    raise RuntimeError(
                        f"[BoyoLoadVideoDirectory] Meta Batch set to {meta_batch.frames_per_batch} frames "
                        f"but only {max_loadable_frames} can fit in memory"
                    )

        # Check completion
        if meta_state["total_frames_read"] >= meta_state["frames_to_read"]:
            if meta_batch is not None:
                meta_batch.inputs.pop(unique_id)
                self.close_inputs()
                meta_batch.has_closed_inputs = True
                meta_batch.total_frames = 0
            raise StopIteration("[BoyoLoadVideoDirectory] All frames have been processed")

        # Determine batch size
        frames_this_batch = min(
            meta_state["frames_to_read"] - meta_state["total_frames_read"],
            meta_batch.frames_per_batch if meta_batch is not None else meta_state["frames_to_read"]
        )

        gen = boyo_frame_generator(
            meta_state["video_path"],
            meta_state["current_frame"] + meta_state["total_frames_read"],
            frames_this_batch,
            meta_state["fps"]
        )

        # Skip the initial info yield
        next(gen)

        frames_list = list(gen)

        if not frames_list:
            if meta_batch is not None:
                meta_batch.inputs.pop(unique_id)
                self.close_inputs()
            raise ValueError("[BoyoLoadVideoDirectory] No frames could be read from the video")

        meta_state["total_frames_read"] += len(frames_list)

        if meta_batch is not None:
            if meta_state["total_frames_read"] >= meta_state["frames_to_read"]:
                meta_batch.inputs.pop(unique_id)
                self.close_inputs()
                meta_batch.has_closed_inputs = True
                meta_batch.total_frames = 0

        frames_tensor = torch.stack(frames_list, dim=0)

        video_info = {
            "source_fps": meta_state["fps"],
            "source_frame_count": meta_state["total_frames"],
            "source_duration": meta_state["duration"],
            "source_width": meta_state["width"],
            "source_height": meta_state["height"],
            "loaded_fps": 1/meta_state["target_frame_time"] if meta_state["target_frame_time"] > 0 else meta_state["fps"],
            "loaded_frame_count": len(frames_list),
            "loaded_duration": len(frames_list) * meta_state["target_frame_time"],
            "loaded_width": meta_state["width"],
            "loaded_height": meta_state["height"],
        }

        current_start_time = (skip_frames + meta_state["total_frames_read"] - len(frames_list)) / meta_state["fps"]
        audio_duration = len(frames_list) / meta_state["fps"] if meta_state["fps"] > 0 else None
        audio_info = boyo_load_audio(video_path, current_start_time, audio_duration)

        return (frames_tensor, len(frames_list), audio_info, video_info, filename)

    class BoyoVideoDirectoryLoader:
        def __init__(self, directory_path: str, pattern: str):
            self.video_paths = []
            self.load_videos(directory_path, pattern)
            self.video_paths.sort()

        def load_videos(self, directory_path: str, pattern: str):
            for file_name in glob.glob(os.path.join(glob.escape(directory_path), pattern), recursive=True):
                if file_name.lower().endswith(ALLOWED_VIDEO_EXT):
                    self.video_paths.append(os.path.abspath(file_name))


NODE_CLASS_MAPPINGS = {
    "BoyoLoadVideoDirectory": BoyoLoadVideoDirectory
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoLoadVideoDirectory": "Boyo Load Video Directory"
}
