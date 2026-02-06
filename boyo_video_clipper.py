"""
BoyoVideoClipper - Surgical video dataset preparation node
"""

import subprocess
import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import wave
import folder_paths
import torch

video_extensions = ['mp4', 'mov', 'avi', 'mkv', 'webm', 'm4v', 'flv']

class BoyoVideoClipper:
    
    def __init__(self):
        self.temp_dir = Path(folder_paths.get_temp_directory()) / "boyo_video_clipper"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and (file_parts[-1].lower() in video_extensions):
                    files.append(f)
        return {
            "required": {
                "video": (sorted(files),),
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 0.01}),
                "target_fps": ("INT", {"default": 16, "min": 1, "max": 120, "step": 1}),
                "required_frames": ("INT", {"default": 81, "min": 1, "max": 9999, "step": 1}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("frames", "audio", "metadata")
    FUNCTION = "clip_video"
    CATEGORY = "Boyo/Video"
    
    def check_ffmpeg(self):
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            raise RuntimeError("FFmpeg not found in PATH")
    
    def probe_video(self, video_path):
        try:
            cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
                   "-show_entries", "stream=width,height,r_frame_rate,duration",
                   "-of", "csv=p=0", str(video_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            parts = result.stdout.strip().split(',')
            width = int(parts[0])
            height = int(parts[1])
            fps_parts = parts[2].split('/')
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
            duration = float(parts[3]) if len(parts) > 3 and parts[3] else 0.0
            return {'width': width, 'height': height, 'fps': fps, 'duration': duration}
        except Exception as e:
            raise RuntimeError(f"Failed to probe video: {e}")
    
    def check_audio_stream(self, video_path):
        try:
            cmd = ["ffprobe", "-v", "error", "-select_streams", "a:0",
                   "-show_entries", "stream=codec_type",
                   "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout.strip() == "audio"
        except:
            return False
    
    def extract_frames(self, video_path, start_time, duration, target_fps, required_frames):
        frames_dir = self.temp_dir / "frames"
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        output_pattern = str(frames_dir / "frame_%04d.png")
        cmd = ["ffmpeg", "-ss", str(start_time), "-i", str(video_path),
               "-t", str(duration), "-vf", f"fps={target_fps}",
               "-frames:v", str(required_frames), "-q:v", "2", output_pattern, "-y"]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Frame extraction failed: {e.stderr}")
        
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        if not frame_files:
            raise RuntimeError("No frames extracted")
        
        first_frame = Image.open(frame_files[0])
        width, height = first_frame.size
        frames_array = np.zeros((len(frame_files), height, width, 3), dtype=np.float32)
        
        for idx, frame_path in enumerate(frame_files):
            img = Image.open(frame_path).convert('RGB')
            frames_array[idx] = np.array(img).astype(np.float32) / 255.0
        
        return torch.from_numpy(frames_array)
    
    def extract_audio(self, video_path, start_time, duration):
        audio_file = self.temp_dir / "audio.wav"
        cmd = ["ffmpeg", "-ss", str(start_time), "-i", str(video_path),
               "-t", str(duration), "-vn", "-acodec", "pcm_s16le",
               "-ar", "44100", "-ac", "2", str(audio_file), "-y"]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return audio_file
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Audio extraction failed: {e.stderr}")
    
    def create_silent_audio(self, duration):
        sample_rate = 44100
        channels = 2
        num_samples = int(duration * sample_rate)
        silent_audio = np.zeros((num_samples, channels), dtype=np.int16)
        audio_file = self.temp_dir / "audio.wav"
        with wave.open(str(audio_file), 'w') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(silent_audio.tobytes())
        return audio_file
    
    def load_audio(self, audio_path):
        with wave.open(str(audio_path), 'r') as wav_file:
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            audio_bytes = wav_file.readframes(n_frames)
            
            if sample_width == 2:
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            else:
                raise RuntimeError(f"Unsupported audio sample width: {sample_width}")
            
            audio_array = audio_array.reshape(-1, n_channels)
            audio_array = audio_array.astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_array).transpose(0, 1)
            audio_tensor = audio_tensor.unsqueeze(0)
            
            return {'waveform': audio_tensor, 'sample_rate': framerate}
    
    def clip_video(self, video, start_time, target_fps, required_frames):
        self.check_ffmpeg()
        
        # Use VHS's exact pattern for resolving uploaded files
        video_path = folder_paths.get_annotated_filepath(video)
        
        if not os.path.exists(video_path):
            raise RuntimeError(f"Video not found: {video_path}")
        
        duration = required_frames / target_fps
        video_info = self.probe_video(video_path)
        end_time = start_time + duration
        
        if end_time > video_info['duration']:
            raise RuntimeError(
                f"Clip window exceeds video duration!\n"
                f"  Video: {video_info['duration']:.2f}s\n"
                f"  Requested: {start_time:.2f}s to {end_time:.2f}s\n"
                f"  Max start: {video_info['duration'] - duration:.2f}s"
            )
        
        print(f"Extracting {required_frames} frames at {target_fps}fps from {start_time:.2f}s")
        frames_tensor = self.extract_frames(video_path, start_time, duration, target_fps, required_frames)
        
        has_audio = self.check_audio_stream(video_path)
        if has_audio:
            audio_file = self.extract_audio(video_path, start_time, duration)
        else:
            audio_file = self.create_silent_audio(duration)
        
        audio_tensor = self.load_audio(audio_file)
        
        metadata = (
            f"Video: {os.path.basename(video_path)}\n"
            f"Resolution: {video_info['width']}x{video_info['height']}\n"
            f"Source FPS: {video_info['fps']:.2f} → Target: {target_fps}\n"
            f"Clip: {start_time:.2f}s to {end_time:.2f}s ({duration:.4f}s)\n"
            f"Frames: {required_frames}\n"
            f"Audio: {'Yes' if has_audio else 'Silent'}\n"
        )
        
        return (frames_tensor, audio_tensor, metadata)
    
    @classmethod
    def IS_CHANGED(s, video, **kwargs):
        image_path = folder_paths.get_annotated_filepath(video)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()
    
    @classmethod
    def VALIDATE_INPUTS(s, video, **kwargs):
        if not folder_paths.exists_annotated_filepath(video):
            return "Invalid video file: {}".format(video)
        return True


NODE_CLASS_MAPPINGS = {"BoyoVideoClipper": BoyoVideoClipper}
NODE_DISPLAY_NAME_MAPPINGS = {"BoyoVideoClipper": "Boyo Video Clipper"}