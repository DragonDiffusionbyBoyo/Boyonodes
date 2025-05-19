import os
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
import nodes

class BoyoAudioEval:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_path": ("STRING", {
                    "default": "./input/audio.wav",
                    "multiline": False,
                    "dynamicPrompt": False
                }),
                "fps": ("FLOAT", {"default": 23.0, "min": 1.0, "max": 60.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT", "STRING")
    RETURN_NAMES = ("frame_count", "video_length_seconds", "metadata")
    FUNCTION = "convert_audio_to_frames"
    CATEGORY = "VideoUtils"

    def convert_audio_to_frames(self, audio_path, fps):
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")

        audio, _ = sf.read(audio_path)
        audio_length = len(audio) / sf.info(audio_path).samplerate

        total_frames_float = audio_length * fps
        total_frames = int(np.ceil(total_frames_float))

        metadata = f"Audio Length: {audio_length:.4f}s, FPS: {fps}, Frames: {total_frames}"

        return (total_frames, audio_length, metadata)