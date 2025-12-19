import os
import torch
from pathlib import Path

class BoyoChatterboxTurboLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "device": (["auto", "cpu", "cuda", "mps"], {"default": "auto"}),
            }
        }
    
    RETURN_TYPES = ("CHATTERBOX_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "Boyo/Audio/TTS"
    
    def load_model(self, device):
        try:
            # Import here to avoid dependency issues if not installed
            from chatterbox.tts_turbo import ChatterboxTurboTTS
        except ImportError as e:
            raise Exception(f"Chatterbox TTS not installed. Please install with: pip install chatterbox-tts\nError: {e}")
        
        # Device selection logic
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        print(f"Loading Chatterbox Turbo TTS model on {device}...")
        print("Downloading from HuggingFace Hub (first run may take a few minutes)...")
        
        try:
            # Always load from HuggingFace Hub - it will cache automatically
            model = ChatterboxTurboTTS.from_pretrained(device)
            print("✅ Model loaded successfully!")
            
            # Store device info for later use
            model._device_info = device
            
            # Check if Perth watermarker is working
            try:
                # Test the watermarker
                test_watermarker = model.watermarker
                print("✅ Perth watermarker available")
            except Exception as e:
                print(f"⚠️ Warning: Perth watermarker issue: {e}")
                print("ℹ️ Audio will be generated without watermarking")
                # Create a dummy watermarker that does nothing
                class DummyWatermarker:
                    def apply_watermark(self, wav, sample_rate):
                        return wav
                model.watermarker = DummyWatermarker()
            
            return (model,)
            
        except Exception as e:
            print(f"❌ Error loading Chatterbox Turbo TTS model: {e}")
            raise Exception(f"Failed to load Chatterbox Turbo TTS model: {e}")

NODE_CLASS_MAPPINGS = {
    "BoyoChatterboxTurboLoader": BoyoChatterboxTurboLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoChatterboxTurboLoader": "Boyo Chatterbox Turbo Loader"
}
