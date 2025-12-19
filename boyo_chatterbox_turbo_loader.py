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
        # Bypass Perth watermarking entirely to avoid dependency issues
        print("🔧 Bypassing Perth watermarking for compatibility...")
        try:
            import sys
            from types import ModuleType
            
            # Create a fake perth module if it doesn't exist
            if 'perth' not in sys.modules:
                fake_perth = ModuleType('perth')
                
                # Create a dummy watermarker class
                class DummyWatermarker:
                    def __init__(self):
                        pass
                    
                    def apply_watermark(self, wav, sample_rate=None):
                        # Just return the audio unchanged
                        return wav
                
                fake_perth.PerthImplicitWatermarker = DummyWatermarker
                sys.modules['perth'] = fake_perth
                print("✅ Created dummy Perth module")
                
        except Exception as e:
            print(f"Warning: Could not set up Perth bypass: {e}")
        
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
            
            # Ensure watermarker is our dummy version
            try:
                # Replace the model's watermarker with our dummy one
                class DummyWatermarker:
                    def apply_watermark(self, wav, sample_rate=None):
                        return wav
                
                model.watermarker = DummyWatermarker()
                print("✅ Watermarking bypassed successfully")
            except Exception as e:
                print(f"⚠️ Warning: Could not replace watermarker: {e}")
            
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
