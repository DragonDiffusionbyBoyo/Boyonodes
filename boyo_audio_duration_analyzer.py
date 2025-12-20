import torch
import numpy as np

class BoyoAudioDurationAnalyzer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("duration_seconds", "info_text")
    FUNCTION = "analyze_duration"
    CATEGORY = "Boyo/Audio/Analysis"
    
    def analyze_duration(self, audio):
        try:
            print("🔍 Analyzing audio duration...")
            
            # Handle different ComfyUI audio formats
            if isinstance(audio, dict):
                if 'waveform' in audio:
                    waveform = audio['waveform']
                    sample_rate = audio.get('sample_rate', 22050)
                elif 'audio' in audio:
                    waveform = audio['audio'] 
                    sample_rate = audio.get('sample_rate', 22050)
                else:
                    raise ValueError(f"No recognizable audio data found in dict: {list(audio.keys())}")
            elif isinstance(audio, (list, tuple)) and len(audio) >= 2:
                # Format might be (waveform, sample_rate)
                waveform, sample_rate = audio[0], audio[1]
            elif hasattr(audio, 'shape'):
                # Direct tensor/array
                waveform = audio
                sample_rate = 22050  # Default sample rate
            else:
                raise ValueError(f"Unsupported audio data format: {type(audio)}")
            
            print(f"Debug: Waveform type: {type(waveform)}")
            print(f"Debug: Sample rate: {sample_rate}")
            
            # Convert to tensor if needed
            if hasattr(waveform, 'cpu'):
                waveform = waveform.cpu()
            if hasattr(waveform, 'numpy'):
                waveform = waveform.numpy()
            elif hasattr(waveform, 'detach'):
                waveform = waveform.detach().numpy()
                
            # Handle different tensor shapes - ComfyUI uses various formats
            print(f"Debug: Waveform shape: {waveform.shape}")
            
            if len(waveform.shape) == 3:  # (batch, channels, samples) or (batch, samples, channels)
                # Check which dimension is likely samples (largest one)
                if waveform.shape[1] > waveform.shape[2]:  # (batch, samples, channels)
                    num_samples = waveform.shape[1]
                    channels = waveform.shape[2]
                else:  # (batch, channels, samples) - more common
                    num_samples = waveform.shape[2]
                    channels = waveform.shape[1]
            elif len(waveform.shape) == 2:  # Could be (channels, samples) or (batch, samples) 
                if waveform.shape[0] <= 8:  # Likely (channels, samples) if first dim is small
                    num_samples = waveform.shape[1]
                    channels = waveform.shape[0]
                else:  # Likely (batch, samples) or (samples, channels)
                    if waveform.shape[1] <= 8:  # Second dim small = (samples, channels)
                        num_samples = waveform.shape[0]
                        channels = waveform.shape[1] 
                    else:  # Both dims large = assume (batch, samples)
                        num_samples = waveform.shape[1]
                        channels = 1
            elif len(waveform.shape) == 1:  # (samples,) - mono
                num_samples = waveform.shape[0]
                channels = 1
            else:
                raise ValueError(f"Unexpected waveform shape: {waveform.shape}")
            
            # Calculate duration
            duration_seconds = float(num_samples) / float(sample_rate)
            
            print(f"✅ Analysis complete - Duration: {duration_seconds:.3f}s")
            
            # Create informative status text
            info_text = f"📊 Audio: {duration_seconds:.2f}s | {num_samples:,} samples @ {sample_rate}Hz"
            if channels > 1:
                info_text += f" | {channels} channels"
            
            return (duration_seconds, info_text)
            
        except Exception as e:
            print(f"❌ Error analyzing audio duration: {e}")
            import traceback
            traceback.print_exc()
            error_text = f"❌ Error: {str(e)}"
            return (0.0, error_text)

NODE_CLASS_MAPPINGS = {
    "BoyoAudioDurationAnalyzer": BoyoAudioDurationAnalyzer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoAudioDurationAnalyzer": "Boyo Audio Duration Analyzer"
}
