import torch
import numpy as np
import tempfile
import os
from pathlib import Path

class BoyoChatterboxTurboGenerate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("CHATTERBOX_MODEL",),
                "text": ("STRING", {"default": "Hello, this is a test of the Chatterbox Turbo text to speech system.", "multiline": True}),
                "reference_audio": ("AUDIO",),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.05, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 1000, "min": 0, "max": 1000, "step": 10}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967295}),
                "min_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "exaggeration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "norm_loudness": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_audio"
    CATEGORY = "Boyo/Audio/TTS"
    
    def set_seed(self, seed):
        if seed != 0:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
    
    def audio_to_temp_file(self, audio_data):
        """Convert ComfyUI audio format to temporary file for Chatterbox"""
        try:
            import soundfile as sf
        except ImportError:
            raise Exception("soundfile library required for audio processing. Install with: pip install soundfile")
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_path = temp_file.name
        temp_file.close()
        
        try:
            # Handle different ComfyUI audio formats
            if isinstance(audio_data, dict):
                if 'waveform' in audio_data:
                    waveform = audio_data['waveform']
                    sample_rate = audio_data.get('sample_rate', 22050)
                elif 'audio' in audio_data:
                    waveform = audio_data['audio']
                    sample_rate = audio_data.get('sample_rate', 22050)
                else:
                    raise ValueError(f"No recognizable audio data found in dict: {list(audio_data.keys())}")
            elif isinstance(audio_data, (list, tuple)) and len(audio_data) >= 2:
                waveform, sample_rate = audio_data[0], audio_data[1]
            elif hasattr(audio_data, 'shape'):
                waveform = audio_data
                sample_rate = 22050  # Default sample rate
            else:
                raise ValueError(f"Unsupported audio data format: {type(audio_data)}")
            
            # Convert to numpy and ensure format
            if hasattr(waveform, 'cpu'):
                waveform = waveform.cpu()
            if hasattr(waveform, 'numpy'):
                waveform = waveform.numpy()
            elif hasattr(waveform, 'detach'):
                waveform = waveform.detach().numpy()
            
            # Ensure float32 dtype
            if waveform.dtype != np.float32:
                waveform = waveform.astype(np.float32)
            
            # Handle tensor shapes
            if len(waveform.shape) == 3:  # (batch, channels, samples)
                waveform = waveform[0, 0]
            elif len(waveform.shape) == 2:
                if waveform.shape[0] <= 2:  # (channels, samples)
                    waveform = waveform[0]
                else:  # (batch, samples) 
                    waveform = waveform[0]
            
            sf.write(temp_path, waveform, int(sample_rate))
            return temp_path
            
        except Exception as e:
            try:
                os.unlink(temp_path)
            except:
                pass
            raise Exception(f"Failed to convert audio to temp file: {e}")
    
    def apply_compatibility_patches(self, model):
        """Apply comprehensive patches for dtype compatibility"""
        try:
            # Patch 1: Model components
            if hasattr(model, 's3gen') and hasattr(model.s3gen, 'tokenizer'):
                tokenizer = model.s3gen.tokenizer
                
                # Fix mel filters
                if hasattr(tokenizer, '_mel_filters'):
                    tokenizer._mel_filters = tokenizer._mel_filters.float()
                
                # Patch log_mel_spectrogram method
                if hasattr(tokenizer, 'log_mel_spectrogram'):
                    original_log_mel = tokenizer.log_mel_spectrogram
                    
                    def patched_log_mel_spectrogram(self, wav):
                        if wav.dtype != torch.float32:
                            wav = wav.float()
                        
                        try:
                            return original_log_mel(wav)
                        except RuntimeError as e:
                            if "expected scalar type Float but found Double" in str(e):
                                # Force float32 throughout the mel spectrogram calculation
                                window = torch.hann_window(self.n_fft, device=wav.device, dtype=torch.float32)
                                stft = torch.stft(wav.float(), n_fft=self.n_fft, hop_length=self.hop_length, 
                                                window=window, return_complex=True)
                                magnitudes = torch.abs(stft).float()
                                mel_filters = self._mel_filters.float()
                                mel_spec = mel_filters.to(self.device) @ magnitudes
                                return torch.log(torch.clamp(mel_spec, min=1e-5))
                            else:
                                raise e
                    
                    import types
                    tokenizer.log_mel_spectrogram = types.MethodType(patched_log_mel_spectrogram, tokenizer)
            
            # Patch 2: Librosa operations
            try:
                import librosa
                original_librosa_load = librosa.load
                
                def patched_librosa_load(*args, **kwargs):
                    result = original_librosa_load(*args, **kwargs)
                    if isinstance(result, tuple):
                        audio, sr = result
                        if hasattr(audio, 'dtype') and audio.dtype != np.float32:
                            audio = audio.astype(np.float32)
                        return audio, sr
                    elif hasattr(result, 'dtype') and result.dtype != np.float32:
                        result = result.astype(np.float32)
                    return result
                
                librosa.load = patched_librosa_load
            except Exception as e:
                print(f"⚠️ Could not patch librosa: {e}")
            
            # Patch 3: Voice encoder
            try:
                if hasattr(model, 've') and hasattr(model.ve, 'embeds_from_wavs'):
                    ve = model.ve
                    original_embeds_from_wavs = ve.embeds_from_wavs
                    
                    def patched_embeds_from_wavs(self, wavs, *args, **kwargs):
                        # Force float32 for input wavs
                        if isinstance(wavs, list):
                            patched_wavs = []
                            for wav in wavs:
                                if hasattr(wav, 'dtype') and wav.dtype != np.float32:
                                    wav = wav.astype(np.float32)
                                patched_wavs.append(wav)
                            wavs = patched_wavs
                        elif hasattr(wavs, 'dtype') and wavs.dtype != np.float32:
                            wavs = wavs.astype(np.float32)
                        
                        return original_embeds_from_wavs(wavs, *args, **kwargs)
                    
                    import types
                    ve.embeds_from_wavs = types.MethodType(patched_embeds_from_wavs, ve)
            except Exception as e:
                print(f"⚠️ Could not patch voice encoder: {e}")
                
        except Exception as e:
            print(f"⚠️ Warning in comprehensive patches: {e}")
    
    def generate_audio(self, model, text, reference_audio, temperature, top_p, top_k, 
                      repetition_penalty, seed=0, min_p=0.0, exaggeration=0.0, norm_loudness=True):
        
        # Set seed if specified
        self.set_seed(seed)
        
        try:
            # Convert audio and apply patches
            temp_ref_path = self.audio_to_temp_file(reference_audio)
            self.apply_compatibility_patches(model)
            
            try:
                # Generate audio using Chatterbox
                wav_tensor = model.generate(
                    text=text,
                    audio_prompt_path=temp_ref_path,
                    temperature=temperature,
                    min_p=min_p,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    norm_loudness=norm_loudness,
                )
                
                print("✅ model.generate completed successfully!")
                
                # Handle model output format
                if isinstance(wav_tensor, tuple) and len(wav_tensor) == 2:
                    sample_rate, audio_np = wav_tensor
                elif hasattr(wav_tensor, 'shape'):
                    audio_np = wav_tensor
                    sample_rate = getattr(model, 'sr', 24000)
                else:
                    raise ValueError(f"Unexpected output format: {type(wav_tensor)}")
                
                # Convert to tensor and preserve quality
                if isinstance(audio_np, np.ndarray):
                    if audio_np.dtype != np.float32:
                        audio_np = audio_np.astype(np.float32)
                    audio_tensor = torch.from_numpy(audio_np)
                else:
                    audio_tensor = audio_np
                    if hasattr(audio_tensor, 'dtype') and audio_tensor.dtype == torch.float64:
                        audio_tensor = audio_tensor.float()
                
                print(f"🔧 APPLYING WORKING CHATTERBOX FORMAT:")
                print(f"   Raw output shape: {audio_tensor.shape}")
                print(f"   Native sample rate: {sample_rate}Hz")
                
                # Apply the EXACT working ChatterBox formatting logic
                # (copied from working base_node.py format_audio_output method)
                
                # Ensure audio has batch dimension
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension
                    print(f"   Added channel dim: {audio_tensor.shape}")
                    
                if audio_tensor.dim() == 2 and audio_tensor.shape[0] != 1:
                    # If it's multi-channel, assume it's already in [channels, samples] format
                    print(f"   Multi-channel detected: {audio_tensor.shape}")
                    pass
                
                # Add batch dimension for ComfyUI
                if audio_tensor.dim() == 2:
                    audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
                    print(f"   Added batch dim: {audio_tensor.shape}")
                
                print(f"   Final ChatterBox format: {audio_tensor.shape} (batch, channels, samples)")
                print(f"   Preserving native sample rate: {sample_rate}Hz")
                
                # Use the EXACT working ChatterBox output format
                audio_output = {
                    "waveform": audio_tensor,
                    "sample_rate": int(sample_rate)  # Use native sample rate like working ChatterBox
                }
                
                print(f"✅ Audio generated with working ChatterBox format!")
                print(f"📊 Final shape: {audio_tensor.shape}, Sample rate: {sample_rate}Hz")
                return (audio_output,)
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_ref_path)
                except:
                    pass
        
        except Exception as e:
            print(f"❌ GENERATION FAILED: {e}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Failed to generate audio: {e}")

NODE_CLASS_MAPPINGS = {
    "BoyoChatterboxTurboGenerate": BoyoChatterboxTurboGenerate
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoChatterboxTurboGenerate": "Boyo Chatterbox Turbo Generate"
}
