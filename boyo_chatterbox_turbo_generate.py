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
                "preserve_native_samplerate": ("BOOLEAN", {"default": True}),
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
    
    def fix_model_dtypes(self, model):
        """Force the model to use float32 throughout to avoid dtype mismatches"""
        try:
            # Fix sub-modules that contain parameters
            sub_models = []
            if hasattr(model, 't3'):
                sub_models.append(('t3', model.t3))
            if hasattr(model, 's3gen'):
                sub_models.append(('s3gen', model.s3gen))
            if hasattr(model, 've'):
                sub_models.append(('ve', model.ve))
            
            for name, sub_model in sub_models:
                print(f"🔧 Fixing dtypes for {name}...")
                
                # Fix parameters
                if hasattr(sub_model, 'named_parameters'):
                    for param_name, param in sub_model.named_parameters():
                        if param.dtype == torch.float64:
                            param.data = param.data.float()
                            print(f"Fixed {name}.{param_name} from float64 to float32")
                
                # Fix buffers  
                if hasattr(sub_model, 'named_buffers'):
                    for buffer_name, buffer in sub_model.named_buffers():
                        if buffer.dtype == torch.float64:
                            buffer.data = buffer.data.float()
                            print(f"Fixed {name}.{buffer_name} from float64 to float32")
                
                # Apply float32 to sub-model
                if hasattr(sub_model, 'float'):
                    sub_model.float()
                    
        except Exception as e:
            print(f"⚠️ Warning: Could not fix all model dtypes: {e}")
        
        return model
    
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
                    # Try to find any tensor-like data
                    for key, value in audio_data.items():
                        if hasattr(value, 'shape') and len(value.shape) >= 1:
                            waveform = value
                            sample_rate = 22050  # Default
                            break
                    else:
                        raise ValueError(f"No recognizable audio data found in dict: {list(audio_data.keys())}")
            elif isinstance(audio_data, (list, tuple)) and len(audio_data) >= 2:
                # Format might be (waveform, sample_rate)
                waveform, sample_rate = audio_data[0], audio_data[1]
            elif hasattr(audio_data, 'shape'):
                # Direct tensor/array
                waveform = audio_data
                sample_rate = 22050  # Default sample rate
            else:
                raise ValueError(f"Unsupported audio data format: {type(audio_data)}")
            
            # Convert to numpy if needed and ensure float32
            if hasattr(waveform, 'cpu'):
                waveform = waveform.cpu()
            if hasattr(waveform, 'numpy'):
                waveform = waveform.numpy()
            elif hasattr(waveform, 'detach'):
                waveform = waveform.detach().numpy()
            
            # Ensure float32 dtype (not float64/double)
            if waveform.dtype != np.float32:
                waveform = waveform.astype(np.float32)
            
            # Handle different tensor shapes
            if len(waveform.shape) == 3:  # (batch, channels, samples)
                waveform = waveform[0, 0]  # Take first batch, first channel
            elif len(waveform.shape) == 2:  # (channels, samples) or (batch, samples)
                if waveform.shape[0] <= 2:  # Likely (channels, samples)
                    waveform = waveform[0]  # Take first channel
                else:  # Likely (batch, samples) 
                    waveform = waveform[0]  # Take first batch
            
            # Write the file
            sf.write(temp_path, waveform, int(sample_rate))
            return temp_path
            
        except Exception as e:
            # Clean up temp file if creation failed
            try:
                os.unlink(temp_path)
            except:
                pass
            raise Exception(f"Failed to convert audio to temp file: {e}")
    
    def apply_comprehensive_patches(self, model):
        """Apply comprehensive patches to fix dtype issues throughout the pipeline"""
        try:
            print("🔧 Applying comprehensive dtype patches...")
            
            # Patch 1: Mel spectrogram calculation
            if hasattr(model, 's3gen') and hasattr(model.s3gen, 'tokenizer'):
                tokenizer = model.s3gen.tokenizer
                
                # Force fix mel filters
                if hasattr(tokenizer, '_mel_filters'):
                    mel_filters = tokenizer._mel_filters
                    if mel_filters.dtype != torch.float32:
                        tokenizer._mel_filters = mel_filters.float()
                        print("✅ Fixed _mel_filters dtype")
                
                # Monkey-patch mel spectrogram function
                if hasattr(tokenizer, 'log_mel_spectrogram'):
                    original_log_mel = tokenizer.log_mel_spectrogram
                    
                    def patched_log_mel_spectrogram(self, wav):
                        # Force float32 throughout
                        if wav.dtype != torch.float32:
                            wav = wav.float()
                        
                        try:
                            return original_log_mel(wav)
                        except RuntimeError as e:
                            if "expected scalar type Float but found Double" in str(e):
                                # Manual implementation with forced float32
                                import torch.nn.functional as F
                                
                                window = torch.hann_window(self.n_fft, device=wav.device, dtype=torch.float32)
                                stft = torch.stft(
                                    wav.float(),
                                    n_fft=self.n_fft,
                                    hop_length=self.hop_length,
                                    window=window,
                                    return_complex=True
                                )
                                magnitudes = torch.abs(stft).float()
                                mel_filters = self._mel_filters.float()
                                mel_spec = mel_filters.to(self.device) @ magnitudes
                                return torch.log(torch.clamp(mel_spec, min=1e-5))
                            else:
                                raise e
                    
                    import types
                    tokenizer.log_mel_spectrogram = types.MethodType(patched_log_mel_spectrogram, tokenizer)
                    print("✅ Applied mel spectrogram patch")
            
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
                print("✅ Applied librosa patch")
            except Exception as e:
                print(f"⚠️ Could not patch librosa: {e}")
            
            # Patch 3: Voice encoder
            try:
                if hasattr(model, 've'):
                    ve = model.ve
                    if hasattr(ve, 'embeds_from_wavs'):
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
                        print("✅ Applied voice encoder patch")
            except Exception as e:
                print(f"⚠️ Could not patch voice encoder: {e}")
                
        except Exception as e:
            print(f"⚠️ Warning in comprehensive patches: {e}")
    
    def generate_audio(self, model, text, reference_audio, temperature, top_p, top_k, 
                      repetition_penalty, seed=0, min_p=0.0, exaggeration=0.0, 
                      norm_loudness=True, preserve_native_samplerate=True):
        
        # Set seed if specified
        self.set_seed(seed)
        
        try:
            print("🎵 === BOYO CHATTERBOX TURBO GENERATION ===")
            print(f"📝 Text: {text[:50]}...")
            print(f"🎛️ Params: temp={temperature}, top_p={top_p}, top_k={top_k}")
            
            # Step 1: Convert audio to temp file
            print("🔄 Converting audio to temp file...")
            temp_ref_path = self.audio_to_temp_file(reference_audio)
            
            try:
                # Step 2: Fix model dtypes and apply patches
                print("🔧 Preparing model...")
                model = self.fix_model_dtypes(model)
                self.apply_comprehensive_patches(model)
                
                # Step 3: Generate with error recovery
                print("🎯 Generating audio...")
                try:
                    wav_tensor = model.generate(
                        text=text,
                        audio_prompt_path=temp_ref_path,
                        temperature=float(temperature),
                        min_p=float(min_p),
                        top_p=float(top_p),
                        top_k=int(top_k),
                        repetition_penalty=float(repetition_penalty),
                        norm_loudness=norm_loudness,
                    )
                    print("✅ Generation completed successfully!")
                    
                except Exception as generate_error:
                    print(f"❌ Generation error: {generate_error}")
                    raise generate_error
                
                # Step 4: Process output with quality preservation
                print("🎛️ Processing output...")
                
                # Handle different return formats
                if isinstance(wav_tensor, tuple) and len(wav_tensor) == 2:
                    sample_rate, audio_np = wav_tensor
                elif hasattr(wav_tensor, 'shape'):
                    audio_np = wav_tensor
                    # Get model's native sample rate for best quality
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
                
                # Format for ComfyUI: (batch, samples, channels)
                if audio_tensor.dim() == 2:
                    if audio_tensor.shape[0] < audio_tensor.shape[1]:  # (channels, samples)
                        audio_tensor = audio_tensor.transpose(0, 1).unsqueeze(0)
                    else:  # (samples, channels)
                        audio_tensor = audio_tensor.unsqueeze(0)
                elif audio_tensor.dim() == 1:
                    # Mono: (samples) -> (batch, samples, channels)
                    audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(-1)
                
                # Final quality settings
                final_tensor = audio_tensor.float()
                
                # Use native sample rate for best quality or fallback to ComfyUI compatible rate
                if preserve_native_samplerate:
                    output_sample_rate = int(sample_rate)
                    print(f"🎵 Using native sample rate: {output_sample_rate}Hz for maximum quality")
                else:
                    # ComfyUI sometimes has issues with high sample rates
                    output_sample_rate = min(int(sample_rate), 48000)  # Cap at 48kHz for compatibility
                    print(f"🎵 Using compatibility sample rate: {output_sample_rate}Hz")
                
                audio_output = {
                    "waveform": final_tensor,
                    "sample_rate": int(12000) 
                }
                
                print(f"✅ Audio generated successfully!")
                print(f"📊 Shape: {final_tensor.shape}, Sample Rate: {output_sample_rate}Hz")
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
