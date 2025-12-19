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
                print(f"Fixing dtypes for {name}...")
                
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
                    print(f"Applied .float() to {name}")
            
            # Specifically target the problematic tokenizer
            if hasattr(model, 's3gen') and hasattr(model.s3gen, 'tokenizer'):
                tokenizer = model.s3gen.tokenizer
                print("Fixing s3gen tokenizer...")
                
                # Fix all attributes that might be tensors
                for attr_name in dir(tokenizer):
                    if not attr_name.startswith('_') or attr_name == '_mel_filters':
                        try:
                            attr = getattr(tokenizer, attr_name)
                            if hasattr(attr, 'dtype') and attr.dtype == torch.float64:
                                setattr(tokenizer, attr_name, attr.float())
                                print(f"Fixed tokenizer.{attr_name} from float64 to float32")
                        except:
                            pass
                            
                # Apply float32 to tokenizer if possible
                if hasattr(tokenizer, 'float'):
                    tokenizer.float()
                    print("Applied .float() to tokenizer")
                    
        except Exception as e:
            print(f"Warning: Could not fix all model dtypes: {e}")
            import traceback
            traceback.print_exc()
        
        return model
    
    def audio_to_temp_file(self, audio_data):
        """Convert ComfyUI audio format to temporary file for Chatterbox"""
        import tempfile
        import os
        
        # Debug: Print the audio_data structure
        print(f"Debug: Audio data type: {type(audio_data)}")
        print(f"Debug: Audio data keys/shape: {audio_data.keys() if isinstance(audio_data, dict) else getattr(audio_data, 'shape', 'no shape')}")
        
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
            
            # Ensure correct shape - ComfyUI might use (batch, channels, samples) or (samples,)
            print(f"Debug: Waveform shape before processing: {waveform.shape}, dtype: {waveform.dtype}")
            
            # Handle different tensor shapes
            if len(waveform.shape) == 3:  # (batch, channels, samples)
                waveform = waveform[0, 0]  # Take first batch, first channel
            elif len(waveform.shape) == 2:  # (channels, samples) or (batch, samples)
                if waveform.shape[0] <= 2:  # Likely (channels, samples)
                    waveform = waveform[0]  # Take first channel
                else:  # Likely (batch, samples) 
                    waveform = waveform[0]  # Take first batch
            # len == 1 means (samples,) which is what we want
            
            print(f"Debug: Final waveform shape: {waveform.shape}, dtype: {waveform.dtype}, sample_rate: {sample_rate}")
            
            # Write the file
            sf.write(temp_path, waveform, int(sample_rate))
            print(f"Debug: Successfully wrote audio file: {temp_path}")
            
            return temp_path
            
        except Exception as e:
            # Clean up temp file if creation failed
            try:
                os.unlink(temp_path)
            except:
                pass
            raise Exception(f"Failed to convert audio to temp file: {e}")
    
    def generate_audio(self, model, text, reference_audio, temperature, top_p, top_k, 
                      repetition_penalty, seed=0, min_p=0.0, exaggeration=0.0, norm_loudness=True):
        
        # Set seed if specified
        self.set_seed(seed)
        
        try:
            print("=== STEP 1: Converting audio to temp file ===")
            # Convert ComfyUI audio to temporary file
            temp_ref_path = self.audio_to_temp_file(reference_audio)
            print(f"✅ Created temp file: {temp_ref_path}")
            
            try:
                print("=== STEP 2: Preparing parameters ===")
                # Ensure all float parameters are float32
                temperature = float(temperature)
                top_p = float(top_p)
                top_k = int(top_k)
                repetition_penalty = float(repetition_penalty)
                min_p = float(min_p)
                exaggeration = float(exaggeration)
                
                print(f"✅ Parameters ready - temp={temperature}, top_p={top_p}, top_k={top_k}")
                
                print("=== STEP 3: Calling model.generate ===")
                print(f"Model type: {type(model)}")
                print(f"Model device: {getattr(model, '_device_info', 'unknown')}")
                
                # Fix model dtype issues before generation
                print("=== STEP 3.1: Fixing model dtypes ===")
                model = self.fix_model_dtypes(model)
                
                print("=== STEP 3.2: Pre-generate model inspection ===")
                # Deep inspection of the problematic components
                try:
                    s3gen = model.s3gen
                    tokenizer = s3gen.tokenizer
                    print(f"Tokenizer type: {type(tokenizer)}")
                    
                    if hasattr(tokenizer, '_mel_filters'):
                        mel_filters = tokenizer._mel_filters
                        print(f"_mel_filters dtype: {mel_filters.dtype}")
                        print(f"_mel_filters device: {mel_filters.device}")
                        print(f"_mel_filters shape: {mel_filters.shape}")
                        
                        # Force fix this specific tensor
                        if mel_filters.dtype != torch.float32:
                            tokenizer._mel_filters = mel_filters.float()
                            print(f"FORCED _mel_filters to float32")
                    
                    # Check other relevant attributes
                    for attr in ['window', 'mel_scale', 'stft']:
                        if hasattr(tokenizer, attr):
                            val = getattr(tokenizer, attr)
                            if hasattr(val, 'dtype'):
                                print(f"tokenizer.{attr} dtype: {val.dtype}")
                                if val.dtype == torch.float64:
                                    setattr(tokenizer, attr, val.float())
                                    print(f"FORCED tokenizer.{attr} to float32")
                    
                except Exception as e:
                    print(f"Error in model inspection: {e}")
                
                print("=== STEP 3.3: Monkey-patch the problematic function ===")
                # Let's monkey-patch the log_mel_spectrogram function to force float32
                try:
                    original_log_mel = tokenizer.log_mel_spectrogram
                    
                    def patched_log_mel_spectrogram(self, wav):
                        print(f"PATCH: Input wav dtype: {wav.dtype}")
                        # Force input to float32
                        if wav.dtype != torch.float32:
                            wav = wav.float()
                            print(f"PATCH: Converted wav to float32")
                        
                        # Call original function but catch the specific error
                        try:
                            return original_log_mel(wav)
                        except RuntimeError as e:
                            if "expected scalar type Float but found Double" in str(e):
                                print(f"PATCH: Caught dtype error, investigating...")
                                
                                # Let's examine what's happening in the failing line
                                # This is line 163: mel_spec = self._mel_filters.to(self.device) @ magnitudes
                                
                                # First get magnitudes (this is what's causing the issue)
                                import torch.nn.functional as F
                                
                                # Replicate the mel spectrogram calculation with forced dtypes
                                window = torch.hann_window(self.n_fft, device=wav.device, dtype=torch.float32)
                                stft = torch.stft(
                                    wav.float(),  # Force float32
                                    n_fft=self.n_fft,
                                    hop_length=self.hop_length,
                                    window=window,
                                    return_complex=True
                                )
                                magnitudes = torch.abs(stft).float()  # Force float32
                                print(f"PATCH: magnitudes dtype: {magnitudes.dtype}")
                                
                                # Now do the multiplication with forced float32
                                mel_filters = self._mel_filters.float()  # Force float32
                                mel_spec = mel_filters.to(self.device) @ magnitudes
                                
                                # Add small epsilon and take log
                                mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
                                return mel_spec
                            else:
                                raise e
                    
                    # Replace the method
                    import types
                    tokenizer.log_mel_spectrogram = types.MethodType(patched_log_mel_spectrogram, tokenizer)
                    print("✅ Applied mel spectrogram patch")
                    
                except Exception as e:
                    print(f"Error applying patch: {e}")
                
                print("=== STEP 3.4: Monkey-patch librosa.load and voice encoder ===")
                # The new error is in voice encoder - let's patch librosa operations
                try:
                    import librosa
                    original_librosa_load = librosa.load
                    
                    def patched_librosa_load(*args, **kwargs):
                        print(f"PATCH: librosa.load called")
                        result = original_librosa_load(*args, **kwargs)
                        if isinstance(result, tuple):
                            audio, sr = result
                            if hasattr(audio, 'dtype') and audio.dtype != np.float32:
                                audio = audio.astype(np.float32)
                                print(f"PATCH: Converted librosa output to float32")
                            return audio, sr
                        elif hasattr(result, 'dtype') and result.dtype != np.float32:
                            result = result.astype(np.float32)
                            print(f"PATCH: Converted librosa output to float32")
                        return result
                    
                    librosa.load = patched_librosa_load
                    print("✅ Applied librosa.load patch")
                    
                    # Also patch the voice encoder's embeds_from_wavs method
                    ve = model.ve
                    original_embeds_from_wavs = ve.embeds_from_wavs
                    
                    def patched_embeds_from_wavs(self, wavs, *args, **kwargs):
                        print(f"PATCH: embeds_from_wavs called")
                        # Force all input wavs to float32
                        if isinstance(wavs, list):
                            patched_wavs = []
                            for wav in wavs:
                                if hasattr(wav, 'dtype') and wav.dtype != np.float32:
                                    wav = wav.astype(np.float32)
                                    print(f"PATCH: Converted wav to float32 in embeds_from_wavs")
                                patched_wavs.append(wav)
                            wavs = patched_wavs
                        elif hasattr(wavs, 'dtype') and wavs.dtype != np.float32:
                            wavs = wavs.astype(np.float32)
                            print(f"PATCH: Converted wav to float32 in embeds_from_wavs")
                        
                        return original_embeds_from_wavs(wavs, *args, **kwargs)
                    
                    ve.embeds_from_wavs = types.MethodType(patched_embeds_from_wavs, ve)
                    print("✅ Applied voice encoder patch")
                    
                except Exception as e:
                    print(f"Error applying voice encoder patch: {e}")

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
                    
                except Exception as generate_error:
                    print(f"❌ ERROR in model.generate: {generate_error}")
                    print(f"Error type: {type(generate_error)}")
                    import traceback
                    traceback.print_exc()
                    raise generate_error
                
                print("=== STEP 4: Processing output ===")
                print(f"Generated wav_tensor type: {type(wav_tensor)}")
                print(f"Generated wav_tensor shape: {getattr(wav_tensor, 'shape', 'no shape')}")
                
                # Handle different possible return formats
                if isinstance(wav_tensor, tuple) and len(wav_tensor) == 2:
                    # Expected format: (sample_rate, audio_np)
                    sample_rate, audio_np = wav_tensor
                    print(f"Tuple format: sample_rate={sample_rate}, audio_np type={type(audio_np)}")
                elif hasattr(wav_tensor, 'shape'):
                    # Just the audio tensor - use model's sample rate
                    audio_np = wav_tensor
                    sample_rate = model.sr  # ChatterboxTurboTTS has .sr attribute
                    print(f"Tensor format: using model.sr={sample_rate}, audio_np type={type(audio_np)}")
                else:
                    raise ValueError(f"Unexpected wav_tensor format: {type(wav_tensor)}")
                
                print(f"sample_rate={sample_rate} (type: {type(sample_rate)})")
                print(f"audio_np type={type(audio_np)}, shape={getattr(audio_np, 'shape', 'no shape')}")
                if hasattr(audio_np, 'dtype'):
                    print(f"audio_np dtype: {audio_np.dtype}")
                
                print("=== STEP 5: Converting to tensor ===")
                # Convert to torch tensor and ensure correct format
                if isinstance(audio_np, np.ndarray):
                    print(f"Converting numpy array - original dtype: {audio_np.dtype}")
                    # Ensure float32 dtype
                    if audio_np.dtype != np.float32:
                        print(f"Converting from {audio_np.dtype} to float32")
                        audio_np = audio_np.astype(np.float32)
                    audio_tensor = torch.from_numpy(audio_np)
                    print(f"Created tensor from numpy - dtype: {audio_tensor.dtype}")
                else:
                    print(f"Audio_np is tensor-like: {type(audio_np)}")
                    audio_tensor = audio_np
                    # Ensure float32 if it's a tensor
                    if hasattr(audio_tensor, 'dtype'):
                        print(f"Original tensor dtype: {audio_tensor.dtype}")
                        if audio_tensor.dtype == torch.float64:
                            print("Converting from float64 to float32")
                            audio_tensor = audio_tensor.float()  # Convert to float32
                
                print(f"audio_tensor dtype: {audio_tensor.dtype}, shape: {audio_tensor.shape}")
                
                print("=== STEP 6: Formatting output ===")
                # ComfyUI expects audio in format (batch, samples, channels) based on the error
                # The error shows ComfyUI trying to do movedim(0, 1) which suggests 3D tensor
                
                if audio_tensor.dim() == 2:
                    # Current shape is (channels, samples), we need (batch, samples, channels)
                    if audio_tensor.shape[0] < audio_tensor.shape[1]:  # Likely (channels, samples)
                        # First transpose to (samples, channels), then add batch dim
                        audio_tensor = audio_tensor.transpose(0, 1)  # (samples, channels) 
                        audio_tensor = audio_tensor.unsqueeze(0)     # (batch=1, samples, channels)
                        print(f"Converted from (channels, samples) to (batch, samples, channels): {audio_tensor.shape}")
                    else:  # Already (samples, channels)
                        audio_tensor = audio_tensor.unsqueeze(0)     # Add batch dimension
                        print(f"Added batch dimension: {audio_tensor.shape}")
                elif audio_tensor.dim() == 1:
                    # Mono audio: (samples) -> (batch=1, samples, channels=1)
                    audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(-1)
                    print(f"Converted mono audio to (batch, samples, channels): {audio_tensor.shape}")
                elif audio_tensor.dim() == 3:
                    # Already correct format
                    print(f"Audio already in 3D format: {audio_tensor.shape}")
                
                print(f"Final audio tensor shape: {audio_tensor.shape}")
                
                # ComfyUI audio format - ensure everything is float32
                try:
                    final_tensor = audio_tensor.float()  # Explicit float32 conversion
                    print(f"Final tensor dtype: {final_tensor.dtype}")
                    
                    audio_output = {
                        "waveform": final_tensor,
                        "sample_rate": int(12000)  # <-- Try 12000 instead of sample_rate
                    }   
                    
                    print("✅ Successfully created audio output!")
                    return (audio_output,)
                    
                except Exception as format_error:
                    print(f"❌ ERROR in output formatting: {format_error}")
                    import traceback
                    traceback.print_exc()
                    raise format_error
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_ref_path)
                    print(f"🗑️ Cleaned up temp file: {temp_ref_path}")
                except:
                    print(f"⚠️ Could not clean up temp file: {temp_ref_path}")
        
        except Exception as e:
            print(f"❌ FINAL ERROR: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            print("=== FULL TRACEBACK ===")
            traceback.print_exc()
            raise Exception(f"Failed to generate audio: {e}")

NODE_CLASS_MAPPINGS = {
    "BoyoChatterboxTurboGenerate": BoyoChatterboxTurboGenerate
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoChatterboxTurboGenerate": "Boyo Chatterbox Turbo Generate"
}
