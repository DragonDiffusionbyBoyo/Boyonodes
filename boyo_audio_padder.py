import torch
import numpy as np
import torchaudio.functional as F

class BoyoAudioPadder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "pre_pad_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 0.1}),
                "post_pad_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 0.1}),
            },
            "optional": {
                "target_duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 0.1}),
                "auto_center": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "FLOAT", "STRING")
    RETURN_NAMES = ("padded_audio", "total_duration", "status_text")
    FUNCTION = "pad_audio"
    CATEGORY = "Boyo/Audio/Processing"
    
    def debug_log(self, stage, waveform, sample_rate, extra_info=""):
        """Comprehensive debugging for sample rate tracking"""
        print(f"\n🔍 DEBUG [{stage}]:")
        print(f"   📊 Tensor Shape: {waveform.shape if hasattr(waveform, 'shape') else 'No shape'}")
        print(f"   📊 Tensor Dtype: {waveform.dtype if hasattr(waveform, 'dtype') else 'No dtype'}")
        print(f"   🎵 Sample Rate: {sample_rate} Hz")
        
        if hasattr(waveform, 'shape') and len(waveform.shape) >= 1:
            total_samples = waveform.numel() if hasattr(waveform, 'numel') else len(waveform)
            duration = total_samples / sample_rate if sample_rate > 0 else 0
            print(f"   ⏱️  Total Samples: {total_samples}")
            print(f"   ⏱️  Calculated Duration: {duration:.3f}s")
            print(f"   📈 Min/Max Values: {waveform.min():.6f} / {waveform.max():.6f}")
        
        if extra_info:
            print(f"   ℹ️  {extra_info}")
    
    def parse_audio_input(self, audio):
        """Safely parse ComfyUI audio input with extensive logging"""
        self.debug_log("INPUT_PARSING", audio if hasattr(audio, 'shape') else torch.tensor([0]), 0, "Raw input received")
        
        waveform = None
        sample_rate = None
        
        if isinstance(audio, dict):
            print(f"🔍 Audio dict keys: {list(audio.keys())}")
            
            if 'waveform' in audio:
                waveform = audio['waveform'] 
                sample_rate = audio.get('sample_rate', None)  # Don't force a default yet
                print(f"✅ Found 'waveform' key with sample_rate: {sample_rate}")
            elif 'audio' in audio:
                waveform = audio['audio']
                sample_rate = audio.get('sample_rate', None)
                print(f"✅ Found 'audio' key with sample_rate: {sample_rate}")
            else:
                # Try to find any tensor-like data
                for key, value in audio.items():
                    if hasattr(value, 'shape') and len(value.shape) >= 1:
                        waveform = value
                        sample_rate = audio.get('sample_rate', None)
                        print(f"✅ Found tensor data in key '{key}' with sample_rate: {sample_rate}")
                        break
                else:
                    raise ValueError(f"No recognizable audio data found in dict: {list(audio.keys())}")
        elif isinstance(audio, (list, tuple)) and len(audio) >= 2:
            waveform, sample_rate = audio[0], audio[1]
            print(f"✅ Tuple format: waveform type={type(waveform)}, sample_rate={sample_rate}")
        elif hasattr(audio, 'shape'):
            waveform = audio
            sample_rate = None  # We don't know the rate yet
            print(f"⚠️ Direct tensor format, no sample rate available")
        else:
            raise ValueError(f"Unsupported audio data format: {type(audio)}")
        
        # If we still don't have a sample rate, this is a problem
        if sample_rate is None:
            raise ValueError("No sample rate found in audio data - cannot proceed safely")
        
        self.debug_log("PARSED_INPUT", waveform, sample_rate, "After parsing input format")
        return waveform, sample_rate
    
    def normalize_tensor_format(self, waveform, sample_rate):
        """Convert to standard (batch, samples, channels) format with minimal processing"""
        self.debug_log("PRE_NORMALIZE", waveform, sample_rate, "Before tensor normalization")
        
        # Convert to tensor if needed, preserving precision
        if isinstance(waveform, np.ndarray):
            print("🔄 Converting numpy array to tensor...")
            if waveform.dtype != np.float32:
                print(f"   Converting dtype from {waveform.dtype} to float32")
                waveform = waveform.astype(np.float32)
            waveform = torch.from_numpy(waveform)
        elif hasattr(waveform, 'cpu'):
            waveform = waveform.cpu().float()
        else:
            waveform = waveform.float()
        
        self.debug_log("TENSOR_CONVERSION", waveform, sample_rate, "After tensor conversion")
        
        original_shape = waveform.shape
        print(f"🔄 Reshaping from {original_shape}...")
        
        # Determine format and reshape to (batch, samples, channels)
        if len(original_shape) == 3:  # Already 3D
            if original_shape[1] > original_shape[2]:  # (batch, samples, channels)
                print("   Already in (batch, samples, channels) format")
                batch_size, num_samples, num_channels = original_shape
            else:  # (batch, channels, samples) - transpose needed
                print("   Converting from (batch, channels, samples) to (batch, samples, channels)")
                waveform = waveform.transpose(1, 2)
                batch_size, num_samples, num_channels = waveform.shape
                
        elif len(original_shape) == 2:
            if original_shape[0] <= 8:  # (channels, samples)
                print("   Converting from (channels, samples) to (batch, samples, channels)")
                waveform = waveform.transpose(0, 1).unsqueeze(0)
                batch_size, num_samples, num_channels = 1, original_shape[1], original_shape[0]
            else:  # (batch, samples) or (samples, channels)
                if original_shape[1] <= 8:  # (samples, channels)
                    print("   Converting from (samples, channels) to (batch, samples, channels)")
                    waveform = waveform.unsqueeze(0)
                    batch_size, num_samples, num_channels = 1, original_shape[0], original_shape[1]
                else:  # (batch, samples) - assume mono
                    print("   Converting from (batch, samples) to (batch, samples, channels=1)")
                    waveform = waveform.unsqueeze(-1)
                    batch_size, num_samples, num_channels = original_shape[0], original_shape[1], 1
                    
        elif len(original_shape) == 1:  # (samples,) - mono
            print("   Converting from (samples,) to (batch=1, samples, channels=1)")
            waveform = waveform.unsqueeze(0).unsqueeze(-1)
            batch_size, num_samples, num_channels = 1, original_shape[0], 1
        else:
            raise ValueError(f"Unexpected waveform shape: {original_shape}")
        
        print(f"✅ Final normalized shape: {waveform.shape}")
        print(f"   Batch={batch_size}, Samples={num_samples}, Channels={num_channels}")
        
        self.debug_log("POST_NORMALIZE", waveform, sample_rate, f"Normalized to batch={batch_size}, samples={num_samples}, channels={num_channels}")
        
        return waveform, batch_size, num_samples, num_channels, sample_rate
    
    def force_mono_output(self, waveform, sample_rate):
        """Force mono output to match working TTS nodes"""
        self.debug_log("PRE_MONO_CONVERSION", waveform, sample_rate, "Before mono conversion")
        
        if waveform.shape[2] > 1:
            print(f"🔧 Converting from {waveform.shape[2]} channels to mono (averaging channels)")
            waveform = waveform.mean(dim=2, keepdim=True)
        else:
            print("✅ Already mono, no conversion needed")
        
        self.debug_log("POST_MONO_CONVERSION", waveform, sample_rate, "After mono conversion")
        return waveform, sample_rate
    
    def create_padding_tensors(self, waveform, pre_pad_samples, post_pad_samples, sample_rate):
        """Create silence padding tensors"""
        batch_size, _, num_channels = waveform.shape
        device = waveform.device
        
        print(f"🔧 Creating padding tensors...")
        print(f"   Pre-padding: {pre_pad_samples} samples")
        print(f"   Post-padding: {post_pad_samples} samples")
        print(f"   Target format: (batch={batch_size}, samples=X, channels={num_channels})")
        
        pre_silence = None
        post_silence = None
        
        if pre_pad_samples > 0:
            pre_silence = torch.zeros(batch_size, pre_pad_samples, num_channels, 
                                    dtype=torch.float32, device=device)
            self.debug_log("PRE_SILENCE", pre_silence, sample_rate, f"Created {pre_pad_samples} samples of pre-silence")
        
        if post_pad_samples > 0:
            post_silence = torch.zeros(batch_size, post_pad_samples, num_channels, 
                                     dtype=torch.float32, device=device)
            self.debug_log("POST_SILENCE", post_silence, sample_rate, f"Created {post_pad_samples} samples of post-silence")
        
        return pre_silence, post_silence
    
    def concatenate_audio(self, waveform, pre_silence, post_silence, sample_rate):
        """Concatenate audio with padding"""
        self.debug_log("PRE_CONCATENATION", waveform, sample_rate, "Before concatenation")
        
        tensors_to_cat = []
        
        if pre_silence is not None:
            tensors_to_cat.append(pre_silence)
            print("📎 Added pre-silence to concatenation list")
        
        tensors_to_cat.append(waveform)
        print("📎 Added main audio to concatenation list")
        
        if post_silence is not None:
            tensors_to_cat.append(post_silence)
            print("📎 Added post-silence to concatenation list")
        
        print(f"📎 Concatenating {len(tensors_to_cat)} tensors along sample axis (dim=1)...")
        padded_waveform = torch.cat(tensors_to_cat, dim=1)
        
        self.debug_log("POST_CONCATENATION", padded_waveform, sample_rate, "After concatenation")
        return padded_waveform
    
    def create_output_dict(self, waveform, sample_rate):
        """Create final output using the EXACT working ChatterBox formatting logic"""
        self.debug_log("PRE_OUTPUT_CREATION", waveform, sample_rate, "Before output creation")
        
        print(f"🔧 USING WORKING CHATTERBOX FORMAT:")
        print(f"   Input sample rate: {sample_rate} Hz (preserving as-is)")
        print(f"   Input shape: {waveform.shape}")
        print(f"   Current format: (batch, samples, channels)")
        
        # Convert from our (batch, samples, channels) to ChatterBox (batch, channels, samples) format
        # ChatterBox expects: (batch, channels, samples)
        # We currently have: (batch, samples, channels)
        
        audio_tensor = waveform.transpose(1, 2)  # (batch, samples, channels) -> (batch, channels, samples)
        print(f"   Converted to ChatterBox format: {audio_tensor.shape} (batch, channels, samples)")
        
        # Apply the EXACT working ChatterBox formatting logic
        # Ensure audio has batch dimension
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension
        if audio_tensor.dim() == 2 and audio_tensor.shape[0] != 1:
            # If it's multi-channel, assume it's already in [channels, samples] format
            pass
        
        # Add batch dimension for ComfyUI
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
        
        print(f"   Final ChatterBox format: {audio_tensor.shape}")
        print(f"   Final sample rate: {sample_rate} Hz (unchanged)")
        
        # Use the EXACT working ChatterBox output format
        output_audio = {
            "waveform": audio_tensor,
            "sample_rate": sample_rate  # Preserve original sample rate like ChatterBox does
        }
        
        self.debug_log("FINAL_OUTPUT", audio_tensor, sample_rate, f"EXACT ChatterBox format applied")
        
        return output_audio, sample_rate
    
    def pad_audio(self, audio, pre_pad_seconds, post_pad_seconds, target_duration=0.0, auto_center=False):
        try:
            print("\n" + "="*60)
            print("🎵 BOYO AUDIO PADDER - COMPREHENSIVE DEBUG VERSION")
            print("="*60)
            
            # Step 1: Parse input audio with extensive logging
            waveform, sample_rate = self.parse_audio_input(audio)
            
            # Step 2: Normalize tensor format
            waveform, batch_size, num_samples, num_channels, sample_rate = self.normalize_tensor_format(waveform, sample_rate)
            
            # Step 3: Calculate original duration
            original_duration = float(num_samples) / float(sample_rate)
            print(f"\n📊 ORIGINAL AUDIO METRICS:")
            print(f"   Duration: {original_duration:.3f}s")
            print(f"   Samples: {num_samples}")
            print(f"   Sample Rate: {sample_rate} Hz")
            print(f"   Channels: {num_channels}")
            
            # Step 4: Handle auto-centering
            if auto_center and target_duration > original_duration:
                total_padding_needed = target_duration - original_duration
                pre_pad_seconds = total_padding_needed / 2.0
                post_pad_seconds = total_padding_needed / 2.0
                print(f"🎯 AUTO-CENTER: {total_padding_needed:.2f}s padding split equally")
            
            # Step 5: Calculate padding samples
            pre_pad_samples = int(pre_pad_seconds * sample_rate)
            post_pad_samples = int(post_pad_seconds * sample_rate)
            
            print(f"\n📏 PADDING CALCULATIONS:")
            print(f"   Pre-padding: {pre_pad_seconds:.3f}s = {pre_pad_samples} samples")
            print(f"   Post-padding: {post_pad_seconds:.3f}s = {post_pad_samples} samples")
            
            # Step 6: Force mono output before padding
            waveform, sample_rate = self.force_mono_output(waveform, sample_rate)
            
            # Step 7: Create padding tensors
            pre_silence, post_silence = self.create_padding_tensors(waveform, pre_pad_samples, post_pad_samples, sample_rate)
            
            # Step 8: Concatenate audio
            padded_waveform = self.concatenate_audio(waveform, pre_silence, post_silence, sample_rate)
            
            # Step 9: Calculate final duration
            total_samples = padded_waveform.shape[1]
            total_duration = float(total_samples) / float(sample_rate)
            
            print(f"\n📊 FINAL AUDIO METRICS:")
            print(f"   Total Duration: {total_duration:.3f}s")
            print(f"   Total Samples: {total_samples}")
            print(f"   Expected Samples: {int(total_duration * sample_rate)}")
            print(f"   Sample Rate: {sample_rate} Hz")
            
            # Step 10: Create output preserving original sample rate
            output_audio, final_sample_rate = self.create_output_dict(padded_waveform, sample_rate)
            
            # Recalculate duration with final sample rate for display
            display_duration = float(total_samples) / float(final_sample_rate)
            
            print(f"\n📊 FINAL AUDIO METRICS:")
            print(f"   Total Duration: {display_duration:.3f}s (at {final_sample_rate}Hz)")
            print(f"   Total Samples: {total_samples}")
            print(f"   Expected Samples: {int(display_duration * final_sample_rate)}")
            print(f"   Final Sample Rate: {final_sample_rate} Hz (preserved from input)")
            
            # Step 11: Create status text with correct timing
            status_parts = []
            status_parts.append(f"📊 Original: {original_duration:.2f}s")
            if pre_pad_seconds > 0 or post_pad_seconds > 0:
                status_parts.append(f"+ Padding: {pre_pad_seconds + post_pad_seconds:.2f}s")
            status_parts.append(f"= Total: {display_duration:.2f}s")
            status_parts.append(f"@ {final_sample_rate}Hz mono")
            
            # Add target comparison using correct duration
            if target_duration > 0:
                diff = display_duration - target_duration
                if abs(diff) < 0.05:
                    status_parts.append("✅ Perfect match!")
                elif diff > 0:
                    status_parts.append(f"⚠️ {diff:.2f}s longer")
                else:
                    status_parts.append(f"⚠️ {-diff:.2f}s shorter")
            
            status_text = " | ".join(status_parts)
            
            print(f"\n✅ PADDING COMPLETE:")
            print(f"   Status: {status_text}")
            print(f"   Output Format: {output_audio['waveform'].shape} @ {output_audio['sample_rate']}Hz")
            print("="*60 + "\n")
            
            return (output_audio, display_duration, status_text)
            
        except Exception as e:
            print(f"\n❌ PADDING ERROR: {e}")
            import traceback
            traceback.print_exc()
            
            # Return original audio with error info
            error_text = f"❌ Error: {str(e)}"
            try:
                # Try to return something reasonable
                if hasattr(audio, 'shape') or (isinstance(audio, dict) and 'waveform' in audio):
                    wform = audio['waveform'] if isinstance(audio, dict) else audio
                    srate = audio.get('sample_rate', 24000) if isinstance(audio, dict) else 24000
                    samples = wform.shape[-1] if len(wform.shape) >= 1 else 0
                    duration = samples / srate
                    return (audio, duration, error_text)
                else:
                    return (audio, 0.0, error_text)
            except:
                return (audio, 0.0, error_text)

NODE_CLASS_MAPPINGS = {
    "BoyoAudioPadder": BoyoAudioPadder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoAudioPadder": "Boyo Audio Padder (Debug)"
}
