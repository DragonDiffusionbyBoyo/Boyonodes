import torch
import numpy as np

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
    
    def pad_audio(self, audio, pre_pad_seconds, post_pad_seconds, target_duration=0.0, auto_center=False):
        try:
            print("🔧 Padding audio...")
            
            # Parse audio format (using same logic as duration analyzer)
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
                waveform, sample_rate = audio[0], audio[1]
            elif hasattr(audio, 'shape'):
                waveform = audio
                sample_rate = 22050
            else:
                raise ValueError(f"Unsupported audio data format: {type(audio)}")
            
            print(f"Debug: Input audio shape: {waveform.shape}, sample_rate: {sample_rate}")
            
            # Convert to tensor if needed and ensure float32
            if isinstance(waveform, np.ndarray):
                waveform = torch.from_numpy(waveform.astype(np.float32))
            elif hasattr(waveform, 'cpu'):
                waveform = waveform.cpu().float()
            else:
                waveform = waveform.float()
            
            # Get original dimensions and duration
            original_shape = waveform.shape
            print(f"Debug: Original tensor shape: {original_shape}")
            
            # Determine audio format and dimensions
            if len(original_shape) == 3:  # (batch, channels, samples) or (batch, samples, channels)
                # Check which dimension is likely samples (largest one)
                if original_shape[1] > original_shape[2]:  # (batch, samples, channels)
                    batch_size, num_samples, num_channels = original_shape
                    sample_axis = 1
                else:  # (batch, channels, samples) - more common
                    batch_size, num_channels, num_samples = original_shape
                    sample_axis = 2
                    # Reshape to (batch, samples, channels) for consistency
                    waveform = waveform.transpose(1, 2)
            elif len(original_shape) == 2:
                if original_shape[0] <= 8:  # (channels, samples)
                    num_channels, num_samples = original_shape
                    batch_size = 1
                    sample_axis = 1
                    # Reshape to (batch, samples, channels) for consistency
                    waveform = waveform.transpose(0, 1).unsqueeze(0)
                else:  # (batch, samples) or (samples, channels)
                    if original_shape[1] <= 8:  # (samples, channels)
                        num_samples, num_channels = original_shape
                        batch_size = 1
                        sample_axis = 0
                        # Reshape to (batch, samples, channels)
                        waveform = waveform.unsqueeze(0)
                    else:  # (batch, samples) - assume mono
                        batch_size, num_samples = original_shape
                        num_channels = 1
                        sample_axis = 1
                        # Add channel dimension
                        waveform = waveform.unsqueeze(-1)
            elif len(original_shape) == 1:  # (samples,) - mono
                num_samples = original_shape[0]
                batch_size = 1
                num_channels = 1
                sample_axis = 0
                # Reshape to (batch, samples, channels)
                waveform = waveform.unsqueeze(0).unsqueeze(-1)
            else:
                raise ValueError(f"Unexpected waveform shape: {original_shape}")
            
            print(f"Debug: Normalized to batch={batch_size}, samples={num_samples}, channels={num_channels}")
            print(f"Debug: Current tensor shape: {waveform.shape}")
            
            # Calculate durations
            original_duration = float(num_samples) / float(sample_rate)
            
            # Handle auto-centering
            if auto_center and target_duration > original_duration:
                total_padding_needed = target_duration - original_duration
                pre_pad_seconds = total_padding_needed / 2.0
                post_pad_seconds = total_padding_needed / 2.0
                print(f"🎯 Auto-center: {total_padding_needed:.2f}s padding split equally")
            
            # Calculate padding samples
            pre_pad_samples = int(pre_pad_seconds * sample_rate)
            post_pad_samples = int(post_pad_seconds * sample_rate)
            
            print(f"Debug: Pre-padding {pre_pad_samples} samples ({pre_pad_seconds:.3f}s)")
            print(f"Debug: Post-padding {post_pad_samples} samples ({post_pad_seconds:.3f}s)")
            
            # Create silence tensors with matching format
            # ComfyUI expects (batch, samples, channels)
            if pre_pad_samples > 0:
                pre_silence = torch.zeros(batch_size, pre_pad_samples, num_channels, dtype=torch.float32, device=waveform.device)
                print(f"Debug: Pre-silence shape: {pre_silence.shape}")
            
            if post_pad_samples > 0:
                post_silence = torch.zeros(batch_size, post_pad_samples, num_channels, dtype=torch.float32, device=waveform.device)
                print(f"Debug: Post-silence shape: {post_silence.shape}")
            
            # Concatenate audio with padding
            tensors_to_cat = []
            
            if pre_pad_samples > 0:
                tensors_to_cat.append(pre_silence)
            
            tensors_to_cat.append(waveform)
            
            if post_pad_samples > 0:
                tensors_to_cat.append(post_silence)
            
            # Concatenate along sample axis (axis 1 for batch, samples, channels)
            padded_waveform = torch.cat(tensors_to_cat, dim=1)
            
            print(f"Debug: Final padded shape: {padded_waveform.shape}")
            
            # Calculate final duration
            total_samples = padded_waveform.shape[1]
            total_duration = float(total_samples) / float(sample_rate)
            
            # Create status text with helpful feedback
            status_parts = []
            status_parts.append(f"📊 Audio: {original_duration:.2f}s")
            
            if pre_pad_seconds > 0 or post_pad_seconds > 0:
                status_parts.append(f"+ Padding: {pre_pad_seconds + post_pad_seconds:.2f}s")
            
            status_parts.append(f"= Total: {total_duration:.2f}s")
            
            # Add comparison to target if specified
            if target_duration > 0:
                diff = total_duration - target_duration
                if abs(diff) < 0.05:  # Within 50ms
                    status_parts.append("✅ Perfect match!")
                elif diff > 0:
                    status_parts.append(f"⚠️ {diff:.2f}s longer than target")
                else:
                    status_parts.append(f"⚠️ {-diff:.2f}s shorter than target")
                    
                # Suggest auto-center if not used
                if not auto_center and target_duration > original_duration:
                    suggested_pad = (target_duration - original_duration) / 2
                    status_parts.append(f"💡 Try auto_center or {suggested_pad:.2f}s each")
            
            status_text = " | ".join(status_parts)
            
            # Create output audio in ComfyUI format
            # Force 12kHz output to match ComfyUI's expected playback rate
            output_sample_rate = 12000  # This fixes the "too fast" audio issue
            
            output_audio = {
                "waveform": padded_waveform,
                "sample_rate": int(output_sample_rate)
            }
            
            print(f"✅ Padding complete - Total duration: {total_duration:.3f}s")
            
            return (output_audio, total_duration, status_text)
            
        except Exception as e:
            print(f"❌ Error padding audio: {e}")
            import traceback
            traceback.print_exc()
            
            # Return original audio and error info
            error_text = f"❌ Error: {str(e)}"
            original_duration = 0.0
            
            # Try to calculate original duration for fallback
            try:
                if hasattr(audio, 'shape') or (isinstance(audio, dict) and 'waveform' in audio):
                    wform = audio['waveform'] if isinstance(audio, dict) else audio
                    srate = audio.get('sample_rate', 22050) if isinstance(audio, dict) else 22050
                    samples = wform.shape[-1] if len(wform.shape) >= 1 else 0
                    original_duration = samples / srate
            except:
                pass
            
            return (audio, original_duration, error_text)

NODE_CLASS_MAPPINGS = {
    "BoyoAudioPadder": BoyoAudioPadder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoAudioPadder": "Boyo Audio Padder"
}
