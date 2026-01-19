import torch
import torchaudio
import numpy as np
from pathlib import Path
import sys
import os
import soundfile as sf

# Add seed-vc to path (assuming it's a submodule in the same directory)
SEED_VC_PATH = Path(__file__).parent / "seed-vc"
if str(SEED_VC_PATH) not in sys.path:
    sys.path.insert(0, str(SEED_VC_PATH))

# Lazy imports - only import when needed
_vc_wrapper = None
_models_loaded = False

def get_vc_wrapper():
    """Load models once and cache"""
    global _vc_wrapper, _models_loaded
    
    if _models_loaded:
        return _vc_wrapper
    
    print("🎤 Loading Seed-VC models (this may take a moment on first run)...")
    
    try:
        from seed_vc_wrapper import SeedVCWrapper
        
        # Determine device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        print(f"🔧 Using device: {device}")
        
        # Initialize wrapper (this loads all models)
        _vc_wrapper = SeedVCWrapper(device=device)
        _models_loaded = True
        
        print("✅ Seed-VC models loaded successfully!")
        return _vc_wrapper
        
    except Exception as e:
        print(f"❌ Failed to load Seed-VC models: {e}")
        import traceback
        traceback.print_exc()
        raise


class BoyoVoiceEnhancer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_voice": ("AUDIO",),
                "target_voice": ("AUDIO",),
                "diffusion_steps": ("INT", {
                    "default": 30,
                    "min": 10,
                    "max": 100,
                    "step": 1,
                    "display": "slider"
                }),
                "length_adjust": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "intelligibility_cfg": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "similarity_cfg": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "convert_style": ("BOOLEAN", {
                    "default": False
                }),
            },
            "optional": {
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.0,
                    "min": 1.0,
                    "max": 3.0,
                    "step": 0.1
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("enhanced_audio",)
    FUNCTION = "enhance_voice"
    CATEGORY = "Boyo/Audio/VoiceEnhancement"
    
    def audio_to_numpy(self, audio_dict, target_sr=22050):
        """Convert ComfyUI audio dict to numpy array at target sample rate"""
        waveform = audio_dict["waveform"]
        sample_rate = audio_dict["sample_rate"]
        
        # Convert to numpy
        if hasattr(waveform, 'cpu'):
            waveform = waveform.cpu()
        if hasattr(waveform, 'numpy'):
            waveform = waveform.numpy()
        elif hasattr(waveform, 'detach'):
            waveform = waveform.detach().numpy()
        
        # Ensure float32 dtype
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)
        
        # Handle tensor shapes (exactly like Chatterbox)
        if len(waveform.shape) == 3:
            waveform = waveform[0, 0]
        elif len(waveform.shape) == 2:
            if waveform.shape[0] <= 2:
                waveform = waveform[0]
            else:
                waveform = waveform[0]
        
        # Resample if needed
        if sample_rate != target_sr:
            print(f"🔄 Resampling from {sample_rate}Hz to {target_sr}Hz")
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sr)
        
        print(f"🔍 Output shape: {waveform.shape}, dtype: {waveform.dtype}")
        return waveform, target_sr
    
    def numpy_to_audio_dict(self, audio_np, sample_rate):
        """Convert numpy array to ComfyUI audio dict format"""
        if isinstance(audio_np, np.ndarray):
            audio_tensor = torch.from_numpy(audio_np).float()
        else:
            audio_tensor = audio_np.float()
        
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        print(f"📤 Output format: {audio_tensor.shape} @ {sample_rate}Hz")
        
        return {
            "waveform": audio_tensor,
            "sample_rate": int(sample_rate)
        }
    
    def enhance_voice(self, video_voice, target_voice, diffusion_steps, length_adjust,
                     intelligibility_cfg, similarity_cfg, convert_style,
                     top_p=0.9, temperature=1.0, repetition_penalty=1.0):
        
        try:
            print("🎙️ Starting voice enhancement...")
            
            vc_wrapper = get_vc_wrapper()
            
            print("📥 Processing source audio (video voice)...")
            source_audio, source_sr = self.audio_to_numpy(video_voice, target_sr=22050)
            
            print("📥 Processing reference audio (target voice)...")
            ref_audio, ref_sr = self.audio_to_numpy(target_voice, target_sr=22050)
            
            import tempfile
            
            source_tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            source_path = source_tmp.name
            source_tmp.close()
            sf.write(source_path, source_audio, source_sr)
            
            ref_tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            ref_path = ref_tmp.name
            ref_tmp.close()
            sf.write(ref_path, ref_audio, ref_sr)
            
            try:
                print(f"🔄 Converting voice (diffusion_steps={diffusion_steps})...")
                
                result_gen = vc_wrapper.convert_voice(
                    source=source_path,
                    target=ref_path,
                    diffusion_steps=diffusion_steps,
                    length_adjust=length_adjust,
                    inference_cfg_rate=similarity_cfg,
                    f0_condition=False,
                    auto_f0_adjust=True,
                    pitch_shift=0,
                    stream_output=False
                )
                
                return_value = None
                try:
                    while True:
                        next(result_gen)
                except StopIteration as e:
                    return_value = e.value
                
                if return_value is None:
                    raise Exception("No audio data returned from conversion")
                
                result = return_value
                
                print("✅ Voice conversion complete!")
                print(f"🔍 Result type: {type(result)}")
                
                enhanced_audio = self.numpy_to_audio_dict(result, 22050)
                
                return (enhanced_audio,)
                
            finally:
                try:
                    os.unlink(source_path)
                    os.unlink(ref_path)
                except:
                    pass
        
        except Exception as e:
            print(f"❌ Voice enhancement failed: {e}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Voice enhancement failed: {e}")


NODE_CLASS_MAPPINGS = {
    "BoyoVoiceEnhancer": BoyoVoiceEnhancer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoVoiceEnhancer": "Boyo Voice Enhancer"
}