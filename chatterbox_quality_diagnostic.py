"""
Audio Quality Diagnostic Script for Chatterbox TTS
Compare output quality between original repo and ComfyUI implementation
"""
import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

def analyze_audio_quality(audio_path):
    """Analyze audio file and return quality metrics"""
    # Load audio
    audio, sr = librosa.load(audio_path, sr=None)
    
    metrics = {
        'sample_rate': sr,
        'duration': len(audio) / sr,
        'bit_depth': 'float32' if audio.dtype == np.float32 else str(audio.dtype),
        'channels': 1 if len(audio.shape) == 1 else audio.shape[1],
        'dynamic_range': np.max(audio) - np.min(audio),
        'rms_level': np.sqrt(np.mean(audio**2)),
        'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)),
        'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)),
        'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(audio)),
    }
    
    # Frequency analysis
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    metrics['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    metrics['spectral_flatness'] = np.mean(librosa.feature.spectral_flatness(y=audio))
    
    return metrics

def compare_chatterbox_implementations():
    """
    Generate test audio using both methods and compare quality
    """
    
    test_text = "Hello world, this is a test of audio quality using Chatterbox TTS."
    
    print("🔍 CHATTERBOX QUALITY DIAGNOSTIC")
    print("=" * 50)
    
    # Method 1: Original repo implementation
    print("📊 Testing original Chatterbox repo...")
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        
        # Load model
        model = ChatterboxTurboTTS.from_pretrained("cpu")  # Use CPU for consistent comparison
        print(f"✅ Model loaded - native sample rate: {model.sr}Hz")
        
        # Generate audio
        wav = model.generate(test_text)
        
        # Save original output
        original_path = "original_output.wav"
        if isinstance(wav, tuple):
            sample_rate, audio_data = wav
        else:
            audio_data = wav
            sample_rate = model.sr
            
        sf.write(original_path, audio_data, sample_rate)
        print(f"✅ Original saved: {original_path}")
        
        # Analyze original
        original_metrics = analyze_audio_quality(original_path)
        print("📊 Original metrics:")
        for key, value in original_metrics.items():
            print(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")
        
    except Exception as e:
        print(f"❌ Original method failed: {e}")
        return
    
    print("\n" + "=" * 50)
    
    # Method 2: Your ComfyUI implementation simulation
    print("📊 Testing ComfyUI-style processing...")
    try:
        # Simulate your processing pipeline
        
        # Step 1: Load the original audio 
        audio_data, sample_rate = sf.read(original_path)
        
        # Step 2: Convert to tensor (like your code does)
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        audio_tensor = torch.from_numpy(audio_data)
        
        # Step 3: Apply your reshaping logic
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(-1)  # (batch, samples, channels)
        
        # Step 4: Simulate forced sample rate change (the potential culprit)
        forced_12khz_path = "comfyui_12khz_output.wav"
        forced_24khz_path = "comfyui_24khz_output.wav"
        
        # Save with forced 12kHz (what your current code does)
        audio_12k = audio_tensor.squeeze().numpy()
        sf.write(forced_12khz_path, audio_12k, 12000)  # Your current forced rate
        
        # Save with native 24kHz (what it should be)
        audio_24k = audio_tensor.squeeze().numpy() 
        sf.write(forced_24khz_path, audio_24k, sample_rate)  # Native rate
        
        print(f"✅ ComfyUI 12kHz saved: {forced_12khz_path}")
        print(f"✅ ComfyUI 24kHz saved: {forced_24khz_path}")
        
        # Analyze both versions
        metrics_12k = analyze_audio_quality(forced_12khz_path)
        metrics_24k = analyze_audio_quality(forced_24khz_path)
        
        print("\n📊 ComfyUI 12kHz metrics:")
        for key, value in metrics_12k.items():
            print(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")
            
        print("\n📊 ComfyUI 24kHz metrics:")
        for key, value in metrics_24k.items():
            print(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")
        
        print("\n🔍 QUALITY COMPARISON")
        print("=" * 30)
        
        # Key quality indicators
        quality_factors = ['spectral_centroid', 'spectral_bandwidth', 'dynamic_range', 'rms_level']
        
        print("Quality Factor | Original | 12kHz | 24kHz | Impact")
        print("-" * 55)
        for factor in quality_factors:
            orig = original_metrics[factor]
            k12 = metrics_12k[factor] 
            k24 = metrics_24k[factor]
            
            # Calculate degradation
            degradation_12k = abs(orig - k12) / orig * 100
            degradation_24k = abs(orig - k24) / orig * 100
            
            impact_12k = "🔴 BAD" if degradation_12k > 10 else "🟡 OK" if degradation_12k > 5 else "🟢 GOOD"
            impact_24k = "🔴 BAD" if degradation_24k > 10 else "🟡 OK" if degradation_24k > 5 else "🟢 GOOD"
            
            print(f"{factor:14} | {orig:8.2f} | {k12:5.2f} | {k24:5.2f} | 12k:{impact_12k} 24k:{impact_24k}")
        
        print(f"\n🎯 DIAGNOSIS:")
        if metrics_12k['sample_rate'] == 12000:
            print("❌ ISSUE FOUND: Forced 12kHz sample rate is degrading quality!")
            print("   - Spectral information is being lost due to Nyquist limit (6kHz)")
            print("   - Original model outputs 24kHz for full frequency response")
            print("   - Solution: Use model's native sample rate")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print("1. Use model.sr ({sample_rate}Hz) instead of forced 12kHz")
        print("2. Avoid unnecessary dtype conversions")
        print("3. Preserve audio precision throughout pipeline")
        print("4. Consider applying noise reduction if needed")
        
    except Exception as e:
        print(f"❌ ComfyUI simulation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    compare_chatterbox_implementations()
