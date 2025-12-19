# Boyo Chatterbox Turbo TTS Nodes for ComfyUI

## Installation

1. **Install Dependencies** (in your ComfyUI environment):
   ```bash
   pip install librosa>=0.10.0 transformers>=4.30.0 safetensors>=0.3.0 huggingface_hub>=0.15.0 perth>=0.1.0 pyloudnorm>=0.1.0 soundfile>=0.12.0
   ```

   **Important**: Do NOT install torch, numpy, or torchaudio as these should use ComfyUI's versions.

2. **Install Chatterbox TTS**:
   ```bash
   pip install chatterbox-tts
   ```

3. **Add the node files** to your ComfyUI custom_nodes directory:
   - `boyo_chatterbox_turbo_loader.py`
   - `boyo_chatterbox_turbo_generate.py`

4. **Update your `__init__.py`** file with the new imports (see `boyo_init_updated.py`)

## Usage

### Basic Workflow:

1. **Load Model**: Add "Boyo Chatterbox Turbo Loader" node
   - Choose device (auto/cpu/cuda/mps)
   - Optional: specify local model path

2. **Load Reference Audio**: Use ComfyUI's standard "Load Audio" node
   - Load a 5+ second audio clip for voice cloning

3. **Generate Speech**: Add "Boyo Chatterbox Turbo Generate" node
   - Connect model from loader
   - Connect audio from load audio node
   - Enter your text (supports paralinguistic tags like [chuckle], [sigh])
   - Adjust generation parameters as needed

4. **Save Audio**: Use ComfyUI's "Save Audio" node to save as FLAC

### Paralinguistic Tags:
You can add emotional expressions to your text:
- `[chuckle]` - Light laughter
- `[laugh]` - Full laughter  
- `[sigh]` - Sighing
- `[gasp]` - Surprise gasp
- `[cough]` - Coughing
- `[clear throat]` - Throat clearing
- `[groan]` - Groaning
- `[sniff]` - Sniffing
- `[shush]` - Shushing sound

### Example Text:
```
Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. It's the SkyNet T-800 series and it's got basically everything.
```

## Parameters:

- **Temperature**: Controls randomness (0.05-2.0, default 0.8)
- **Top P**: Nucleus sampling (0.0-1.0, default 0.95)
- **Top K**: Top-k sampling (0-1000, default 1000)
- **Repetition Penalty**: Reduces repetition (1.0-2.0, default 1.2)
- **Seed**: For reproducible results (0 for random)
- **Exaggeration**: Emotion intensity (0.0-1.0, default 0.0)
- **Normalize Loudness**: Standardize audio volume

## Troubleshooting:

- **Import Error**: Make sure chatterbox-tts is installed in your ComfyUI environment
- **CUDA Issues**: Try setting device to "cpu" if GPU causes problems
- **Audio Format Issues**: Ensure reference audio is at least 5 seconds long
- **Generation Slow**: This is normal for CPU inference, try GPU if available

## Notes:

- First run will download model weights from HuggingFace (~2GB)
- Model supports voice cloning from reference audio
- Generated audio will match the reference voice characteristics
- Works with ComfyUI's audio pipeline (loads MP3/WAV/FLAC, saves as FLAC)
