# Boyonodes
Essential ComfyUI nodes for semantic image editing, audio processing, LoRA management, and automated workflow generation. Streamlines complex pipelines with intelligent automation and robust error handling.

## 🚀 Quick Installation
```bash
git clone https://github.com/DragonDiffusionbyBoyo/Boyonodes.git
cp -r Boyonodes /path/to/ComfyUI/custom_nodes/
```

**Restart ComfyUI** after installation. Most nodes work immediately with no additional dependencies.

## 📋 Installation Requirements

### Core Nodes (No additional dependencies)
- Semantic Image Editing nodes
- LoRA Management System  
- Workflow Enhancement nodes
- Basic Utility nodes

### Audio Processing Nodes (Basic)
```bash
pip install librosa transformers safetensors huggingface_hub pyloudnorm soundfile
pip install chatterbox-tts --no-deps
pip install resemble-perth --no-deps
```

### Advanced Audio Enhancement (Seed-VC)
**See installation instructions at the top of this README**

### Mandelbrot Video Generator
```bash
pip install numpy==1.26 matplotlib pillow tqdm torch
```

### FFmpeg (for video output)
- **Windows**: Download from ffmpeg.org or `choco install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt-get install ffmpeg`

## 🎵 Audio Processing & TTS

### Chatterbox Turbo TTS Integration
High-quality voice cloning and text-to-speech generation with emotion support.

**Key Features:**
- **Voice Cloning**: Clone any voice from 5+ second reference audio
- **Emotion Tags**: Natural expressions like `[chuckle]`, `[sigh]`, `[gasp]`
- **GPU Accelerated**: CUDA support for fast generation
- **24kHz Quality**: Professional audio output

**Available Emotion Tags:**
`[clear throat]` `[sigh]` `[shush]` `[cough]` `[groan]` `[sniff]` `[gasp]` `[chuckle]` `[laugh]`

**Workflow:**
1. **Boyo Chatterbox Turbo Loader** - Load TTS model
2. **Boyo Chatterbox Turbo Generate** - Create speech with emotion
3. Standard ComfyUI audio nodes for processing

### Audio Padding & Synchronization
**Finally solved the community's lip-sync timing challenge!** Precision audio padding for perfect video-audio synchronization.

**Key Nodes:**
- **BoyoAudioDurationAnalyzer** - Extract precise duration from any audio tensor
- **BoyoAudioPadder** - Intelligent silence padding with auto-centering

**Why These Nodes Succeed:**
- ✅ Handles all ComfyUI audio formats (`(batch,channels,samples)` vs `(batch,samples,channels)`)
- ✅ Automatic 12kHz output for proper playback speed  
- ✅ Intelligent feedback: `📊 Audio: 3.2s + Padding: 7.3s = Total: 10.5s ✅ Perfect match!`
- ✅ Memory-efficient processing for large files
- ✅ Auto-centering with target duration matching

**Quick Workflow:**
```
Load Audio → BoyoAudioPadder (target_duration: 10.5s, auto_center: true) → Save Audio
```

---

## 🎨 Semantic Image Editing System

Perfect for **Kontext**, **Qwen Image Edit**, and **HiDream E1.1** workflows with automated iteration and dataset creation.

### Core Editing Nodes

**Boyo Image Grab**
- Auto-monitors directories for newest images
- Enables seamless iterative editing chains
- Perfect for progressive semantic modifications
- Real-time directory monitoring

**Boyo Paired Image Saver**
- Saves original/edited pairs with sequential naming
- Dataset creation for training workflows
- ControlNet format compatibility
- Organized file management

**Boyo Incontext Saver**
- Specialized for semantic editing outputs
- Dataset-ready organization
- Maintains editing relationships

**Boyo Universal Image + Prompt Saver**
- Strips metadata for clean publication
- Saves actual prompts (crucial for wildcard workflows)
- Creates organized image/text pairs
- Publication-ready outputs

### Workflow Example
```
Original → Semantic Edit → Paired Saver → Image Grab (auto-feeds next iteration)
```

---

## 🎯 LoRA Management System

Revolutionary paired LoRA management with intelligent prompt handling for complex workflows requiring multiple LoRA types.

### Boyo LoRA JSON Builder
Create and save LoRA configurations with flexible prompt management.

**Key Features:**
- Supports paired LoRAs (high/low noise variants)
- Multiple prompts per configuration
- Handles utility LoRAs (no prompts needed)
- Auto-saves to organized directory

### Boyo LoRA Paired Loader
Load multiple LoRA configurations simultaneously with advanced prompt strategies.

**Key Features:**
- **3 simultaneous config slots** for layered effects
- **Prompt strategies**: Mute, Concatenate, Merge per config
- **Prompt modes**: First Only, Cycle Through, Random (seed-based)
- **6 LoRA path outputs** + **4 prompt string outputs**
- Direct connection to standard LoRA loaders

### Boyo LoRA Config Inspector
Preview and analyze LoRA configurations before loading.

**Sample Output:**
```
📋 LoRA Configuration: Character_Cyborg
🎯 LoRA FILES:
  📈 High Noise: ✅ cyborg_char_v2.safetensors
  📉 Low Noise: ✅ cyborg_char_v2_low.safetensors
  🎭 Type: PAIRED LoRA

💬 PROMPTS (3 total):
  1. cyborg woman, metallic skin, glowing eyes
  2. android female, chrome details, futuristic
  3. robotic humanoid, synthetic appearance

💡 USAGE RECOMMENDATIONS:
  • Use 'Cycle Through' for variety
  • Use 'Random' for experimentation
```

---

## 📖 AI Storyboard Generation

Automated storyboard creation using local ollama models for consistent multi-scene video workflows.

### Boyo Storyboard Prompt
Intelligent prompt generator for structured storyboard sequences.

**Key Features:**
- **Model-agnostic trigger words** - works with any LoRA/video model
- **Two modes**: 6-scene storyboards or traveling prompt sequences
- **Consistent character/style** across all scenes
- **Optimized for abliterated coder models** (Qwen 30B A3B Coder recommended)

**System Prompt 1 (Standard):** 6 image + 6 video prompts for Next Scene LoRA workflows  
**System Prompt 2 (Traveling):** 6 images + 6 multi-line video sequences for extended content

### Boyo Storyboard Output
Parses ollama JSON responses into 12 separate prompt outputs for direct workflow integration.

**Workflow:**
```
Storyboard Prompt → ollama Generate → Storyboard Output → 12 individual prompts
```

---

## 🛠️ Utility & Enhancement Nodes

### Asset Downloader System
**One-click workflow dependency installation.** Drop JSON manifest files to automatically download custom nodes, models, and dependencies.

**Features:**
- Automatic GitHub repository cloning
- Python dependency installation via pip  
- Model downloads from direct URLs
- Available in Simple (auto-detect) and Advanced (custom paths) versions

### Workflow Enhancement
- **Boyo Empty Latent** - Smart aspect ratio calculator
- **Load Image List** - Batch image processor for mass operations
- **Boyo VAE Decode** - Stealth NSFW filtering for controlled environments
- **Boyo Tiled VAE Decode** - Memory-efficient large image processing

### Creative Tools
- **Mandelbrot Video Generator** - Fractal art for creative projects
- **BoyoVision Node** - Qwen2.5VL vision with abliterated model compatibility

---

## 📚 Workflow Examples

### Multi-LoRA Character Generation
1. Create configs for utility (lightning), character, and style LoRAs
2. Load all three simultaneously in Paired Loader
3. Set strategies: utility = "Mute", character/style = "Concatenate"  
4. Get combined prompts and all LoRA paths in one node

### Iterative Semantic Editing
1. Load initial image
2. Apply semantic edit (Kontext/Qwen/HiDream)
3. Boyo Paired Image Saver stores original + edit
4. Boyo Image Grab auto-feeds edit for next iteration
5. Repeat for progressive modifications

### Video Lip-Sync Workflow
1. Load Video → VideoHelperSuite Info → get duration
2. Generate TTS audio → BoyoAudioPadder with auto-center
3. Perfect timing match for lip-sync models

### Storyboard-to-Video Pipeline
1. Configure story/character in Storyboard Prompt  
2. Generate via ollama → parse with Storyboard Output
3. Connect 6 image outputs to Next Scene LoRA
4. Connect 6 video outputs to video generation

---

## 🔧 Troubleshooting

### Audio Issues
- **Audio too fast**: Node automatically outputs 12kHz for proper ComfyUI playback
- **TTS loading errors**: Ensure dependencies installed with `--no-deps` flags
- **CUDA problems**: Set device to "cpu" in loader node

### LoRA Management
- **Config not loading**: Use Inspector node to verify file paths and JSON syntax
- **Missing LoRA files**: Check paths use forward slashes, verify file existence
- **Prompt issues**: Inspector shows available prompts and recommendations

### Semantic Editing
- **Image Grab not updating**: Verify directory path exists, check auto_refresh enabled
- **Paired Saver failing**: Confirm output directory exists and has write permissions
- **Slow performance**: Organize files into smaller subdirectories

### Storyboard Generation  
- **Poor outputs**: Use recommended ollama models (Qwen 30B A3B Coder Abliterated)
- **JSON parse errors**: Add verbosity instructions in additional_details field
- **Avoid**: Google models (Gemma), Meta coding variants, thinking models

---

## 🏗️ Node Categories

- **Boyo/Audio/TTS** - Text-to-speech and voice cloning
- **Boyo/Audio/Analysis** - Audio duration and analysis tools
- **Boyo/Audio/Processing** - Padding and timing control  
- **Boyo/Image/Editing** - Semantic editing workflow tools
- **Boyo/LoRA** - LoRA management and configuration
- **Boyo/Storyboard** - AI storyboard generation
- **Boyo/Utility** - General workflow enhancement tools

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature-name`)
3. Commit changes
4. Push to branch  
5. Open pull request

Documentation for new features is appreciated.

---

## 📄 License

MIT License - see LICENSE file for details.

**Built by DragonDiffusionbyBoyo for the semantic editing revolution.**

---

*Note: Vision nodes currently disabled due to dependency conflicts. Resolution in progress.*
