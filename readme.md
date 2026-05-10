[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/E1E01W3X6E)

# Boyonodes

Custom ComfyUI nodes for audio/TTS, video generation, LoRA management, looping workflows, semantic image editing, and general pipeline utilities.

Full documentation for every node is in [NODESREADME.md](NODESREADME.md).

---

## Installation

```bash
git clone https://github.com/DragonDiffusionbyBoyo/Boyonodes.git
cp -r Boyonodes /path/to/ComfyUI/custom_nodes/
```

Restart ComfyUI. Most nodes need no additional dependencies.

### Audio & TTS nodes
```bash
pip install librosa transformers safetensors huggingface_hub pyloudnorm soundfile
pip install chatterbox-tts --no-deps
pip install resemble-perth --no-deps
```

### Voice enhancement (Seed-VC)
```bash
git submodule add https://github.com/Plachtaa/seed-vc.git seed-vc
git submodule update --init --recursive
pip install hydra-core omegaconf munch descript-audio-codec
pip install -r requirements.txt
```
A portable batch installer is provided: `Portable_auto_install_nodes.bat`

### LUT node
```bash
pip install colour-science
```

### Qwen grounding node
```bash
pip install qwen-vl-utils
```

### FFmpeg (video output)
- **Windows**: `choco install ffmpeg` or download from ffmpeg.org
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt-get install ffmpeg`

---

## Nodes — Quick Reference

### Audio & TTS
| Node | Summary |
|------|---------|
| Boyo Chatterbox Turbo Loader | Load the Chatterbox Turbo TTS model onto GPU or CPU. |
| Boyo Chatterbox Turbo Generate | Generate speech from text with emotion tags and optional voice cloning. |
| Boyo Audio Duration Analyzer | Extract the precise duration in seconds from any ComfyUI audio tensor. |
| Boyo Audio Padder | Pad audio with silence to hit a target duration; essential for lip-sync timing. |
| Boyo Voice Enhancer | Enhance or convert voice characteristics via Seed-VC. |
| Boyo Audio Eval | Convert an audio file's duration to a frame count at a given FPS. |

### Image
| Node | Summary |
|------|---------|
| Boyo Image Grab | Monitor a directory and automatically load the newest image for iterative editing chains. |
| Boyo Paired Image Saver | Save original/edited image pairs with sequential naming for datasets and ControlNet. |
| Boyo Image Crop | Tile-crop a large image into overlapping patches and save them to disk. |
| Boyo Qwen VL Grounding | Run Qwen2.5-VL visual grounding to detect and annotate objects with bounding boxes. |

### LoRA Management
| Node | Summary |
|------|---------|
| Boyo LoRA JSON Builder | Create and save LoRA configuration JSON files with paired high/low noise paths and multiple prompts. |
| Boyo LoRA Paired Loader | Load up to three LoRA configs simultaneously with per-slot prompt strategies. |
| Boyo LoRA Config Inspector | Preview a LoRA config file and get usage recommendations before loading. |
| Boyo LoRA Config Processor | Process a LoRA config and handle prompt cycling, merging, or muting logic. |
| Boyo LoRA Path Forwarder | Buffer and forward LoRA paths to standard ComfyUI LoRA loader nodes. |
| Boyo LoRA Info Sender | Simple dropdown selector that outputs the chosen LoRA filename as a string. |
| Boyo FramePack LoRA Loader | Apply a LoRA to a FramePack/Hunyuan Video model via path string. |

### Storyboard
| Node | Summary |
|------|---------|
| Boyo Storyboard Prompt | Generate structured multi-scene storyboard prompts for an ollama model. |
| Boyo Storyboard Output | Parse the ollama JSON response into 6 image and 6 video prompt outputs. |
| Boyo Storyboard JSON Parser | Alternative JSON parser for storyboard responses, identical 12-output structure. |

### Video
| Node | Summary |
|------|---------|
| Boyo Video Clipper | Clip a video from the input directory to an exact frame count at a given FPS and start time. |
| Boyo Video Cutter | Remove overlap frames from an image sequence at specified positions. |
| Boyo Video Length Calculator | Calculate total frames, loop count, and trim positions for a target video duration. |
| Boyo Video Paired Saver | Render an image batch to video and save it alongside its prompt text. |
| Boyo Load Video Directory | Load all video files from a directory as a batch. |
| Boyo Frame Counter | Compute `frames_processed = counter × chunk_size + offset` for audio/video sync in loops. |
| Boyo Overlap Switch | Output a different overlap value for the first loop iteration versus subsequent ones. |
| Boyo Watermarks | Stamp a watermark image onto every frame of a video batch. |

### Looping & Flow Control
| Node | Summary |
|------|---------|
| Boyo While Loop Start/End | While loop nodes with custom execution management for complex iterative workflows. |
| Boyo For Loop Start/End | For loop nodes replicating EasyUse loop logic without the EasyUse dependency. |
| Boyo Loop Reset | Reset one or more loop counters to zero when triggered by a completion signal. |
| Boyo Loop Counter | Increment and track a loop iteration count across executions. |
| Boyo Math Int | Perform basic integer arithmetic inside a loop or workflow. |
| Boyo Compare | Compare two values and output a boolean for conditional branching. |
| Boyo Prompt Loop | Iterate through prompts from a `.txt` file sequentially, randomly, or at a fixed index. |
| Boyo Loop Collector / Loop Image Saver | Accumulate images across loop iterations and optionally save them to disk. |

### Latent & Conditioning
| Node | Summary |
|------|---------|
| Boyo Latent Cache Updater | Write latent data to the loop cache and pass it through unchanged, avoiding graph cycles. |
| Boyo Latent Switch | Select the start latent on loop iteration 0 and the cached latent on all subsequent ones. |
| Boyo Latent Passthrough / Execution Barrier | Pass a latent through unchanged; use the barrier variant to enforce execution order. |
| Boyo Painter SVI | Merge PainterI2V motion amplitude conditioning with SVI context preservation for infinite video. |
| Boyo VACE Injector | Inject VACE control data directly into a model for use without conditioning nodes. |
| Boyo VACE Viewer | Inspect and preview VACE control data attached to a model. |

### Utilities
| Node | Summary |
|------|---------|
| Boyo Resolution Calc | Output width and height from a base width and a named aspect ratio preset. |
| Boyo Apply LUT | Apply a `.cube`, `.3dl`, or `.spi3d` colour-grading LUT to an image. |
| Boyo Mask To Image | Convert a mask tensor to a 3-channel image for nodes that expect IMAGE input. |
| Boyo Asset Grabber Simple | Download nodes, models, and pip packages listed in a JSON manifest (auto-detects paths). |
| Boyo Asset Grabber Advanced | Same as Simple but with explicit custom path overrides. |
| Boyo Prompt Relay Encode | Port of kijai's Prompt Relay — encode temporally-segmented prompts for video models. |
| Boyo Prompt Relay Encode Timeline | Timeline-based variant of Prompt Relay for fine-grained per-segment control. |
| Boyo Prompt Relay LoRA Gate | Apply a LoRA to a single temporal segment using Gaussian gating, as an extension of Prompt Relay. |

---

## Licence

MIT — see LICENSE file.

*Built by DragonDiffusionbyBoyo.*
