# NODESREADME — Boyonodes Detailed Reference

Full node-by-node documentation. For a one-line summary of every node see [README.md](README.md).

---

## Audio & TTS

### Boyo Chatterbox Turbo Loader
**File:** `boyo_chatterbox_turbo_loader.py` | **Category:** Boyo/Audio/TTS

Loads the Chatterbox Turbo text-to-speech model into memory. Run this once and feed the model handle into the generate node.

**Inputs**
- `device` — `cuda` or `cpu`

**Outputs**
- `model` — Chatterbox model handle

**Notes:** Install with `pip install chatterbox-tts --no-deps` and `pip install resemble-perth --no-deps`. The `--no-deps` flags are required to avoid conflicts with ComfyUI's own dependency stack.

---

### Boyo Chatterbox Turbo Generate
**File:** `boyo_chatterbox_turbo_generate.py` | **Category:** Boyo/Audio/TTS

Generate speech from text. Supports emotion tags inline in the transcript and optional voice cloning from a reference audio clip.

**Inputs**
- `model` — handle from Loader node
- `text` — transcript; embed emotion tags anywhere e.g. `Hello [chuckle] how are you?`
- `reference_audio` *(optional)* — 5+ second audio clip to clone the voice from
- `exaggeration` — emotion intensity multiplier
- `cfg_weight` — classifier-free guidance weight
- `seed`

**Outputs**
- `audio` — ComfyUI AUDIO tensor at 24 kHz

**Supported emotion tags:** `[clear throat]` `[sigh]` `[shush]` `[cough]` `[groan]` `[sniff]` `[gasp]` `[chuckle]` `[laugh]`

---

### Boyo Audio Duration Analyzer
**File:** `boyo_audio_duration_analyzer.py` | **Category:** Boyo/Audio/Analysis

Extracts the precise duration from a ComfyUI audio tensor. Handles both `(batch, channels, samples)` and `(batch, samples, channels)` layouts automatically.

**Inputs**
- `audio` — ComfyUI AUDIO tensor

**Outputs**
- `duration_seconds` FLOAT
- `sample_rate` INT
- `info` STRING — human-readable summary

---

### Boyo Audio Padder
**File:** `boyo_audio_padder.py` | **Category:** Boyo/Audio/Processing

Pads audio with silence to match a target duration. Solves the classic ComfyUI lip-sync timing problem where generated audio is shorter than the video clip.

**Inputs**
- `audio` — source audio tensor
- `target_duration` FLOAT — desired total length in seconds
- `auto_center` BOOLEAN — pad equally before and after instead of only at the end
- `sample_rate` INT — output sample rate (default 12 kHz for correct ComfyUI playback speed)

**Outputs**
- `audio` — padded audio tensor
- `info` STRING — e.g. `📊 Audio: 3.2s + Padding: 7.3s = Total: 10.5s ✅ Perfect match!`

**Typical workflow:** `TTS Generate → Boyo Audio Padder (target = video length) → VHS Save Audio`

---

### Boyo Voice Enhancer
**File:** `boyo_voice_enhancer.py` | **Category:** Boyo/Audio/TTS

Enhances or converts voice characteristics using the Seed-VC model. Requires the seed-vc submodule and its dependencies.

**Inputs**
- `audio` — source audio tensor
- `reference_audio` — target voice reference
- Various Seed-VC inference parameters (diffusion steps, pitch shift, etc.)

**Outputs**
- `audio` — enhanced/converted audio tensor

**Setup:** `git submodule add https://github.com/Plachtaa/seed-vc.git seed-vc && git submodule update --init --recursive`

---

### Boyo Audio Eval
**File:** `BoyoAudioEval.py` | **Category:** VideoUtils

Reads an audio file from disk and converts its duration to a frame count. Useful for sizing video generation to match audio length before any audio tensor is loaded into the workflow.

**Inputs**
- `audio_path` STRING — absolute or relative path to a WAV file
- `fps` FLOAT — target frames per second

**Outputs**
- `frame_count` INT
- `video_length_seconds` FLOAT
- `metadata` STRING — human-readable summary

---

## Image

### Boyo Image Grab
**File:** `boyo_image_grab.py` | **Category:** Boyo/Image/Editing

Monitors a directory and loads the most recently modified image. Designed for iterative semantic editing chains where the output of one pass becomes the input of the next automatically.

**Inputs**
- `directory` STRING
- `auto_refresh` BOOLEAN
- `file_extension` — filter by type

**Outputs**
- `image` IMAGE
- `filename` STRING

---

### Boyo Paired Image Saver
**File:** `boyo_paired_image_saver.py` | **Category:** Boyo/Image/Editing

Saves an original and an edited image as a sequentially numbered pair. Compatible with ControlNet dataset formats. Use alongside Boyo Image Grab to build an iterative editing pipeline.

**Inputs**
- `original_image` IMAGE
- `edited_image` IMAGE
- `output_directory` STRING
- `filename_prefix` STRING

**Outputs:** none (output node)

---

### Boyo Image Crop
**File:** `boyo_image_crop.py` | **Category:** Boyo

Tile-crops a large image into overlapping patches and saves them to a directory. Useful for building tiled datasets from high-resolution source material.

**Inputs**
- `image` IMAGE
- `crop_width` / `crop_height` INT — patch size
- `overlap` INT — pixel overlap between adjacent patches
- `output_path` STRING — destination directory

**Outputs**
- `status` STRING — summary of patches written

---

### Boyo Qwen VL Grounding
**File:** `boyo_qwen_grounding.py` | **Category:** (Boyo/Vision implied)

Runs Qwen2.5-VL visual grounding to locate objects in an image and draw annotated bounding boxes. Supports 4-bit quantisation for lower VRAM usage.

**Inputs**
- `image` IMAGE
- `model_checkpoint` — Qwen2.5-VL model path
- `prompt` STRING — grounding query e.g. `"the red car"`
- `quantize_4bit` BOOLEAN

**Outputs**
- `image` IMAGE — annotated with bounding boxes
- `grounding_json` STRING — raw detection results

**Dependency:** `pip install qwen-vl-utils`

---

## LoRA Management

### Boyo LoRA JSON Builder
**File:** `boyo_lora_json_builder.py` | **Category:** Boyo/LoRA

Creates and saves LoRA configuration JSON files. Each config can hold a paired high-noise and low-noise LoRA path plus an arbitrary list of trigger prompts. Utility LoRAs (e.g. lightning/turbo) can omit prompts entirely.

**Inputs**
- `config_name` STRING
- `high_noise_lora` STRING — filename from your loras folder
- `low_noise_lora` STRING *(optional)*
- `prompts` STRING — one prompt per line
- `save_directory` STRING

**Outputs**
- `config_path` STRING — path to the written JSON file

---

### Boyo LoRA Paired Loader
**File:** `boyo_lora_paired_loader.py` | **Category:** Boyo/LoRA

Loads up to three LoRA configs at once and resolves prompts from each according to a per-slot strategy. Outputs all six LoRA paths and four combined prompt strings ready for downstream nodes.

**Inputs**
- `config_1/2/3` — config file names (dropdowns from your config directory)
- `strategy_1/2/3` — `Mute`, `Concatenate`, or `Merge`
- `prompt_mode` — `First Only`, `Cycle Through`, or `Random`
- `seed` INT — used for deterministic random prompt selection

**Outputs**
- `lora_high_1–3` STRING — high-noise LoRA paths
- `lora_low_1–3` STRING — low-noise LoRA paths
- `prompt_1–4` STRING — resolved combined prompts

**Typical usage:** slot 1 = utility LoRA (strategy: Mute), slots 2–3 = character + style (strategy: Concatenate).

---

### Boyo LoRA Config Inspector
**File:** `boyo_lora_config_inspector.py` | **Category:** Boyo/LoRA

Reads a config file and prints a detailed breakdown: paired vs single LoRA type, all prompt strings, file existence checks, and usage recommendations. Connect before loading to debug config issues.

**Inputs**
- `config_file` STRING

**Outputs**
- `report` STRING — formatted inspection report

---

### Boyo LoRA Config Processor
**File:** `boyo_lora_config_processor.py` | **Category:** Boyo/LoRA

The logic half of a split architecture. Reads a config and applies the chosen prompt mode and strategy, outputting resolved paths and prompt text. Pair with Boyo LoRA Path Forwarder.

**Inputs**
- `prompt_mode` — `First Only`, `Cycle Through`, `Random`
- `config_1/2/3` *(optional)*
- `strategy_1/2/3`
- `prepend_text` / `append_text` STRING
- `seed` INT

**Outputs**
- Resolved LoRA paths and prompt strings

---

### Boyo LoRA Path Forwarder
**File:** `boyo_lora_path_forwarder.py` | **Category:** Boyo/LoRA

Buffers LoRA filenames and forwards them to standard ComfyUI LoRA loader nodes. Caches the LoRA list at startup to prevent dynamic type-change issues during long batch runs.

**Inputs**
- `lora_high` / `lora_low` STRING — filenames from Processor or Paired Loader

**Outputs**
- `lora_high` / `lora_low` STRING — same filenames, now stable for downstream loaders

---

### Boyo LoRA Info Sender
**File:** `boyo_lora_info_sender.py` | **Category:** Boyo/LoRA Tools

Minimal LoRA selector. Presents a dropdown of all available LoRAs and outputs just the selected filename as a string. Designed to feed index-switcher or routing nodes.

**Inputs**
- `lora_name` — dropdown from loras folder

**Outputs**
- `lora_filename` STRING

---

### Boyo FramePack LoRA Loader
**File:** `nodes.py` | **Category:** BoyoNodes

Applies a LoRA to a FramePack (Hunyuan Video) model. Accepts a direct file path rather than a dropdown, standardises key format, and returns the patched model.

**Inputs**
- `model` FramePackMODEL
- `lora_path` STRING — absolute path to `.safetensors`
- `lora_strength` FLOAT (0–2, default 1.0)

**Outputs**
- `model` FramePackMODEL

---

## Storyboard

### Boyo Storyboard Prompt
**File:** `boyo_storyboard_prompt.py` | **Category:** Boyo/Storyboard

Constructs a system prompt and user prompt that instructs an ollama model to generate a structured 6-scene storyboard in JSON. Two modes are available: standard (6 image + 6 video prompts) and travelling (6 images + 6 multi-line video sequences for extended content).

**Inputs**
- `story_outline` STRING
- `character_description` STRING
- `style_description` STRING
- `mode` — `Standard` or `Travelling`
- `additional_details` STRING *(optional)*
- `model_trigger_word` STRING — LoRA/video model trigger

**Outputs**
- `system_prompt` STRING
- `user_prompt` STRING

**Recommended model:** Qwen 30B A3B Coder Abliterated (via ollama). Avoid Gemma, Meta coding variants, and thinking models.

---

### Boyo Storyboard Output
**File:** `boyo_storyboard_output.py` | **Category:** Boyo/Storyboard

Parses the JSON string returned by an ollama node and splits it into 12 individual prompt outputs for direct connection to image and video generation nodes.

**Inputs**
- `json_string` STRING — raw JSON from ollama

**Outputs**
- `image_1` … `image_6` STRING
- `video_1` … `video_6` STRING

---

### Boyo Storyboard JSON Parser
**File:** `boyo_storyboard_json_parser.py` | **Category:** Boyo/Storyboard

Alternative parser for storyboard JSON with the same 12-output structure. Use if the primary output node has trouble with a particular model's JSON formatting.

**Inputs / Outputs:** identical to Boyo Storyboard Output.

---

## Video

### Boyo Video Clipper
**File:** `boyo_video_clipper.py` | **Category:** Boyo/Video

Clips a video from ComfyUI's input directory to an exact frame count starting from a given time. Uses FFmpeg internally. Useful for preparing dataset clips at a precise resolution and frame rate.

**Inputs**
- `video` — dropdown from input directory
- `start_time` FLOAT — seconds
- `target_fps` INT
- `required_frames` INT

**Outputs**
- `IMAGE` batch — decoded frames
- `frame_count` INT

---

### Boyo Video Cutter
**File:** `boyo_video_cutter.py` | **Category:** Boyo/Video

Removes specific frames from an image sequence, typically the overlapping frames at chunk junctions in looped video generation. Pairs directly with Boyo Video Length Calculator.

**Inputs**
- `images` IMAGE — full frame batch
- `trim_positions` STRING — frame indices to remove, e.g. `"94,95,96|188,189,190"` (pipe separates junction groups)
- `debug_mode` BOOLEAN

**Outputs**
- `trimmed_images` IMAGE
- `cut_info` STRING
- `frames_removed` INT

---

### Boyo Video Length Calculator
**File:** `boyo_video_length_calculator.py` | **Category:** Boyo/Video

Looks up a hardcoded table to return the total frame count, number of loop iterations, and junction trim positions needed to produce a target video duration. Works in 5-second increments up to 60 seconds.

**Inputs**
- `target_seconds` INT — 5 to 60, step 5

**Outputs**
- `total_frames` INT
- `loops_needed` INT
- `trim_positions` STRING — feed directly into Boyo Video Cutter
- `info` STRING

---

### Boyo Video Paired Saver
**File:** `boyo_video_paired_saver.py` | **Category:** Boyo/Video

Renders an image batch to a video file via FFmpeg and saves a matching `.txt` prompt file alongside it. Supports optional audio muxing. Useful for building video-text pair datasets.

**Inputs**
- `images` IMAGE — frame batch
- `enhanced_prompt` STRING
- `folder_name` / `filename_prefix` STRING
- `fps` FLOAT
- `codec` — `libx264`, `libx265`, or `av1`
- `quality` — `high`, `medium`, or `low`
- `audio` AUDIO *(optional)*

**Outputs:** none (output node)

---

### Boyo Load Video Directory
**File:** `BoyoLoadVideoDirectory.py` | **Category:** Boyo/Video

Loads every video file in a specified directory as an image batch. Supports common formats (mp4, mov, avi, mkv, webm).

**Inputs**
- `directory` STRING
- `image_load_cap` INT — max frames to load
- `skip_first_images` INT

**Outputs**
- `image` IMAGE
- `mask` MASK
- `int` — frame count

---

### Boyo Frame Counter
**File:** `boyo_frame_counter.py` | **Category:** Boyo/Video

Computes `frames_processed = (counter × chunk_size) + offset`. Feed the loop counter and your chunk size to get the correct `frames_processed` value for nodes like VHS Video Combine in looped video workflows.

**Inputs**
- `counter` INT — current loop iteration
- `chunk_size` INT — frames per iteration (e.g. 89)
- `offset` INT *(optional, default 0)*

**Outputs**
- `frames_processed` INT

---

### Boyo Overlap Switch
**File:** `boyo_overlap_switch.py` | **Category:** Boyo/Video

Outputs a different overlap value depending on whether this is the first loop iteration or a subsequent one. Prevents the first generated chunk from overlapping nothing.

**Inputs**
- `counter` INT
- `first_overlap` INT — typically 0
- `subsequent_overlap` INT — your optimal blend value (default 13)

**Outputs**
- `overlap` INT

---

### Boyo Watermarks
**File:** `BoyoWatermarks.py` | **Category:** BoyoNodes/video

Stamps a watermark image onto every frame of a video batch. The watermark is resized to specified dimensions, composited at a chosen corner with adjustable opacity.

**Inputs**
- `frames` IMAGE — video frame batch
- `watermark` IMAGE — single watermark frame
- `wm_width` / `wm_height` INT
- `corner` — `bottom-right`, `bottom-left`, `top-right`, `top-left`
- `opacity` FLOAT (0–1, default 0.85)
- `padding` INT — pixel gap from edge

**Outputs**
- `frames` IMAGE

---

## Looping & Flow Control

### Boyo While Loop Start / End
**File:** `BoyoBastardLoops.py` | **Category:** Boyo/Loops

While loop nodes with custom execution management. The Start node initialises loop state; the End node evaluates the continue condition and re-queues the Start node if true. Handles arbitrary typed pass-through values.

**Notes:** These patch ComfyUI's execution engine to support genuine cycles. Use with care in complex graphs.

---

### Boyo For Loop Start / End
**File:** `boyo_for_loops_exact.py` | **Category:** Boyo/Loops

For loop nodes that replicate EasyUse's loop logic without requiring EasyUse as a dependency. Includes an internal tagged cache to persist values across iterations.

**Inputs (Start)**
- `total` INT — number of iterations
- `initial_value0–4` — any type, pass-through values

**Outputs (End)**
- `flow` — loop control signal
- `value0–4` — values from last iteration

---

### Boyo Loop Reset
**File:** `boyo_loop_reset.py` | **Category:** Boyo/Loops

Resets named loop counters to zero when a trigger signal arrives. Use at the end of a workflow to restart a bastard loop from the beginning.

**Inputs**
- `trigger` ANY — completion signal (e.g. from a save node)
- `reset_mode` — `immediate` or `delayed`
- `loop_id_1–3` STRING *(optional)* — named loop IDs to reset

**Outputs**
- `trigger` ANY — pass-through

---

### Boyo Loop Counter / Boyo Math Int / Boyo Compare
**File:** `boyo_for_loops_exact.py` | **Category:** Boyo/Loops

Utility nodes for loop arithmetic. Loop Counter increments on each pass. Math Int performs addition, subtraction, multiplication, and division on integers. Compare outputs a boolean from two integer inputs.

---

### Boyo Prompt Loop
**File:** `BoyoPromptLoop.py` | **Category:** Boyo/Loops

Iterates through prompts stored in `.txt` files in a `prompts/` subfolder. Three modes: sequential (one per loop pass), random (seeded), or single (fixed index).

**Inputs**
- `text_file` — dropdown of `.txt` files in `prompts/` directory
- `mode` — `sequential`, `random`, or `single`
- `start_seed` INT
- `index` INT — used in single mode

**Outputs**
- `prompt` STRING
- `index` INT — current position

---

### Boyo Loop Collector / Boyo Loop Image Saver
**File:** `BoyoLoopCollector.py` | **Category:** Boyo/Loops

Loop Collector accumulates images across loop iterations into a growing batch. Loop Image Saver writes each collected image to disk with sequential filenames as they arrive.

---

## Latent & Conditioning

### Boyo Latent Cache Updater
**File:** `boyo_latent_cache_updater.py` | **Category:** Boyo/Latent

Writes a latent tensor into the Boyo loop cache keyed by a string ID, then passes the tensor through unchanged. Avoids graph cycles by separating the write from the next iteration's read (handled by Boyo Latent Switch).

**Inputs**
- `latent` LATENT
- `cache_key` STRING — must match the key used in Latent Switch
- `counter` INT *(optional)*

**Outputs**
- `latent` LATENT — unchanged pass-through

---

### Boyo Latent Switch
**File:** `boyo_latent_switch.py` | **Category:** Boyo/Latent

On iteration 0 outputs `start_latent`; on all subsequent iterations reads the cached latent written by Boyo Latent Cache Updater. This is the clean way to feed the previous frame's latent back into a sampler without a cycle.

**Inputs**
- `counter` INT
- `start_latent` LATENT — used on first pass
- `cache_key` STRING — must match Cache Updater key
- `next_latent` LATENT *(optional)*

**Outputs**
- `latent` LATENT

---

### Boyo Latent Passthrough / Boyo Execution Barrier
**File:** `boyo_latent_passthrough.py` | **Category:** Boyo/Latent

Passthrough passes a latent through with no modification. Execution Barrier adds an explicit dependency edge — use it to enforce a specific ordering between otherwise unconnected graph branches.

---

### Boyo Painter SVI
**File:** `boyo_painter_svi.py` | **Category:** conditioning/video_models

Merges PainterI2V motion amplitude conditioning with WanImageToVideoSVIPro context preservation. Intended for samplers 2 and onwards in infinite-length video generation, after the first PainterI2V pass has established motion.

**Inputs**
- `positive` / `negative` CONDITIONING
- `length` INT — frame count for this chunk
- `anchor_samples` LATENT — latent from the first PainterI2V sampler
- `motion_amplitude` FLOAT (1.0–2.0)
- `motion_latent_count` INT
- `prev_samples` LATENT *(optional)* — previous chunk for context blending

**Outputs**
- `positive` / `negative` CONDITIONING
- `latent` LATENT

---

### Boyo VACE Injector
**File:** `BoyoControl.py` | **Category:** Boyo/Control

Injects VACE control data directly into model attributes, bypassing conditioning entirely. Supports standard ComfyUI MODEL or WanVideoWrapper's WANVIDEOMODEL.

**Inputs**
- `control_image` IMAGE
- `vace_strength` FLOAT (0–2)
- `vace_start_percent` / `vace_end_percent` FLOAT
- `num_frames` INT
- `model` MODEL *(optional)*
- `wanvideomodel` WANVIDEOMODEL *(optional)*
- `vae` VAE *(optional)*

**Outputs**
- `model` MODEL

---

### Boyo VACE Viewer
**File:** `BoyoControl.py` | **Category:** Boyo/Control

Reads VACE control attributes from a model and displays them as a formatted string for debugging injection results.

**Inputs**
- `model` MODEL

**Outputs**
- `info` STRING

---

## Utilities

### Boyo Resolution Calc
**File:** `BoyoResolutionCalc.py` | **Category:** Boyonodes

Takes a base width and a named aspect ratio and returns width and height both rounded to the nearest multiple of 8.

**Inputs**
- `width` INT
- `aspect_ratio` — `1:1 Square`, `4:3`, `3:2`, `16:9 HD`, `21:9`, `2:3`, `3:4`, `9:16 Tiktok-Shorts`

**Outputs**
- `width` INT
- `height` INT

---

### Boyo Apply LUT
**File:** `boyo_lut.py` | **Category:** BoyoNodes/Colour

Applies a colour-grading LUT to an image. Scans the `luts/` subfolder at startup for `.cube`, `.3dl`, and `.spi3d` files.

**Inputs**
- `image` IMAGE
- `lut_file` — dropdown from `luts/` folder
- `strength` FLOAT — blend between original and graded (0–1)

**Outputs**
- `image` IMAGE

**Dependency:** `pip install colour-science`

**Setup:** Drop your LUT files into `ComfyUI/custom_nodes/Boyonodes/luts/`.

---

### Boyo Mask To Image
**File:** `boyo_mask_normalise.py` | **Category:** BoyoNodes/Masks

Converts a mask tensor of any standard shape to a 3-channel IMAGE tensor. Written specifically to work around DiffuEraser's broken `len(video_mask) > 3` check, which incorrectly rejects any video longer than 3 frames.

**Inputs**
- `mask` MASK — accepts `[h,w]`, `[b,h,w]`, `[b,h,w,1]`, or `[b,1,h,w]`

**Outputs**
- `image` IMAGE — shape `[b,h,w,3]`

---

### Boyo Asset Grabber Simple
**File:** `boyo_asset_grabber_simple.py` | **Category:** Boyo/Utility

Reads a JSON manifest and automatically installs custom nodes (via git clone), Python packages (via pip), and downloads model files. Detects ComfyUI's base path automatically.

**Inputs**
- `manifest_json` STRING — JSON config defining assets to install
- `execute` BOOLEAN — dry-run when false

**Outputs**
- `log` STRING — installation report

---

### Boyo Asset Grabber Advanced
**File:** `boyo_asset_grabber_advanced.py` | **Category:** Boyo/Utility

Same as Simple but with explicit path override inputs for custom nodes directory, models directory, etc. Use when ComfyUI is installed in a non-standard location.

---

### Boyo Prompt Relay Encode
**File:** `boyo_prompt_relay_nodes.py` | **Category:** (conditioning)

Port of kijai's ComfyUI-PromptRelay into Boyonodes. Encodes a set of prompts with temporally-varying Gaussian segment masks so different prompts influence different time steps of a video generation.

**Inputs**
- `clip` CLIP
- `prompts` — list of text segments
- `segment_lengths` — duration of each segment as a proportion of total steps

**Outputs**
- `conditioning` CONDITIONING

**Attribution:** Original logic by Jukka Seppänen (kijai), ported with attribution.

---

### Boyo Prompt Relay Encode Timeline
**File:** `boyo_prompt_relay_nodes.py` | **Category:** (conditioning)

Timeline-based variant of Prompt Relay. Accepts explicit step-range definitions per prompt segment instead of proportional lengths, for finer control.

---

### Boyo Prompt Relay LoRA Gate
**File:** `boyo_prompt_relay_nodes.py` | **Category:** (conditioning)

Applies a LoRA selectively to a single temporal segment using the same Gaussian gating mechanism as Prompt Relay. Enables LoRA influence to be confined to a specific portion of the video.

**Inputs**
- `model` MODEL
- `lora_name` STRING
- `strength` FLOAT
- `segment_start` / `segment_end` FLOAT — 0–1 proportional range

**Outputs**
- `model` MODEL

---

*For installation instructions and one-line node summaries see [README.md](README.md).*
