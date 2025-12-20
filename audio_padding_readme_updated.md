## Audio Padding Nodes

Perfect for synchronising audio with video for lip-sync workflows and timing-critical applications.

### BoyoAudioDurationAnalyzer

Extracts precise duration information from ComfyUI audio tensors.

**Inputs:**
- `audio` (AUDIO) - Any ComfyUI audio tensor

**Outputs:**
- `duration_seconds` (FLOAT) - Precise duration in seconds
- `info_text` (STRING) - Detailed audio information

**Use Case:** Get exact audio duration for padding calculations and workflow debugging.

### BoyoAudioPadder  

Add silence before/after audio with intelligent feedback and auto-centering. Automatically adjusts sample rate for proper ComfyUI playback compatibility.

**Inputs:**
- `audio` (AUDIO) - Input audio to pad
- `pre_pad_seconds` (FLOAT) - Silence before audio (0.0-300.0s)
- `post_pad_seconds` (FLOAT) - Silence after audio (0.0-300.0s)
- `target_duration` (FLOAT, optional) - Target total duration for comparison
- `auto_center` (BOOLEAN, optional) - Automatically center audio in target duration

**Outputs:**
- `padded_audio` (AUDIO) - Audio with padding applied (12kHz for proper playback speed)
- `total_duration` (FLOAT) - Final duration after padding
- `status_text` (STRING) - Helpful feedback and suggestions

**Example Status Messages:**
- `📊 Audio: 3.2s + Padding: 7.3s = Total: 10.5s ✅ Perfect match!`
- `📊 Audio: 5.1s + Padding: 2.4s = Total: 7.5s ⚠️ 3.0s shorter than target 💡 Try auto_center or 1.5s each`

### Quick Start Workflow

**Simple Auto-Centering:**
1. **Load Audio** → **BoyoAudioPadder**
2. Set `target_duration` to your desired length (e.g., 10.5s)
3. Enable `auto_center` = true
4. Connect to **Save Audio (FLAC)**
5. Perfect timing achieved!

### Advanced Workflow for Lip-Sync

1. **Load Video** → **VideoHelperSuite: Video Info** → `duration` = 10.5s
2. **Load Audio** → **BoyoAudioDurationAnalyzer** → verify original duration
3. **Load Audio** → **BoyoAudioPadder** with:
   - `target_duration` = 10.5s (from video)
   - `auto_center` = true (for centered placement)
   - Or manual: `pre_pad` = 3.65s, `post_pad` = 3.65s  
4. **BoyoAudioPadder** → **Save Audio** → **Perfect 10.5s audio** ready for lip-sync models!

### Manual Precision Workflow

For exact control over audio placement:
- **Start placement**: `pre_pad` = 0s, `post_pad` = (target - audio duration)
- **End placement**: `pre_pad` = (target - audio duration), `post_pad` = 0s
- **Custom timing**: Set exact pre/post values for specific sync points

### Technical Features

- **Smart Format Detection**: Works with all ComfyUI audio formats and tensor shapes
- **Sample Rate Optimization**: Automatically outputs 12kHz for proper ComfyUI playback speed
- **Numpy 1.26.4 Compatible**: Tested with established dependency constraints
- **Intelligent Feedback**: Status text guides you to perfect timing
- **Auto-Centering**: Automatically calculate equal padding for target duration
- **Error Handling**: Graceful fallbacks with helpful error messages
- **Memory Efficient**: Handles large audio files without excessive RAM usage
- **Precise Timing**: Float32 precision for sample-accurate padding

### Why These Nodes Succeed Where Others Failed

- **Comprehensive Format Support**: Handles ComfyUI's inconsistent audio tensor shapes (`(batch, channels, samples)` vs `(batch, samples, channels)`)
- **Proper Sample Rate Handling**: Fixes the common "audio too fast" problem by outputting 12kHz for ComfyUI compatibility
- **Intelligent Guidance**: Status messages tell you exactly what you're getting instead of silent failures
- **Modular Design**: Works seamlessly with existing VideoHelperSuite and audio nodes

### Categories

- **Boyo/Audio/Analysis**: BoyoAudioDurationAnalyzer
- **Boyo/Audio/Processing**: BoyoAudioPadder
