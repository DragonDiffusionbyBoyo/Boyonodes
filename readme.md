# Boyonodes

Essential nodes for semantic image editing workflows with **Kontext**, **Qwen Image Edit**, and **HiDream E1.1** models. Streamlines iterative editing, dataset creation, and batch processing for next-generation semantic editing pipelines.

## Semantic Image Editing Workflow Nodes

### Boyo Image Grab
Automatically monitors directories and grabs the most recent image file - perfect for iterative semantic editing workflows where you want to chain multiple edits together without manual file management.

**Key Features:**
- Auto-detects newest image in specified directory
- Enables seamless iterative editing chains
- Perfect for progressive semantic modifications
- Supports all major image formats
- Real-time directory monitoring

**Inputs:**
- `directory_path`: Full path to monitor (e.g., "C:\MyImages")
- `file_extensions`: Comma-separated formats (default: "jpg,jpeg,png,bmp,tiff,webp")
- `auto_refresh`: Enable automatic monitoring (default: True)
- `refresh_trigger`: Manual refresh control (default: 0)

**Outputs:**
- `image`: Ready-to-use image tensor
- `filename`: Just the filename
- `full_path`: Complete file path
- `timestamp`: Modification timestamp

**Usage:**
Perfect for workflows like: Original → "add a hat" → "make it red" → "change lighting" where each step automatically feeds into the next.

### Boyo Paired Image Saver
Saves original and edited image pairs with organised sequential naming - essential for creating datasets from semantic editing workflows.

**Key Features:**
- Maintains original/edited image relationships
- Sequential batch numbering
- Organised dataset creation
- ControlNet data format compatible
- Custom directory placement (inside or outside ComfyUI)

**Inputs:**
- `original_image`: Source image
- `controlnet_image`: Edited/processed image
- `folder_name`: Target directory
- `filename_prefix`: File naming prefix

**Output Format:**
```
generated001_original.png
generated001_controlnet.png
generated002_original.png
generated002_controlnet.png
```

### Boyo Incontext Saver
Specialised saver for semantic editing model outputs with proper dataset formatting.

**Key Features:**
- Handles source and diffusion outputs
- Dataset-ready file organisation
- Batch processing support
- Maintains editing relationships

**Inputs:**
- `source_image`: Original input
- `diffusion_output`: Semantic edit result
- `folder_name`: Output directory

### Boyo Universal Image + Prompt Saver
Saves clean images with accompanying prompt files - perfect for publication and documentation.

**Key Features:**
- Strips metadata for clean publication
- Saves actual prompts (crucial for wildcard/enhanced workflows)
- Creates organised image/text pairs
- Protects workflow secrets whilst maintaining documentation

**Perfect for:**
- Publishing images without exposing ComfyUI metadata
- Documenting wildcard-generated prompts you never see
- Creating training datasets with prompt annotations
- Maintaining reproducible results

## Workflow Enhancement Nodes

### Boyo Empty Latent
Smart aspect ratio calculator - input width, select ratio, automatically calculates height.

**Features:**
- Model-aware resolution calculation
- Common aspect ratio presets
- Prevents invalid dimensions

### Load Image List
Batch image processor for mass workflows.

**Features:**
- Mass resize, pad, or crop operations
- Perfect for upscaler pipelines
- Handles hundreds of images automatically
- Set-and-forget batch processing

## Utility Nodes

### Boyo VAE Decode Nodes
Stealth NSFW filtering for controlled environments.

**Standard VAE Decode:**
- Hidden NSFW detection
- Disguisable as normal VAE node
- Perfect for educational/workplace deployment

**Tiled VAE Decode:**
- Memory-efficient processing for large images
- Seamless tile blending
- VRAM-friendly for high-resolution workflows
- Same stealth NSFW protection

### Mandelbrot Video Generator
Fractal art generation for creative projects.

**Features:**
- Random Mandelbrot and Julia sets
- Video or frame sequence output
- Multiple resolution options
- Grayscale mode available

## Installation

**Quick Install:**
```bash
git clone https://github.com/DragonDiffusionbyBoyo/Boyonodes.git
cp -r Boyonodes /path/to/ComfyUI/custom_nodes/
```

**Most nodes** require only ComfyUI itself.

**Mandelbrot Video Node** requires additional dependencies:
```bash
pip install numpy==1.26 matplotlib pillow tqdm torch
```

**FFmpeg** (for video output):
- Windows: Download from ffmpeg.org or `choco install ffmpeg`
- macOS: `brew install ffmpeg`
- Linux: `sudo apt-get install ffmpeg`

Restart ComfyUI after installation.

## Semantic Editing Workflow Examples

**Basic Iterative Editing:**
1. Load initial image
2. Run through Kontext/Qwen/HiDream with prompt
3. Boyo Paired Image Saver stores original + edit
4. Boyo Image Grab automatically picks up the edit
5. Feed back for next iteration
6. Repeat for progressive modifications

**Dataset Creation:**
1. Use semantic editing models on source images
2. Boyo Paired Image Saver organises results
3. Creates properly formatted training datasets
4. Maintains source/target relationships

**Publication Workflow:**
1. Create final edited images
2. Boyo Universal Saver strips metadata
3. Saves clean images + prompt documentation
4. Ready for sharing without exposing workflows

## Troubleshooting

**Boyo Image Grab issues:**
- Verify directory path exists and is accessible
- Check file permissions
- Ensure auto_refresh is enabled
- Use forward slashes in Windows paths

**Paired Saver issues:**
- Confirm output directory exists
- Check file write permissions
- Verify input image formats

**Performance tips:**
- Large directories may slow Image Grab scanning
- Organise files into smaller subdirectories
- Use specific file extensions to reduce scanning

## Contributing

Fork, branch (`git checkout -b feature-name`), commit, push, and open a pull request. Documentation for new features is appreciated.

## License

MIT License - see LICENSE file for details.

Built by DragonDiffusionbyBoyo for the semantic editing revolution.