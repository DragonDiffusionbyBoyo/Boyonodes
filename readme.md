The Vae node is a sneaky little node perfect for deployment in Schools or work environments where you do not want the kiddywinkles creating NSFW content. Just rename the node to VAE decode and it looks like a normal node but hidden inside is an NSFW detector. Once hidden in the workflow there are no settings to undo the NSFW detection so cannot be worked around unless you remove the node. The node looks innocent once renamed so is virtually undetectable.
I have placed an example workflow for you to see how to connect it. Simple stuff really, but once connected just rename.

![nope](https://github.com/user-attachments/assets/bc93bf19-79f4-43ae-8f84-4d6bc3bd9f00)

The Boyo Empty Latent node lets you supply the width then you just choose you aspect ratio and the node calcs the height for you. This is obviously model dependant. If the model does not support your resolution choice then the image will be garbage.

The Boyo Paired Saver node is designed for batch workflows where you're generating images from enhanced prompts. It saves both the generated image and the enhanced prompt text with matching sequential filenames (e.g., `prompt001.png` and `prompt001.txt`). Perfect for workflows using prompt injectors with Ollama enhancement - you can keep track of which enhanced prompt generated which image. Just connect your image output and enhanced prompt text, set a folder name and filename prefix, and it handles the rest. Files are saved in ComfyUI's output directory in your specified subfolder.

The Image Saver node is WIP and this is just the first iteration. A path must be supplied for this to work but essentially you can save your files outside of the comfy environment. The second iteration will do loads of cool things so keep your eyes open for the updates.

The load image list lets you take images from a folder and then mass resize pad or crop for input into your workflow. I use this as an example of SD Upscaler where I resize all images prior to feeding into the upscaler workflow. I plop like 400 images in a folder then run the workflow and go to bed. Wake up all done. 

Table of Contents

- Installation
- MandelbrotVideoNode
  - Overview
  - Inputs
  - Outputs
  - Usage
  - Troubleshooting
- Boyo Paired Saver
  - Overview
  - Inputs
  - Usage
- Contributing
- License

Installation
Just plop this folder in custom nodes and then restart comfy. Most nodes (VAE, Boyo Empty Latent, Image Saver, Load Image List, Boyo Paired Saver) have no install requirements other than ComfyUI itself. The MandelbrotVideoNode needs some extra stuff to work:

Clone the Repository:
```bash
git clone https://github.com/DragonDiffusionbyBoyo/Boyonodes.git
```

Move to ComfyUI Custom Nodes:
```bash
cp -r Boyonodes /path/to/ComfyUI/custom_nodes/
```
Typically, this is ComfyUI/custom_nodes/.

Install Dependencies for MandelbrotVideoNode:
- Ensure you have Python 3.8+ and ComfyUI installed.
- Install required Python packages:
```bash
pip install numpy==1.26 matplotlib pillow tqdm torch
```

Install ffmpeg for video output:
- On Windows: Download from ffmpeg.org or use `choco install ffmpeg`.
- On macOS: `brew install ffmpeg`
- On Linux: `sudo apt-get install ffmpeg` (Ubuntu/Debian) or equivalent.

Restart ComfyUI:
Launch or restart ComfyUI to load the nodes.

## MandelbrotVideoNode
### Overview
This node spits out random Mandelbrot or Julia set images or videos, perfect for trippy art projects in your ComfyUI workflow. Every time you run it, you get a fresh, wild fractal pattern. Check the Installation section for the stuff you need to make it work.

### Inputs
- **resolution**: Pick your size:
  - square (512x512)
  - widescreen (720x480)
  - vertical (480x836)
- **output_type**: What you want out:
  - video: Gets you an MP4.
  - frames: Gives you a stack of images to play with.
- **grayscale**: Check this for black-and-white vibes (default: off).

### Outputs
- **frames**: A tensor of images (RGB, shape like (batch, height, width, 3)). Empty if you chose video.
- **video_path**: Where the MP4 or frame folder lives.

### Usage
1. Find Boyo Mandelbrot Generator in ComfyUI and drop it in your workflow.
2. Set your resolution and output type:
   - Want frames? Pick frames and connect to SaveImage or VHS_VideoCombine.
   - Want a video? Pick video for a quick MP4.
   - Grayscale? Flip it on for that retro look.
3. Run it. You'll get a new fractal every time.
4. Frames land in ComfyUI/output/mandelbrot_frames/, videos as mandelbrot_XXXX.mp4 in ComfyUI/output/.

### Troubleshooting
- **Error: "Frame output directory not found"**:
  - Make sure mandelbrot_video.py and mandelbrot_generator.py are in the Boyonodes folder.
  - Check you can write to ComfyUI/output/.
- **Error: "Cannot handle this data type"**:
  - Peek at mandelbrot_frames/ and ensure images match your resolution (e.g., 512x512 for square).
  - Console will show tensor shapes—should be (batch, height, width, 3).
- **No output**:
  - Run `ffmpeg -version` to confirm it's installed.
  - Check ComfyUI console for errors.
- **Same fractal every time**:
  - It's built to be random. If it's not, look for a fixed random.seed() in mandelbrot_video.py.
Boyo VAE Decode Nodes
Overview
Both VAE decode nodes pack sneaky NSFW detection that's perfect for controlled environments. They look like standard VAE nodes but secretly filter inappropriate content - brilliant for schools or workplaces where you need to keep things clean.
Standard VAE Decode
Your basic VAE decode with hidden NSFW filtering. Rename it to "VAE Decode" and it looks completely innocent whilst secretly protecting against unwanted content. No settings to fiddle with once it's in your workflow.
Tiled VAE Decode
Takes the standard node's stealth approach but adds memory-efficient tiled processing. Perfect for large images that would normally crash your VRAM. It splits the latent into overlapping tiles, decodes each separately, then seamlessly blends them back together. All with the same NSFW detection running on the final result.
Inputs:

samples: Your latent tensor (same as standard VAE)
vae: Your VAE model (same as standard VAE)
horizontal_tiles: How many tiles across (1-8, default: 2)
vertical_tiles: How many tiles down (1-8, default: 2)
overlap: Pixel overlap between tiles (16-128, default: 32)

How it works:

Splits your latent into overlapping tiles
Decodes each tile individually (memory efficient!)
Blends tiles together with weighted averaging for seamless results
Runs NSFW detection on the final reconstructed image
Swaps in alternative image if inappropriate content detected

Usage Tips

Start small: Try 2x2 tiles first, increase if you're still hitting memory limits
Increase overlap: If you see tile seams, bump up the overlap value
High resolution: Perfect for 2K+ images that normally cause VRAM issues
Batch processing: Great for processing loads of large images overnight
Stealth mode: Rename both nodes to look like standard VAE decodes for deployment

Both nodes are virtually undetectable once renamed and integrated into workflows. The NSFW filtering can't be disabled without removing the node entirely.
## Boyo Paired Saver
### Overview
This node is perfect for batch workflows where you're using prompt injectors with text enhancement (like Ollama). It saves your generated image and the enhanced prompt text with matching sequential filenames, so you can always see which enhanced prompt created which image. Brilliant for keeping track of your workflow outputs when you're churning through hundreds of prompts.

### Inputs
- **image**: The generated image from your workflow
- **enhanced_prompt**: The enhanced/extended prompt text (usually from Ollama or similar)
- **folder_name**: Name of the subfolder in ComfyUI's output directory (e.g., "batch_output")
- **filename_prefix**: Prefix for your files (e.g., "enhanced_prompts")

### Usage
1. Drop the Boyo Paired Saver node into your workflow
2. Connect your image generation output to the "image" input
3. Connect your enhanced prompt text (from Ollama, etc.) to the "enhanced_prompt" input
4. Set your folder name - this becomes a subfolder in ComfyUI/output/
5. Set your filename prefix - this becomes the start of your filenames
6. Run your workflow

Files get saved as:
- `ComfyUI/output/your_folder_name/your_prefix001.png`
- `ComfyUI/output/your_folder_name/your_prefix001.txt`
- `ComfyUI/output/your_folder_name/your_prefix002.png`
- `ComfyUI/output/your_folder_name/your_prefix002.txt`
- And so on...

The node automatically handles sequential numbering and continues from where it left off if you run multiple batches.

If you ask me technical stuff I am likely to ask you to ask me one on sport as I vibe coded this. 

Boyo Image Grab
Overview
This node automatically monitors a directory and grabs the most recently added or modified image file. Perfect for automation workflows where external processes are creating images and you want ComfyUI to seamlessly pick up the latest one without manual intervention. The node continuously monitors for new files and automatically switches to the newest image as soon as it appears.
Inputs

directory_path: Full path to the directory you want to monitor (e.g., "C:\MyImages" or "/home/user/generated_images/")
file_extensions (optional): Comma-separated list of image formats to look for (default: "jpg,jpeg,png,bmp,tiff,webp")
auto_refresh (optional): When enabled, automatically checks for new files on each workflow execution (default: True)
refresh_trigger (optional): Change this number to manually force the node to refresh and check for new images (default: 0)

Outputs

image: The loaded image tensor ready for use in your workflow
filename: Just the filename of the loaded image (e.g., "latest_image.png")
full_path: Complete path to the loaded image file
timestamp: File modification timestamp (useful for debugging or chaining with other nodes)

Usage

Drop the Boyo Image Grab node into your workflow (found under "Boyo/loaders")
Set the directory_path to the folder you want to monitor
Optionally customise file_extensions if you only want specific image types
Connect the image output to wherever you need the loaded image in your workflow
Run your workflow - the node will automatically grab the newest image from the specified directory

Advanced Usage

Automation Workflows: Perfect for scenarios where other software (image editors, AI tools, cameras) drops images into a folder and you want ComfyUI to automatically process the latest one
Manual Control: Set auto_refresh to False and use the refresh_trigger input if you want manual control over when to check for new files
File Filtering: Customise file_extensions to "png,jpg" if you only want specific formats, or add exotic formats like "exr,hdr" for specialised workflows
Batch Processing: The timestamp output can be useful for conditional processing or sorting in complex workflows

Troubleshooting

Error: "Directory path does not exist":

Double-check your directory path is correct and accessible
On Windows, use forward slashes or double backslashes: "C:/MyImages/" or "C:\MyImages\"
Ensure ComfyUI has read permissions for the directory


Error: "No image files found":

Check that the directory contains images with the specified extensions
Verify the file_extensions setting matches your image file types


Node not updating with new images:

Ensure auto_refresh is enabled (True)
Try changing the refresh_trigger value to force a manual refresh
Check the console for any error messages


Performance with large directories:

The node scans all files to find the newest one, so very large directories (thousands of files) may slow things down
Consider organising files into smaller subdirectories for better performance

Contributing
Got ideas? Fork the repo, make a branch (`git checkout -b cool-new-thing`), commit your stuff (`git commit -m "Added cool thing"`), push it (`git push origin cool-new-thing`), and open a pull request. Add some docs for your changes, and we're golden.

License
MIT License. Check the LICENSE file for the nitty-gritty.

Built by DragonDiffusionbyBoyo. Toss a star if you dig it!


