The Vae node is a sneaky little node perfect for deployment in Schools or work environments where you do not want the kiddywinkles creating NSFW content. Just rename the node to VAE decode and it looks like a normal node but hidden inside is an NSFW detector. Once hidden in the workflow there are no settings to undo the NSFW detection so cannot be worked around unless you remove the node. The node looks innocent once renamed so is virtually undetectable.
I have placed an example workflow for you to see how to connect it. Simple stuff really, but once connected just rename.

![nope](https://github.com/user-attachments/assets/bc93bf19-79f4-43ae-8f84-4d6bc3bd9f00)



The Boyo Empty Latent node lets you supply the width then you just choose you aspect ratio and the node calcs the height for you. This is obviously model dependant. If the model does not support your resolution choice then the image will be garbage. 
The Image Saver node is WIP and this is just the first iteration. A path must be supplied for this to work but essentially you can save your files outside of the comfy environment. The second iteration will do loads of cool things so keep your eyes open for the updates.
The load image list lets you take images from a folder and then mass resize pad or crop for input into your workflow. I use this as an example of SD Upscaler where I resize all images prior to feeding into the upscaler workflow. I plop like 400 images in a folder then run the workflow and go to bed. Wake up all done. 
Table of Contents

Installation
MandelbrotVideoNode
Overview
Inputs
Outputs
Usage
Troubleshooting


Contributing
License

Installation
Just plop this folder in custom nodes and then restart comfy. Most nodes (VAE, Boyo Empty Latent, Image Saver, Load Image List) have no install requirements other than ComfyUI itself. The MandelbrotVideoNode needs some extra stuff to work:

Clone the Repository:git clone https://github.com/DragonDiffusionbyBoyo/Boyonodes.git


Move to ComfyUI Custom Nodes:cp -r Boyonodes /path/to/ComfyUI/custom_nodes/

Typically, this is ComfyUI/custom_nodes/.
Install Dependencies for MandelbrotVideoNode:
Ensure you have Python 3.8+ and ComfyUI installed.
Install required Python packages:pip install numpy==1.26 matplotlib pillow tqdm torch


Install ffmpeg for video output:
On Windows: Download from ffmpeg.org or use choco install ffmpeg.
On macOS: brew install ffmpeg
On Linux: sudo apt-get install ffmpeg (Ubuntu/Debian) or equivalent.




Restart ComfyUI:
Launch or restart ComfyUI to load the nodes.



MandelbrotVideoNode
Overview
This node spits out random Mandelbrot or Julia set images or videos, perfect for trippy art projects in your ComfyUI workflow. Every time you run it, you get a fresh, wild fractal pattern. Check the Installation section for the stuff you need to make it work.
Inputs

resolution: Pick your size:
square (512x512)
widescreen (720x480)
vertical (480x836)


output_type: What you want out:
video: Gets you an MP4.
frames: Gives you a stack of images to play with.


grayscale: Check this for black-and-white vibes (default: off).

Outputs

frames: A tensor of images (RGB, shape like (batch, height, width, 3)). Empty if you chose video.
video_path: Where the MP4 or frame folder lives.

Usage

Find Boyo Mandelbrot Generator in ComfyUI and drop it in your workflow.
Set your resolution and output type:
Want frames? Pick frames and connect to SaveImage or VHS_VideoCombine.
Want a video? Pick video for a quick MP4.
Grayscale? Flip it on for that retro look.


Run it. You’ll get a new fractal every time.
Frames land in ComfyUI/output/mandelbrot_frames/, videos as mandelbrot_XXXX.mp4 in ComfyUI/output/.

Troubleshooting

Error: "Frame output directory not found":
Make sure mandelbrot_video.py and mandelbrot_generator.py are in the Boyonodes folder.
Check you can write to ComfyUI/output/.


Error: "Cannot handle this data type":
Peek at mandelbrot_frames/ and ensure images match your resolution (e.g., 512x512 for square).
Console will show tensor shapes—should be (batch, height, width, 3).


No output:
Run ffmpeg -version to confirm it’s installed.
Check ComfyUI console for errors.


Same fractal every time:
It’s built to be random. If it’s not, look for a fixed random.seed() in mandelbrot_video.py.

If you ask me technical stuff I am likely to ask you to ask me one on sport as I vibe coded this. 

Contributing
Got ideas? Fork the repo, make a branch (git checkout -b cool-new-thing), commit your stuff (git commit -m "Added cool thing"), push it (git push origin cool-new-thing), and open a pull request. Add some docs for your changes, and we’re golden.
License
MIT License. Check the LICENSE file for the nitty-gritty.

Built by DragonDiffusionbyBoyo. Toss a star if you dig it!
