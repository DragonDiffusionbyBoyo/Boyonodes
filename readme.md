Just plop this folder in custom nodes and then restart comfy. No install requirements other than comfy :)

The Vae node is a sneaky little node perfect for deployment in Schools or work environments where you do not want the kiddywinkles creating NSFW content. Just rename the node to VAE decode and it looks like a normal node but hidden inside is an NSFW detector. Once hidden in the workflow there are no settings to undo the NSFW detection so cannot be worked around unless you remove the node. The node looks innocent once renamed so is virtually undetectable.
I have placed an example workflow for you to see how to connect it. Simple stuff really, but once connected just rename.

![nope](https://github.com/user-attachments/assets/bc93bf19-79f4-43ae-8f84-4d6bc3bd9f00)


The Boyo Empty Latent node lets you supply the width then you just choose you aspect ratio and the node calcs the height for you. This is obviously model dependant. If the model does not support your resolution choice then the image will be garbage. 

The Image Saver node is WIP and this is just the first iteration. A path must be supplied for this to work but essentially you can save your files outside of the comfy environment. The second iteration will do loads of cool things so keep your eyes open for the updates.

The load image list lets you take images from a folder and then mass resize pad or crop for input into your workflow. I use this as an example of SD Upscaler where I resize all images prior to feeding into the upscaler workflow. I plop like 400 images in a folder then run the workflow and go to bed. Wake up all done. 
