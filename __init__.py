from .boyolatent import NODE_CLASS_MAPPINGS as BOYOLATENT_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as BOYOLATENT_DISPLAY_NAME_MAPPINGS
from .boyo_vae_decode import BoyoVAEDecode
from .boyo_saver import BoyoSaver
from .boyo_load_image_list import BoyoLoadImageList
from .BoyoAudioEval import BoyoAudioEval
from .boyo_paired_saver import BoyoPairedSaver
from .boyo_tiled_vae_decode import BoyoTiledVAEDecode
from .Boyomandelbrot import NODE_CLASS_MAPPINGS as MANDELBROT_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as MANDELBROT_DISPLAY_NAME_MAPPINGS
from .BoyoBastardLoops import NODE_CLASS_MAPPINGS as BASTARDLOOPS_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as BASTARDLOOPS_DISPLAY_NAME_MAPPINGS
from .BoyoPromptLoop import NODE_CLASS_MAPPINGS as PROMPTLOOP_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as PROMPTLOOP_DISPLAY_NAME_MAPPINGS
from .BoyoLoopCollector import NODE_CLASS_MAPPINGS as LOOPCOLLECTOR_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as LOOPCOLLECTOR_DISPLAY_NAME_MAPPINGS

# Import the existing custom nodes
from .BoyoControl import NODE_CLASS_MAPPINGS as BOYOCONTROL_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as BOYOCONTROL_DISPLAY_NAME_MAPPINGS

# Import the new BoyoImageGrab node
from .boyo_image_grab import NODE_CLASS_MAPPINGS as BOYOIMAGEGRAB_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as BOYOIMAGEGRAB_DISPLAY_NAME_MAPPINGS

# Import the paired image saver nodes
from .boyo_paired_image_saver import BoyoPairedImageSaver, BoyoIncontextSaver

NODE_CLASS_MAPPINGS = {
    "BoyoVAEDecode": BoyoVAEDecode,
    "BoyoSaver": BoyoSaver,
    "BoyoLoadImageList": BoyoLoadImageList,
    "BoyoAudioEval": BoyoAudioEval,
    "BoyoTiledVAEDecode": BoyoTiledVAEDecode,
    "BoyoPairedSaver": BoyoPairedSaver,
    "BoyoPairedImageSaver": BoyoPairedImageSaver,
    "BoyoIncontextSaver": BoyoIncontextSaver
}

# Update the mappings with all custom nodes
NODE_CLASS_MAPPINGS.update(BOYOLATENT_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(MANDELBROT_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(BOYOCONTROL_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(BASTARDLOOPS_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(PROMPTLOOP_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(LOOPCOLLECTOR_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(BOYOIMAGEGRAB_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoVAEDecode": "Boyo VAE Decode",
    "BoyoSaver": "Boyo Saver",
    "BoyoLoadImageList": "Boyo Load Image List",
    "BoyoAudioEval": "Boyo Audio Evaluator",
    "BoyoTiledVAEDecode": "Boyo Tiled VAE Decode",
    "BoyoPairedSaver": "Boyo Paired Saver",
    "BoyoPairedImageSaver": "Boyo Paired Image Saver",
    "BoyoIncontextSaver": "Boyo Incontext Saver"
}

# Update the display name mappings with all custom nodes
NODE_DISPLAY_NAME_MAPPINGS.update(BOYOLATENT_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(MANDELBROT_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(BOYOCONTROL_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(BASTARDLOOPS_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(PROMPTLOOP_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(LOOPCOLLECTOR_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(BOYOIMAGEGRAB_DISPLAY_NAME_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']