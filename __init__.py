from .boyolatent import NODE_CLASS_MAPPINGS as BOYOLATENT_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as BOYOLATENT_DISPLAY_NAME_MAPPINGS
from .boyo_vae_decode import BoyoVAEDecode
from .boyo_saver import BoyoSaver
from .boyo_load_image_list import BoyoLoadImageList
from .BoyoAudioEval import BoyoAudioEval
from .boyo_paired_saver import BoyoPairedSaver
from .boyo_tiled_vae_decode import BoyoTiledVAEDecode
from .Boyomandelbrot import NODE_CLASS_MAPPINGS as MANDELBROT_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as MANDELBROT_DISPLAY_NAME_MAPPINGS

# Import the new custom node
from .BoyoControl import NODE_CLASS_MAPPINGS as BOYOCONTROL_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as BOYOCONTROL_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS = {
    "BoyoVAEDecode": BoyoVAEDecode,
    "BoyoSaver": BoyoSaver,
    "BoyoLoadImageList": BoyoLoadImageList,
    "BoyoAudioEval": BoyoAudioEval,
    "BoyoTiledVAEDecode": BoyoTiledVAEDecode,
    "BoyoPairedSaver": BoyoPairedSaver
}

# Update the mappings with the new custom node
NODE_CLASS_MAPPINGS.update(BOYOLATENT_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(MANDELBROT_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(BOYOCONTROL_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoVAEDecode": "Boyo VAE Decode",
    "BoyoSaver": "Boyo Saver",
    "BoyoLoadImageList": "Boyo Load Image List",
    "BoyoAudioEval": "Boyo Audio Evaluator",
    "BoyoTiledVAEDecode": "Boyo Tiled VAE Decode",
    "BoyoPairedSaver": "Boyo Paired Saver"
}

# Update the display name mappings with the new custom node
NODE_DISPLAY_NAME_MAPPINGS.update(BOYOLATENT_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(MANDELBROT_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(BOYOCONTROL_DISPLAY_NAME_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
