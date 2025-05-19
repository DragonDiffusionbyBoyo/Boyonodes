from .boyolatent import NODE_CLASS_MAPPINGS as BOYOLATENT_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as BOYOLATENT_DISPLAY_NAME_MAPPINGS
from .boyo_vae_decode import BoyoVAEDecode
from .boyo_saver import BoyoSaver
from .boyo_load_image_list import BoyoLoadImageList
from .BoyoAudioEval import BoyoAudioEval
from .Boyomandelbrot import NODE_CLASS_MAPPINGS as MANDELBROT_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as MANDELBROT_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS = {
    "BoyoVAEDecode": BoyoVAEDecode,
    "BoyoSaver": BoyoSaver,
    "BoyoLoadImageList": BoyoLoadImageList,
    "BoyoAudioEval": BoyoAudioEval
}

NODE_CLASS_MAPPINGS.update(BOYOLATENT_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(MANDELBROT_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoVAEDecode": "Boyo VAE Decode",
    "BoyoSaver": "Boyo Saver",
    "BoyoLoadImageList": "Boyo Load Image List",
    "BoyoAudioEval": "Boyo Audio Evaluator"
}

NODE_DISPLAY_NAME_MAPPINGS.update(BOYOLATENT_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(MANDELBROT_DISPLAY_NAME_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']