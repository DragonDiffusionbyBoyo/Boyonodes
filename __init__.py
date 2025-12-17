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
from .boyo_video_paired_saver import BoyoVideoPairedSaver, BoyoVideoSaver
# Import the existing custom nodes
from .BoyoControl import NODE_CLASS_MAPPINGS as BOYOCONTROL_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as BOYOCONTROL_DISPLAY_NAME_MAPPINGS

# Import the new BoyoImageGrab node
from .boyo_image_grab import NODE_CLASS_MAPPINGS as BOYOIMAGEGRAB_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as BOYOIMAGEGRAB_DISPLAY_NAME_MAPPINGS


# Import the paired image saver nodes
from .boyo_paired_image_saver import BoyoPairedImageSaver, BoyoIncontextSaver
# Import Lora Loader stuff
from .boyo_lora_json_builder import BoyoLoRAJSONBuilder
from .boyo_lora_paired_loader import BoyoLoRAPairedLoader
from .boyo_lora_config_inspector import BoyoLoRAConfigInspector
from .boyo_lora_config_processor import BoyoLoRAConfigProcessor
from .boyo_lora_path_forwarder import BoyoLoRAPathForwarder

# Import the BoyoImageCrop node
from .boyo_image_crop import BoyoImageCrop

#Import the Video-Image-Storyboard Node
from .boyo_storyboard_prompt import BoyoStoryboardPrompt
from .boyo_storyboard_output import BoyoStoryboardOutput

# Import the loop reset nodes
from .boyo_loop_reset import NODE_CLASS_MAPPINGS as BOYOLOOPRESET_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as BOYOLOOPRESET_DISPLAY_NAME_MAPPINGS
# Import the custom for loop nodes
from .boyo_for_loops_exact import NODE_CLASS_MAPPINGS as BOYOFORLOOPS_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as BOYOFORLOOPS_DISPLAY_NAME_MAPPINGS

# Import the latent passthrough nodes
from .boyo_latent_passthrough import NODE_CLASS_MAPPINGS as BOYOPASSTHROUGH_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as BOYOPASSTHROUGH_DISPLAY_NAME_MAPPINGS

from .boyo_asset_grabber_simple import BoyoAssetGrabberSimple
from .boyo_asset_grabber_advanced import BoyoAssetGrabberAdvanced

# 🏴‍☠️ Import Z-Image IP-Adapter nodes (hijacked from SD3)
from .zimage_ip_adapter_nodes import ZIMAGE_IP_ADAPTER_CLASS_MAPPINGS, ZIMAGE_IP_ADAPTER_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS = {
    "BoyoVAEDecode": BoyoVAEDecode,
    "BoyoSaver": BoyoSaver,
    "BoyoLoadImageList": BoyoLoadImageList,
    "BoyoAudioEval": BoyoAudioEval,
    "BoyoTiledVAEDecode": BoyoTiledVAEDecode,
    "BoyoPairedSaver": BoyoPairedSaver,
    "BoyoPairedImageSaver": BoyoPairedImageSaver,
    "BoyoVideoPairedSaver": BoyoVideoPairedSaver,
    "BoyoVideoSaver": BoyoVideoSaver,
    "BoyoLoRAJSONBuilder": BoyoLoRAJSONBuilder,
    "BoyoLoRAPairedLoader": BoyoLoRAPairedLoader,
    "BoyoLoRAConfigInspector": BoyoLoRAConfigInspector,
    "BoyoLoRAConfigProcessor": BoyoLoRAConfigProcessor,
    "BoyoLoRAPathForwarder": BoyoLoRAPathForwarder,
    "BoyoIncontextSaver": BoyoIncontextSaver,
    "BoyoStoryboardPrompt": BoyoStoryboardPrompt,
    "BoyoStoryboardOutput": BoyoStoryboardOutput,
    "BoyoAssetGrabberSimple": BoyoAssetGrabberSimple,
    "BoyoAssetGrabberAdvanced": BoyoAssetGrabberAdvanced,
    "BoyoImageCrop": BoyoImageCrop
}

# Update the mappings with all custom nodes
NODE_CLASS_MAPPINGS.update(BOYOLATENT_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(MANDELBROT_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(BOYOCONTROL_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(BASTARDLOOPS_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(PROMPTLOOP_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(LOOPCOLLECTOR_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(BOYOIMAGEGRAB_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(BOYOLOOPRESET_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(BOYOFORLOOPS_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(BOYOPASSTHROUGH_CLASS_MAPPINGS)
# 🏴‍☠️ Add hijacked Z-Image IP-Adapter nodes
NODE_CLASS_MAPPINGS.update(ZIMAGE_IP_ADAPTER_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoVAEDecode": "Boyo VAE Decode",
    "BoyoSaver": "Boyo Saver",
    "BoyoLoadImageList": "Boyo Load Image List",
    "BoyoAudioEval": "Boyo Audio Evaluator",
    "BoyoTiledVAEDecode": "Boyo Tiled VAE Decode",
    "BoyoPairedSaver": "Boyo Paired Saver",
    "BoyoPairedImageSaver": "Boyo Paired Image Saver",
    "BoyoVideoPairedSaver": "Boyo Video Paired Saver",
    "BoyoVideoSaver": "Boyo Video Saver",
    "BoyoLoRAJSONBuilder": "Boyo LoRA JSON Builder",
    "BoyoLoRAPairedLoader": "Boyo LoRA Paired Loader",
    "BoyoLoRAConfigInspector": "Boyo LoRA Config Inspector",
    "BoyoLoRAConfigProcessor": "Boyo LoRA Config Processor",
    "BoyoLoRAPathForwarder": "Boyo LoRA Path Forwarder",
    "BoyoIncontextSaver": "Boyo Incontext Saver",
    "BoyoStoryboardPrompt": "Boyo Storyboard Prompt",
    "BoyoStoryboardOutput": "Boyo Storyboard Output",
    "BoyoAssetGrabberSimple": "Boyo Asset Grabber (Simple)",
    "BoyoAssetGrabberAdvanced": "Boyo Asset Grabber (Advanced)",
    "BoyoImageCrop": "Boyo Image Crop"
}

# Update the display name mappings with all custom nodes
NODE_DISPLAY_NAME_MAPPINGS.update(BOYOLATENT_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(MANDELBROT_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(BOYOCONTROL_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(BASTARDLOOPS_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(PROMPTLOOP_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(LOOPCOLLECTOR_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(BOYOIMAGEGRAB_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(BOYOLOOPRESET_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(BOYOFORLOOPS_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(BOYOPASSTHROUGH_DISPLAY_NAME_MAPPINGS)
# 🏴‍☠️ Add hijacked Z-Image IP-Adapter display names
NODE_DISPLAY_NAME_MAPPINGS.update(ZIMAGE_IP_ADAPTER_DISPLAY_NAME_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
