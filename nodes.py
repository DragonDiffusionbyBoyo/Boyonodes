```python
import os
import torch
from comfy.utils import load_torch_file
import comfy.model_management as mm
from comfy.sd import load_lora_for_models
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def standardize_lora_key_format(lora_sd):
    new_sd = {}
    for k, v in lora_sd.items():
        k = k.replace('transformer.', 'diffusion_model.')  # Align with Hunyuan Video key format
        new_sd[k] = v
    return new_sd

class BoyoFramePackLoRA:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("FramePackMODEL",),
                "lora_path": ("STRING", {"default": "", "tooltip": "Path to LoRA safetensors file"}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("FramePackMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_lora"
    CATEGORY = "BoyoNodes"
    DESCRIPTION = "Loads LoRA weights for Hunyuan Video transformer"

    def load_lora(self, model, lora_path, lora_strength):
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA file not found: {lora_path}")

        log.info(f"Loading LoRA: {os.path.basename(lora_path)} with strength: {lora_strength}")
        
        # Load and standardize LoRA weights
        lora_sd = load_torch_file(lora_path, safe_load=True)
        lora_sd = standardize_lora_key_format(lora_sd)

        # Apply LoRA using ComfyUI's standard method
        model, _ = load_lora_for_models(model, None, lora_sd, lora_strength, 0)

        return (model,)

# Register the node for ComfyUI
NODE_CLASS_MAPPINGS = {
    "BoyoFramePackLoRA": BoyoFramePackLoRA
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoFramePackLoRA": "Boyo FramePack LoRA Loader"
}
```