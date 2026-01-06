import folder_paths as comfy_paths
from typing import Dict, Any, Tuple

class BoyoLorainforsender:
    """
    Simple LoRA selector that outputs just the selected LoRA filename as a string.
    Stripped down version for index switcher feeding.
    """
    
    def __init__(self):
        self.type = "BoyoLorainforsender"
        
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        # Get available LoRA files (same as dragos node)
        lora_files = comfy_paths.get_filename_list("loras")
        
        return {
            "required": {
                "lora_name": (lora_files, {
                    "default": lora_files[0] if lora_files else "None"
                }),
            }
        }
    
    # Just one string output
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_filename",)
    
    FUNCTION = "send_lora_name"
    CATEGORY = "Boyo/LoRA Tools"
    
    def send_lora_name(self, lora_name: str) -> Tuple[str]:
        """Output the selected LoRA filename as a string."""
        
        # Debug output
        print(f"BoyoLorainforsender selected: {lora_name}")
        
        # Return the filename exactly as-is (including .safetensors)
        return (lora_name,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "BoyoLorainforsender": BoyoLorainforsender
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoLorainforsender": "Boyo LoRA Info Sender"
}
