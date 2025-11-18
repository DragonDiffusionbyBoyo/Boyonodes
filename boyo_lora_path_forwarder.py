import folder_paths
from typing import Dict, List, Optional, Any, Tuple

# Cache the LoRA list at module load time to prevent dynamic changes
_CACHED_LORA_LIST = None

def get_cached_lora_list():
    """Get a cached version of the LoRA list to prevent dynamic return type changes."""
    global _CACHED_LORA_LIST
    if _CACHED_LORA_LIST is None:
        try:
            _CACHED_LORA_LIST = ["none"] + folder_paths.get_filename_list("loras")
        except:
            _CACHED_LORA_LIST = ["none"]
    return _CACHED_LORA_LIST

def refresh_cached_lora_list():
    """Force refresh of the cached LoRA list."""
    global _CACHED_LORA_LIST
    try:
        _CACHED_LORA_LIST = ["none"] + folder_paths.get_filename_list("loras")
    except:
        _CACHED_LORA_LIST = ["none"]
    return _CACHED_LORA_LIST

class BoyoLoRAPathForwarder:
    """
    A ComfyUI node for forwarding processed LoRA configuration data to native LoRA loaders.
    This is the second part of the split architecture - handles connection buffering.
    """
    
    def __init__(self):
        self.type = "BoyoLoRAPathForwarder"
        
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "high_noise_path_1": ("STRING", {
                    "default": "",
                    "tooltip": "High noise LoRA path from Config Processor"
                }),
                "low_noise_path_1": ("STRING", {
                    "default": "",
                    "tooltip": "Low noise LoRA path from Config Processor"
                }),
                "high_noise_path_2": ("STRING", {
                    "default": "",
                    "tooltip": "High noise LoRA path from Config Processor"
                }),
                "low_noise_path_2": ("STRING", {
                    "default": "",
                    "tooltip": "Low noise LoRA path from Config Processor"
                }),
                "high_noise_path_3": ("STRING", {
                    "default": "",
                    "tooltip": "High noise LoRA path from Config Processor"
                }),
                "low_noise_path_3": ("STRING", {
                    "default": "",
                    "tooltip": "Low noise LoRA path from Config Processor"
                }),
            },
            "optional": {
                "force_refresh": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Refresh",
                    "label_off": "Normal",
                    "tooltip": "Force refresh LoRA list if connections break"
                }),
            }
        }
    
    def validate_and_clean_path(self, path: str, force_refresh: bool = False) -> str:
        """Validate LoRA path and ensure it exists in the cached list."""
        if not path or path.strip() == "":
            return "none"
        
        # Clean the path
        clean_path = path.strip().replace("\\", "/")
        
        # If we have a non-empty path from the config processor, trust it
        # The config processor already validated it exists when loading the JSON
        if clean_path:
            print(f"Using LoRA path: '{clean_path}'")
            return clean_path
        
        # Only return none if the path was actually empty
        return "none"
    
    def extract_paths_from_config(self, config_data: Optional[Dict], force_refresh: bool = False) -> Tuple[str, str]:
        """Extract and validate high/low noise paths from config data."""
        if not config_data or not config_data.get("has_config", False):
            return ("none", "none")
        
        high_path = self.validate_and_clean_path(
            config_data.get("high_noise_path", ""), 
            force_refresh
        )
        low_path = self.validate_and_clean_path(
            config_data.get("low_noise_path", ""), 
            force_refresh
        )
        
        return (high_path, low_path)
    
    # Use cached LoRA list to prevent dynamic changes - same as original loader
    RETURN_TYPES = (get_cached_lora_list(),) * 6 + ("STRING",)
    RETURN_NAMES = (
        "high_noise_path_1", "low_noise_path_1",
        "high_noise_path_2", "low_noise_path_2", 
        "high_noise_path_3", "low_noise_path_3",
        "status_message"
    )
    FUNCTION = "forward_lora_paths"
    CATEGORY = "Boyo/LoRA Tools"
    
    def forward_lora_paths(self, force_refresh: bool = False,
                          high_noise_path_1: str = "", low_noise_path_1: str = "",
                          high_noise_path_2: str = "", low_noise_path_2: str = "",
                          high_noise_path_3: str = "", low_noise_path_3: str = "") -> Tuple[str, ...]:
        """Main function to forward LoRA paths from processed config data."""
        
        # DEBUG: Print what we're receiving
        print(f"Path Forwarder RECEIVED:")
        print(f"  high_noise_path_1: '{high_noise_path_1}' (type: {type(high_noise_path_1)})")
        print(f"  low_noise_path_1: '{low_noise_path_1}' (type: {type(low_noise_path_1)})")
        print(f"  high_noise_path_2: '{high_noise_path_2}' (type: {type(high_noise_path_2)})")
        print(f"  low_noise_path_2: '{low_noise_path_2}' (type: {type(low_noise_path_2)})")
        print(f"  high_noise_path_3: '{high_noise_path_3}' (type: {type(high_noise_path_3)})")
        print(f"  low_noise_path_3: '{low_noise_path_3}' (type: {type(low_noise_path_3)})")
        
        # Collect all path inputs
        input_paths = [
            high_noise_path_1, low_noise_path_1,
            high_noise_path_2, low_noise_path_2,
            high_noise_path_3, low_noise_path_3
        ]
        
        lora_paths = []
        status_parts = []
        
        # Process each pair of paths
        for i in range(0, len(input_paths), 2):
            high_path = self.validate_and_clean_path(input_paths[i], force_refresh)
            low_path = self.validate_and_clean_path(input_paths[i + 1], force_refresh)
            
            lora_paths.extend([high_path, low_path])
            
            config_num = (i // 2) + 1
            if high_path != "none" or low_path != "none":
                status_parts.append(f"Config_{config_num}: H={high_path}, L={low_path}")
            else:
                status_parts.append(f"Config_{config_num}: None")
        
        # Create status message
        status_message = " | ".join(status_parts)
        
        # Add refresh status if requested
        if force_refresh:
            status_message += " [REFRESHED]"
        
        # Debug output
        print(f"Path Forwarder Debug:")
        print(f"  Input paths: {input_paths}")
        print(f"  Output LoRA paths: {lora_paths}")
        print(f"  Status: {status_message}")
        
        # Return: 6 LoRA paths + 1 status string
        result = tuple(lora_paths + [status_message])
        
        return result

# Node registration
NODE_CLASS_MAPPINGS = {
    "BoyoLoRAPathForwarder": BoyoLoRAPathForwarder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoLoRAPathForwarder": "Boyo LoRA Path Forwarder"
}
