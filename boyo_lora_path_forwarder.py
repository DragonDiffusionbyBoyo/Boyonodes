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
                "force_refresh": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Refresh",
                    "label_off": "Normal",
                    "tooltip": "Force refresh LoRA list if connections break"
                }),
            },
            "optional": {
                "config_data_1": ("DICT", {
                    "tooltip": "Processed LoRA config data from Config Processor"
                }),
                "config_data_2": ("DICT", {
                    "tooltip": "Processed LoRA config data from Config Processor"
                }),
                "config_data_3": ("DICT", {
                    "tooltip": "Processed LoRA config data from Config Processor"
                }),
            }
        }
    
    def validate_and_clean_path(self, path: str, force_refresh: bool = False) -> str:
        """Validate LoRA path and ensure it exists in the cached list."""
        if not path or path.strip() == "":
            return "none"
        
        # Get current LoRA list (refresh if requested)
        if force_refresh:
            lora_list = refresh_cached_lora_list()
        else:
            lora_list = get_cached_lora_list()
        
        # Clean the path
        clean_path = path.strip().replace("\\", "/")
        
        # Check if path exists in available LoRAs
        if clean_path in lora_list:
            return clean_path
        
        # If not found, try to find a partial match
        for available_lora in lora_list:
            if available_lora != "none" and clean_path in available_lora:
                print(f"LoRA path '{clean_path}' not found exactly, using '{available_lora}'")
                return available_lora
        
        # If still not found, log warning and return none
        print(f"Warning: LoRA path '{clean_path}' not found in available LoRAs")
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
                          config_data_1: Optional[Dict] = None,
                          config_data_2: Optional[Dict] = None, 
                          config_data_3: Optional[Dict] = None) -> Tuple[str, ...]:
        """Main function to forward LoRA paths from processed config data."""
        
        lora_paths = []
        status_parts = []
        
        # Process each config data slot
        config_data_list = [config_data_1, config_data_2, config_data_3]
        
        for i, config_data in enumerate(config_data_list, 1):
            high_path, low_path = self.extract_paths_from_config(config_data, force_refresh)
            
            lora_paths.extend([high_path, low_path])
            
            # Build status message
            if config_data and config_data.get("has_config", False):
                config_name = config_data.get("name", f"Config_{i}")
                status_parts.append(f"{config_name}: H={high_path}, L={low_path}")
            else:
                status_parts.append(f"Config_{i}: None")
        
        # Create status message
        status_message = " | ".join(status_parts)
        
        # Add refresh status if requested
        if force_refresh:
            status_message += " [REFRESHED]"
        
        # Debug output
        print(f"Path Forwarder Debug:")
        print(f"  LoRA paths: {lora_paths}")
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
