import json
import os
import folder_paths
from typing import Dict, List, Optional, Any, Tuple

class BoyoLoRAJSONBuilder:
    """
    A ComfyUI node for creating and saving LoRA configuration JSON files.
    Supports paired LoRAs (high/low noise) with associated trigger prompts.
    """
    
    def __init__(self):
        self.type = "BoyoLoRAJSONBuilder"
        
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        # Get the available LoRA files for the dropdowns
        lora_files = ["None"] + folder_paths.get_filename_list("loras")
        
        return {
            "required": {
                "name": ("STRING", {
                    "default": "my_lora_config",
                    "multiline": False,
                    "placeholder": "Configuration name (becomes filename)"
                }),
            },
            "optional": {
                "prompts": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Enter prompts, one per line\nLeave empty if no prompts needed"
                }),
                "high_noise_lora": (lora_files, {
                    "default": "None"
                }),
                "low_noise_lora": (lora_files, {
                    "default": "None"
                }),
                "save_config": ("BOOLEAN", {
                    "default": False,
                    "label_on": "SAVE NOW",
                    "label_off": "Ready to Save"
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status", "config_path")
    OUTPUT_NODE = True
    FUNCTION = "build_config"
    CATEGORY = "Boyo/LoRA Tools"
    
    def get_config_directory(self) -> str:
        """Get or create the lora_configs directory within the custom node pack."""
        # Find the custom nodes directory
        custom_nodes_path = folder_paths.get_folder_paths("custom_nodes")[0] if folder_paths.get_folder_paths("custom_nodes") else None
        
        if not custom_nodes_path:
            # Fallback to ComfyUI root if custom_nodes not found
            custom_nodes_path = os.path.join(folder_paths.base_path, "custom_nodes")
        
        # Create the lora_configs directory for your node pack
        config_dir = os.path.join(custom_nodes_path, "Boyonodes", "lora_configs")
        
        # Create directory if it doesn't exist
        os.makedirs(config_dir, exist_ok=True)
        
        print(f"LoRA config directory: {config_dir}")  # Debug output
        
        return config_dir
    
    def parse_prompts(self, prompts_text: str) -> List[str]:
        """Parse the multi-line prompts text into a list of prompts."""
        if not prompts_text or prompts_text.strip() == "":
            return []
        
        # Split by lines and clean up
        prompts = [line.strip() for line in prompts_text.strip().split('\n')]
        # Remove empty lines
        prompts = [prompt for prompt in prompts if prompt]
        
        return prompts
    
    def get_lora_path(self, lora_name: str) -> Optional[str]:
        """Get the full path for a LoRA file, or None if not selected."""
        if lora_name == "None" or not lora_name:
            return None
        
        # Get the LoRA directory path
        lora_paths = folder_paths.get_folder_paths("loras")
        if not lora_paths:
            return None
        
        # Return the relative path from the LoRA directory
        return os.path.join("models", "loras", lora_name)
    
    def create_config_json(self, name: str, prompts: List[str], 
                          high_noise_path: Optional[str], 
                          low_noise_path: Optional[str]) -> Dict[str, Any]:
        """Create the configuration dictionary."""
        return {
            "name": name,
            "high_noise_path": high_noise_path,
            "low_noise_path": low_noise_path,
            "prompts": prompts
        }
    
    def save_config(self, config: Dict[str, Any], filename: str) -> Tuple[bool, str]:
        """Save the configuration to a JSON file."""
        try:
            config_dir = self.get_config_directory()
            
            # Ensure filename has .json extension
            if not filename.endswith('.json'):
                filename += '.json'
            
            config_path = os.path.join(config_dir, filename)
            
            # Save the JSON file
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            return True, config_path
            
        except Exception as e:
            return False, str(e)
    
    def build_config(self, name: str, prompts: str = "", 
                    high_noise_lora: str = "None", 
                    low_noise_lora: str = "None",
                    save_config: bool = False) -> Tuple[str, str]:
        """Main function to build and optionally save the LoRA configuration."""
        
        # Parse inputs
        parsed_prompts = self.parse_prompts(prompts)
        high_noise_path = self.get_lora_path(high_noise_lora)
        low_noise_path = self.get_lora_path(low_noise_lora)
        
        # Create configuration
        config = self.create_config_json(name, parsed_prompts, high_noise_path, low_noise_path)
        
        # Generate status message
        status_parts = []
        status_parts.append(f"Config '{name}' prepared")
        
        if parsed_prompts:
            status_parts.append(f"with {len(parsed_prompts)} prompt(s)")
        else:
            status_parts.append("with no prompts")
        
        if high_noise_path:
            status_parts.append(f"High: {os.path.basename(high_noise_path)}")
        if low_noise_path:
            status_parts.append(f"Low: {os.path.basename(low_noise_path)}")
        
        if not high_noise_path and not low_noise_path:
            status_parts.append("(utility config - no LoRAs)")
        
        # Save if requested
        config_path = ""
        if save_config:
            success, result = self.save_config(config, name)
            if success:
                status_parts.append("✓ SAVED")
                config_path = result
            else:
                status_parts.append(f"✗ SAVE FAILED: {result}")
        else:
            status_parts.append("(ready to save)")
        
        status = " | ".join(status_parts)
        
        return (status, config_path)

# Node registration
NODE_CLASS_MAPPINGS = {
    "BoyoLoRAJSONBuilder": BoyoLoRAJSONBuilder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoLoRAJSONBuilder": "Boyo LoRA JSON Builder"
}
