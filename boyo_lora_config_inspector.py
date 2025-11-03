import json
import os
import folder_paths
from typing import Dict, List, Optional, Any, Tuple

class BoyoLoRAConfigInspector:
    """
    A ComfyUI node for previewing the contents of LoRA configuration JSON files.
    Displays config details to help users make informed choices.
    """
    
    def __init__(self):
        self.type = "BoyoLoRAConfigInspector"
        
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        # Get available config files
        config_files = cls.get_config_files()
        
        return {
            "required": {
                "config_to_inspect": (config_files, {
                    "default": "None",
                    "tooltip": "Select a LoRA configuration to preview its contents"
                }),
            },
            "optional": {
                "refresh_trigger": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999,
                    "tooltip": "Change this value to refresh the config list"
                })
            }
        }
    
    @classmethod
    def get_config_files(cls) -> List[str]:
        """Get list of available LoRA configuration files."""
        try:
            # Find the custom nodes directory
            custom_nodes_path = folder_paths.get_folder_paths("custom_nodes")[0] if folder_paths.get_folder_paths("custom_nodes") else None
            
            if not custom_nodes_path:
                custom_nodes_path = os.path.join(folder_paths.base_path, "custom_nodes")
            
            config_dir = os.path.join(custom_nodes_path, "Boyonodes", "lora_configs")
            
            if not os.path.exists(config_dir):
                return ["None"]
            
            # Get all .json files
            json_files = [f for f in os.listdir(config_dir) if f.endswith('.json')]
            json_files.sort()  # Sort alphabetically
            
            return ["None"] + json_files
            
        except Exception as e:
            print(f"Error loading config files: {e}")
            return ["None"]
    
    def load_config(self, config_filename: str) -> Optional[Dict[str, Any]]:
        """Load a specific configuration file."""
        if config_filename == "None" or not config_filename:
            return None
        
        try:
            # Find the custom nodes directory
            custom_nodes_path = folder_paths.get_folder_paths("custom_nodes")[0] if folder_paths.get_folder_paths("custom_nodes") else None
            
            if not custom_nodes_path:
                custom_nodes_path = os.path.join(folder_paths.base_path, "custom_nodes")
            
            config_dir = os.path.join(custom_nodes_path, "Boyonodes", "lora_configs")
            config_path = os.path.join(config_dir, config_filename)
            
            if not os.path.exists(config_path):
                return None
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            return config
            
        except Exception as e:
            print(f"Error loading config '{config_filename}': {e}")
            return None
    
    def format_config_display(self, config: Dict[str, Any], filename: str) -> str:
        """Format the configuration data for display."""
        if not config:
            return f"❌ Config '{filename}' not found or invalid"
        
        # Build display text
        display_lines = []
        display_lines.append(f"📋 LoRA Configuration: {config.get('name', 'Unknown')}")
        display_lines.append("=" * 50)
        
        # LoRA Files section
        display_lines.append("\n🎯 LoRA FILES:")
        high_noise = config.get("high_noise_path")
        low_noise = config.get("low_noise_path")
        
        if high_noise:
            # Extract just the filename for cleaner display
            high_filename = high_noise.split('/')[-1].split('\\')[-1]
            display_lines.append(f"  📈 High Noise: {high_filename}")
        else:
            display_lines.append(f"  📈 High Noise: ❌ Not set")
        
        if low_noise:
            # Extract just the filename for cleaner display
            low_filename = low_noise.split('/')[-1].split('\\')[-1]
            display_lines.append(f"  📉 Low Noise:  {low_filename}")
        else:
            display_lines.append(f"  📉 Low Noise:  ❌ Not set")
        
        # Configuration type assessment
        if high_noise and low_noise:
            if high_noise == low_noise:
                display_lines.append(f"  🔧 Type: UTILITY LoRA (same file for both)")
            else:
                display_lines.append(f"  🎭 Type: PAIRED LoRA (different high/low files)")
        elif high_noise or low_noise:
            display_lines.append(f"  🎯 Type: SINGLE LoRA")
        else:
            display_lines.append(f"  ⚠️  Type: CONFIG ONLY (no LoRA files)")
        
        # Prompts section
        prompts = config.get("prompts", [])
        display_lines.append(f"\n💬 PROMPTS ({len(prompts)} total):")
        
        if prompts:
            for i, prompt in enumerate(prompts, 1):
                # Truncate long prompts for display
                display_prompt = prompt if len(prompt) <= 60 else prompt[:57] + "..."
                display_lines.append(f"  {i}. {display_prompt}")
        else:
            display_lines.append(f"  ❌ No prompts (utility/silent LoRA)")
        
        # Usage recommendations
        display_lines.append(f"\n💡 USAGE RECOMMENDATIONS:")
        
        if not prompts:
            display_lines.append(f"  • Use 'Mute' strategy (no prompts needed)")
            display_lines.append(f"  • Perfect for utility/enhancement LoRAs")
        elif len(prompts) == 1:
            display_lines.append(f"  • Use 'First Only' mode (single prompt)")
            display_lines.append(f"  • Good for character or style LoRAs")
        else:
            display_lines.append(f"  • Use 'Cycle Through' for variety")
            display_lines.append(f"  • Use 'Random' for experimentation")
            display_lines.append(f"  • Use 'First Only' for consistency")
        
        # File status check
        display_lines.append(f"\n🔍 FILE STATUS:")
        lora_dir = folder_paths.get_folder_paths("loras")[0] if folder_paths.get_folder_paths("loras") else ""
        
        if high_noise:
            # Check if high noise file exists
            relative_path = high_noise.replace("models\\loras\\", "").replace("models/loras/", "")
            full_path = os.path.join(lora_dir, relative_path.replace("/", os.sep))
            status = "✅ Found" if os.path.exists(full_path) else "❌ Missing"
            display_lines.append(f"  📈 High Noise: {status}")
        
        if low_noise:
            # Check if low noise file exists
            relative_path = low_noise.replace("models\\loras\\", "").replace("models/loras/", "")
            full_path = os.path.join(lora_dir, relative_path.replace("/", os.sep))
            status = "✅ Found" if os.path.exists(full_path) else "❌ Missing"
            display_lines.append(f"  📉 Low Noise: {status}")
        
        return "\n".join(display_lines)
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("config_summary", "raw_json", "file_status")
    OUTPUT_NODE = True
    FUNCTION = "inspect_config"
    CATEGORY = "Boyo/LoRA Tools"
    
    def inspect_config(self, config_to_inspect: str, refresh_trigger: int = 0) -> Tuple[str, str, str]:
        """Main function to inspect and display LoRA configuration details."""
        
        if config_to_inspect == "None":
            return (
                "👋 Select a LoRA configuration to inspect its contents.",
                "No configuration selected",
                "Ready to inspect"
            )
        
        # Load the configuration
        config = self.load_config(config_to_inspect)
        
        if not config:
            return (
                f"❌ Failed to load configuration: {config_to_inspect}",
                "Error: Configuration not found or invalid JSON",
                "Error"
            )
        
        # Generate formatted display
        summary = self.format_config_display(config, config_to_inspect)
        
        # Generate raw JSON for technical inspection
        raw_json = json.dumps(config, indent=2, ensure_ascii=False)
        
        # Generate file status summary
        high_noise = config.get("high_noise_path")
        low_noise = config.get("low_noise_path")
        prompts = config.get("prompts", [])
        
        status_parts = []
        status_parts.append(f"Config: {config.get('name', 'Unknown')}")
        
        if high_noise and low_noise:
            status_parts.append("Type: Paired LoRA")
        elif high_noise or low_noise:
            status_parts.append("Type: Single LoRA")
        else:
            status_parts.append("Type: Config only")
        
        status_parts.append(f"Prompts: {len(prompts)}")
        
        file_status = " | ".join(status_parts)
        
        return (summary, raw_json, file_status)

# Node registration
NODE_CLASS_MAPPINGS = {
    "BoyoLoRAConfigInspector": BoyoLoRAConfigInspector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoLoRAConfigInspector": "Boyo LoRA Config Inspector"
}
