import json
import os
import folder_paths
import random
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

class BoyoLoRAPairedLoader:
    """
    A ComfyUI node for loading multiple LoRA configurations simultaneously
    with flexible prompt handling strategies.
    """
    
    def __init__(self):
        self.type = "BoyoLoRAPairedLoader"
        self.prompt_counters = {}  # Track cycling counters per config
        
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        # Get available config files
        config_files = cls.get_config_files()
        
        prompt_modes = ["First Only", "Cycle Through", "Random"]
        prompt_strategies = ["Mute", "Concatenate", "Merge"]
        
        return {
            "required": {
                "prompt_mode": (prompt_modes, {
                    "default": "First Only"
                }),
            },
            "optional": {
                "prepend_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Text to add before all prompts"
                }),
                "append_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Text to add after all prompts"
                }),
                "lora_config_1": (config_files, {
                    "default": "None"
                }),
                "prompt_strategy_1": (prompt_strategies, {
                    "default": "Concatenate"
                }),
                "lora_config_2": (config_files, {
                    "default": "None"
                }),
                "prompt_strategy_2": (prompt_strategies, {
                    "default": "Concatenate"
                }),
                "lora_config_3": (config_files, {
                    "default": "None"
                }),
                "prompt_strategy_3": (prompt_strategies, {
                    "default": "Concatenate"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Seed for random prompt selection"
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
                print(f"Config file not found: {config_path}")
                return None
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            return config
            
        except Exception as e:
            print(f"Error loading config '{config_filename}': {e}")
            return None
    
    def get_prompt_from_config(self, config: Dict[str, Any], prompt_mode: str, 
                              config_name: str, seed: int) -> str:
        """Get a prompt from config based on the selected mode."""
        if not config or not config.get("prompts"):
            return ""
        
        prompts = config["prompts"]
        if not prompts:
            return ""
        
        if prompt_mode == "First Only":
            return prompts[0]
        
        elif prompt_mode == "Random":
            # Use seed for consistent randomness
            random.seed(seed + hash(config_name))
            return random.choice(prompts)
        
        elif prompt_mode == "Cycle Through":
            # Initialize counter if not exists
            if config_name not in self.prompt_counters:
                self.prompt_counters[config_name] = 0
            
            # Get current prompt and increment counter
            current_index = self.prompt_counters[config_name] % len(prompts)
            self.prompt_counters[config_name] += 1
            
            return prompts[current_index]
        
        return ""
    
    def apply_prompt_strategy(self, prompt: str, strategy: str) -> str:
        """Apply the selected prompt strategy."""
        if strategy == "Mute":
            return ""
        elif strategy == "Concatenate":
            return prompt
        elif strategy == "Merge":
            # For now, merge is the same as concatenate
            # TODO: Implement intelligent merging logic later
            return prompt
        
        return prompt
    
    def clean_prompt_combination(self, prompts: List[str], prepend: str = "", append: str = "") -> str:
        """Combine multiple prompts intelligently with prepend and append text."""
        # Filter out empty prompts
        valid_prompts = [p.strip() for p in prompts if p.strip()]
        
        if not valid_prompts and not prepend.strip() and not append.strip():
            return ""
        
        # Build final prompt with prepend/append
        parts = []
        
        # Add prepend if provided
        if prepend.strip():
            parts.append(prepend.strip())
        
        # Add the combined LoRA prompts
        if valid_prompts:
            parts.append(", ".join(valid_prompts))
        
        # Add append if provided
        if append.strip():
            parts.append(append.strip())
        
        return ", ".join(parts)
    
    # Use cached LoRA list to prevent dynamic changes
    RETURN_TYPES = (get_cached_lora_list(),) * 6 + ("STRING",) * 4
    RETURN_NAMES = (
        "high_noise_path_1", "low_noise_path_1", 
        "high_noise_path_2", "low_noise_path_2", 
        "high_noise_path_3", "low_noise_path_3",
        "prompt_1", "prompt_2", "prompt_3", "combined_prompts"
    )
    FUNCTION = "load_lora_configs"
    CATEGORY = "Boyo/LoRA Tools"
    
    def load_lora_configs(self, prompt_mode: str, 
                         prepend_text: str = "", append_text: str = "",
                         lora_config_1: str = "None", prompt_strategy_1: str = "Concatenate",
                         lora_config_2: str = "None", prompt_strategy_2: str = "Concatenate", 
                         lora_config_3: str = "None", prompt_strategy_3: str = "Concatenate",
                         seed: int = 0) -> Tuple[str, ...]:
        """Main function to load LoRA configurations and return paths and prompts."""
        
        lora_paths = []  # First 6 outputs: all LoRA paths
        prompts = []     # Last 4 outputs: all prompts
        all_prompts = [] # For combined output
        
        # Process each config slot
        configs_and_strategies = [
            (lora_config_1, prompt_strategy_1),
            (lora_config_2, prompt_strategy_2), 
            (lora_config_3, prompt_strategy_3)
        ]
        
        for i, (config_name, strategy) in enumerate(configs_and_strategies, 1):
            # Load configuration
            config = self.load_config(config_name)
            
            if config:
                # Get paths - preserve subdirectory structure relative to loras folder
                high_noise_path = config.get("high_noise_path")
                if high_noise_path:
                    # Extract the path relative to models/loras/
                    if "models\\loras\\" in high_noise_path:
                        high_noise_path = high_noise_path.split("models\\loras\\", 1)[1]
                    elif "models/loras/" in high_noise_path:
                        high_noise_path = high_noise_path.split("models/loras/", 1)[1]
                    # Convert backslashes to forward slashes for consistency
                    high_noise_path = high_noise_path.replace("\\", "/")
                    if not high_noise_path or high_noise_path.strip() == "":
                        high_noise_path = "none"
                else:
                    high_noise_path = "none"
                    
                low_noise_path = config.get("low_noise_path")
                if low_noise_path:
                    # Extract the path relative to models/loras/
                    if "models\\loras\\" in low_noise_path:
                        low_noise_path = low_noise_path.split("models\\loras\\", 1)[1]
                    elif "models/loras/" in low_noise_path:
                        low_noise_path = low_noise_path.split("models/loras/", 1)[1]
                    # Convert backslashes to forward slashes for consistency
                    low_noise_path = low_noise_path.replace("\\", "/")
                    if not low_noise_path or low_noise_path.strip() == "":
                        low_noise_path = "none"
                else:
                    low_noise_path = "none"
                
                # Get prompt based on mode
                raw_prompt = self.get_prompt_from_config(config, prompt_mode, config_name, seed)
                
                # Apply strategy
                final_prompt = self.apply_prompt_strategy(raw_prompt, strategy)
                
                # Add paths to LoRA outputs
                lora_paths.extend([high_noise_path, low_noise_path])
                
                # Add prompt to prompt outputs
                prompts.append(final_prompt)
                
                # Collect for combined output (only if not muted)
                if final_prompt:
                    all_prompts.append(final_prompt)
                    
            else:
                # Use "none" for missing config to match LoRA loader format
                lora_paths.extend(["none", "none"])
                prompts.append("")
        
        # Create combined prompts with prepend/append
        combined_prompts = self.clean_prompt_combination(all_prompts, prepend_text, append_text)
        prompts.append(combined_prompts)
        
        # Return: 6 LoRA paths + 4 prompt strings
        result = tuple(lora_paths + prompts)
        
        # Debug output
        print(f"LoRA Loader Debug - Returning: {result}")
        print(f"LoRA paths: {lora_paths}")
        print(f"Prompts: {prompts}")
        
        return result

# Node registration
NODE_CLASS_MAPPINGS = {
    "BoyoLoRAPairedLoader": BoyoLoRAPairedLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoLoRAPairedLoader": "Boyo LoRA Paired Loader"
}
