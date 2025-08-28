import os
import random
import logging

class BoyoPromptLoop:
    """Prompt iteration management for bastard loops"""
    
    @classmethod
    def INPUT_TYPES(cls):
        # Try to find prompts folder, create if doesn't exist
        prompts_folder = os.path.join(os.path.dirname(__file__), 'prompts')
        if not os.path.exists(prompts_folder):
            os.makedirs(prompts_folder)
            # Create a sample file
            sample_file = os.path.join(prompts_folder, 'sample_prompts.txt')
            with open(sample_file, 'w', encoding='utf-8') as f:
                f.write("a beautiful landscape\n")
                f.write("a majestic mountain\n")
                f.write("a serene lake\n")
                f.write("a vibrant sunset\n")
                f.write("a mystical forest\n")
        
        # Get available text files
        text_files = []
        try:
            text_files = [f for f in os.listdir(prompts_folder) if f.endswith('.txt')]
        except:
            text_files = ["No .txt files found"]
        
        if not text_files:
            text_files = ["No .txt files found"]
        
        return {
            "required": {
                "text_file": (text_files,),
                "mode": (["sequential", "random", "single"],),
                "start_seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFF
                }),
                "loop_count": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
            },
            "optional": {
                "prefix": ("STRING", {"default": ""}),
                "suffix": ("STRING", {"default": ""}),
                "single_prompt": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
            }
        }

    RETURN_TYPES = ("PROMPT_LOOP_CONFIG",)
    RETURN_NAMES = ("prompt_config",)
    FUNCTION = "setup_prompt_loop"
    CATEGORY = "boyo/bastardloops"

    def setup_prompt_loop(self, text_file, mode, start_seed, loop_count, prefix="", suffix="", single_prompt=""):
        """Set up the prompt loop configuration"""
        
        prompts = []
        
        if mode == "single":
            # Use the single prompt for all iterations
            if single_prompt.strip():
                base_prompt = single_prompt.strip()
                for i in range(loop_count):
                    prompts.append(f"{prefix}{base_prompt}{suffix}")
            else:
                # Fallback to a default prompt
                for i in range(loop_count):
                    prompts.append(f"{prefix}a beautiful image{suffix}")
        
        else:
            # Load prompts from file
            prompts_folder = os.path.join(os.path.dirname(__file__), 'prompts')
            file_path = os.path.join(prompts_folder, text_file)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                
                if not lines:
                    # Fallback if file is empty
                    lines = ["a beautiful image", "a stunning artwork", "a masterpiece"]
                
                total_lines = len(lines)
                
                if mode == "sequential":
                    # Sequential mode: start from seed position and loop through
                    for i in range(loop_count):
                        line_index = (start_seed + i) % total_lines
                        base_prompt = lines[line_index]
                        prompts.append(f"{prefix}{base_prompt}{suffix}")
                
                elif mode == "random":
                    # Random mode: use seed for reproducible randomness
                    random.seed(start_seed)
                    for i in range(loop_count):
                        # Re-seed for each iteration to get different but reproducible results
                        random.seed(start_seed + i)
                        base_prompt = random.choice(lines)
                        prompts.append(f"{prefix}{base_prompt}{suffix}")
            
            except Exception as e:
                logging.warning(f"BoyoPromptLoop: Could not read file {text_file}: {e}")
                # Fallback prompts
                for i in range(loop_count):
                    prompts.append(f"{prefix}a beautiful image (iteration {i+1}){suffix}")
        
        # Create configuration object
        config = {
            "prompts": prompts,
            "mode": mode,
            "start_seed": start_seed,
            "loop_count": loop_count,
            "prefix": prefix,
            "suffix": suffix,
            "text_file": text_file if mode != "single" else None,
            "single_prompt": single_prompt if mode == "single" else None
        }
        
        logging.info(f"BoyoPromptLoop: Generated {len(prompts)} prompts in {mode} mode")
        for i, prompt in enumerate(prompts[:3]):  # Log first 3 prompts
            logging.info(f"  Prompt {i+1}: {prompt}")
        if len(prompts) > 3:
            logging.info(f"  ... and {len(prompts)-3} more")
        
        return (config,)

    @classmethod
    def IS_CHANGED(cls, text_file, mode, start_seed, loop_count, prefix, suffix, single_prompt):
        # Force update when any parameter changes
        return float(f"{start_seed}.{hash((text_file, mode, loop_count, prefix, suffix, single_prompt)) % 1000}")

class BoyoPromptInjector:
    """Injects prompts from the loop configuration into the workflow"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_config": ("PROMPT_LOOP_CONFIG",),
                "iteration_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999
                }),
            },
            "optional": {
                "fallback_prompt": ("STRING", {
                    "default": "a beautiful image",
                    "multiline": True
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "inject_prompt"
    CATEGORY = "boyo/bastardloops"

    def inject_prompt(self, prompt_config, iteration_index, fallback_prompt="a beautiful image"):
        """Inject the appropriate prompt for the current iteration"""
        
        if not isinstance(prompt_config, dict) or "prompts" not in prompt_config:
            logging.warning("BoyoPromptInjector: Invalid prompt_config received")
            return (fallback_prompt,)
        
        prompts = prompt_config["prompts"]
        
        if not prompts:
            logging.warning("BoyoPromptInjector: No prompts in config")
            return (fallback_prompt,)
        
        # Get the prompt for this iteration
        if iteration_index < len(prompts):
            selected_prompt = prompts[iteration_index]
        else:
            # Loop back if we run out of prompts
            selected_prompt = prompts[iteration_index % len(prompts)]
        
        logging.info(f"BoyoPromptInjector: Iteration {iteration_index} -> '{selected_prompt}'")
        
        return (selected_prompt,)

# Node mappings
NODE_CLASS_MAPPINGS = {
    "BoyoPromptLoop": BoyoPromptLoop,
    "BoyoPromptInjector": BoyoPromptInjector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoPromptLoop": "Boyo Prompt Loop",
    "BoyoPromptInjector": "Boyo Prompt Injector"
}