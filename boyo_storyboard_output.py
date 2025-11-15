import json

class BoyoStoryboardOutput:
    """
    Output node for parsing structured storyboard JSON from ollama into separate prompt outputs.
    Takes JSON input and provides 12 separate text outputs (6 image + 6 video prompts).
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_input": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Paste JSON output from ollama here..."
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",) * 12
    RETURN_NAMES = (
        "image_scene1", "image_scene2", "image_scene3", "image_scene4", "image_scene5", "image_scene6",
        "video_scene1", "video_scene2", "video_scene3", "video_scene4", "video_scene5", "video_scene6"
    )
    FUNCTION = "parse_storyboard"
    CATEGORY = "Boyo/Storyboard"
    
    def parse_storyboard(self, json_input):
        """
        Parse JSON storyboard into separate prompt outputs.
        Returns 12 strings: 6 image prompts + 6 video prompts.
        """
        
        # Initialize empty outputs
        outputs = [""] * 12
        
        try:
            # Clean and parse the JSON
            json_text = json_input.strip()
            
            # Handle potential markdown code blocks
            if json_text.startswith("```json"):
                json_text = json_text.replace("```json", "").replace("```", "").strip()
            elif json_text.startswith("```"):
                json_text = json_text.replace("```", "").strip()
            
            # Parse the JSON
            data = json.loads(json_text)
            
            # Extract image prompts (outputs 0-5)
            if "imagePrompts" in data:
                image_prompts = data["imagePrompts"]
                for i in range(1, 7):  # scene1 through scene6
                    scene_key = f"scene{i}"
                    if scene_key in image_prompts:
                        outputs[i-1] = image_prompts[scene_key]
                    else:
                        outputs[i-1] = f"[Missing image prompt for scene {i}]"
            
            # Extract video prompts (outputs 6-11)
            if "videoPrompts" in data:
                video_prompts = data["videoPrompts"]
                for i in range(1, 7):  # scene1 through scene6
                    scene_key = f"scene{i}"
                    if scene_key in video_prompts:
                        outputs[5+i] = video_prompts[scene_key]  # Start at index 6
                    else:
                        outputs[5+i] = f"[Missing video prompt for scene {i}]"
        
        except json.JSONDecodeError as e:
            # If JSON parsing fails, provide error information
            error_msg = f"[JSON Parse Error: {str(e)}]"
            for i in range(12):
                outputs[i] = error_msg
        
        except Exception as e:
            # Handle any other errors
            error_msg = f"[Parse Error: {str(e)}]"
            for i in range(12):
                outputs[i] = error_msg
        
        return tuple(outputs)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "BoyoStoryboardOutput": BoyoStoryboardOutput
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoStoryboardOutput": "Boyo Storyboard Output"
}
