import json

class BoyoStoryboardJsonParser:
    """
    Parses JSON output from the Unified Video Storyboard Director Agent
    and splits it into individual image and video prompts for each scene.
    
    Outputs:
    - 6 image prompts (image_1 through image_6)
    - 6 video prompts (video_1 through video_6)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_input": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "dynamicPrompts": False
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",) * 12
    RETURN_NAMES = (
        "image_1", "image_2", "image_3", "image_4", "image_5", "image_6",
        "video_1", "video_2", "video_3", "video_4", "video_5", "video_6"
    )
    FUNCTION = "parse_json"
    CATEGORY = "Boyo/Storyboard"
    
    def parse_json(self, json_input):
        """
        Parse the JSON and extract image and video prompts for all 6 scenes.
        """
        try:
            # Parse the JSON input
            data = json.loads(json_input.strip())
            
            # Validate structure
            if "scenes" not in data:
                raise ValueError("JSON missing 'scenes' array")
            
            scenes = data["scenes"]
            
            if len(scenes) != 6:
                raise ValueError(f"Expected 6 scenes, got {len(scenes)}")
            
            # Extract prompts
            image_prompts = []
            video_prompts = []
            
            for i, scene in enumerate(scenes, 1):
                # Validate scene number matches
                if scene.get("scene_number") != i:
                    print(f"Warning: Scene {i} has scene_number {scene.get('scene_number')}")
                
                # Extract image prompt
                image_prompt = scene.get("image_prompt", "")
                if not image_prompt:
                    print(f"Warning: Scene {i} missing image_prompt")
                    image_prompt = f"[Missing image prompt for scene {i}]"
                
                # Extract video prompt
                video_prompt = scene.get("video_prompt", "")
                if not video_prompt:
                    print(f"Warning: Scene {i} missing video_prompt")
                    video_prompt = f"[Missing video prompt for scene {i}]"
                
                image_prompts.append(image_prompt)
                video_prompts.append(video_prompt)
            
            # Return all 12 outputs (6 image + 6 video)
            return tuple(image_prompts + video_prompts)
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON: {str(e)}"
            print(f"BoyoStoryboardJsonParser Error: {error_msg}")
            # Return error message for all outputs
            return (error_msg,) * 12
            
        except (ValueError, KeyError) as e:
            error_msg = f"JSON structure error: {str(e)}"
            print(f"BoyoStoryboardJsonParser Error: {error_msg}")
            # Return error message for all outputs
            return (error_msg,) * 12
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"BoyoStoryboardJsonParser Error: {error_msg}")
            # Return error message for all outputs
            return (error_msg,) * 12


NODE_CLASS_MAPPINGS = {
    "BoyoStoryboardJsonParser": BoyoStoryboardJsonParser
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoStoryboardJsonParser": "Boyo Storyboard JSON Parser"
}
