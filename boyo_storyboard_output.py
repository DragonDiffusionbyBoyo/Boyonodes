import json
import re

class BoyoStoryboardOutput:
    """
    Output node for parsing structured storyboard JSON from ollama into separate prompt outputs.
    Takes JSON input and provides 12 separate text outputs (6 image + 6 video prompts).
    For OVI mode, integrates bypassed speech back into video prompts with <S></S> formatting.
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
            },
            "optional": {
                "speech_scene1": ("STRING", {"default": ""}),
                "speech_scene2": ("STRING", {"default": ""}),
                "speech_scene3": ("STRING", {"default": ""}),
                "speech_scene4": ("STRING", {"default": ""}),
                "speech_scene5": ("STRING", {"default": ""}),
                "speech_scene6": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING",) * 12
    RETURN_NAMES = (
        "image_scene1", "image_scene2", "image_scene3", "image_scene4", "image_scene5", "image_scene6",
        "video_scene1", "video_scene2", "video_scene3", "video_scene4", "video_scene5", "video_scene6"
    )
    FUNCTION = "parse_storyboard"
    CATEGORY = "Boyo/Storyboard"
    
    def parse_storyboard(self, json_input, speech_scene1="", speech_scene2="", speech_scene3="", speech_scene4="", speech_scene5="", speech_scene6=""):
        """
        Parse JSON storyboard into separate prompt outputs.
        Returns 12 strings: 6 image prompts + 6 video prompts.
        For OVI mode, integrates speech back into video prompts.
        """
        
        # Initialize empty outputs
        outputs = [""] * 12
        
        # Collect speech inputs
        speech_inputs = [speech_scene1, speech_scene2, speech_scene3, speech_scene4, speech_scene5, speech_scene6]
        
        # Check if we have speech inputs (indicates OVI mode)
        has_speech = any(speech.strip() for speech in speech_inputs)
        
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
                        base_video_prompt = video_prompts[scene_key]
                        
                        # If we have speech for this scene (OVI mode), integrate it
                        if has_speech and i <= len(speech_inputs) and speech_inputs[i-1].strip():
                            ovi_video_prompt = self._create_ovi_prompt(base_video_prompt, speech_inputs[i-1].strip())
                            outputs[5+i] = ovi_video_prompt  # Start at index 6
                        else:
                            outputs[5+i] = base_video_prompt  # Standard video prompt
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
    
    def _create_ovi_prompt(self, base_video_prompt, speech_text):
        """
        Create an OVI-formatted video prompt by integrating speech and audio descriptions.
        """
        # Add speech formatting
        speech_formatted = f"<S>{speech_text}</S>"
        
        # Generate basic audio description based on speech
        audio_description = self._generate_audio_description(speech_text)
        
        # Combine base prompt with speech and audio
        ovi_prompt = f"{base_video_prompt} Character speaks: {speech_formatted}. {audio_description}"
        
        return ovi_prompt
    
    def _generate_audio_description(self, speech_text):
        """
        Generate basic audio description based on speech content and common patterns.
        """
        # Basic audio patterns based on speech characteristics
        audio_elements = []
        
        # Determine voice characteristics based on speech content
        if any(word in speech_text.lower() for word in ['!', 'amazing', 'wonderful', 'fantastic', 'incredible']):
            audio_elements.append("excited voice")
        elif any(word in speech_text.lower() for word in ['?', 'what', 'how', 'where', 'when', 'why']):
            audio_elements.append("curious voice")
        elif any(word in speech_text.lower() for word in ['whisper', 'quiet', 'soft']):
            audio_elements.append("soft whispered voice")
        elif any(word in speech_text.lower() for word in ['shout', 'loud', 'call']):
            audio_elements.append("loud clear voice")
        else:
            audio_elements.append("natural speaking voice")
        
        # Add common ambient sounds (can be made more sophisticated)
        ambient_sounds = ["ambient room tone", "soft background atmosphere", "subtle environmental sounds"]
        audio_elements.append(ambient_sounds[0])  # Default to room tone
        
        # Format as Audio: description
        return f"Audio: {', '.join(audio_elements)}"


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "BoyoStoryboardOutput": BoyoStoryboardOutput
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoStoryboardOutput": "Boyo Storyboard Output"
}
