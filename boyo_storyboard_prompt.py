import json

class BoyoStoryboardPrompt:
    """
    Input node for generating structured storyboard prompts using ollama models.
    Now accepts Movie Director output format for temporal reasoning and story structure.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "movie_director_output": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Paste Movie Director output here (from GPT, Gemini, Claude, etc.)"
                }),
                "image_trigger_word": ("STRING", {
                    "default": "Next Scene:",
                    "placeholder": "Trigger word for image model (e.g., 'Next Scene:', 'Character:', etc.)"
                }),
                "video_trigger_word": ("STRING", {
                    "default": "Next Scene:",
                    "placeholder": "Trigger word for video model (e.g., 'Motion:', 'Camera:', etc.)"
                }),
                "system_prompt_type": (["System Prompt 1 (6 Scenes)", "System Prompt 2 (Traveling Prompts)"], {
                    "default": "System Prompt 1 (6 Scenes)"
                }),
                "traveling_prompt_count": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "display": "number"
                }),
            },
            "optional": {
                "additional_details": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Any additional instructions..."
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("formatted_prompt",)
    FUNCTION = "generate_prompt"
    CATEGORY = "Boyo/Storyboard"
    
    def generate_prompt(self, movie_director_output, image_trigger_word, video_trigger_word, system_prompt_type, traveling_prompt_count, additional_details=""):
        """
        Generate a formatted prompt for ollama based on Movie Director output.
        """
        
        # Parse the Movie Director output
        parsed_data = self._parse_movie_director_output(movie_director_output)
        
        # System Prompt selection
        if system_prompt_type == "System Prompt 1 (6 Scenes)":
            system_prompt = self._get_system_prompt_1(image_trigger_word, video_trigger_word, parsed_data)
        elif system_prompt_type == "System Prompt 2 (Traveling Prompts)":
            system_prompt = self._get_system_prompt_2(image_trigger_word, video_trigger_word, traveling_prompt_count, parsed_data)
        else:
            system_prompt = self._get_system_prompt_1(image_trigger_word, video_trigger_word, parsed_data)  # Default fallback
        
        # Add additional details if provided
        if additional_details.strip():
            system_prompt += f"\n\nAdditional Instructions: {additional_details}"
        
        return (system_prompt,)
    
    def _parse_movie_director_output(self, output):
        """
        Parse Movie Director output format into structured data.
        Handles both GPT and Gemini style outputs.
        """
        parsed = {
            "story_concept": "",
            "main_character": "",
            "style_setting": "",
            "scenes": []
        }
        
        lines = output.split('\n')
        current_section = None
        current_scene_text = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect section headers
            if line.startswith("STORY CONCEPT:"):
                current_section = "story_concept"
                parsed["story_concept"] = line.replace("STORY CONCEPT:", "").strip()
            elif line.startswith("MAIN CHARACTER:"):
                current_section = "main_character"
                parsed["main_character"] = line.replace("MAIN CHARACTER:", "").strip()
            elif line.startswith("STYLE & SETTING:"):
                current_section = "style_setting"
                parsed["style_setting"] = line.replace("STYLE & SETTING:", "").strip()
            elif line.startswith("SCENE BREAKDOWN:"):
                current_section = "scenes"
            elif line.startswith("Scene ") and current_section == "scenes":
                # Save previous scene if exists
                if current_scene_text:
                    parsed["scenes"].append(current_scene_text.strip())
                current_scene_text = line
            elif current_section == "scenes" and line:
                # Continue building current scene
                current_scene_text += " " + line
            elif current_section and line:
                # Continue building current section
                if current_section == "story_concept":
                    parsed["story_concept"] += " " + line
                elif current_section == "main_character":
                    parsed["main_character"] += " " + line
                elif current_section == "style_setting":
                    parsed["style_setting"] += " " + line
        
        # Don't forget the last scene
        if current_scene_text:
            parsed["scenes"].append(current_scene_text.strip())
            
        return parsed
    
    def _get_system_prompt_1(self, image_trigger_word, video_trigger_word, parsed_data):
        """
        System Prompt 1: Generates 6 image prompts and 6 corresponding video prompts using Movie Director structure.
        """
        return f"""You are an expert AI visual prompt engineer, tasked with converting structured movie direction into specific image and video prompts for AI generation. You have been provided with a complete story breakdown from a film director including character details, style guidelines, and scene-by-scene structure.

CRITICAL UNDERSTANDING: Use the Movie Director's scene breakdowns as the foundation for your prompts. Each scene description provides the keyframe moment - your job is to convert these into detailed visual prompts while maintaining consistency.

MOVIE DIRECTOR INPUT:
Story: {parsed_data.get('story_concept', 'Not provided')}
Character: {parsed_data.get('main_character', 'Not provided')}
Style: {parsed_data.get('style_setting', 'Not provided')}

Scene Structure:
{chr(10).join(parsed_data.get('scenes', []))}

CRITICAL FORMATTING REQUIREMENTS:
- Output ONLY valid JSON format
- No explanatory text outside the JSON structure
- Each prompt must be a single line with NO line breaks within the prompt text
- Use proper JSON escaping for quotes and special characters
- Start each image prompt with: "{image_trigger_word}"
- Start each video prompt with: "{video_trigger_word}"

Structure your response as:
{{
  "imagePrompts": {{
    "scene1": "{image_trigger_word} detailed static scene description...",
    "scene2": "{image_trigger_word} detailed static scene description...", 
    "scene3": "{image_trigger_word} detailed static scene description...",
    "scene4": "{image_trigger_word} detailed static scene description...",
    "scene5": "{image_trigger_word} detailed static scene description...",
    "scene6": "{image_trigger_word} detailed static scene description..."
  }},
  "videoPrompts": {{
    "scene1": "{video_trigger_word} action transitioning from scene 1 to scene 2 with camera work...",
    "scene2": "{video_trigger_word} action transitioning from scene 2 to scene 3 with camera work...",
    "scene3": "{video_trigger_word} action transitioning from scene 3 to scene 4 with camera work...", 
    "scene4": "{video_trigger_word} action transitioning from scene 4 to scene 5 with camera work...",
    "scene5": "{video_trigger_word} action transitioning from scene 5 to scene 6 with camera work...",
    "scene6": "{video_trigger_word} final scene completion with camera work..."
  }}
}}

IMAGE PROMPT GUIDELINES:
- Convert each Movie Director scene description into a detailed static visual
- Maintain the character description exactly as provided
- Use the style and setting details consistently
- Focus on the specific keyframe moment described in each scene
- Include lighting, composition, and atmospheric details
- NO camera movements or motion descriptions

VIDEO PROMPT GUIDELINES:
- Video prompts show the ACTION that transitions from current scene to the next scene
- Include character actions, movement, and story progression
- Add camera movements (pan, tilt, zoom, dolly, tracking shots, handheld) to follow the action
- Show what the character DOES to get from the current scene moment to the next scene moment
- Think of video prompts as the "in-between animation" that connects the static image keyframes
- Final scene (scene 6) should show completion/resolution action rather than transition

CONSISTENCY REQUIREMENTS:
- Maintain character appearance exactly as described by the Movie Director
- Keep consistent style, lighting mood, and color palette from the Movie Director's guidelines
- Each scene should flow logically based on the Movie Director's structure
- Character positioning and state should make sense in the sequence"""

    def _get_system_prompt_2(self, image_trigger_word, video_trigger_word, traveling_prompt_count, parsed_data):
        """
        System Prompt 2: Generates 6 image prompts and 6 video prompts with multiple traveling sub-prompts using Movie Director structure.
        """
        return f"""You are an expert AI visual prompt engineer, tasked with converting structured movie direction into specific image and traveling video prompts for AI generation. You have been provided with a complete story breakdown from a film director including character details, style guidelines, and scene-by-scene structure.

CRITICAL UNDERSTANDING: Use the Movie Director's scene breakdowns as the foundation for your prompts. Each scene description provides the keyframe moment - your job is to convert these into detailed visual prompts and extended traveling video sequences while maintaining consistency.

MOVIE DIRECTOR INPUT:
Story: {parsed_data.get('story_concept', 'Not provided')}
Character: {parsed_data.get('main_character', 'Not provided')}
Style: {parsed_data.get('style_setting', 'Not provided')}

Scene Structure:
{chr(10).join(parsed_data.get('scenes', []))}

CRITICAL FORMATTING REQUIREMENTS:
- Output ONLY valid JSON format
- No explanatory text outside the JSON structure
- Each image prompt must be a single line with NO line breaks
- Each video prompt contains {traveling_prompt_count} sub-prompts separated by \\n newlines
- Use proper JSON escaping for quotes and special characters
- Start each image prompt with: "{image_trigger_word}"
- Start each video sub-prompt with: "{video_trigger_word}"

Structure your response as:
{{
  "imagePrompts": {{
    "scene1": "{image_trigger_word} detailed static scene description...",
    "scene2": "{image_trigger_word} detailed static scene description...", 
    "scene3": "{image_trigger_word} detailed static scene description...",
    "scene4": "{image_trigger_word} detailed static scene description...",
    "scene5": "{image_trigger_word} detailed static scene description...",
    "scene6": "{image_trigger_word} detailed static scene description..."
  }},
  "videoPrompts": {{
    "scene1": "{video_trigger_word} camera movement description\\n{video_trigger_word} camera movement description\\n{video_trigger_word} camera movement description",
    "scene2": "{video_trigger_word} camera movement description\\n{video_trigger_word} camera movement description\\n{video_trigger_word} camera movement description",
    "scene3": "{video_trigger_word} camera movement description\\n{video_trigger_word} camera movement description\\n{video_trigger_word} camera movement description",
    "scene4": "{video_trigger_word} camera movement description\\n{video_trigger_word} camera movement description\\n{video_trigger_word} camera movement description",
    "scene5": "{video_trigger_word} camera movement description\\n{video_trigger_word} camera movement description\\n{video_trigger_word} camera movement description",
    "scene6": "{video_trigger_word} camera movement description\\n{video_trigger_word} camera movement description\\n{video_trigger_word} camera movement description"
  }}
}}

IMAGE PROMPT GUIDELINES:
- Convert each Movie Director scene description into a detailed static visual
- Maintain the character description exactly as provided
- Use the style and setting details consistently
- Focus on the specific keyframe moment described in each scene
- Include lighting, composition, and atmospheric details
- NO camera movements, motion blur, or temporal effects

VIDEO TRAVELING PROMPT GUIDELINES:
- Each scene gets {traveling_prompt_count} sequential sub-prompts (each ~5 seconds)
- Video prompts show the ACTION that transitions from current scene to the next scene
- Include character actions, movement, and story progression within the scene
- Add camera movements (pan, tilt, zoom, dolly, tracking, handheld) to follow the action
- Each sub-prompt advances both the story and cinematography
- Sub-prompts should flow smoothly together, showing continuous action within that scene
- Think of video prompts as the "in-between animation" that connects the static image keyframes
- Show what the character DOES in this scene to get to the next story beat

CONSISTENCY REQUIREMENTS:
- Maintain character appearance exactly as described by the Movie Director
- Keep consistent style, lighting mood, and color palette from the Movie Director's guidelines
- Each scene should flow logically based on the Movie Director's structure
- Within each scene, all {traveling_prompt_count} sub-prompts should maintain visual continuity"""


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "BoyoStoryboardPrompt": BoyoStoryboardPrompt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoStoryboardPrompt": "Boyo Storyboard Prompt"
}
