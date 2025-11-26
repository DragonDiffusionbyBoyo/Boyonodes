import json
import re

class BoyoStoryboardPrompt:
    """
    Input node for generating structured storyboard prompts using ollama models.
    Now accepts Movie Director output format for temporal reasoning and story structure.
    Includes OVI audio integration with speech extraction and bypass.
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
                "system_prompt_type": (["System Prompt 1 (6 Scenes)", "System Prompt 2 (Traveling Prompts)", "System Prompt 3 (OVI Audio)"], {
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
    
    RETURN_TYPES = ("STRING",) * 7  # Main prompt + 6 speech outputs
    RETURN_NAMES = ("formatted_prompt", "speech_scene1", "speech_scene2", "speech_scene3", "speech_scene4", "speech_scene5", "speech_scene6")
    FUNCTION = "generate_prompt"
    CATEGORY = "Boyo/Storyboard"
    
    def generate_prompt(self, movie_director_output, image_trigger_word, video_trigger_word, system_prompt_type, traveling_prompt_count, additional_details=""):
        """
        Generate a formatted prompt for ollama based on Movie Director output.
        For OVI mode, extracts speech and returns separately.
        """
        
        # Initialize speech outputs (empty for non-OVI modes)
        speech_outputs = [""] * 6
        
        # Parse the Movie Director output
        if system_prompt_type == "System Prompt 3 (OVI Audio)":
            parsed_data, speech_outputs = self._parse_ovi_movie_director_output(movie_director_output)
        else:
            parsed_data = self._parse_movie_director_output(movie_director_output)
        
        # System Prompt selection
        if system_prompt_type == "System Prompt 1 (6 Scenes)":
            system_prompt = self._get_system_prompt_1(image_trigger_word, video_trigger_word, parsed_data)
        elif system_prompt_type == "System Prompt 2 (Traveling Prompts)":
            system_prompt = self._get_system_prompt_2(image_trigger_word, video_trigger_word, traveling_prompt_count, parsed_data)
        elif system_prompt_type == "System Prompt 3 (OVI Audio)":
            system_prompt = self._get_system_prompt_3_ovi(image_trigger_word, video_trigger_word, parsed_data)
        else:
            system_prompt = self._get_system_prompt_1(image_trigger_word, video_trigger_word, parsed_data)  # Default fallback
        
        # Add additional details if provided
        if additional_details.strip():
            system_prompt += f"\n\nAdditional Instructions: {additional_details}"
        
        return tuple([system_prompt] + speech_outputs)
    
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
    
    def _parse_ovi_movie_director_output(self, output):
        """
        Parse OVI Movie Director output, extracting speech and cleaning scenes.
        Returns parsed data with cleaned scenes AND extracted speech.
        """
        parsed = {
            "story_concept": "",
            "main_character": "",
            "style_setting": "",
            "scenes": []
        }
        
        speech_outputs = [""] * 6
        
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
        
        # Extract speech from scenes and clean them
        cleaned_scenes = []
        for i, scene in enumerate(parsed["scenes"]):
            # Extract speech using regex
            speech_match = re.search(r'<SPEECH>(.*?)</SPEECH>', scene)
            if speech_match and i < 6:
                speech_outputs[i] = speech_match.group(1)
            
            # Remove speech tags from scene
            cleaned_scene = re.sub(r'<SPEECH>.*?</SPEECH>', '', scene).strip()
            # Clean up any double spaces
            cleaned_scene = re.sub(r'\s+', ' ', cleaned_scene)
            cleaned_scenes.append(cleaned_scene)
        
        parsed["scenes"] = cleaned_scenes
        return parsed, speech_outputs
    
    def _get_system_prompt_1(self, image_trigger_word, video_trigger_word, parsed_data):
        """
        System Prompt 1: Direct instructions for generating 6 image and 6 video prompts.
        """
        return f"""Convert the Movie Director scenes into image and video prompts. Output valid JSON only.

MOVIE DIRECTOR INPUT:
Story: {parsed_data.get('story_concept', 'Not provided')}
Character: {parsed_data.get('main_character', 'Not provided')}
Style: {parsed_data.get('style_setting', 'Not provided')}

Scenes:
{chr(10).join(parsed_data.get('scenes', []))}

TASK:
Create exactly 6 image prompts and 6 video prompts from these scenes.

IMAGE PROMPTS: 
Start each with "{image_trigger_word}" then describe the static visual moment. Include character appearance, location details, lighting, and composition. Focus on the keyframe moment from each scene.

VIDEO PROMPTS:
Start each with "{video_trigger_word}" then describe character movement and actions within that location. Show what the character does during 5 seconds in that scene. Include gestures, expressions, and natural movement.

OUTPUT FORMAT:
Return JSON with "imagePrompts" and "videoPrompts" sections. Each section has scene1 through scene6. Use proper JSON formatting."""

    def _get_system_prompt_2(self, image_trigger_word, video_trigger_word, traveling_prompt_count, parsed_data):
        """
        System Prompt 2: Generates 6 image prompts and 6 video prompts with multiple traveling sub-prompts using Movie Director structure.
        """
        return f"""Convert the Movie Director scenes into image and traveling video prompts. Output valid JSON only.

MOVIE DIRECTOR INPUT:
Story: {parsed_data.get('story_concept', 'Not provided')}
Character: {parsed_data.get('main_character', 'Not provided')}
Style: {parsed_data.get('style_setting', 'Not provided')}

Scenes:
{chr(10).join(parsed_data.get('scenes', []))}

TASK:
Create exactly 6 image prompts and 6 traveling video prompts from these scenes.

IMAGE PROMPTS:
Start each with "{image_trigger_word}" then describe the static visual moment. Include character appearance, location details, lighting, and composition.

VIDEO PROMPTS:
Each video prompt contains {traveling_prompt_count} sub-prompts separated by newlines. Start each sub-prompt with "{video_trigger_word}". Show progression of actions within the same location over multiple shots.

OUTPUT FORMAT:
Return JSON with "imagePrompts" and "videoPrompts" sections. Each section has scene1 through scene6. For video prompts, separate the {traveling_prompt_count} sub-prompts with \\n characters."""

    def _get_system_prompt_3_ovi(self, image_trigger_word, video_trigger_word, parsed_data):
        """
        System Prompt 3: Generates 6 image prompts and 6 video prompts for OVI processing (speech handled separately).
        """
        return f"""Convert the Movie Director scenes into image and video prompts for OVI processing. Output valid JSON only.

MOVIE DIRECTOR INPUT:
Story: {parsed_data.get('story_concept', 'Not provided')}
Character: {parsed_data.get('main_character', 'Not provided')}
Style: {parsed_data.get('style_setting', 'Not provided')}

Scenes:
{chr(10).join(parsed_data.get('scenes', []))}

TASK:
Create exactly 6 image prompts and 6 video prompts from these scenes. Speech is handled separately.

IMAGE PROMPTS:
Start each with "{image_trigger_word}" then describe the static visual moment. Include character appearance, location details, lighting, and composition.

VIDEO PROMPTS:
Start each with "{video_trigger_word}" then describe character physical actions and movements within that location. Show gestures, expressions, and body language suitable for 10-second OVI scenes. Focus on movement that matches speaking.

OUTPUT FORMAT:
Return JSON with "imagePrompts" and "videoPrompts" sections. Each section has scene1 through scene6."""


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "BoyoStoryboardPrompt": BoyoStoryboardPrompt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoStoryboardPrompt": "Boyo Storyboard Prompt"
}
