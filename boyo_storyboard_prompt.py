import json
import re

class BoyoStoryboardPrompt:
    """
    Input node for generating structured storyboard prompts using ollama models.
    Now accepts Movie Director output format for temporal reasoning and story structure.
    ROBUST PARSER: Handles variations from GPT, Gemini, Claude, etc.
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
    
    RETURN_TYPES = ("STRING", "STRING", "STRING") + ("STRING",) * 6
    RETURN_NAMES = ("formatted_prompt", "image_trigger_word", "video_trigger_word", "speech_scene1", "speech_scene2", "speech_scene3", "speech_scene4", "speech_scene5", "speech_scene6")
    FUNCTION = "generate_prompt"
    CATEGORY = "Boyo/Storyboard"
    
    def generate_prompt(self, movie_director_output, image_trigger_word, video_trigger_word, system_prompt_type, traveling_prompt_count, additional_details=""):
        """
        Generate a formatted prompt for ollama based on Movie Director output.
        Uses robust parsing to handle different LLM output formats.
        """
        
        # Initialize speech outputs (empty for non-OVI modes)
        speech_outputs = [""] * 6
        
        # Parse the Movie Director output with robust parser
        if system_prompt_type == "System Prompt 3 (OVI Audio)":
            parsed_data, speech_outputs = self._parse_ovi_movie_director_output_robust(movie_director_output)
        else:
            parsed_data = self._parse_movie_director_output_robust(movie_director_output)
        
        # System Prompt selection
        if system_prompt_type == "System Prompt 1 (6 Scenes)":
            system_prompt = self._get_system_prompt_1(image_trigger_word, video_trigger_word, parsed_data)
        elif system_prompt_type == "System Prompt 2 (Traveling Prompts)":
            system_prompt = self._get_system_prompt_2(image_trigger_word, video_trigger_word, traveling_prompt_count, parsed_data)
        elif system_prompt_type == "System Prompt 3 (OVI Audio)":
            system_prompt = self._get_system_prompt_3_ovi(image_trigger_word, video_trigger_word, parsed_data)
        else:
            system_prompt = self._get_system_prompt_1(image_trigger_word, video_trigger_word, parsed_data)
        
        # Add additional details if provided
        if additional_details.strip():
            system_prompt += f"\n\nAdditional Instructions: {additional_details}"
        
        return tuple([system_prompt, image_trigger_word, video_trigger_word] + speech_outputs)
    
    def _parse_movie_director_output_robust(self, output):
        """
        ROBUST parser that handles variations from different LLMs.
        Uses flexible regex patterns to extract story elements.
        """
        parsed = {
            "story_concept": "",
            "main_character": "",
            "style_setting": "",
            "scenes": []
        }
        
        # Fallback: if parsing fails, use entire input as story concept
        fallback_text = output.strip()
        
        try:
            # Multiple patterns for story concept
            story_patterns = [
                r"STORY CONCEPT:\s*(.+?)(?=MAIN CHARACTER|CHARACTER|STYLE|SCENE|$)",
                r"Story Concept:\s*(.+?)(?=Main Character|Character|Style|Scene|$)",
                r"Story:\s*(.+?)(?=Character|Main Character|Style|Scene|$)",
                r"STORY:\s*(.+?)(?=CHARACTER|MAIN CHARACTER|STYLE|SCENE|$)",
                r"Concept:\s*(.+?)(?=Character|Main Character|Style|Scene|$)"
            ]
            
            # Multiple patterns for character
            character_patterns = [
                r"MAIN CHARACTER:\s*(.+?)(?=STYLE|SCENE|STORY|$)",
                r"Main Character:\s*(.+?)(?=Style|Scene|Story|$)", 
                r"CHARACTER:\s*(.+?)(?=STYLE|SCENE|STORY|$)",
                r"Character:\s*(.+?)(?=Style|Scene|Story|$)"
            ]
            
            # Multiple patterns for style/setting
            style_patterns = [
                r"STYLE & SETTING:\s*(.+?)(?=SCENE|STORY|CHARACTER|$)",
                r"Style & Setting:\s*(.+?)(?=Scene|Story|Character|$)",
                r"STYLE:\s*(.+?)(?=SCENE|STORY|CHARACTER|$)",
                r"Style:\s*(.+?)(?=Scene|Story|Character|$)",
                r"SETTING:\s*(.+?)(?=SCENE|STORY|CHARACTER|$)",
                r"Setting:\s*(.+?)(?=Scene|Story|Character|$)"
            ]
            
            # Try each pattern until one works
            for pattern in story_patterns:
                match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
                if match:
                    parsed["story_concept"] = match.group(1).strip()
                    break
                    
            for pattern in character_patterns:
                match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
                if match:
                    parsed["main_character"] = match.group(1).strip()
                    break
                    
            for pattern in style_patterns:
                match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
                if match:
                    parsed["style_setting"] = match.group(1).strip()
                    break
            
            # Extract scenes with specific pattern for timestamp format
            scene_patterns = [
                r"SCENE BREAKDOWN:(.+?)$",
                r"Scene Breakdown:(.+?)$", 
                r"SCENES:(.+?)$",
                r"Scenes:(.+?)$"
            ]
            
            scene_section = ""
            for pattern in scene_patterns:
                match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
                if match:
                    scene_section = match.group(1).strip()
                    break
            
            # If no scene section found, try to find individual scenes in the whole text
            if not scene_section:
                scene_section = output
            
            # UPDATED: Extract individual scenes with timestamp pattern
            scenes_found = []
            
            # Primary pattern for: **Scene 1 (0-5s): [composition]** - Description
            timestamp_pattern = r"\*\*Scene\s+\d+\s*\([^)]+\):[^*]+\*\*\s*-\s*(.+?)(?=\*\*Scene\s+\d+|\*\*TEMPORAL|$)"
            matches = re.findall(timestamp_pattern, scene_section, re.IGNORECASE | re.DOTALL)
            
            if matches:
                scenes_found = [match.strip() for match in matches]
            else:
                # Fallback: Try without markdown
                simple_timestamp_pattern = r"Scene\s+\d+\s*\([^)]+\):[^-]+-\s*(.+?)(?=Scene\s+\d+|$)"
                matches = re.findall(simple_timestamp_pattern, scene_section, re.IGNORECASE | re.DOTALL)
                
                if matches:
                    scenes_found = [match.strip() for match in matches]
                else:
                    # Final fallback: standard scene patterns
                    standard_patterns = [
                        r"Scene\s+\d+:?\s*(.+?)(?=Scene\s+\d+|$)",
                        r"SCENE\s+\d+:?\s*(.+?)(?=SCENE\s+\d+|$)",
                        r"\d+\.\s*(.+?)(?=\d+\.|$)"
                    ]
                    
                    for pattern in standard_patterns:
                        matches = re.findall(pattern, scene_section, re.IGNORECASE | re.DOTALL)
                        if matches:
                            scenes_found = [match.strip() for match in matches]
                            break
            
            # If still no scenes, try splitting on common delimiters
            if not scenes_found:
                # Try splitting on newlines and look for scene-like content
                lines = scene_section.split('\n')
                current_scene = ""
                for line in lines:
                    line = line.strip()
                    if re.match(r'(Scene|SCENE|\d+\.)', line, re.IGNORECASE):
                        if current_scene:
                            scenes_found.append(current_scene.strip())
                        current_scene = line
                    elif current_scene and line:
                        current_scene += " " + line
                        
                # Add the last scene
                if current_scene:
                    scenes_found.append(current_scene.strip())
            
            parsed["scenes"] = scenes_found
            
            # If we got nothing, use fallback strategy
            if not any([parsed["story_concept"], parsed["main_character"], parsed["style_setting"], parsed["scenes"]]):
                # Use entire input as story concept and try to extract scenes from it
                parsed["story_concept"] = fallback_text
                # Basic scene extraction as fallback
                lines = fallback_text.split('\n')
                scenes_fallback = [line.strip() for line in lines if line.strip() and len(line.strip()) > 20]
                parsed["scenes"] = scenes_fallback[:6]  # Take first 6 substantial lines
                
        except Exception as e:
            # Complete fallback: treat entire input as story concept
            parsed["story_concept"] = fallback_text
            parsed["scenes"] = [fallback_text]
        
        return parsed
    
    def _parse_ovi_movie_director_output_robust(self, output):
        """
        Robust OVI parser with speech extraction.
        """
        parsed_data = self._parse_movie_director_output_robust(output)
        speech_outputs = [""] * 6
        
        # Extract speech from scenes
        cleaned_scenes = []
        for i, scene in enumerate(parsed_data["scenes"]):
            # Extract speech using regex
            speech_match = re.search(r'<SPEECH>(.*?)</SPEECH>', scene, re.IGNORECASE | re.DOTALL)
            if speech_match and i < 6:
                speech_outputs[i] = speech_match.group(1).strip()
            
            # Remove speech tags from scene
            cleaned_scene = re.sub(r'<SPEECH>.*?</SPEECH>', '', scene, flags=re.IGNORECASE | re.DOTALL).strip()
            cleaned_scene = re.sub(r'\s+', ' ', cleaned_scene)
            cleaned_scenes.append(cleaned_scene)
        
        parsed_data["scenes"] = cleaned_scenes
        return parsed_data, speech_outputs
    
    # [Rest of the methods remain the same...]
    def _get_system_prompt_1(self, image_trigger_word, video_trigger_word, parsed_data):
        """System Prompt 1: Direct instructions for generating 6 image and 6 video prompts."""
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
        """System Prompt 2: Traveling prompts with explicit line break requirements."""
        return f"""Convert Movie Director scenes into JSON format. Do NOT copy the template - create actual content.

MOVIE DIRECTOR INPUT:
Story: {parsed_data.get('story_concept', 'Not provided')}
Character: {parsed_data.get('main_character', 'Not provided')}
Style: {parsed_data.get('style_setting', 'Not provided')}

SCENES TO PROCESS:
{chr(10).join(parsed_data.get('scenes', []))}

CRITICAL REQUIREMENTS:
1. IMAGE PROMPTS: Single line each, start with "{image_trigger_word}"
2. VIDEO PROMPTS: EXACTLY {traveling_prompt_count} traveling sub-prompts per scene
3. Each video sub-prompt MUST start with "{video_trigger_word}" 
4. Separate video sub-prompts with \\n - EACH SUB-PROMPT ON NEW LINE
5. Output valid JSON only

TRAVELING VIDEO FORMAT:
Each video prompt = {traveling_prompt_count} separate actions on new lines:
"scene1": "{video_trigger_word} first action\\n{video_trigger_word} second action\\n{video_trigger_word} third action"

JSON STRUCTURE REQUIRED:
{{
  "imagePrompts": {{
    "scene1": "{image_trigger_word} single line description",
    "scene2": "{image_trigger_word} single line description",
    "scene3": "{image_trigger_word} single line description",
    "scene4": "{image_trigger_word} single line description",
    "scene5": "{image_trigger_word} single line description",
    "scene6": "{image_trigger_word} single line description"
  }},
  "videoPrompts": {{
    "scene1": "{video_trigger_word} action one\\n{video_trigger_word} action two\\n{video_trigger_word} action three",
    "scene2": "{video_trigger_word} action one\\n{video_trigger_word} action two\\n{video_trigger_word} action three",
    "scene3": "{video_trigger_word} action one\\n{video_trigger_word} action two\\n{video_trigger_word} action three",
    "scene4": "{video_trigger_word} action one\\n{video_trigger_word} action two\\n{video_trigger_word} action three",
    "scene5": "{video_trigger_word} action one\\n{video_trigger_word} action two\\n{video_trigger_word} action three",
    "scene6": "{video_trigger_word} action one\\n{video_trigger_word} action two\\n{video_trigger_word} action three"
  }}
}}

MANDATORY: Each scene must have EXACTLY {traveling_prompt_count} video sub-prompts. Create content based on Movie Director scenes above."""

    def _get_system_prompt_3_ovi(self, image_trigger_word, video_trigger_word, parsed_data):
        """System Prompt 3: OVI processing (speech handled separately)."""
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
