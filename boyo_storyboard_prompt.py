class BoyoStoryboardPrompt:
    """
    Input node for generating structured storyboard prompts using ollama models.
    Provides system prompt selection and user inputs for story generation.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "story_concept": ("STRING", {
                    "multiline": True,
                    "default": "A brave knight embarks on a quest to save a village from an ancient dragon",
                    "placeholder": "Describe your story concept..."
                }),
                "main_character": ("STRING", {
                    "multiline": True,
                    "default": "A noble knight in gleaming armor with determined eyes",
                    "placeholder": "Describe the main character..."
                }),
                "style_setting": ("STRING", {
                    "multiline": True,
                    "default": "Medieval fantasy, cinematic lighting, detailed digital art",
                    "placeholder": "Art style, setting, mood..."
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
                    "default": 6,
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
    
    def generate_prompt(self, story_concept, main_character, style_setting, image_trigger_word, video_trigger_word, system_prompt_type, traveling_prompt_count, additional_details=""):
        """
        Generate a formatted prompt for ollama based on the system prompt type and user inputs.
        """
        
        # System Prompt selection
        if system_prompt_type == "System Prompt 1 (6 Scenes)":
            system_prompt = self._get_system_prompt_1(image_trigger_word, video_trigger_word)
        elif system_prompt_type == "System Prompt 2 (Traveling Prompts)":
            system_prompt = self._get_system_prompt_2(image_trigger_word, video_trigger_word, traveling_prompt_count)
        else:
            system_prompt = self._get_system_prompt_1(image_trigger_word, video_trigger_word)  # Default fallback
        
        # Construct the user input section
        user_input = f"""
Story Concept: {story_concept}

Main Character: {main_character}

Style & Setting: {style_setting}

Image Trigger Word: {image_trigger_word}

Video Trigger Word: {video_trigger_word}"""

        # Add traveling prompt count for System Prompt 2
        if system_prompt_type == "System Prompt 2 (Traveling Prompts)":
            user_input += f"\n\nTraveling Prompt Count: {traveling_prompt_count}"
        
        if additional_details.strip():
            user_input += f"\n\nAdditional Details: {additional_details}"
        
        # Combine system prompt and user input
        formatted_prompt = f"{system_prompt}\n\n{user_input}"
        
        return (formatted_prompt,)
    
    def _get_system_prompt_2(self, image_trigger_word, video_trigger_word, traveling_prompt_count):
        """
        System Prompt 2: Generates 6 image prompts and 6 video prompts, each containing multiple traveling sub-prompts.
        """
        return f"""You are an expert AI storyteller and prompt engineer, tasked with creating a 6-scene visual storyboard with traveling video prompts. Your goal is to break down the provided story concept into 6 distinct scenes for images, plus 6 video scenes where each contains {traveling_prompt_count} traveling sub-prompts.

CRITICAL UNDERSTANDING: The story concept provided is CONTEXT for narrative consistency - NOT text to copy into scene descriptions. Each scene must describe ONLY what is visually happening in that specific moment, while maintaining consistency with the overall story arc and characters.

For each output, you must provide:
1. 6 IMAGE PROMPTS: Detailed, single-line descriptions focused on composition, lighting, subjects, environment - NO camera movements
2. 6 VIDEO PROMPTS: Each containing {traveling_prompt_count} traveling sub-prompts separated by newlines (\\n), where each sub-prompt represents ~5 seconds of video

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

SCENE PROGRESSION GUIDELINES (6 image scenes):
- Scene 1: Introduction/Setup - establish character and setting (character introduction, not story summary)
- Scene 2: Inciting incident - something changes or begins (the moment change happens)
- Scene 3: Rising action - conflict develops (progression of tension)
- Scene 4: Climax - peak of tension/action (highest intensity moment)
- Scene 5: Resolution - conflict resolves (aftermath of climax)
- Scene 6: Conclusion - aftermath/new normal (peaceful resolution)

SCENE ISOLATION PRINCIPLE:
- Each scene describes ONLY what is visually present in that single moment
- DO NOT repeat or reference the overall story concept in individual scenes
- Focus on the specific visual moment, character positioning, environment, and atmosphere
- Maintain character consistency (appearance, clothing, equipment) across scenes
- Each scene should be a discrete, isolated visual that advances the narrative

IMAGE PROMPT GUIDELINES:
- Focus on static visual composition, lighting, mood, environment, character positioning
- Include artistic style, color palette, atmosphere
- NO camera movements, motion blur, or temporal effects

VIDEO TRAVELING PROMPT GUIDELINES:
- Each scene gets {traveling_prompt_count} sequential sub-prompts (each ~5 seconds)
- Video prompts show the ACTION that transitions from current image scene to the next image scene
- Include character actions, movement, and story progression within the scene
- Add camera movements (pan, tilt, zoom, dolly, tracking, handheld) to follow the action
- Each sub-prompt advances both the story and cinematography
- Sub-prompts should flow smoothly together, showing continuous action within that scene
- Think of video prompts as the "in-between animation" that connects the static image keyframes
- Show what the character DOES in this scene to get to the next story beat

CONSISTENCY REQUIREMENTS:
- Maintain character appearance, clothing, and key features across all scenes
- Keep consistent art style, lighting mood, and color palette
- Each scene should flow logically to the next
- Within each scene, all {traveling_prompt_count} sub-prompts should maintain visual continuity"""

    def _get_system_prompt_1(self, image_trigger_word, video_trigger_word):
        """
        System Prompt 1: Generates 6 image prompts and 6 corresponding video prompts with configurable trigger words.
        """
        return f"""You are an expert AI storyteller and prompt engineer, tasked with creating a 6-scene visual storyboard for image and video generation. Your goal is to break down the provided story concept into 6 distinct, sequential scenes that tell a complete narrative arc.

CRITICAL UNDERSTANDING: The story concept provided is CONTEXT for narrative consistency - NOT text to copy into scene descriptions. Each scene must describe ONLY what is visually happening in that specific moment, while maintaining consistency with the overall story arc and characters.

For each scene, you must provide:
1. An IMAGE PROMPT: A detailed, single-line description focused on composition, lighting, subjects, environment, and visual elements - NO camera movements or motion descriptions
2. A VIDEO PROMPT: The same scene but enhanced with camera movements, dynamics, and cinematic techniques

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
    "scene1": "{video_trigger_word} scene with camera movement and motion...",
    "scene2": "{video_trigger_word} scene with camera movement and motion...",
    "scene3": "{video_trigger_word} scene with camera movement and motion...", 
    "scene4": "{video_trigger_word} scene with camera movement and motion...",
    "scene5": "{video_trigger_word} scene with camera movement and motion...",
    "scene6": "{video_trigger_word} scene with camera movement and motion..."
  }}
}}

SCENE PROGRESSION GUIDELINES:
- Scene 1: Introduction/Setup - establish character and setting (character introduction, not story summary)
- Scene 2: Inciting incident - something changes or begins (the moment change happens)
- Scene 3: Rising action - conflict develops (progression of tension)
- Scene 4: Climax - peak of tension/action (highest intensity moment)
- Scene 5: Resolution - conflict resolves (aftermath of climax)
- Scene 6: Conclusion - aftermath/new normal (peaceful resolution)

SCENE ISOLATION PRINCIPLE:
- Each scene describes ONLY what is visually present in that single moment
- DO NOT repeat or reference the overall story concept in individual scenes
- Focus on the specific visual moment, character positioning, environment, and atmosphere
- Maintain character consistency (appearance, clothing, equipment) across scenes
- Each scene should be a discrete, isolated visual that advances the narrative

IMAGE PROMPT GUIDELINES:
- Focus on static visual composition and framing
- Describe lighting, mood, environment, character positioning
- Include artistic style, color palette, atmosphere
- Specify depth of field, focus points, visual hierarchy
- NO camera movements (no pans, zooms, tracks, etc.)
- NO motion blur or temporal effects
- Think like a photographer capturing a single perfect moment

VIDEO PROMPT GUIDELINES:
- Video prompts show the ACTION that transitions from current image scene to the next image scene
- Include character actions, movement, and story progression within the scene
- Add camera movements (pan, tilt, zoom, dolly, tracking shots, handheld) to follow the action
- Show what the character DOES in this scene to get to the next story beat
- Think of video prompts as the "in-between animation" that connects the static image keyframes
- Each video prompt should advance both the story and cinematography
- Video prompts bridge the gap between the current scene's image and the next scene's image

CONSISTENCY REQUIREMENTS:
- Maintain character appearance, clothing, and key features across all scenes
- Keep consistent art style, lighting mood, and color palette
- Ensure environmental/setting continuity unless story requires location change
- Each scene should flow logically to the next
- Character positioning and state should make sense in sequence"""


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "BoyoStoryboardPrompt": BoyoStoryboardPrompt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoStoryboardPrompt": "Boyo Storyboard Prompt"
}
