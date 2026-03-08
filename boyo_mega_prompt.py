import os
import random


def _scan_folder(subfolder_name):
    """Scan a category subfolder and return ['<none>'] + sorted list of .txt files."""
    folder = os.path.join(os.path.dirname(__file__), 'mega_prompts', subfolder_name)
    if not os.path.isdir(folder):
        return ['<none>']
    files = sorted([f for f in os.listdir(folder) if f.endswith('.txt')])
    return ['<none>'] + files


def _pick_line(subfolder_name, filename, seed, mode):
    """Read a file and return a single line based on seed and mode."""
    if filename == '<none>':
        return None
    folder = os.path.join(os.path.dirname(__file__), 'mega_prompts', subfolder_name)
    file_path = os.path.join(folder, filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
    except (FileNotFoundError, IOError):
        return None
    if not lines:
        return None
    if mode == 'random':
        random.seed(seed)
        return random.choice(lines)
    else:
        return lines[seed % len(lines)]


class BoyoMegaPrompt:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode":           (["sequential", "random"],),
                "scene_file":     (_scan_folder('scene'),),
                "scene_seed":     ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "subject_file":   (_scan_folder('subject'),),
                "subject_seed":   ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "action_file":    (_scan_folder('action'),),
                "action_seed":    ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "clothing_file":  (_scan_folder('clothing'),),
                "clothing_seed":  ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "mood_file":      (_scan_folder('mood'),),
                "mood_seed":      ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            },
            "optional": {
                "prepend": ("STRING", {"default": "", "multiline": False}),
                "append":  ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate"
    CATEGORY = "DJZ-Nodes"

    def generate(
        self,
        mode,
        scene_file,    scene_seed,
        subject_file,  subject_seed,
        action_file,   action_seed,
        clothing_file, clothing_seed,
        mood_file,     mood_seed,
        prepend="",
        append=""
    ):
        slots = [
            ('scene',    scene_file,    scene_seed),
            ('subject',  subject_file,  subject_seed),
            ('action',   action_file,   action_seed),
            ('clothing', clothing_file, clothing_seed),
            ('mood',     mood_file,     mood_seed),
        ]

        parts = []

        if prepend.strip():
            parts.append(prepend.strip())

        for subfolder, filename, seed in slots:
            result = _pick_line(subfolder, filename, seed, mode)
            if result:
                parts.append(result)

        if append.strip():
            parts.append(append.strip())

        return (" , ".join(parts),)

    @classmethod
    def IS_CHANGED(
        cls,
        mode,
        scene_file,    scene_seed,
        subject_file,  subject_seed,
        action_file,   action_seed,
        clothing_file, clothing_seed,
        mood_file,     mood_seed,
        prepend="",
        append=""
    ):
        return (
            mode,
            scene_file,    scene_seed,
            subject_file,  subject_seed,
            action_file,   action_seed,
            clothing_file, clothing_seed,
            mood_file,     mood_seed,
            prepend,
            append,
        )


NODE_CLASS_MAPPINGS = {
    "BoyoMegaPrompt": BoyoMegaPrompt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoMegaPrompt": "Boyo Mega Prompt"
}
