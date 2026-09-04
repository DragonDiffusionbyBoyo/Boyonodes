"""BoyoH3TriggerInjector — injects per-scene trigger words (or any prefix text)
into a multi-scene H3 prompt string separated by ---.

Scene 1 prefix -> top of first block
Scene 2 prefix -> top of second block
Scene 3 prefix -> top of third block

Blank boxes are ignored — no content is inserted for that scene.
Safe to wire in permanently and leave all boxes empty when not needed.
"""

from __future__ import annotations

SEPARATOR = "\n---\n"


class BoyoH3TriggerInjector:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Multi-scene H3 prompt string separated by ---"
                }),
            },
            "optional": {
                "scene_1_triggers": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Trigger words or prefix text for Scene 1. Leave blank to skip."
                }),
                "scene_2_triggers": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Trigger words or prefix text for Scene 2. Leave blank to skip."
                }),
                "scene_3_triggers": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Trigger words or prefix text for Scene 3. Leave blank to skip."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "inject"
    CATEGORY = "H3/Promptor"

    def inject(self, prompt: str, scene_1_triggers: str = "",
               scene_2_triggers: str = "", scene_3_triggers: str = "") -> tuple[str]:

        triggers = [
            (scene_1_triggers or "").strip(),
            (scene_2_triggers or "").strip(),
            (scene_3_triggers or "").strip(),
        ]

        # Split on --- handling variations in surrounding whitespace
        blocks = [b.strip() for b in prompt.replace("\r\n", "\n").split("---") if b.strip()]

        if not blocks:
            return (prompt,)

        result = []
        for i, block in enumerate(blocks):
            if i < len(triggers) and triggers[i]:
                result.append(triggers[i] + "\n" + block)
            else:
                result.append(block)

        return (SEPARATOR.join(result),)


NODE_CLASS_MAPPINGS = {
    "BoyoH3TriggerInjector": BoyoH3TriggerInjector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoH3TriggerInjector": "Boyo H3 Trigger Injector",
}
