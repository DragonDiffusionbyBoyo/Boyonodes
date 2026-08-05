"""
BoyoMultilineText / BoyoGetLineByNumber

Replacement for the multiline text widget ComfyUI ripped out, plus a
companion node to pull a single line back out by (Python, 0-based) index.

Blank lines are treated as junk formatting, not content: BoyoGetLineByNumber
strips whitespace from every line and drops any line that's empty after
stripping BEFORE indexing, so the index always lines up with the Nth
non-blank line, not the Nth line as typed.

Out-of-range indices never throw. Negative clamps to 0, too-high clamps to
the last available line, and an all-blank/empty source returns "" rather
than blowing up the graph.
"""


class BoyoMultilineText:
    """Dumb, honest multiline text source. No filtering happens here."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "get_text"
    CATEGORY = "BoyoNodes/text"

    def get_text(self, text):
        return (text,)


class BoyoGetLineByNumber:
    """
    Grabs a single line out of a text block by index, after stripping
    whitespace and dropping blank lines. Index is Python-style (0-based).

    line_index has both a widget default and an optional input socket;
    the socket, if connected, overrides the widget value - useful for
    driving line selection from a counter/loop node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "line_index": ("INT", {"default": 0, "min": 0, "max": 999999}),
            },
            "optional": {
                "line_index_override": ("INT", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("line",)
    FUNCTION = "get_line"
    CATEGORY = "BoyoNodes/text"

    def get_line(self, text, line_index, line_index_override=None):
        index = line_index_override if line_index_override is not None else line_index

        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]

        if not lines:
            return ("",)

        if index < 0:
            index = 0
        elif index >= len(lines):
            index = len(lines) - 1

        return (lines[index],)


NODE_CLASS_MAPPINGS = {
    "BoyoMultilineText": BoyoMultilineText,
    "BoyoGetLineByNumber": BoyoGetLineByNumber,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoMultilineText": "Boyo Multiline Text",
    "BoyoGetLineByNumber": "Boyo Get Line By Number",
}
