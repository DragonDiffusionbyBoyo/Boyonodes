class BoyoOverlapSwitch:
    """
    Switches overlap value based on counter for smooth video transitions.
    Loop 0: overlap = 0 (first generation has nothing to overlap with)
    Loop 1+: overlap = optimal_value (blend previous with current for smooth transitions)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "counter": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "first_overlap": ("INT", {"default": 0, "min": 0, "max": 100}),
                "subsequent_overlap": ("INT", {"default": 13, "min": 0, "max": 100}),
            }
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("overlap",)
    FUNCTION = "switch_overlap"
    CATEGORY = "Boyo/Video"
    
    def switch_overlap(self, counter, first_overlap=0, subsequent_overlap=13):
        """
        Switch overlap value based on counter:
        counter = 0: return first_overlap (usually 0)
        counter > 0: return subsequent_overlap (your optimal blend value)
        """
        if counter == 0:
            overlap_value = first_overlap
            print(f"[Boyo] BoyoOverlapSwitch: First iteration (counter {counter}) → overlap = {overlap_value}")
        else:
            overlap_value = subsequent_overlap
            print(f"[Boyo] BoyoOverlapSwitch: Subsequent iteration (counter {counter}) → overlap = {overlap_value}")
        
        return (overlap_value,)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "BoyoOverlapSwitch": BoyoOverlapSwitch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoOverlapSwitch": "Boyo Overlap Switch"
}
