class BoyoFrameCounter:
    """
    Calculates frames_processed for video loops based on counter and chunk size.
    Essential for audio/video sync in infinite generation loops.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "counter": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "chunk_size": ("INT", {"default": 89, "min": 1, "max": 1000}),
            },
            "optional": {
                "offset": ("INT", {"default": 0, "min": 0, "max": 1000}),
            }
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("frames_processed",)
    FUNCTION = "calculate_frames"
    CATEGORY = "Boyo/Video"
    
    def calculate_frames(self, counter, chunk_size, offset=0):
        """
        Calculate frames_processed = (counter * chunk_size) + offset
        
        counter: Current loop iteration (0, 1, 2, ...)
        chunk_size: Number of frames per loop iteration (e.g., 89)
        offset: Starting frame offset (usually 0)
        """
        frames_processed = (counter * chunk_size) + offset
        
        print(f"[Boyo] BoyoFrameCounter: Loop {counter}, chunk_size {chunk_size} → frames_processed = {frames_processed}")
        
        return (frames_processed,)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "BoyoFrameCounter": BoyoFrameCounter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoFrameCounter": "Boyo Frame Counter"
}
