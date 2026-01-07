import comfy.utils
from typing import Any

# Import the cache system from the loop implementation
try:
    from .boyo_for_loops_exact import boyo_cache, boyo_update_cache, boyo_log_node_info
except ImportError:
    # Fallback for standalone testing
    class MockCache:
        def __init__(self):
            self._data = {}
        def get(self, key, default=None):
            return self._data.get(key, default)
        def __setitem__(self, key, value):
            self._data[key] = value
    
    boyo_cache = MockCache()
    def boyo_update_cache(k, tag, v):
        boyo_cache[k] = (tag, v)
    def boyo_log_node_info(node_name, message):
        print(f"[{node_name}] {message}")

class BoyoLatentSwitch:
    """
    A node that switches between two latent inputs based on a counter value.
    When counter = 0, uses start_latent (for loop initialization)
    When counter > 0, uses cached latent from previous iteration (for loop continuation)
    Integrates with Boyo loop cache system to avoid graph cycles.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "counter": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "start_latent": ("LATENT",),
                "cache_key": ("STRING", {"default": "latent_loop_cache"}),
            },
            "optional": {
                "next_latent": ("LATENT",),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "switch_latent"
    CATEGORY = "Boyo/Latent"
    
    def switch_latent(self, counter, start_latent, cache_key="latent_loop_cache", next_latent=None):
        """
        Switch between latent inputs based on counter value
        counter = 0: return start_latent and cache it
        counter > 0: return cached latent from previous iteration
        
        If next_latent is provided, update the cache for the next iteration
        """
        
        if counter == 0:
            # First iteration: use start latent and cache it
            boyo_update_cache(cache_key, "latent", start_latent)
            boyo_log_node_info("BoyoLatentSwitch", f"First iteration: Using start latent, cached to '{cache_key}'")
            return (start_latent,)
        else:
            # Subsequent iterations: get from cache
            cached_data = boyo_cache.get(cache_key, (None, None))
            if cached_data[1] is None:
                # Fallback to start_latent if cache miss
                boyo_log_node_info("BoyoLatentSwitch", f"Cache miss for '{cache_key}', falling back to start latent")
                result_latent = start_latent
            else:
                result_latent = cached_data[1]
                boyo_log_node_info("BoyoLatentSwitch", f"Iteration {counter}: Retrieved latent from cache '{cache_key}'")
            
            # If next_latent is provided, update cache for next iteration
            if next_latent is not None:
                boyo_update_cache(cache_key, "latent", next_latent)
                boyo_log_node_info("BoyoLatentSwitch", f"Updated cache '{cache_key}' with next latent for iteration {counter + 1}")
            
            return (result_latent,)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "BoyoLatentSwitch": BoyoLatentSwitch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoLatentSwitch": "Boyo Latent Switch"
}
