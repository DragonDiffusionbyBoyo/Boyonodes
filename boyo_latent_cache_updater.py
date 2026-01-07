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

class BoyoLatentCacheUpdater:
    """
    Updates the cache with latent data without creating graph cycles.
    Takes latent input, caches it, and passes it through unchanged.
    Perfect for capturing sampler outputs for next loop iteration.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "cache_key": ("STRING", {"default": "latent_loop_cache"}),
            },
            "optional": {
                "counter": ("INT", {"default": 0, "min": 0, "max": 999999}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "update_cache"
    CATEGORY = "Boyo/Latent"
    
    def update_cache(self, latent, cache_key="latent_loop_cache", counter=0):
        """
        Update the cache with the provided latent and pass it through unchanged.
        This allows capturing sampler outputs without creating graph cycles.
        """
        
        # Update the cache with the new latent
        boyo_update_cache(cache_key, "latent", latent)
        boyo_log_node_info("BoyoLatentCacheUpdater", f"Updated cache '{cache_key}' with latent for next iteration (counter: {counter})")
        
        # Pass the latent through unchanged
        return (latent,)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "BoyoLatentCacheUpdater": BoyoLatentCacheUpdater
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoLatentCacheUpdater": "Boyo Latent Cache Updater"
}
