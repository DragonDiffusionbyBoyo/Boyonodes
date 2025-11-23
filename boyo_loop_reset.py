"""
Custom Loop Reset Node for Boyo's Video Workflow
Standalone implementation without EasyUse dependencies
Resets loop counters to zero when triggered by completion signals
"""

import time

# Use ComfyUI's standard any type instead of custom implementation
any_type = "*"

# Simple cache implementation
_cache = {}

def cache_get(key, default=None):
    return _cache.get(key, default)

def cache_update(key, value):
    _cache[key] = value
    
def cache_remove(key):
    if key in _cache:
        del _cache[key]

class BoyoLoopReset:
    """
    Master reset node that can force all connected loop nodes back to zero.
    Triggered by completion signals from workflow sections.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": (any_type, {}),  # Completion signal from workflow
                "reset_mode": (["immediate", "delayed"], {"default": "immediate"}),
            },
            "optional": {
                "loop_id_1": ("STRING", {"default": ""}),
                "loop_id_2": ("STRING", {"default": ""}),
                "loop_id_3": ("STRING", {"default": ""}),
                "loop_id_4": ("STRING", {"default": ""}),
                "loop_id_5": ("STRING", {"default": ""}),
                "delay_seconds": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 10.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = (any_type, "INT", "STRING")
    RETURN_NAMES = ("trigger_out", "reset_count", "status")
    FUNCTION = "reset_loops"
    CATEGORY = "Boyo/Logic"
    
    def reset_loops(self, trigger, reset_mode, loop_id_1="", loop_id_2="", loop_id_3="", 
                   loop_id_4="", loop_id_5="", delay_seconds=0.1):
        """
        Reset specified loop counters to zero
        """
        
        # Collect all non-empty loop IDs
        loop_ids = [lid for lid in [loop_id_1, loop_id_2, loop_id_3, loop_id_4, loop_id_5] if lid.strip()]
        
        if not loop_ids:
            print("BoyoLoopReset: No loop IDs provided - nothing to reset")
            return (trigger, 0, "No loops specified")
        
        # Add delay if requested
        if reset_mode == "delayed" and delay_seconds > 0:
            time.sleep(delay_seconds)
        
        reset_count = 0
        failed_resets = []
        
        # Reset each specified loop
        for loop_id in loop_ids:
            try:
                # Check if cache entry exists
                cached_data = cache_get(loop_id, None)
                if cached_data is not None:
                    # Reset the counter to 0
                    # Based on forLoopEnd implementation, we need to reset the iteration count
                    if isinstance(cached_data, dict) and 'current_iteration' in cached_data:
                        cached_data['current_iteration'] = 0
                        cache_update(loop_id, cached_data)
                    else:
                        # Simple counter reset
                        cache_update(loop_id, 0)
                    
                    reset_count += 1
                    print(f"BoyoLoopReset: Reset loop '{loop_id}' to zero")
                else:
                    # Initialize cache entry if it doesn't exist
                    cache_update(loop_id, 0)
                    reset_count += 1
                    print(f"BoyoLoopReset: Initialized loop '{loop_id}' to zero")
                    
            except Exception as e:
                failed_resets.append(loop_id)
                print(f"BoyoLoopReset: Failed to reset loop '{loop_id}': {str(e)}")
        
        # Generate status message
        if failed_resets:
            status = f"Reset {reset_count} loops, failed: {', '.join(failed_resets)}"
        else:
            status = f"Successfully reset {reset_count} loops"
        
        log_node_info("BoyoLoopReset", status)
        
        return (trigger, reset_count, status)


class BoyoLoopCounter:
    """
    Simple counter that can be reset by the BoyoLoopReset node
    Alternative to complex loop nodes when you just need a basic counter
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": (any_type, {}),
                "counter_id": ("STRING", {"default": "counter_1"}),
                "increment": ("INT", {"default": 1, "min": 0, "max": 100}),
                "max_count": ("INT", {"default": 6, "min": 1, "max": 1000}),
            },
            "optional": {
                "reset_to_zero": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("INT", "BOOLEAN", any_type)
    RETURN_NAMES = ("count", "is_complete", "trigger_out")
    FUNCTION = "count"
    CATEGORY = "Boyo/Logic"
    
    def count(self, trigger, counter_id, increment, max_count, reset_to_zero=False):
        """
        Increment counter and check if complete
        """
        
        # Handle manual reset
        if reset_to_zero:
            cache_update(counter_id, 0)
            print(f"BoyoLoopCounter: Manually reset counter '{counter_id}' to zero")
            return (0, False, trigger)
        
        # Get current count from cache
        current_count = cache_get(counter_id, 0)
        
        # Increment counter
        new_count = current_count + increment
        
        # Update cache
        cache_update(counter_id, new_count)
        
        # Check if complete
        is_complete = new_count >= max_count
        
        print(f"BoyoLoopCounter: Counter '{counter_id}': {new_count}/{max_count}")
        
        return (new_count, is_complete, trigger)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "BoyoLoopReset": BoyoLoopReset,
    "BoyoLoopCounter": BoyoLoopCounter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoLoopReset": "Boyo Loop Reset",
    "BoyoLoopCounter": "Boyo Loop Counter",
}
