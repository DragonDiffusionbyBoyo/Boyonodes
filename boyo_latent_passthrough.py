"""
Boyo Latent Passthrough Node - Forces clean execution boundary between samplers
"""

import torch
import copy

class BoyoLatentPassthrough:
    """
    Simple latent passthrough that forces a clean execution boundary.
    Specifically designed to break state contamination between WAN samplers in loops.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            },
            "optional": {
                "force_copy": ("BOOLEAN", {"default": True}),
                "clear_cache": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent_out",)
    FUNCTION = "passthrough_latent"
    CATEGORY = "Boyo/Utils"
    
    def passthrough_latent(self, latent, force_copy=True, clear_cache=False):
        """
        Pass latent through with optional deep copy to break state references
        """
        
        # Optional cache clearing for extra paranoia
        if clear_cache:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Force a proper copy to break any lingering state references
        if force_copy:
            # Deep copy the latent to ensure no state contamination
            latent_out = {}
            for key, value in latent.items():
                if isinstance(value, torch.Tensor):
                    # Clone tensor to new memory location
                    latent_out[key] = value.clone().detach()
                else:
                    # Deep copy other values
                    latent_out[key] = copy.deepcopy(value)
        else:
            # Simple passthrough
            latent_out = latent
        
        # Force garbage collection to clean up any orphaned state
        import gc
        gc.collect()
        
        return (latent_out,)


class BoyoExecutionBarrier:
    """
    More aggressive execution barrier that completely isolates execution contexts
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_data": ("*",),
            },
            "optional": {
                "barrier_strength": (["light", "medium", "heavy"], {"default": "medium"}),
                "debug_output": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output_data",)
    FUNCTION = "execution_barrier"
    CATEGORY = "Boyo/Utils"
    
    def execution_barrier(self, input_data, barrier_strength="medium", debug_output=False):
        """
        Create execution barrier with varying levels of isolation
        """
        
        if debug_output:
            print(f"[BoyoExecutionBarrier] Creating {barrier_strength} barrier")
        
        if barrier_strength == "light":
            # Just pass through with minimal processing
            return (input_data,)
            
        elif barrier_strength == "medium":
            # Force a copy if possible
            if isinstance(input_data, dict) and "samples" in input_data:
                # Latent-like data
                output = {}
                for key, value in input_data.items():
                    if isinstance(value, torch.Tensor):
                        output[key] = value.clone().detach()
                    else:
                        output[key] = copy.deepcopy(value)
                return (output,)
            else:
                return (copy.deepcopy(input_data),)
                
        elif barrier_strength == "heavy":
            # Nuclear option - force garbage collection and memory cleanup
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Deep copy everything
            if isinstance(input_data, dict) and "samples" in input_data:
                output = {}
                for key, value in input_data.items():
                    if isinstance(value, torch.Tensor):
                        # Clone, detach, and move to ensure clean memory
                        tensor_copy = value.clone().detach()
                        if torch.cuda.is_available() and value.is_cuda:
                            tensor_copy = tensor_copy.cuda()
                        output[key] = tensor_copy
                    else:
                        output[key] = copy.deepcopy(value)
                
                # Force another GC after copy
                gc.collect()
                return (output,)
            else:
                result = copy.deepcopy(input_data)
                gc.collect()
                return (result,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "BoyoLatentPassthrough": BoyoLatentPassthrough,
    "BoyoExecutionBarrier": BoyoExecutionBarrier,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoLatentPassthrough": "Boyo Latent Passthrough",
    "BoyoExecutionBarrier": "Boyo Execution Barrier",
}
