import torch
import comfy.model_management
import comfy.utils
import node_helpers
import comfy.latent_formats

class BoyoPainterSVI:
    """
    Merged node combining PainterI2V motion amplitude enhancement with WanImageToVideoSVIPro 
    context preservation for infinite length video generation. Designed for use in samplers 2+ 
    after initial PainterI2V processing.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 4}),
                "anchor_samples": ("LATENT",),
                "motion_amplitude": ("FLOAT", {"default": 1.15, "min": 1.0, "max": 2.0, "step": 0.05}),
                "motion_latent_count": ("INT", {"default": 1, "min": 0, "max": 128, "step": 1}),
            },
            "optional": {
                "prev_samples": ("LATENT",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "conditioning/video_models"

    def execute(self, positive, negative, length, anchor_samples, motion_amplitude, motion_latent_count, prev_samples=None):
        # SVI Context Preservation Logic
        anchor_latent = anchor_samples["samples"].clone()
        B, C, T, H, W = anchor_latent.shape
        empty_latent = torch.zeros([B, 16, ((length - 1) // 4) + 1, H, W], 
                                 device=comfy.model_management.intermediate_device())
        total_latents = (length - 1) // 4 + 1
        device = anchor_latent.device
        dtype = anchor_latent.dtype
        
        # Context concatenation from SVI
        if prev_samples is None or motion_latent_count == 0:
            padding_size = total_latents - anchor_latent.shape[2]
            image_cond_latent = anchor_latent
        else:
            motion_latent = prev_samples["samples"][:, :, -motion_latent_count:].clone()
            padding_size = total_latents - anchor_latent.shape[2] - motion_latent.shape[2]
            image_cond_latent = torch.cat([anchor_latent, motion_latent], dim=2)
        
        padding = torch.zeros(1, C, padding_size, H, W, dtype=dtype, device=device)
        padding = comfy.latent_formats.Wan21().process_out(padding)
        image_cond_latent = torch.cat([image_cond_latent, padding], dim=2)
        
        # PainterI2V Motion Amplitude Enhancement
        if motion_amplitude > 1.0 and image_cond_latent.shape[2] > 1:
            # Split into base frame and subsequent frames
            base_latent = image_cond_latent[:, :, 0:1]      # First frame (anchor)
            subsequent_latent = image_cond_latent[:, :, 1:] # All subsequent frames
            
            # Apply motion enhancement to subsequent frames relative to base
            if subsequent_latent.shape[2] > 0:
                diff = subsequent_latent - base_latent
                diff_mean = diff.mean(dim=(1, 3, 4), keepdim=True)
                diff_centered = diff - diff_mean
                scaled_latent = base_latent + diff_centered * motion_amplitude + diff_mean
                
                # Clamp to prevent extreme values and recombine
                scaled_latent = torch.clamp(scaled_latent, -6, 6)
                image_cond_latent = torch.cat([base_latent, scaled_latent], dim=2)
        
        # Create conditioning mask
        mask = torch.ones((1, 1, empty_latent.shape[2], H, W), device=device, dtype=dtype)
        mask[:, :, :1] = 0.0
        
        # Apply conditioning values
        positive = node_helpers.conditioning_set_values(
            positive, {"concat_latent_image": image_cond_latent, "concat_mask": mask}
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"concat_latent_image": image_cond_latent, "concat_mask": mask}
        )
        
        # Reference latent enhancement from PainterI2V
        ref_latent = anchor_latent[:, :, 0:1]  # Use first frame of anchor as reference
        positive = node_helpers.conditioning_set_values(
            positive, {"reference_latents": [ref_latent]}, append=True
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"reference_latents": [torch.zeros_like(ref_latent)]}, append=True
        )
        
        out_latent = {"samples": empty_latent}
        return (positive, negative, out_latent)

# Node registration
NODE_CLASS_MAPPINGS = {
    "BoyoPainterSVI": BoyoPainterSVI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoPainterSVI": "Boyo Painter SVI (Motion + Infinite Length)",
}
