import torch
import math
import logging
import comfy.sample
import comfy.samplers
import comfy.model_management as mm
import comfy.utils
from comfy.utils import common_upscale

# Set up logging
log = logging.getLogger(__name__)

class BoyoWanFunImageSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "control_image": ("IMAGE",),
                "control_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "control_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "control_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "vae": ("VAE",),
                "control_processing": (["vae_encode", "direct"], {"default": "direct"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "BoyoResearch"

    def process_control_image_direct(self, control_image, latent_shape, device):
        """FIXED: Process control image directly - simplified"""
        log.info("Processing control image with direct method")
        
        # Get target dimensions from latent
        batch_size, channels, frames, latent_h, latent_w = latent_shape
        
        # Convert B,H,W,C -> B,C,H,W
        control_image = control_image.permute(0, 3, 1, 2).to(device)
        
        # Resize to latent spatial dimensions (multiply by 8 for proper size)
        target_h, target_w = latent_h * 8, latent_w * 8
        
        control_resized = torch.nn.functional.interpolate(
            control_image, 
            size=(target_h, target_w), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Normalize to [-1, 1]
        if control_resized.max() > 1.0:
            control_resized = (control_resized / 255.0) * 2.0 - 1.0
        else:
            control_resized = control_resized * 2.0 - 1.0
        
        # Add temporal dimension: B,C,H,W -> B,C,T,H,W
        control_resized = control_resized.unsqueeze(2)  # Add time dim
        
        # FIXED: Resize to match latent channels (32 channels for Wan Fun Control models)
        # Based on config: in_dim=48 means 32 control + 16 latent = 48 total
        if control_resized.shape[1] == 1:  # Grayscale
            control_latents = control_resized.repeat(1, 32, 1, 1, 1)
        elif control_resized.shape[1] == 3:  # RGB
            # Pad to 32 channels
            padding = torch.zeros(batch_size, 29, frames, target_h, target_w, device=device, dtype=control_resized.dtype)
            control_latents = torch.cat([control_resized, padding], dim=1)
        else:
            # Interpolate to 32 channels
            control_latents = torch.nn.functional.interpolate(
                control_resized.view(batch_size, -1, target_h, target_w),
                size=(32, target_h, target_w),
                mode='trilinear' if control_resized.dim() == 5 else 'bilinear'
            ).view(batch_size, 32, frames, target_h, target_w)
        
        # Downsample to latent space (8x downsampling)
        control_latents = torch.nn.functional.interpolate(
            control_latents.view(batch_size * frames, 16, target_h, target_w),
            size=(latent_h, latent_w),
            mode='bilinear',
            align_corners=False
        ).view(batch_size, 16, frames, latent_h, latent_w)
        
        log.info(f"Control latents shape: {control_latents.shape}")
        return control_latents

    def process_control_image_vae(self, control_image, vae, device):
        """FIXED: Process control image through VAE encoding"""
        log.info("Processing control image with VAE encoding")
        
        # Convert B,H,W,C -> B,C,H,W
        control_image = control_image.permute(0, 3, 1, 2).to(device)
        
        # Ensure RGB (3 channels)
        if control_image.shape[1] == 1:  # Grayscale to RGB
            control_image = control_image.repeat(1, 3, 1, 1)
        elif control_image.shape[1] == 4:  # RGBA to RGB
            control_image = control_image[:, :3]
        
        # Normalize to [0, 1] if needed
        if control_image.max() > 1.0:
            control_image = control_image / 255.0
        
        # FIXED: Don't add time dimension - VAE expects 4D tensor
        log.info(f"VAE input shape: {control_image.shape}")
        
        try:
            # Move VAE to same device
            vae.to(device)
            
            # Encode through VAE (this returns latent distribution)
            with torch.no_grad():
                encoded = vae.encode(control_image)
                
                # Handle different VAE return types
                if hasattr(encoded, 'sample'):
                    control_latents = encoded.sample()
                elif hasattr(encoded, 'latent_dist'):
                    if hasattr(encoded.latent_dist, 'sample'):
                        control_latents = encoded.latent_dist.sample()
                    else:
                        control_latents = encoded.latent_dist.mean
                else:
                    control_latents = encoded
            
            # Add time dimension if needed: B,C,H,W -> B,C,T,H,W
            if control_latents.dim() == 4:
                control_latents = control_latents.unsqueeze(2)
            
            log.info(f"VAE encoded control shape: {control_latents.shape}")
            return control_latents
            
        except Exception as e:
            log.error(f"VAE encoding failed: {e}")
            log.error("Falling back to direct processing")
            # Fall back to direct processing
            latent_shape = (control_image.shape[0], 16, 1, control_image.shape[2]//8, control_image.shape[3]//8)
            return self.process_control_image_direct(control_image.permute(0, 2, 3, 1), latent_shape, device)

    def patch_model_for_control(self, model_patcher, control_latents, control_start_percent, control_end_percent):
        """FIXED: Correct model patching based on official WanVideoSampler logic"""
        
        actual_model = model_patcher.model
        original_apply_model = actual_model.apply_model
        
        log.info(f"Patching model with control range: {control_start_percent:.2f} - {control_end_percent:.2f}")
        
        def apply_model_with_control(x, t, **kwargs):
            # Calculate current step percentage from timestep
            if hasattr(t, 'item'):
                t_value = t.item() if t.numel() == 1 else t.mean().item()
            else:
                t_value = float(t)
                
            # Normalize timestep (assuming t goes from ~1000 to 0)
            current_step_percentage = 1.0 - (t_value / 1000.0)
            
            # Apply control based on step percentage
            apply_control = (control_start_percent <= current_step_percentage <= control_end_percent) or \
                           (control_end_percent > 0 and current_step_percentage == 0 and current_step_percentage >= control_start_percent)
            
            # Prepare control data to match x dimensions
            control_for_concat = control_latents.to(x.device, x.dtype)
            
            # Handle batch dimension (x might be [2, 16, 1, 128, 128] for pos/neg batching)
            if control_for_concat.shape[0] != x.shape[0]:
                control_for_concat = control_for_concat.repeat(x.shape[0], 1, 1, 1, 1)
            
            # Ensure spatial dimensions match
            if control_for_concat.shape[3:] != x.shape[3:]:
                log.debug(f"Resizing control from {control_for_concat.shape} to match {x.shape}")
                control_for_concat = torch.nn.functional.interpolate(
                    control_for_concat.view(-1, *control_for_concat.shape[2:]),
                    size=x.shape[3:],
                    mode='bilinear',
                    align_corners=False
                ).view(control_for_concat.shape[:2] + x.shape[2:])
            
            if apply_control:
                log.debug(f"APPLYING CONTROL at step {current_step_percentage:.3f}")
                # FIXED: Based on official WanVideoSampler - concatenate [control_latents, image_cond]
                # image_cond_input = torch.cat([control_latents.to(z), image_cond.to(z)])
                image_cond_input = torch.cat([control_for_concat, x], dim=1)
            else:
                log.debug(f"NO CONTROL at step {current_step_percentage:.3f}")
                # FIXED: When no control, concatenate [zeros, image_cond] 
                # image_cond_input = torch.cat([torch.zeros_like(control_latents, dtype=dtype), image_cond.to(z)])
                zero_control = torch.zeros_like(control_for_concat)
                image_cond_input = torch.cat([zero_control, x], dim=1)
            
            # Pass as list like the official sampler does: y=[image_cond_input]
            kwargs['y'] = [image_cond_input]
            
            # Call original model
            try:
                return original_apply_model(x, t, **kwargs)
            except Exception as e:
                log.error(f"Model forward failed: {e}")
                log.error(f"x shape: {x.shape}, y shape: {kwargs.get('y', ['None'])[0].shape if 'y' in kwargs else 'None'}")
                raise e
        
        # Create patched model
        patched_model_patcher = model_patcher.clone()
        patched_model_patcher.model.apply_model = apply_model_with_control
        
        return patched_model_patcher

    def sample(self, model, positive, negative, latent_image, control_image, 
               control_strength, control_start_percent, control_end_percent,
               seed, steps, cfg, sampler_name, scheduler, denoise, 
               vae=None, control_processing="direct"):
        
        device = mm.get_torch_device()
        
        log.info(f"=== BoyoWanFun Control Sampler ===")
        log.info(f"Control processing: {control_processing}")
        log.info(f"Control strength: {control_strength}")
        log.info(f"Control range: {control_start_percent:.2f} - {control_end_percent:.2f}")
        
        # Validate inputs
        if control_processing == "vae_encode" and vae is None:
            raise ValueError("VAE must be provided when control_processing is 'vae_encode'")
        
        # Get latent info
        latent = latent_image["samples"]
        latent_shape = latent.shape
        log.info(f"Latent shape: {latent_shape}")
        
        # Process the control image
        try:
            if control_processing == "vae_encode":
                control_latents = self.process_control_image_vae(control_image, vae, device)
            else:  # "direct"
                control_latents = self.process_control_image_direct(control_image, latent_shape, device)
            
            # Apply control strength
            control_latents = control_latents * control_strength
            
        except Exception as e:
            log.error(f"Control processing failed: {e}")
            # Fallback: create zeros
            control_latents = torch.zeros_like(latent)
        
        # Only patch the model if control_strength > 0
        if control_strength > 0.01:  # Small threshold to avoid float precision issues
            patched_model = self.patch_model_for_control(
                model, control_latents, control_start_percent, control_end_percent
            )
        else:
            # Use original model without any patching
            patched_model = model
        
        # FIXED: Use ComfyUI's sample function directly instead of manual KSampler
        sampler_obj = comfy.samplers.sampler_object(sampler_name)
        
        # Calculate sigmas
        sigmas = comfy.samplers.calculate_sigmas(
            patched_model.get_model_object("model_sampling"), 
            scheduler, 
            steps
        ).to(device)
        
        # Handle denoise
        if denoise < 1.0:
            if denoise <= 0.0:
                sigmas = torch.FloatTensor([]).to(device)
            else:
                new_steps = int(steps / denoise)
                full_sigmas = comfy.samplers.calculate_sigmas(
                    patched_model.get_model_object("model_sampling"), 
                    scheduler, 
                    new_steps
                ).to(device)
                sigmas = full_sigmas[-(steps + 1):]
        
        # Generate noise
        noise = torch.randn_like(latent).to(device)
        
        # FIXED: Use the proper sample function
        samples = comfy.samplers.sample(
            model=patched_model,
            noise=noise,
            positive=positive,
            negative=negative,
            cfg=cfg,
            device=device,
            sampler=sampler_obj,
            sigmas=sigmas,
            model_options={},
            latent_image=latent.to(device),
            seed=seed
        )
        
        return ({"samples": samples},)


class BoyoWanFunEmptyLatent:
    """Create empty latents for Fun Control models - UNCHANGED (this works)"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 32, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 1024, "min": 32, "max": 8192, "step": 32}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "BoyoResearch"

    def generate(self, width, height, batch_size):
        device = mm.get_torch_device()
        
        # Video models typically use 8x downsampling
        latent_width = width // 8
        latent_height = height // 8
        
        # Fun models typically use 16 channels, 1 frame for single image
        channels = 16
        frames = 1
        
        # Create random latent directly on CUDA
        latent = torch.randn(
            batch_size, channels, frames, latent_height, latent_width,
            dtype=torch.float32,
            device=device
        )
        
        log.info(f"Created empty latent: {latent.shape} for {width}x{height} image on {latent.device}")
        
        return ({"samples": latent},)


NODE_CLASS_MAPPINGS = {
    "BoyoWanFunImageSampler": BoyoWanFunImageSampler,
    "BoyoWanFunEmptyLatent": BoyoWanFunEmptyLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoWanFunImageSampler": "Boyo Wan Fun Image Sampler",
    "BoyoWanFunEmptyLatent": "Boyo Wan Fun Empty Latent",
}