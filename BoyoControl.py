import torch
import math
import logging
import comfy.model_management as mm
from comfy.utils import common_upscale

# Set up logging
log = logging.getLogger(__name__)

# Constants from WanVideoWrapper
VAE_STRIDE = (4, 8, 8)
PATCH_SIZE = (1, 2, 2)

class BoyoVACEInjector:
    """
    A node that injects VACE control data directly into the model at the model level.
    This bypasses conditioning entirely and sets the control data as model attributes
    where VACE models expect to find them.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_image": ("IMAGE",),
                "vace_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "vace_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "vace_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "num_frames": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            },
            "optional": {
                "model": ("MODEL",),
                "wanvideomodel": ("WANVIDEOMODEL",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "inject_vace_control"
    CATEGORY = "BoyoResearch"

    def process_control_to_vace(self, control_image, num_frames, vae=None):
        """
        Process control image into VACE format (96 channels) 
        Reuse the working logic from our previous conditioning approach
        """
        log.info("Processing control image to VACE format")
        
        device = mm.get_torch_device()
        
        # Use control image's actual dimensions
        actual_height, actual_width = control_image.shape[1:3]
        
        # Ensure dimensions are compatible with VAE (multiple of 16)
        actual_width = (actual_width // 16) * 16
        actual_height = (actual_height // 16) * 16
        log.info(f"Processing at dimensions: {actual_width}x{actual_height}")
        
        # Target latent shape for VACE
        target_shape = (16, (num_frames - 1) // VAE_STRIDE[0] + 1,
                        actual_height // VAE_STRIDE[1],
                        actual_width // VAE_STRIDE[2])
        
        # Process control image - reuse our working tensor processing
        input_frames = control_image.clone()[:num_frames, :, :, :3]
        input_frames = common_upscale(input_frames.movedim(-1, 1), actual_width, actual_height, "lanczos", "disabled").movedim(1, -1)
        
        # Convert to proper tensor format: [B, C, T, H, W] - use our working approach
        input_frames = input_frames.permute(0, 3, 1, 2)  # [num_frames, C, H, W]
        input_frames = input_frames.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, C, num_frames, H, W]
        input_frames = input_frames.to(device, dtype=torch.float32)
        
        log.info(f"Input frames shape: {input_frames.shape}")
        
        if vae is not None:
            try:
                # Use VAE encoding if available
                log.info("Using VAE encoding for control image")
                
                # Encode through VAE
                with torch.no_grad():
                    if hasattr(vae, 'encode_video'):
                        # Video VAE
                        encoded = vae.encode_video(input_frames)
                    else:
                        # Regular VAE - process frame by frame
                        frames_2d = input_frames.squeeze(2)  # Remove time dimension
                        log.info(f"Frames 2D shape for VAE: {frames_2d.shape}")
                        
                        encoded = vae.encode(frames_2d)
                        if hasattr(encoded, 'sample'):
                            encoded = encoded.sample()
                        elif hasattr(encoded, 'latent_dist'):
                            encoded = encoded.latent_dist.sample() if hasattr(encoded.latent_dist, 'sample') else encoded.latent_dist.mean
                        elif isinstance(encoded, dict) and 'samples' in encoded:
                            encoded = encoded['samples']
                        
                        # Add time dimension back
                        if encoded.dim() == 4:  # B,C,H,W
                            encoded = encoded.unsqueeze(2)  # B,C,T,H,W
                
                vace_context = encoded.to(device, dtype=torch.float32)
                log.info(f"VAE encoded shape: {vace_context.shape}")
                
            except Exception as e:
                log.warning(f"VAE encoding failed: {e}, falling back to direct processing")
                vace_context = self.direct_process_control(input_frames, device)
        else:
            log.info("No VAE provided, using direct processing")
            vace_context = self.direct_process_control(input_frames, device)
        
        # Ensure we have the right number of channels (96 for VACE)
        if vace_context.shape[1] != 96:
            # Resize to 96 channels
            if vace_context.shape[1] < 96:
                # Pad with zeros
                padding = torch.zeros(vace_context.shape[0], 96 - vace_context.shape[1], 
                                    *vace_context.shape[2:], device=device, dtype=vace_context.dtype)
                vace_context = torch.cat([vace_context, padding], dim=1)
            else:
                # Truncate to 96 channels
                vace_context = vace_context[:, :96]
        
        log.info(f"Final VACE context shape: {vace_context.shape}")
        
        # Return the processed VACE tensor
        return vace_context

    def direct_process_control(self, input_frames, device):
        """
        Direct processing without VAE - creates 96-channel control representation
        """
        log.info("Direct processing control image")
        
        # Handle tensor shape - use our working logic
        if input_frames.dim() == 6:
            B, C, T, H, W, extra = input_frames.shape
            input_frames = input_frames.squeeze(-1)  # Remove extra dimension
        elif input_frames.dim() == 5:
            B, C, T, H, W = input_frames.shape
        else:
            log.error(f"Unexpected input tensor dimensions: {input_frames.shape}")
            # Reshape to expected format
            input_frames = input_frames.view(1, 3, 1, input_frames.shape[-2], input_frames.shape[-1])
            B, C, T, H, W = input_frames.shape
        
        log.info(f"Processing tensor shape: B={B}, C={C}, T={T}, H={H}, W={W}")
        
        # Create base control representation
        if C == 3:  # RGB
            # Create multiple representations of the control signal
            control_base = input_frames.repeat(1, 32, 1, 1, 1)  # 3*32 = 96 channels
        elif C == 1:  # Grayscale
            control_base = input_frames.repeat(1, 96, 1, 1, 1)
        else:
            # Interpolate to 96 channels
            control_base = torch.nn.functional.interpolate(
                input_frames.view(B, C, T * H * W),
                size=(96, T * H * W),
                mode='linear',
                align_corners=False
            ).view(B, 96, T, H, W)
        
        # Normalize to match expected VACE range
        control_base = (control_base - 0.5) * 2.0  # [0,1] -> [-1,1]
        
        return control_base

    def inject_vace_control(self, control_image, vace_strength, vace_start_percent, vace_end_percent, num_frames, model=None, wanvideomodel=None, vae=None):
        """
        Main function to inject VACE control directly into the model
        Supports both native ComfyUI models and WanVideoWrapper models
        """
        device = mm.get_torch_device()
        
        log.info(f"=== BoyoVACE Injector ===")
        
        # Check that we have at least one model input
        if model is None and wanvideomodel is None:
            raise ValueError("Either 'model' or 'wanvideomodel' must be provided")
        
        # Determine which model to use
        if wanvideomodel is not None:
            log.info("Using WanVideoWrapper model input")
            target_model = self.convert_wanvideomodel_to_model(wanvideomodel)
            log.info(f"Converted WanVideoModel to ComfyUI Model: {type(target_model)}")
        else:
            log.info("Using native ComfyUI model input")
            target_model = model.clone()
        
        log.info(f"Target model type: {type(target_model)}")
        log.info(f"Control image shape: {control_image.shape}")
        log.info(f"VACE strength: {vace_strength}")
        log.info(f"VACE range: {vace_start_percent:.2f} - {vace_end_percent:.2f}")
        
        try:
            if vace_strength > 0.01:  # Only process if strength is meaningful
                # Process control image to VACE format
                vace_tensor = self.process_control_to_vace(control_image, num_frames, vae)
                
                # Apply strength scaling
                vace_tensor = vace_tensor * vace_strength
                
                # Remove batch dimension to match expected format
                vace_tensor_final = vace_tensor.squeeze(0)  # [96, 1, H, W] format
                
                log.info(f"Final VACE tensor shape for injection: {vace_tensor_final.shape}")
                
                # Set VACE attributes on the target model
                self.set_model_attributes(target_model, vace_tensor_final, vace_start_percent, vace_end_percent)
                
                log.info("Successfully injected VACE control into model")
            else:
                log.info("VACE strength too low, skipping injection")
            
            return (target_model,)
            
        except Exception as e:
            log.error(f"VACE injection failed: {e}")
            log.error("Returning original model without VACE control")
            import traceback
            traceback.print_exc()
            return (model if model is not None else self.convert_wanvideomodel_to_model(wanvideomodel),)

    def convert_wanvideomodel_to_model(self, wanvideomodel):
        """
        Convert a WANVIDEOMODEL back to a standard ComfyUI MODEL like the normal diffusion model loader
        This mimics what the standard loader does - extract just the basic model without WanWrapper functionality
        """
        log.info("Converting WANVIDEOMODEL to ComfyUI MODEL")
        
        try:
            # Extract the ComfyUI ModelPatcher from the WanVideoWrapper
            # This is what connects the WanVideoWrapper to ComfyUI's sampling system
            if hasattr(wanvideomodel, 'model'):
                # wanvideomodel is the ModelPatcher that wraps the WanVideoModel
                # wanvideomodel.model is the WanVideoModel (BaseModel subclass)
                # wanvideomodel.model.diffusion_model is the actual WanModel with VACE data
                
                # Clone the ModelPatcher to get a clean ComfyUI-compatible model
                clean_model = wanvideomodel.clone()
                log.info(f"Cloned ModelPatcher from WanVideoWrapper: {type(clean_model)}")
                
                # The VACE attributes should already be on the diffusion_model
                # So this should preserve them while giving us ComfyUI compatibility
                return clean_model
            
            else:
                log.warning("WanVideoWrapper has unexpected structure, trying direct return")
                return wanvideomodel
                
        except Exception as e:
            log.error(f"Model conversion failed: {e}")
            import traceback
            traceback.print_exc()
            log.info("Falling back to using WanVideoWrapper directly")
            return wanvideomodel

    def add_comfyui_compatibility(self, model):
        """Add missing ComfyUI methods and attributes to WanVideoModel for full compatibility"""
        
        # Add load_device attribute if missing
        if not hasattr(model, 'load_device'):
            model.load_device = mm.get_torch_device()
            log.info("Added load_device attribute")
        
        # Add offload_device attribute if missing
        if not hasattr(model, 'offload_device'):
            model.offload_device = mm.unet_offload_device()
            log.info("Added offload_device attribute")
        
        # Add model_options attribute if missing
        if not hasattr(model, 'model_options'):
            model.model_options = {}
            log.info("Added model_options attribute")
        
        # Add ModelPatcher-specific attributes
        if not hasattr(model, 'hook_mode'):
            model.hook_mode = "default"
            log.info("Added hook_mode attribute")
        
        # Create a nested model structure that ComfyUI expects
        if not hasattr(model, 'model'):
            # Create a mock model object with latent_format
            class MockModel:
                def __init__(self):
                    from comfy.latent_formats import LatentFormat
                    class VACELatentFormat(LatentFormat):
                        def __init__(self):
                            self.scale_factor = 0.18215
                            self.latent_channels = 16  # VACE models use 16 channels
                            self.latent_rgb_factors = [0.5, 0.5, 0.5]
                            self.taesd_decoder_name = None
                    
                    self.latent_format = VACELatentFormat()
            
            model.model = MockModel()
            log.info("Added mock model.model with latent_format")
        
        def get_model_object(name):
            """Compatibility method for ComfyUI model interface"""
            if name == "latent_format":
                return model.model.latent_format
            elif name == "model_sampling":
                # Return basic model sampling
                if hasattr(model, 'diffusion_model') and hasattr(model.diffusion_model, 'dtype'):
                    from comfy.model_sampling import ModelSamplingDiscreteFlow
                    sampling = ModelSamplingDiscreteFlow()
                    sampling.set_parameters(shift=1.0)  # Default shift for VACE
                    return sampling
                return None
            return None
        
        def model_dtype():
            """Return the model's dtype"""
            if hasattr(model, 'diffusion_model') and hasattr(model.diffusion_model, 'dtype'):
                return model.diffusion_model.dtype
            return torch.bfloat16  # Default to bf16
        
        # Add the missing methods
        model.get_model_object = get_model_object
        model.model_dtype = model_dtype
        
        log.info("Added ComfyUI compatibility: load_device, offload_device, model_options, hook_mode, model.model.latent_format, get_model_object, model_dtype")

    def copy_vace_attributes(self, source_model, target_model):
        """Copy VACE attributes from source to target model"""
        vace_attrs = [
            'src_ref_images', 'vace_context', 'ref_images', 'vace_control',
            'image_embeds', 'vace_embeds', 'vace_start_percent', 'vace_end_percent',
            'vace_strength', 'has_vace_control'
        ]
        
        for attr in vace_attrs:
            if hasattr(source_model, attr):
                setattr(target_model, attr, getattr(source_model, attr))
                log.debug(f"Copied VACE attribute: {attr}")

    def set_model_attributes(self, model, vace_tensor, start_percent, end_percent):
        """
        Set VACE control attributes on the model using multiple naming patterns
        """
        log.info("Setting VACE attributes on model")
        
        # Try to access the actual model instance at all levels
        models_to_patch = [model]
        
        if hasattr(model, 'model'):
            models_to_patch.append(model.model)
            log.info(f"Found model.model: {type(model.model)}")
        
        # Check for diffusion_model at multiple levels
        diffusion_model = None
        if hasattr(model, 'diffusion_model'):
            diffusion_model = model.diffusion_model
        elif hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
            diffusion_model = model.model.diffusion_model
        
        if diffusion_model:
            models_to_patch.append(diffusion_model)
            log.info(f"Found diffusion_model: {type(diffusion_model)}")
        
        log.info(f"Will patch {len(models_to_patch)} model levels")
        
        # Try various attribute names that VACE models might expect
        attribute_names = [
            'src_ref_images',      # Most likely from official code
            'vace_context',        # Alternative naming
            'ref_images',          # Shortened version
            'vace_control',        # Control-focused naming
            'image_embeds',        # Generic embedding name
            'vace_embeds'          # Direct VACE reference
        ]
        
        # VACE timing and control parameters
        timing_attrs = {
            'vace_start_percent': start_percent,
            'vace_end_percent': end_percent,
            'vace_strength': 1.0,  # Strength already applied to tensor
            'has_vace_control': True
        }
        
        for model_obj in models_to_patch:
            # Set the main VACE tensor under multiple names
            for attr_name in attribute_names:
                try:
                    setattr(model_obj, attr_name, vace_tensor)
                    log.info(f"Set {attr_name} on {type(model_obj)}")
                except Exception as e:
                    log.debug(f"Failed to set {attr_name}: {e}")
            
            # Set timing and control parameters
            for attr_name, attr_value in timing_attrs.items():
                try:
                    setattr(model_obj, attr_name, attr_value)
                    log.debug(f"Set {attr_name} = {attr_value}")
                except Exception as e:
                    log.debug(f"Failed to set {attr_name}: {e}")
        
        log.info("Completed setting VACE attributes")


class BoyoVACEViewer:
    """
    Debug node to inspect what VACE data is attached to a model
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "view_model_vace"
    CATEGORY = "BoyoResearch"

    def view_model_vace(self, model):
        """
        View what VACE data is attached to the model
        """
        log.info("=== VACE Model Viewer ===")
        log.info(f"Model type: {type(model)}")
        
        # Check various model levels
        models_to_check = [model]
        if hasattr(model, 'model'):
            models_to_check.append(model.model)
            log.info(f"Found model.model: {type(model.model)}")
        
        if hasattr(model, 'diffusion_model'):
            models_to_check.append(model.diffusion_model)
        elif hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
            models_to_check.append(model.model.diffusion_model)
            log.info(f"Found diffusion_model: {type(model.model.diffusion_model)}")
        
        # VACE attribute names to check
        vace_attrs = [
            'src_ref_images', 'vace_context', 'ref_images', 'vace_control',
            'image_embeds', 'vace_embeds', 'vace_start_percent', 'vace_end_percent',
            'vace_strength', 'has_vace_control'
        ]
        
        for i, model_obj in enumerate(models_to_check):
            log.info(f"\nChecking model level {i}: {type(model_obj)}")
            
            found_attrs = []
            for attr in vace_attrs:
                if hasattr(model_obj, attr):
                    attr_value = getattr(model_obj, attr)
                    if torch.is_tensor(attr_value):
                        found_attrs.append(f"{attr}: tensor {attr_value.shape}")
                    else:
                        found_attrs.append(f"{attr}: {type(attr_value)} = {attr_value}")
            
            if found_attrs:
                log.info(f"  Found VACE attributes:")
                for attr_info in found_attrs:
                    log.info(f"    {attr_info}")
            else:
                log.info(f"  No VACE attributes found")
        
        return (model,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "BoyoVACEInjector": BoyoVACEInjector,
    "BoyoVACEViewer": BoyoVACEViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoVACEInjector": "Boyo VACE Injector",
    "BoyoVACEViewer": "Boyo VACE Model Viewer",
}