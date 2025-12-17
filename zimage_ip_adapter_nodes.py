"""
Z-Image IP-Adapter ComfyUI Nodes

Adapted from SD3 IP-Adapter nodes to work with:
1. Z-Image transformer architecture (30 layers, 3840 hidden dim)
2. Hijacked SD3 IP-Adapter weights (converted to Z-Image format)

Based on the SD3 implementation but modified for Z-Image compatibility.
"""

import os
import logging

import torch
import folder_paths
from safetensors.torch import load_file as open_safetensors

# Import the resampler from the SD3 pack (we can reuse this with modifications)
from .resampler import TimeResampler
from .zimage_attention_wrapper import ZImageBlockIPWrapper, ZImageIPAttnProcessor

# Use the same folder structure as SD3 IP-Adapter
MODELS_DIR = os.path.join(folder_paths.models_dir, "ipadapter")
if "ipadapter" not in folder_paths.folder_names_and_paths:
    current_paths = [MODELS_DIR]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["ipadapter"]
folder_paths.folder_names_and_paths["ipadapter"] = (
    current_paths,
    folder_paths.supported_pt_extensions,
)


def patch_zimage_model(
    patcher,
    ip_procs,
    resampler: TimeResampler,
    clip_embeds,
    weight=1.0,
    start=0.0,
    end=1.0,
):
    """
    Patches a Z-Image model to add the IP-Adapter functionality.
    
    Similar to SD3's patch() but adapted for Z-Image architecture.
    """
    
    # Get Z-Image transformer (different from SD3's mmdit)
    zimage_transformer = patcher.model.diffusion_model
    print(f"🔍 Z-Image transformer type: {type(zimage_transformer).__name__}")
    print(f"🔍 Z-Image transformer attributes: {[attr for attr in dir(zimage_transformer) if 'layer' in attr.lower() or 'block' in attr.lower()]}")
    
    # Z-Image might have different timestep settings
    timestep_schedule_max = patcher.model.model_config.sampling_settings.get(
        "timesteps", 1000
    )
    
    # IP-Adapter options shared between layers
    ip_options = {
        "hidden_states": None,
        "t_emb": None,
        "weight": weight,
    }

    def zimage_wrapper(forward, args):
        """Wrapper for Z-Image forward pass with IP-Adapter integration."""
        print(f"🏴‍☠️ IP-Adapter wrapper called! t_percent = {1 - args['timestep'].flatten()[0].cpu().item()}")
        # Calculate timestep percentage (0 to 1)
        t_percent = 1 - args["timestep"].flatten()[0].cpu().item()
        
        if start <= t_percent <= end:
            batch_size = args["input"].shape[0] // len(args["cond_or_uncond"])
            
            # Get image embeddings for current conditioning
            embeds = clip_embeds[args["cond_or_uncond"]]
            embeds = torch.repeat_interleave(embeds, batch_size, dim=0)
            
            # Convert timestep for resampler
            timestep = args["timestep"] * timestep_schedule_max
            
            # Run resampler to get IP-Adapter tokens
            image_emb, t_emb = resampler(embeds, timestep, need_temb=True)
            print(f"🎨 Generated IP tokens: {image_emb.shape}, t_emb: {t_emb.shape}")
            # Store for IP-Adapter layers
            ip_options["hidden_states"] = image_emb
            ip_options["t_emb"] = t_emb
        else:
            ip_options["hidden_states"] = None
            ip_options["t_emb"] = None

        return forward(args["input"], args["timestep"], **args["c"])

    # Set the wrapper
    patcher.set_model_unet_function_wrapper(zimage_wrapper)
    
    # Patch Z-Image transformer blocks (different structure from SD3)
    # Z-Image has 30 layers vs SD3's 38 joint_blocks
    if hasattr(zimage_transformer, 'layers'):
        blocks = zimage_transformer.layers
    elif hasattr(zimage_transformer, 'transformer_blocks'):
        blocks = zimage_transformer.transformer_blocks
    else:
        raise ValueError("Cannot find Z-Image transformer blocks")
    
    
    # Patch each block with IP-Adapter
    # DEBUG: Show what blocks we found FIRST
    print(f"🔧 Found {len(blocks)} Z-Image blocks")
    for i, block in enumerate(blocks[:3]):  # Just show first 3
        print(f"   Block {i}: {type(block).__name__}")
    
    # Patch each block with IP-Adapter
    for i, block in enumerate(blocks):
        if i < len(ip_procs):  # Make sure we have IP-Adapter for this layer
            wrapper = ZImageBlockIPWrapper(block, ip_procs[i], ip_options)
            
            # Use appropriate patch method for Z-Image
            # This might need adjustment based on Z-Image's exact structure
            try:
                patcher.set_model_patch_replace(wrapper, "dit", "joint_blocks", i)
                print(f"✅ Patched block {i}")
                print(f"   Original block type: {type(block).__name__}")
                print(f"   Wrapper created: {type(wrapper).__name__}")
            except Exception as e:
                print(f"❌ Failed to patch block {i}: {e}")

class ZImageIPAdapter:
    """
    Z-Image IP-Adapter model loader.
    
    Loads hijacked SD3 weights that have been converted for Z-Image.
    """
    
    def __init__(self, checkpoint: str, device):
        self.device = device
        
        # Load hijacked weights (our converted format)
        if checkpoint.endswith("safetensors") or checkpoint.endswith("sft"):
            # Load safetensors format
            full_state_dict = open_safetensors(
                os.path.join(MODELS_DIR, checkpoint), device=self.device
            )
        else:
            # Load .pt format (our hijacked weights)
            full_state_dict = torch.load(
                os.path.join(MODELS_DIR, checkpoint),
                map_location=self.device,
                weights_only=True,
            )
        
        # Handle our hijacked format
        if 'ip_adapter_state_dict' in full_state_dict:
            # This is our hijacked format
            state_dict = full_state_dict['ip_adapter_state_dict']
            self.conversion_info = full_state_dict.get('conversion_info', {})
            logging.info("Loaded hijacked SD3->Z-Image weights")
        else:
            # Assume it's a direct state dict
            state_dict = full_state_dict
            self.conversion_info = {}
        
        # Separate image projector and IP-Adapter weights
        self.image_proj_state = {}
        self.ip_adapter_state = {}
        
        for key, value in state_dict.items():
            if key.startswith('image_projector.'):
                # Convert key names for TimeResampler compatibility
                new_key = key.replace('image_projector.', '')
                self.image_proj_state[new_key] = value
            elif key.startswith('to_k_ip.') or key.startswith('to_v_ip.'):
                # These are our converted projection layers
                self.ip_adapter_state[key] = value
            else:
                # Put other weights in appropriate category
                if 'resampler' in key or 'proj' in key or 'latents' in key:
                    self.image_proj_state[key] = value
                else:
                    self.ip_adapter_state[key] = value
        
        # Create TimeResampler with Z-Image dimensions
        self.resampler = TimeResampler(
            dim=1280,           # Internal dimension (from SD3)
            depth=4,            # Resampler depth (from SD3)
            dim_head=64,        # Attention head dimension
            heads=20,           # Number of attention heads
            num_queries=64,     # Number of IP tokens (64 from SD3)
            embedding_dim=1152, # SigLIP input dimension
            output_dim=3840,    # Z-Image hidden dimension (expanded from SD3's 2432)
            ff_mult=4,
            timestep_in_dim=320,
            timestep_flip_sin_to_cos=True,
            timestep_freq_shift=0,
        )
        
        self.resampler.eval()
        self.resampler.to(self.device, dtype=torch.float16)
        
        # Load TimeResampler weights
        try:
            self.resampler.load_state_dict(self.image_proj_state, strict=False)
            logging.info("Loaded TimeResampler weights")
        except Exception as e:
            logging.warning(f"Failed to load some TimeResampler weights: {e}")
        
        # Create IP-Adapter processors for Z-Image (30 layers)
        num_layers = 30  # Z-Image has 30 layers
        
        self.procs = torch.nn.ModuleList([
            ZImageIPAttnProcessor(
                hidden_size=3840,                    # Z-Image hidden dimension
                cross_attention_dim=3840,            # Z-Image cross attention
                ip_hidden_states_dim=3840,           # IP-Adapter hidden states
                ip_encoder_hidden_states_dim=3840,   # IP encoder states
                head_dim=120,                        # Z-Image head dim (3840/32)
                timesteps_emb_dim=1280,              # Time embedding dimension
            ).to(self.device, dtype=torch.float16)
            for _ in range(num_layers)
        ])
        
        # Load IP-Adapter processor weights
        try:
            self._load_ip_processor_weights()
            logging.info("Loaded IP-Adapter processor weights")
        except Exception as e:
            logging.warning(f"Failed to load some IP-Adapter weights: {e}")
    
    def _load_ip_processor_weights(self):
        """Load weights for IP-Adapter processors."""
        
        # Group weights by layer
        layer_weights = {}
        for key, value in self.ip_adapter_state.items():
            if '.weight' in key:
                # Extract layer number and weight type
                if key.startswith('to_k_ip.'):
                    layer_num = int(key.split('.')[1])
                    if layer_num not in layer_weights:
                        layer_weights[layer_num] = {}
                    layer_weights[layer_num]['to_k_ip.weight'] = value
                    
                elif key.startswith('to_v_ip.'):
                    layer_num = int(key.split('.')[1])
                    if layer_num not in layer_weights:
                        layer_weights[layer_num] = {}
                    layer_weights[layer_num]['to_v_ip.weight'] = value
        
        # Load weights into processors
        for layer_idx, weights in layer_weights.items():
            if layer_idx < len(self.procs):
                proc = self.procs[layer_idx]
                
                # Load K projection
                if 'to_k_ip.weight' in weights:
                    proc.to_k_ip.weight.data = weights['to_k_ip.weight']
                    
                # Load V projection  
                if 'to_v_ip.weight' in weights:
                    proc.to_v_ip.weight.data = weights['to_v_ip.weight']


class IPAdapterZImageLoader:
    """ComfyUI node to load Z-Image IP-Adapter."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter": (folder_paths.get_filename_list("ipadapter"),),
                "provider": (["cuda", "cpu", "mps"],),
            }
        }

    RETURN_TYPES = ("IP_ADAPTER_ZIMAGE",)
    RETURN_NAMES = ("ipadapter",)
    FUNCTION = "load_model"
    CATEGORY = "BoyoNodes/ZImage"

    def load_model(self, ipadapter, provider):
        logging.info("Loading Z-Image IP-Adapter model (hijacked from SD3)")
        model = ZImageIPAdapter(ipadapter, provider)
        return (model,)


class ApplyIPAdapterZImage:
    """ComfyUI node to apply Z-Image IP-Adapter to model."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "ipadapter": ("IP_ADAPTER_ZIMAGE",),
                "image_embed": ("CLIP_VISION_OUTPUT",),
                "weight": (
                    "FLOAT",
                    {"default": 0.7, "min": -2.0, "max": 5.0, "step": 0.05},
                ),
                "start_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "end_percent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter"
    CATEGORY = "BoyoNodes/ZImage"

    def apply_ipadapter(
        self, model, ipadapter, image_embed, weight, start_percent, end_percent
    ):
        # Clone model for modification
        new_model = model.clone()
        
        # Prepare image embeddings
        # Use penultimate hidden states (like SD3 does)
        image_embed_tensor = image_embed.penultimate_hidden_states
        
        # Add unconditional embedding (zeros) for CFG
        embeds = torch.cat([
            image_embed_tensor, 
            torch.zeros_like(image_embed_tensor)
        ], dim=0).to(ipadapter.device, dtype=torch.float16)
        
        # Apply IP-Adapter patch to model
        patch_zimage_model(
            new_model,
            ipadapter.procs,
            ipadapter.resampler,
            embeds,
            weight=weight,
            start=start_percent,
            end=end_percent,
        )
        
        return (new_model,)


# Export for ComfyUI registration
ZIMAGE_IP_ADAPTER_CLASS_MAPPINGS = {
    "BoyoIPAdapterZImageLoader": IPAdapterZImageLoader,        # ← Add "Boyo" prefix
    "BoyoApplyIPAdapterZImage": ApplyIPAdapterZImage,          # ← Add "Boyo" prefix
}

ZIMAGE_IP_ADAPTER_DISPLAY_NAME_MAPPINGS = {
    "BoyoIPAdapterZImageLoader": "Boyo Load Z-Image IP-Adapter",
    "BoyoApplyIPAdapterZImage": "Boyo Apply Z-Image IP-Adapter",
}