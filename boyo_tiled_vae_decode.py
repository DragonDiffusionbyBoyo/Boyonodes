import torch
import numpy as np
from PIL import Image
from transformers import pipeline
import torchvision.transforms as T
import os

class BoyoTiledVAEDecode:
    def __init__(self):
        self.classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
        self.alternative_image_path = os.path.join(os.path.dirname(__file__), "images", "example.png")
        if not os.path.exists(self.alternative_image_path):
            raise FileNotFoundError(f"Alternative image not found at {self.alternative_image_path}")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "horizontal_tiles": ("INT", {"default": 2, "min": 1, "max": 8}),
                "vertical_tiles": ("INT", {"default": 2, "min": 1, "max": 8}),
                "overlap": ("INT", {"default": 32, "min": 16, "max": 128}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode_tiled_and_check_nsfw"
    CATEGORY = "Boyonodes"

    def tensor2pil(self, image):
        """Convert tensor to PIL Image"""
        return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    def pil2tensor(self, image):
        """Convert PIL Image to tensor"""
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    def check_nsfw_on_decoded_image(self, decoded_tensor):
        """Check a single decoded image tensor for NSFW content and replace if needed"""
        # Convert tensor to PIL for NSFW classification
        transform = T.ToPILImage()
        
        # Handle different tensor formats - ensure we have [C, H, W]
        if len(decoded_tensor.shape) == 4:  # [1, H, W, C]
            pil_image = transform(decoded_tensor.squeeze(0).permute(2, 0, 1))
        elif len(decoded_tensor.shape) == 3:  # [H, W, C]
            pil_image = transform(decoded_tensor.permute(2, 0, 1))
        else:
            raise ValueError(f"Unexpected tensor shape: {decoded_tensor.shape}")
        
        # Run NSFW classification
        result = self.classifier(pil_image)
        
        # Check if NSFW
        for r in result:
            if r["label"] == "nsfw" and r["score"] > 0.9:
                print(f"NSFW content detected (score: {r['score']:.3f}), replacing with alternative image")
                # Load and resize alternative image to match original
                alternative_image = Image.open(self.alternative_image_path).convert('RGB')
                width, height = pil_image.size
                alt_pil = alternative_image.resize((width, height), resample=Image.LANCZOS)
                
                # Convert back to tensor format matching input
                alt_tensor = torch.from_numpy(np.array(alt_pil).astype(np.float32) / 255.0)
                if len(decoded_tensor.shape) == 4:  # [1, H, W, C]
                    return alt_tensor.unsqueeze(0)
                else:  # [H, W, C]
                    return alt_tensor
        
        return decoded_tensor

    def decode_tiled_and_check_nsfw(self, samples, vae, horizontal_tiles, vertical_tiles, overlap):
        # Get the latent samples
        latent_samples = samples["samples"]
        batch, channels, height, width = latent_samples.shape
        
        print(f"Starting tiled VAE decode: {batch}x{channels}x{height}x{width}")
        print(f"Tiles: {vertical_tiles}x{horizontal_tiles}, Overlap: {overlap}")
        
        # Get VAE scale factors
        scale_factor = 8  # Standard VAE scale factor, adjust if needed
        output_height = height * scale_factor
        output_width = width * scale_factor
        
        # Calculate tile sizes with overlap
        base_tile_height = (height + (vertical_tiles - 1) * overlap) // vertical_tiles
        base_tile_width = (width + (horizontal_tiles - 1) * overlap) // horizontal_tiles
        
        print(f"Base tile size: {base_tile_height}x{base_tile_width}")
        print(f"Output image size: {output_height}x{output_width}")
        
        # Initialize output tensor and weight tensor
        output = None
        weights = None
        
        # Process each tile
        for v in range(vertical_tiles):
            for h in range(horizontal_tiles):
                # Calculate tile boundaries in latent space
                h_start = h * (base_tile_width - overlap)
                v_start = v * (base_tile_height - overlap)
                
                # Adjust end positions for edge tiles
                h_end = (
                    min(h_start + base_tile_width, width)
                    if h < horizontal_tiles - 1
                    else width
                )
                v_end = (
                    min(v_start + base_tile_height, height)
                    if v < vertical_tiles - 1
                    else height
                )
                
                # Calculate actual tile dimensions
                tile_height = v_end - v_start
                tile_width = h_end - h_start
                
                print(f"Processing tile {v}x{h}: latent ({v_start}:{v_end}, {h_start}:{h_end}) = {tile_height}x{tile_width}")
                
                # Extract tile from latent samples
                tile = latent_samples[:, :, v_start:v_end, h_start:h_end]
                
                # Create tile latents dict
                tile_latents = {"samples": tile}
                
                # Decode the tile
                decoded_tile = vae.decode(tile_latents["samples"])
                
                # Initialize output tensors on first tile
                if output is None:
                    # decoded_tile should be in format [batch, height, width, channels]
                    num_channels = decoded_tile.shape[-1]
                    output = torch.zeros(
                        (batch, output_height, output_width, num_channels),
                        device=decoded_tile.device,
                        dtype=decoded_tile.dtype,
                    )
                    weights = torch.zeros(
                        (batch, output_height, output_width, 1),
                        device=decoded_tile.device,
                        dtype=decoded_tile.dtype,
                    )
                
                # Calculate output tile boundaries
                out_v_start = v_start * scale_factor
                out_v_end = v_end * scale_factor
                out_h_start = h_start * scale_factor
                out_h_end = h_end * scale_factor
                
                # Calculate output tile dimensions
                tile_out_height = out_v_end - out_v_start
                tile_out_width = out_h_end - out_h_start
                
                print(f"  Output position: ({out_v_start}:{out_v_end}, {out_h_start}:{out_h_end}) = {tile_out_height}x{tile_out_width}")
                
                # Create weight mask for this tile
                tile_weights = torch.ones(
                    (batch, tile_out_height, tile_out_width, 1),
                    device=decoded_tile.device,
                    dtype=decoded_tile.dtype,
                )
                
                # Calculate overlap regions in output space
                overlap_out = overlap * scale_factor
                
                # Apply horizontal blending weights
                if h > 0:  # Left overlap
                    h_blend = torch.linspace(0, 1, overlap_out, device=decoded_tile.device)
                    tile_weights[:, :, :overlap_out, :] *= h_blend.view(1, 1, -1, 1)
                if h < horizontal_tiles - 1:  # Right overlap
                    h_blend = torch.linspace(1, 0, overlap_out, device=decoded_tile.device)
                    tile_weights[:, :, -overlap_out:, :] *= h_blend.view(1, 1, -1, 1)
                
                # Apply vertical blending weights
                if v > 0:  # Top overlap
                    v_blend = torch.linspace(0, 1, overlap_out, device=decoded_tile.device)
                    tile_weights[:, :overlap_out, :, :] *= v_blend.view(1, -1, 1, 1)
                if v < vertical_tiles - 1:  # Bottom overlap
                    v_blend = torch.linspace(1, 0, overlap_out, device=decoded_tile.device)
                    tile_weights[:, -overlap_out:, :, :] *= v_blend.view(1, -1, 1, 1)
                
                # Add weighted tile to output
                output[:, out_v_start:out_v_end, out_h_start:out_h_end, :] += (
                    decoded_tile * tile_weights
                )
                
                # Add weights to weight tensor
                weights[:, out_v_start:out_v_end, out_h_start:out_h_end, :] += tile_weights
        
        # Normalise by weights to complete the weighted average
        output = output / (weights + 1e-8)
        
        print("Tiled decode complete, checking for NSFW content...")
        
        # Check each batch item for NSFW content
        for i in range(batch):
            output[i] = self.check_nsfw_on_decoded_image(output[i])
        
        print("NSFW check complete")
        
        return (output,)