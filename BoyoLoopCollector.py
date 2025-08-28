import torch
import numpy as np
import logging
from PIL import Image, ImageDraw, ImageFont
import os
import math

def tensor_to_pil(tensor):
    """Convert tensor to PIL Image"""
    if tensor.dim() == 4:
        # Batch of images, take the first one
        tensor = tensor[0]
    
    # Convert from CHW to HWC if needed
    if tensor.dim() == 3 and tensor.shape[0] in [1, 3, 4]:
        tensor = tensor.permute(1, 2, 0)
    
    # Ensure values are in 0-255 range
    if tensor.max() <= 1.0:
        tensor = tensor * 255.0
    
    tensor = tensor.clamp(0, 255).byte()
    
    # Convert to numpy
    np_image = tensor.cpu().numpy()
    
    if np_image.shape[2] == 1:
        # Grayscale
        np_image = np_image.squeeze(2)
        return Image.fromarray(np_image, mode='L')
    elif np_image.shape[2] == 3:
        # RGB
        return Image.fromarray(np_image, mode='RGB')
    elif np_image.shape[2] == 4:
        # RGBA
        return Image.fromarray(np_image, mode='RGBA')
    else:
        # Fallback - convert to RGB
        np_image = np_image[:, :, :3]
        return Image.fromarray(np_image, mode='RGB')

def pil_to_tensor(pil_image):
    """Convert PIL Image to tensor"""
    np_image = np.array(pil_image).astype(np.float32) / 255.0
    
    if len(np_image.shape) == 2:
        # Grayscale
        np_image = np.expand_dims(np_image, axis=2)
    
    tensor = torch.from_numpy(np_image)
    
    # Add batch dimension
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    return tensor

class BoyoLoopCollector:
    """Collects and aggregates images from bastard loop iterations"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_result": ("BASTARD_LOOP_RESULT",),
                "target_output_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10
                }),
                "grid_columns": ("INT", {
                    "default": 0,  # 0 = auto-calculate
                    "min": 0,
                    "max": 20
                }),
                "spacing": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 100
                }),
                "add_labels": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "background_color": ("STRING", {
                    "default": "#000000"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("grid_image", "individual_images")
    FUNCTION = "collect_loop_images"
    CATEGORY = "boyo/bastardloops"
    OUTPUT_IS_LIST = (False, True)

    def collect_loop_images(self, loop_result, target_output_index, grid_columns, spacing, add_labels, background_color="#000000"):
        """Collect images from loop iterations and create grid"""
        
        if not isinstance(loop_result, dict) or not loop_result.get("success", False):
            logging.error("BoyoLoopCollector: Invalid or failed loop result")
            # Return empty images
            empty_tensor = torch.zeros((1, 512, 512, 3))
            return (empty_tensor, [empty_tensor])
        
        iterations = loop_result.get("iterations", [])
        if not iterations:
            logging.error("BoyoLoopCollector: No iterations in loop result")
            empty_tensor = torch.zeros((1, 512, 512, 3))
            return (empty_tensor, [empty_tensor])
        
        collected_images = []
        collected_tensors = []
        
        # Extract images from each iteration
        for i, iteration_data in enumerate(iterations):
            iteration_outputs = iteration_data.get("outputs", {})
            
            # Find images in the outputs
            images_found = False
            for node_id, node_outputs in iteration_outputs.items():
                if node_outputs and len(node_outputs) > target_output_index:
                    potential_image = node_outputs[target_output_index]
                    
                    if isinstance(potential_image, list) and len(potential_image) > 0:
                        potential_image = potential_image[0]
                    
                    if isinstance(potential_image, torch.Tensor) and potential_image.dim() >= 3:
                        # Found an image tensor
                        collected_tensors.append(potential_image)
                        pil_image = tensor_to_pil(potential_image)
                        collected_images.append(pil_image)
                        images_found = True
                        logging.info(f"BoyoLoopCollector: Collected image from iteration {i+1}, node {node_id}")
                        break
            
            if not images_found:
                logging.warning(f"BoyoLoopCollector: No image found in iteration {i+1}")
                # Create placeholder image
                placeholder = Image.new('RGB', (512, 512), color='gray')
                draw = ImageDraw.Draw(placeholder)
                draw.text((256, 256), f"No Image\nIteration {i+1}", fill='white', anchor='mm')
                collected_images.append(placeholder)
                collected_tensors.append(pil_to_tensor(placeholder))
        
        if not collected_images:
            logging.error("BoyoLoopCollector: No images collected from any iteration")
            empty_tensor = torch.zeros((1, 512, 512, 3))
            return (empty_tensor, [empty_tensor])
        
        # Create grid
        grid_image = self.create_image_grid(collected_images, grid_columns, spacing, add_labels, background_color)
        grid_tensor = pil_to_tensor(grid_image)
        
        logging.info(f"BoyoLoopCollector: Created grid with {len(collected_images)} images")
        
        return (grid_tensor, collected_tensors)

    def create_image_grid(self, images, grid_columns, spacing, add_labels, background_color):
        """Create a grid from collected images"""
        
        if not images:
            return Image.new('RGB', (512, 512), color='gray')
        
        num_images = len(images)
        
        # Calculate grid dimensions
        if grid_columns == 0:
            # Auto-calculate columns (roughly square grid)
            grid_columns = math.ceil(math.sqrt(num_images))
        
        grid_rows = math.ceil(num_images / grid_columns)
        
        # Get image dimensions (assume all images are the same size)
        img_width, img_height = images[0].size
        
        # Calculate label height
        label_height = 30 if add_labels else 0
        
        # Calculate grid dimensions
        total_width = grid_columns * img_width + (grid_columns - 1) * spacing
        total_height = grid_rows * (img_height + label_height) + (grid_rows - 1) * spacing
        
        # Create grid image
        try:
            grid_img = Image.new('RGB', (total_width, total_height), color=background_color)
        except:
            # Fallback to black if color parsing fails
            grid_img = Image.new('RGB', (total_width, total_height), color='black')
        
        # Place images in grid
        for i, img in enumerate(images):
            row = i // grid_columns
            col = i % grid_columns
            
            x = col * (img_width + spacing)
            y = row * (img_height + label_height + spacing)
            
            # Paste image
            grid_img.paste(img, (x, y))
            
            # Add label if requested
            if add_labels:
                draw = ImageDraw.Draw(grid_img)
                label_text = f"#{i+1}"
                
                # Try to use a font, fallback to default if not available
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    try:
                        font = ImageFont.load_default()
                    except:
                        font = None
                
                label_y = y + img_height + 5
                draw.text((x + img_width//2, label_y), label_text, fill='white', anchor='mt', font=font)
        
        return grid_img

class BoyoLoopImageSaver:
    """Saves individual images from loop iterations"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "bastard_loop"}),
                "save_grid": ("BOOLEAN", {"default": True}),
                "save_individuals": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("save_path",)
    FUNCTION = "save_loop_images"
    CATEGORY = "boyo/bastardloops"
    OUTPUT_NODE = True

    def save_loop_images(self, images, filename_prefix, save_grid, save_individuals):
        """Save loop images to disk"""
        
        # Get ComfyUI output directory
        from folder_paths import get_output_directory
        output_dir = get_output_directory()
        
        saved_files = []
        
        if isinstance(images, list):
            # Multiple images (individual images from collector)
            if save_individuals:
                for i, img_tensor in enumerate(images):
                    pil_img = tensor_to_pil(img_tensor)
                    filename = f"{filename_prefix}_{i+1:03d}.png"
                    filepath = os.path.join(output_dir, filename)
                    pil_img.save(filepath)
                    saved_files.append(filename)
                    logging.info(f"BoyoLoopImageSaver: Saved {filename}")
        else:
            # Single image (grid from collector)
            if save_grid:
                pil_img = tensor_to_pil(images)
                filename = f"{filename_prefix}_grid.png"
                filepath = os.path.join(output_dir, filename)
                pil_img.save(filepath)
                saved_files.append(filename)
                logging.info(f"BoyoLoopImageSaver: Saved {filename}")
        
        if saved_files:
            return (f"Saved {len(saved_files)} files: {', '.join(saved_files)}",)
        else:
            return ("No files saved",)

# Node mappings
NODE_CLASS_MAPPINGS = {
    "BoyoLoopCollector": BoyoLoopCollector,
    "BoyoLoopImageSaver": BoyoLoopImageSaver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoLoopCollector": "Boyo Loop Collector",
    "BoyoLoopImageSaver": "Boyo Loop Image Saver"
}