import os
import torch
import re
import json
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import folder_paths


def tensor_to_pil(image_tensor, batch_index=0) -> Image:
    """Convert tensor of shape [batch, height, width, channels] to PIL Image"""
    image_tensor = image_tensor[batch_index].unsqueeze(0)
    i = 255.0 * image_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img


def pil_to_tensor(image: Image) -> torch.Tensor:
    """Convert PIL Image to tensor of shape [1, height, width, channels]"""
    image_np = np.array(image).astype(np.float32) / 255.0
    if len(image_np.shape) == 2:  # Greyscale
        image_np = np.expand_dims(image_np, axis=-1)
        image_np = np.repeat(image_np, 3, axis=-1)
    return torch.from_numpy(image_np).unsqueeze(0)


class BoyoQwenVLGrounding:
    def __init__(self):
        self.model_checkpoint = None
        self.processor = None
        self.model = None
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.bf16_support = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "text_input": ("STRING", {
                    "default": "person", 
                    "multiline": False,
                    "tooltip": "Object or region to find and ground (e.g., 'bikini top', 'eyes', 'person')"
                }),
                "model": (
                    [
                        "Qwen2.5-VL-7B-Instruct-abliterated",
                        "Qwen2.5-VL-3B-Instruct",
                        "Qwen2.5-VL-7B-Instruct",
                    ],
                    {"default": "Qwen2.5-VL-7B-Instruct-abliterated"},
                ),
                "prompt_style": (
                    [
                        "Find {object} with grounding",
                        "Locate {object} with bounding boxes",
                        "Describe the image in detail with grounding",
                        "Where is {object}? Provide bounding boxes.",
                        "Detect {object} and return coordinates",
                    ],
                    {"default": "Find {object} with grounding"},
                ),
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "none"},
                ),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 512, "min": 128, "max": 2048, "step": 1},
                ),
                "seed": ("INT", {"default": -1}),
                "draw_boxes": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Draw bounding boxes on output image for visual verification"
                }),
                "debug_mode": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Print extensive debugging information"
                }),
            },
        }

    RETURN_TYPES = ("BBOX", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("bboxes", "coordinates", "raw_response", "annotated_image")
    FUNCTION = "inference"
    CATEGORY = "BoyoNodes/QwenVL"

    def parse_json_grounding(self, response_text, debug=False):
        """
        Parse JSON format grounding response from Qwen models.
        Expected format: [{"bbox_2d": [x1, y1, x2, y2], "label": "object"}, ...]
        """
        bboxes = []
        labels = []
        
        if debug:
            print("\n" + "="*80)
            print("PARSING JSON GROUNDING RESPONSE")
            print("="*80)
        
        try:
            # Try to extract JSON from code blocks if present
            json_pattern = r'```json\s*(.*?)\s*```'
            json_match = re.search(json_pattern, response_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                if debug:
                    print("Found JSON in code block:")
                    print(json_str)
            else:
                # Try to find JSON array or object directly
                # Look for array pattern
                array_pattern = r'\[\s*\{.*?\}\s*\]'
                array_match = re.search(array_pattern, response_text, re.DOTALL)
                if array_match:
                    json_str = array_match.group(0)
                    if debug:
                        print("Found JSON array:")
                        print(json_str)
                else:
                    if debug:
                        print("No JSON format detected in response")
                    return bboxes, labels
            
            # Parse the JSON
            data = json.loads(json_str)
            
            if debug:
                print(f"\nParsed JSON successfully. Found {len(data)} entries")
            
            # Extract bboxes and labels
            for idx, item in enumerate(data):
                if "bbox_2d" in item and "label" in item:
                    bbox = item["bbox_2d"]
                    label = item["label"]
                    
                    if len(bbox) == 4:
                        # Coordinates are already in pixel space, just convert to float
                        bboxes.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])
                        labels.append(label)
                        
                        if debug:
                            print(f"  {idx}. {label}: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
                    else:
                        if debug:
                            print(f"  Warning: bbox has wrong format: {bbox}")
                else:
                    if debug:
                        print(f"  Warning: item missing bbox_2d or label: {item}")
            
        except json.JSONDecodeError as e:
            if debug:
                print(f"JSON parsing error: {e}")
        except Exception as e:
            if debug:
                print(f"Error parsing JSON grounding: {e}")
        
        return bboxes, labels

    def parse_token_grounding(self, response_text, image_width, image_height, debug=False):
        """
        Parse special token format grounding response.
        Tries multiple token formats as different models may use different tokens.
        """
        bboxes = []
        labels = []
        
        if debug:
            print("\n" + "="*80)
            print("PARSING TOKEN-BASED GROUNDING RESPONSE")
            print("="*80)
        
        # Try Format 1: <|object_ref_start|>...<|object_ref_end|> and <|box_start|>...<|box_end|>
        label_pattern_1 = r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|>'
        bbox_pattern_1 = r'<\|box_start\|>(.*?)<\|box_end\|>'
        
        # Try Format 2: <|extra_0|>...<|extra_1|> and <|extra_2|>...<|extra_3|>
        label_pattern_2 = r'<\|extra_0\|>(.*?)<\|extra_1\|>'
        bbox_pattern_2 = r'<\|extra_2\|>(.*?)<\|extra_3\|>'
        
        # Try each format
        for fmt_num, (label_pattern, bbox_pattern) in enumerate([
            (label_pattern_1, bbox_pattern_1),
            (label_pattern_2, bbox_pattern_2)
        ], 1):
            label_matches = re.findall(label_pattern, response_text)
            bbox_matches = re.findall(bbox_pattern, response_text)
            
            if debug:
                print(f"\nFormat {fmt_num} attempt:")
                print(f"  Label pattern: {label_pattern}")
                print(f"  Bbox pattern: {bbox_pattern}")
                print(f"  Labels found: {len(label_matches)} - {label_matches}")
                print(f"  Bboxes found: {len(bbox_matches)} - {bbox_matches}")
            
            if bbox_matches:  # Found something!
                if debug:
                    print(f"  ✓ Format {fmt_num} matched!")
                
                for bbox_str in bbox_matches:
                    try:
                        # Parse coordinates - format should be "x1,y1,x2,y2"
                        coords = bbox_str.strip().replace('(', '').replace(')', '').split(',')
                        if len(coords) == 4:
                            # Convert from [0-1000] normalized to pixel coordinates
                            x1 = float(coords[0]) / 1000.0 * image_width
                            y1 = float(coords[1]) / 1000.0 * image_height
                            x2 = float(coords[2]) / 1000.0 * image_width
                            y2 = float(coords[3]) / 1000.0 * image_height
                            
                            bboxes.append([x1, y1, x2, y2])
                            if debug:
                                print(f"  Parsed bbox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                    except Exception as e:
                        if debug:
                            print(f"  Error parsing bbox '{bbox_str}': {e}")
                        continue
                
                # Match labels to bboxes
                labels = label_matches[:len(bboxes)]
                break  # Found valid format, stop trying others
        
        return bboxes, labels

    def parse_grounding_response(self, response_text, image_width, image_height, debug=False):
        """
        Main parsing function that tries multiple formats.
        Priority: JSON format > Special tokens > Fallback
        """
        if debug:
            print("\n" + "="*80)
            print("PARSING GROUNDING RESPONSE")
            print("="*80)
            print(f"Raw response length: {len(response_text)} characters")
            print(f"Image dimensions: {image_width}x{image_height}")
            print("\nFull raw response:")
            print(response_text)
            print("="*80)
        
        # Try JSON format first (most common for Qwen models)
        bboxes, labels = self.parse_json_grounding(response_text, debug=debug)
        
        # If JSON parsing didn't work, try special tokens
        if not bboxes:
            if debug:
                print("\nJSON parsing found nothing, trying token-based parsing...")
            bboxes, labels = self.parse_token_grounding(response_text, image_width, image_height, debug=debug)
        
        if debug:
            print(f"\n{'='*80}")
            print(f"PARSING COMPLETE")
            print(f"Total bboxes extracted: {len(bboxes)}")
            print(f"Total labels extracted: {len(labels)}")
            print(f"{'='*80}\n")
        
        return bboxes, labels

    def draw_bounding_boxes(self, image_pil, bboxes, labels):
        """Draw bounding boxes on image for visualization"""
        draw = ImageDraw.Draw(image_pil)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
        
        for idx, (bbox, label) in enumerate(zip(bboxes, labels)):
            color = colors[idx % len(colors)]
            x1, y1, x2, y2 = bbox
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label
            label_text = f"{idx}.{label}"
            text_bbox = draw.textbbox((x1, y1), label_text, font=font)
            text_bg = [text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2]
            draw.rectangle(text_bg, fill=color)
            draw.text((x1, y1), label_text, fill='white', font=font)
        
        return image_pil

    def inference(
        self,
        image,
        text_input,
        model,
        prompt_style,
        quantization,
        keep_model_loaded,
        temperature,
        max_new_tokens,
        seed,
        draw_boxes,
        debug_mode,
    ):
        if seed != -1:
            torch.manual_seed(seed)
        
        # Handle different model sources
        if "abliterated" in model:
            model_id = f"huihui-ai/{model}"
        else:
            model_id = f"qwen/{model}"
        
        self.model_checkpoint = os.path.join(
            folder_paths.models_dir, "LLM", os.path.basename(model_id)
        )

        if debug_mode:
            print("\n" + "="*80)
            print("BOYO QWEN VL GROUNDING - DEBUG MODE")
            print("="*80)
            print(f"Model ID: {model_id}")
            print(f"Model checkpoint path: {self.model_checkpoint}")
            print(f"Text input: {text_input}")
            print(f"Prompt style: {prompt_style}")
            print(f"="*80 + "\n")

        # Download model if needed
        if not os.path.exists(self.model_checkpoint):
            print(f"Model not found locally. Downloading from {model_id}...")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=model_id,
                local_dir=self.model_checkpoint,
                local_dir_use_symlinks=False,
            )
            print(f"Model downloaded to {self.model_checkpoint}")

        # Load processor if needed
        if self.processor is None:
            if debug_mode:
                print("Loading processor...")
            min_pixels = 256*28*28
            max_pixels = 1024*28*28 
            self.processor = AutoProcessor.from_pretrained(
                self.model_checkpoint,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            if debug_mode:
                print("Processor loaded successfully")

        # Load model if needed
        if self.model is None:
            if debug_mode:
                print(f"Loading model with quantization: {quantization}...")
            
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                quantization_config = None

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_checkpoint,
                torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map="auto",
                quantization_config=quantization_config,
            )
            if debug_mode:
                print("Model loaded successfully")

        # Convert tensor to PIL
        pil_image = tensor_to_pil(image)
        image_width, image_height = pil_image.size
        
        if debug_mode:
            print(f"\nImage dimensions: {image_width}x{image_height}")
        
        # Format the grounding prompt using the selected style
        if "{object}" in prompt_style:
            grounding_prompt = prompt_style.replace("{object}", text_input)
        else:
            grounding_prompt = prompt_style
        
        if debug_mode:
            print(f"Formatted grounding prompt: '{grounding_prompt}'")

        with torch.no_grad():
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": grounding_prompt},
                    ],
                }
            ]

            # Prepare input
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            if debug_mode:
                print("\nApplied chat template:")
                print(text[:500] + "..." if len(text) > 500 else text)
            
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            if debug_mode:
                print("\nGenerating response...")

            # Generate
            try:
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] 
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                # Decode with special tokens preserved
                result_with_tokens = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                raw_response = result_with_tokens[0] if result_with_tokens else ""
                
                if debug_mode:
                    print("\n" + "="*80)
                    print("MODEL RAW OUTPUT (with special tokens):")
                    print("="*80)
                    print(raw_response)
                    print("="*80)
                
            except Exception as e:
                error_msg = f"Error during model inference: {str(e)}"
                print(error_msg)
                if debug_mode:
                    import traceback
                    traceback.print_exc()
                return ([], "", error_msg, image)

            # Parse bounding boxes
            bboxes, labels = self.parse_grounding_response(
                raw_response, image_width, image_height, debug=debug_mode
            )
            
            # Format coordinates string
            if len(bboxes) == 0:
                coords_str = "No bounding boxes detected"
                if debug_mode:
                    print("\n⚠️ WARNING: No bounding boxes were extracted from the response")
                    print("This could mean:")
                    print("  1. The model didn't produce grounding output")
                    print("  2. The prompt format isn't triggering grounding mode")
                    print("  3. Try a different prompt style")
            else:
                coords_str = ""
                for idx, (bbox, label) in enumerate(zip(bboxes, labels)):
                    coords_str += f"{idx}. {label}: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]\n"
                if debug_mode:
                    print(f"\n✓ Successfully extracted {len(bboxes)} bounding boxes")
            
            # Draw boxes if requested
            if draw_boxes and len(bboxes) > 0:
                annotated_pil = pil_image.copy()
                annotated_pil = self.draw_bounding_boxes(annotated_pil, bboxes, labels)
                annotated_tensor = pil_to_tensor(annotated_pil)
            else:
                annotated_tensor = image

            # Clean up if not keeping model loaded
            if not keep_model_loaded:
                if debug_mode:
                    print("\nUnloading model...")
                del self.processor
                del self.model
                self.processor = None
                self.model = None
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            return (bboxes, coords_str, raw_response, annotated_tensor)


# Node registration
NODE_CLASS_MAPPINGS = {
    "BoyoQwenVLGrounding": BoyoQwenVLGrounding,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoQwenVLGrounding": "Boyo Qwen VL Grounding",
}