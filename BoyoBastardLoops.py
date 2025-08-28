import nodes
import torch
import comfy.model_management
import copy
import logging
import sys
import traceback
import uuid
import asyncio
from execution import full_type_name
from collections import OrderedDict
import execution

def get_input_data(inputs, class_def, unique_id, outputs={}, prompt={}, extra_data={}):
    """Extract and prepare input data for node execution"""
    valid_inputs = class_def.INPUT_TYPES()
    input_data_all = {}
    
    for x in inputs:
        input_data = inputs[x]
        if isinstance(input_data, list):
            input_unique_id = input_data[0]
            output_index = input_data[1]
            if input_unique_id not in outputs:
                input_data_all[x] = (None,)
                continue
            obj = outputs[input_unique_id][output_index]
            input_data_all[x] = obj
        else:
            if ("required" in valid_inputs and x in valid_inputs["required"]) or ("optional" in valid_inputs and x in valid_inputs["optional"]):
                input_data_all[x] = [input_data]

    if "hidden" in valid_inputs:
        h = valid_inputs["hidden"]
        for x in h:
            if h[x] == "PROMPT":
                input_data_all[x] = [prompt]
            if h[x] == "EXTRA_PNGINFO":
                input_data_all[x] = [extra_data.get('extra_pnginfo', None)]
            if h[x] == "UNIQUE_ID":
                input_data_all[x] = [unique_id]
    
    return input_data_all

def get_output_data(obj, input_data_all):
    """Execute node and collect outputs"""
    results = []
    uis = []
    return_values = map_node_over_list(obj, input_data_all, obj.FUNCTION, allow_interrupt=True)

    for r in return_values:
        if isinstance(r, dict):
            if 'ui' in r:
                uis.append(r['ui'])
            if 'result' in r:
                results.append(r['result'])
        else:
            results.append(r)

    output = []
    if len(results) > 0:
        # Check which outputs need concatenating
        output_is_list = [False] * len(results[0])
        if hasattr(obj, "OUTPUT_IS_LIST"):
            output_is_list = obj.OUTPUT_IS_LIST

        # Merge node execution results
        for i, is_list in zip(range(len(results[0])), output_is_list):
            if is_list:
                output.append([x for o in results for x in o[i]])
            else:
                output.append([o[i] for o in results])

    ui = dict()    
    if len(uis) > 0:
        ui = {k: [y for x in uis for y in x[k]] for k in uis[0].keys()}
    
    return output, ui

def map_node_over_list(obj, input_data_all, func, allow_interrupt=False):
    """Map node execution over input lists"""
    # Check if node wants the lists
    input_is_list = False
    if hasattr(obj, "INPUT_IS_LIST"):
        input_is_list = obj.INPUT_IS_LIST

    if len(input_data_all) == 0:
        max_len_input = 0
    else:
        max_len_input = max([len(x) for x in input_data_all.values()])
    
    # Get a slice of inputs, repeat last input when list isn't long enough
    def slice_dict(d, i):
        d_new = dict()
        for k,v in d.items():
            d_new[k] = v[i if len(v) > i else -1]
        return d_new
    
    results = []
    if input_is_list:
        if allow_interrupt:
            nodes.before_node_execution()
        results.append(getattr(obj, func)(**input_data_all))
    elif max_len_input == 0:
        if allow_interrupt:
            nodes.before_node_execution()
        results.append(getattr(obj, func)())
    else:
        for i in range(max_len_input):
            if allow_interrupt:
                nodes.before_node_execution()
            results.append(getattr(obj, func)(**slice_dict(input_data_all, i)))
    
    return results

def format_value(x):
    """Format values for error reporting"""
    if x is None:
        return None
    elif isinstance(x, (int, float, bool, str)):
        return x
    else:
        return str(x)

def recursive_execute(prompt, outputs, current_item, extra_data, executed, prompt_id, outputs_ui, object_storage):
    """Recursively execute nodes in dependency order"""
    unique_id = current_item
    inputs = prompt[unique_id]['inputs']
    class_type = prompt[unique_id]['class_type']
    class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
    
    if unique_id in outputs:
        return (True, None, None)

    # Execute dependencies first
    for x in inputs:
        input_data = inputs[x]
        if isinstance(input_data, list):
            input_unique_id = input_data[0]
            output_index = input_data[1]
            if input_unique_id not in outputs:
                result = recursive_execute(prompt, outputs, input_unique_id, extra_data, executed, prompt_id, outputs_ui, object_storage)
                if result[0] is not True:
                    return result

    input_data_all = None
    try:
        input_data_all = get_input_data(inputs, class_def, unique_id, outputs, prompt, extra_data)

        obj = object_storage.get((unique_id, class_type), None)
        if obj is None:
            obj = class_def()
            object_storage[(unique_id, class_type)] = obj

        output_data, output_ui = get_output_data(obj, input_data_all)
        outputs[unique_id] = output_data
        if len(output_ui) > 0:
            outputs_ui[unique_id] = output_ui

    except comfy.model_management.InterruptProcessingException as iex:
        logging.info("Processing interrupted")
        error_details = {"node_id": unique_id}
        return (False, error_details, iex)
    
    except Exception as ex:
        typ, _, tb = sys.exc_info()
        exception_type = full_type_name(typ)
        input_data_formatted = {}
        if input_data_all is not None:
            input_data_formatted = {}
            for name, inputs in input_data_all.items():
                input_data_formatted[name] = [format_value(x) for x in inputs]

        output_data_formatted = {}
        for node_id, node_outputs in outputs.items():
            output_data_formatted[node_id] = [[format_value(x) for x in l] for l in node_outputs]

        logging.error(f"!!! Exception during BoyoBastardLoops processing!!! {ex}")
        logging.error(traceback.format_exc())

        error_details = {
            "node_id": unique_id,
            "exception_message": str(ex),
            "exception_type": exception_type,
            "traceback": traceback.format_tb(tb),
            "current_inputs": input_data_formatted,
            "current_outputs": output_data_formatted
        }
        return (False, error_details, ex)

    executed.add(unique_id)
    return (True, None, None)

def recursive_will_execute(prompt, outputs, current_item, memo={}):
    """Determine which nodes will be executed"""
    unique_id = current_item

    if unique_id in memo:
        return memo[unique_id]

    inputs = prompt[unique_id]['inputs']
    will_execute = []
    if unique_id in outputs:
        return []

    for x in inputs:
        input_data = inputs[x]
        if isinstance(input_data, list):
            input_unique_id = input_data[0]
            output_index = input_data[1]
            if input_unique_id not in outputs:
                will_execute += recursive_will_execute(prompt, outputs, input_unique_id, memo)

    memo[unique_id] = will_execute + [unique_id]
    return memo[unique_id]

class BoyoBastardExecutor:
    """The bastardized execution engine for looping workflows (copied from ttN)"""
    
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the executor state"""
        self.outputs = {}
        self.object_storage = {}
        self.outputs_ui = {}
        self.status_messages = []
        self.success = True
        self.old_prompt = {}

    def add_message(self, event, data, broadcast: bool = False):
        """Add status message"""
        self.status_messages.append((event, data))

    def handle_execution_error(self, prompt_id, prompt, current_outputs, executed, error, ex):
        """Handle execution errors"""
        node_id = error["node_id"]
        class_type = prompt[node_id]["class_type"]

        if isinstance(ex, comfy.model_management.InterruptProcessingException):
            mes = {
                "prompt_id": prompt_id,
                "node_id": node_id,
                "node_type": class_type,
                "executed": list(executed),
            }
            self.add_message("execution_interrupted", mes, broadcast=True)
        else:
            mes = {
                "prompt_id": prompt_id,
                "node_id": node_id,
                "node_type": class_type,
                "executed": list(executed),
                "exception_message": error["exception_message"],
                "exception_type": error["exception_type"],
                "traceback": error["traceback"],
                "current_inputs": error["current_inputs"],
                "current_outputs": error["current_outputs"],
            }
            self.add_message("execution_error", mes, broadcast=False)
        
        # Clean up subsequent outputs
        to_delete = []
        for o in self.outputs:
            if (o not in current_outputs) and (o not in executed):
                to_delete += [o]
                if o in self.old_prompt:
                    d = self.old_prompt.pop(o)
                    del d
        for o in to_delete:
            d = self.outputs.pop(o)
            del d
        
        raise Exception(ex)

    def execute(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
        """Execute using ttN's exact method"""
        nodes.interrupt_processing(False)

        self.status_messages = []
        self.add_message("execution_start", {"prompt_id": prompt_id}, broadcast=False)

        with torch.inference_mode():
            # Delete cached outputs if nodes don't exist for them
            to_delete = []
            for o in self.outputs:
                if o not in prompt:
                    to_delete += [o]
            for o in to_delete:
                d = self.outputs.pop(o)
                del d
            
            to_delete = []
            for o in self.object_storage:
                if o[0] not in prompt:
                    to_delete += [o]
                else:
                    p = prompt[o[0]]
                    if o[1] != p['class_type']:
                        to_delete += [o]
            for o in to_delete:
                d = self.object_storage.pop(o)
                del d

            current_outputs = set(self.outputs.keys())
            comfy.model_management.cleanup_models()
            
            self.add_message("execution_cached", {"nodes": list(current_outputs), "prompt_id": prompt_id}, broadcast=False)
            
            executed = set()
            to_execute = []

            for node_id in list(execute_outputs):
                to_execute += [(0, node_id)]

            while len(to_execute) > 0:
                # Always execute the output that depends on the least amount of unexecuted nodes first
                memo = {}
                to_execute = sorted(list(map(lambda a: (len(recursive_will_execute(prompt, self.outputs, a[-1], memo)), a[-1]), to_execute)))
                output_node_id = to_execute.pop(0)[-1]

                # Execute the node
                self.success, error, ex = recursive_execute(prompt, self.outputs, output_node_id, extra_data, executed, prompt_id, self.outputs_ui, self.object_storage)
                if self.success is not True:
                    self.handle_execution_error(prompt_id, prompt, current_outputs, executed, error, ex)
                    break

            for x in executed:
                self.old_prompt[x] = copy.deepcopy(prompt[x])

            if comfy.model_management.DISABLE_SMART_MEMORY:
                comfy.model_management.unload_all_models()

class BoyoChainBastardLoops:
    """End-of-chain bastard loops that pull the entire workflow"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # This connects to VAE Decode output!
                "loop_count": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
            },
            "optional": {
                "prompt_config": ("PROMPT_LOOP_CONFIG",),
            },
            "hidden": {
                "prompt": ("PROMPT",),
                "extra_pnginfo": ("EXTRA_PNGINFO",),
                "unique_id": ("UNIQUE_ID",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("grid_image", "individual_images")
    FUNCTION = "chain_bastard_loop"
    CATEGORY = "boyo/bastardloops"
    OUTPUT_IS_LIST = (False, True)

    @staticmethod
    def _get_nodes_to_keep(nodeID, prompt):
        """Get all nodes needed to execute the target node (copied from ttN)"""
        nodes_to_keep = OrderedDict([(nodeID, None)])
        toCheck = [nodeID]

        while toCheck:
            current_node_id = toCheck.pop()
            current_node = prompt[current_node_id]

            for input_key in current_node["inputs"]:
                value = current_node["inputs"][input_key]

                if isinstance(value, list) and len(value) == 2:
                    input_node_id = value[0]

                    if input_node_id not in nodes_to_keep:
                        nodes_to_keep[input_node_id] = None
                        toCheck.append(input_node_id)

        return list(reversed(list(nodes_to_keep.keys())))

    def get_relevant_prompt(self, prompt, unique_id):
        """Get a clean workflow with only the nodes needed for our target (copied from ttN)"""
        nodes_to_keep = self._get_nodes_to_keep(unique_id, prompt)
        new_prompt = {node_id: prompt[node_id] for node_id in nodes_to_keep if node_id != unique_id}  # Exclude ourselves
        return new_prompt

    def update_prompt_with_text(self, prompt, new_prompt_text):
        """Update CLIPTextEncode nodes with new prompt text"""
        updated_count = 0
        for node_id, node_data in prompt.items():
            if node_data.get("class_type") == "CLIPTextEncode":
                if "inputs" in node_data and "text" in node_data["inputs"]:
                    logging.info(f"🔥 BoyoChainBastardLoops: Updating CLIPTextEncode node {node_id}")
                    logging.info(f"  Old: '{node_data['inputs']['text'][:50]}...'")
                    node_data["inputs"]["text"] = new_prompt_text
                    logging.info(f"  New: '{new_prompt_text[:50]}...'")
                    updated_count += 1
        
        logging.info(f"🔥 BoyoChainBastardLoops: Updated {updated_count} CLIPTextEncode nodes")
        return prompt

    def execute_prompt(self, prompt, extra_data, iteration_prompt, iteration_num, vae_decode_node_id):
        """Execute a single iteration with a specific prompt (using ttN's method)"""
        prompt_id = uuid.uuid4()

        # Try to get the current event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # Already inside an event loop
            import threading
            result_container = {}

            def run_coroutine():
                coro = execution.validate_prompt(prompt_id, prompt, None)
                result_container["result"] = asyncio.run(coro)

            thread = threading.Thread(target=run_coroutine)
            thread.start()
            thread.join()
            valid = result_container["result"]
        else:
            # Safe to run directly
            valid = loop.run_until_complete(execution.validate_prompt(prompt_id, prompt, None))
        
        if valid[0]:
            logging.info(f'🔥 BoyoChainBastardLoops: Iteration {iteration_num} -> "{iteration_prompt[:50]}..."')

            # Create executor and execute!
            executor = BoyoBastardExecutor()
            executor.execute(prompt, prompt_id, extra_data, valid[2])

            # Extract the image from VAE Decode node
            if vae_decode_node_id in executor.outputs:
                outputs = executor.outputs[vae_decode_node_id]
                if outputs and len(outputs) > 0:
                    image_tensor = outputs[0]  # First output should be the image
                    logging.info(f"🔥 BoyoChainBastardLoops: Successfully collected image from iteration {iteration_num}")
                    return image_tensor
                else:
                    logging.warning(f"🔥 BoyoChainBastardLoops: No outputs from VAE Decode node {vae_decode_node_id}")
            else:
                logging.warning(f"🔥 BoyoChainBastardLoops: VAE Decode node {vae_decode_node_id} not found in outputs")
                
        else:
            raise Exception(f"Prompt validation failed: {valid[1]}")
        
        return None

    def find_vae_decode_node(self, prompt, unique_id):
        """Find the VAE Decode node that feeds into our images input"""
        our_node = prompt[unique_id]
        images_input = our_node["inputs"]["images"]
        
        if isinstance(images_input, list) and len(images_input) == 2:
            vae_decode_node_id = images_input[0]
            logging.info(f"🔥 BoyoChainBastardLoops: Found VAE Decode node: {vae_decode_node_id}")
            return vae_decode_node_id
        
        logging.error("🔥 BoyoChainBastardLoops: Could not find VAE Decode node")
        return None

    def tensor_to_pil(self, tensor):
        """Convert tensor to PIL Image"""
        import numpy as np
        from PIL import Image
        
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
        else:
            # Fallback - convert to RGB
            np_image = np_image[:, :, :3]
            return Image.fromarray(np_image, mode='RGB')

    def pil_to_tensor(self, pil_image):
        """Convert PIL Image to tensor"""
        import numpy as np
        
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        
        if len(np_image.shape) == 2:
            # Grayscale
            np_image = np.expand_dims(np_image, axis=2)
        
        tensor = torch.from_numpy(np_image)
        
        # Add batch dimension
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        return tensor

    def create_image_grid(self, pil_images, spacing=10):
        """Create a grid from PIL images"""
        import math
        from PIL import Image, ImageDraw, ImageFont
        
        if not pil_images:
            return Image.new('RGB', (512, 512), color='gray')
        
        num_images = len(pil_images)
        
        # Calculate grid dimensions (roughly square)
        grid_columns = math.ceil(math.sqrt(num_images))
        grid_rows = math.ceil(num_images / grid_columns)
        
        # Get image dimensions (assume all images are the same size)
        img_width, img_height = pil_images[0].size
        
        # Calculate grid dimensions
        total_width = grid_columns * img_width + (grid_columns - 1) * spacing
        total_height = grid_rows * img_height + (grid_rows - 1) * spacing
        
        # Create grid image
        grid_img = Image.new('RGB', (total_width, total_height), color='black')
        
        # Place images in grid
        for i, img in enumerate(pil_images):
            row = i // grid_columns
            col = i % grid_columns
            
            x = col * (img_width + spacing)
            y = row * (img_height + spacing)
            
            grid_img.paste(img, (x, y))
        
        return grid_img

    def chain_bastard_loop(self, images, loop_count, prompt_config=None, prompt=None, extra_pnginfo=None, unique_id=None):
        """Execute the chain bastard loop - sitting at the end of the chain like ttN XY Plot"""
        
        logging.info(f"🔥 BoyoChainBastardLoops: Starting chain bastard loop!")
        logging.info(f"🔥 BoyoChainBastardLoops: Received {images.shape[0]} input images")
        
        # Try to get workflow data from multiple sources
        workflow_prompt = None
        
        if prompt is not None:
            workflow_prompt = prompt
            logging.info(f"🔥 BoyoChainBastardLoops: Got workflow from hidden parameter")
        else:
            # Emergency hack - since we're at the end of the chain, the workflow just executed
            # Let's try to get it from ComfyUI's execution context
            try:
                # Try server instance
                import server
                if hasattr(server, 'PromptServer') and hasattr(server.PromptServer, 'instance'):
                    if hasattr(server.PromptServer.instance, 'last_prompt'):
                        workflow_prompt = server.PromptServer.instance.last_prompt
                        logging.info(f"🔥 BoyoChainBastardLoops: Got workflow from PromptServer.last_prompt")
            except Exception as e:
                logging.warning(f"🔥 BoyoChainBastardLoops: PromptServer attempt failed: {e}")
            
            if workflow_prompt is None:
                # Try execution module globals
                try:
                    import execution
                    # Look for current_prompt in execution module
                    for attr_name in dir(execution):
                        attr_value = getattr(execution, attr_name)
                        if isinstance(attr_value, dict) and unique_id in attr_value:
                            workflow_prompt = attr_value
                            logging.info(f"🔥 BoyoChainBastardLoops: Got workflow from execution.{attr_name}")
                            break
                except Exception as e:
                    logging.warning(f"🔥 BoyoChainBastardLoops: Execution module scan failed: {e}")
            
            if workflow_prompt is None:
                # Desperate frame scanning
                try:
                    import sys
                    import inspect
                    
                    # Look through the call stack for workflow data
                    for frame_info in inspect.stack():
                        frame_locals = frame_info.frame.f_locals
                        frame_globals = frame_info.frame.f_globals
                        
                        # Check locals for 'prompt' that contains our unique_id
                        if 'prompt' in frame_locals:
                            potential_prompt = frame_locals['prompt']
                            if isinstance(potential_prompt, dict) and unique_id in potential_prompt:
                                workflow_prompt = potential_prompt
                                logging.info(f"🔥 BoyoChainBastardLoops: Got workflow from frame locals in {frame_info.function}")
                                break
                        
                        # Check globals too
                        if 'prompt' in frame_globals:
                            potential_prompt = frame_globals['prompt']
                            if isinstance(potential_prompt, dict) and unique_id in potential_prompt:
                                workflow_prompt = potential_prompt
                                logging.info(f"🔥 BoyoChainBastardLoops: Got workflow from frame globals in {frame_info.function}")
                                break
                except Exception as e:
                    logging.warning(f"🔥 BoyoChainBastardLoops: Frame scanning failed: {e}")
        
        if workflow_prompt is None:
            logging.error("🔥 BoyoChainBastardLoops: COULD NOT GET WORKFLOW DATA FROM ANY SOURCE!")
            logging.error("🔥 BoyoChainBastardLoops: Will return the original image as fallback")
            return (images, [images[0]])
        
        logging.info(f"🔥 BoyoChainBastardLoops: Got workflow with {len(workflow_prompt)} nodes")
        
        # Find the VAE Decode node that feeds us
        vae_decode_node_id = self.find_vae_decode_node(workflow_prompt, unique_id)
        if not vae_decode_node_id:
            logging.error("🔥 BoyoChainBastardLoops: Could not find VAE Decode node - returning original image")
            return (images, [images[0]])
        
        # Get prompts for iterations
        iteration_prompts = []
        if prompt_config and isinstance(prompt_config, dict) and "prompts" in prompt_config:
            iteration_prompts = prompt_config["prompts"][:loop_count]
            logging.info(f"🔥 BoyoChainBastardLoops: Using {len(iteration_prompts)} prompts from config")
        else:
            iteration_prompts = [f"a beautiful image (iteration {i+1})" for i in range(loop_count)]
            logging.info(f"🔥 BoyoChainBastardLoops: Using {len(iteration_prompts)} fallback prompts")
        
        # Get the relevant prompt (all nodes except ourselves)
        base_prompt = self.get_relevant_prompt(workflow_prompt, unique_id)
        logging.info(f"🔥 BoyoChainBastardLoops: Base prompt has {len(base_prompt)} nodes")
        
        # Execute each iteration
        collected_images = []
        pil_images = []
        
        for i, iteration_prompt in enumerate(iteration_prompts):
            logging.info(f"🔥 BoyoChainBastardLoops: Starting iteration {i+1}/{len(iteration_prompts)}")
            
            # Create a copy of the base prompt for this iteration
            iteration_prompt_data = copy.deepcopy(base_prompt)
            
            # Update the prompt text in CLIPTextEncode nodes
            iteration_prompt_data = self.update_prompt_with_text(iteration_prompt_data, iteration_prompt)
            
            # Execute this iteration
            result_image = self.execute_prompt(iteration_prompt_data, extra_pnginfo, iteration_prompt, i+1, vae_decode_node_id)
            
            if result_image is not None:
                collected_images.append(result_image)
                pil_images.append(self.tensor_to_pil(result_image))
                logging.info(f"🔥 BoyoChainBastardLoops: Iteration {i+1} completed successfully")
            else:
                logging.warning(f"🔥 BoyoChainBastardLoops: Iteration {i+1} failed to generate image")
        
        # Create grid
        if pil_images:
            grid_pil = self.create_image_grid(pil_images)
            grid_tensor = self.pil_to_tensor(grid_pil)
            
            logging.info(f"🔥 BoyoChainBastardLoops: SUCCESS! Created grid with {len(collected_images)} images")
            return (grid_tensor, collected_images)
        else:
            logging.error("🔥 BoyoChainBastardLoops: No images were generated!")
            empty_tensor = torch.zeros((1, 512, 512, 3))
            return (empty_tensor, [empty_tensor])

# Node mappings
NODE_CLASS_MAPPINGS = {
    "BoyoChainBastardLoops": BoyoChainBastardLoops
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoChainBastardLoops": "Boyo Chain Bastard Loops"
}