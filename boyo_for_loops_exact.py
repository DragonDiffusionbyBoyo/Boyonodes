"""
Boyo For Loop Nodes - Exact replication of EasyUse loops with reset capability
Consolidates all necessary utilities and replicates the exact logic from EasyUse
"""

import numpy as np
import torch
import comfy.utils
from typing import Iterator, List, Tuple, Dict, Any, Union, Optional

# ==================== CONSOLIDATED UTILITIES ====================

class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False

class TautologyStr(str):
    def __ne__(self, other):
        return False

class ByPassTypeTuple(tuple):
    def __getitem__(self, index):
        if index > 0:
            index = 0
        item = super().__getitem__(index)
        if isinstance(item, str):
            return TautologyStr(item)
        return item

# Cache implementation
import itertools
from typing import Optional

class TaggedCache:
    def __init__(self, tag_settings: Optional[dict] = None):
        self._tag_settings = tag_settings or {}
        self._data = {}

    def __getitem__(self, key):
        for tag_data in self._data.values():
            if key in tag_data:
                return tag_data[key]
        raise KeyError(f'Key `{key}` does not exist')

    def __setitem__(self, key, value: tuple):
        # if key already exists, pop old value
        for tag_data in self._data.values():
            if key in tag_data:
                tag_data.pop(key, None)
                break

        tag = value[0]
        if tag not in self._data:
            try:
                from cachetools import LRUCache
                default_size = 20
                if 'ckpt' in tag:
                    default_size = 5
                elif tag in ['latent', 'image']:
                    default_size = 100
                self._data[tag] = LRUCache(maxsize=self._tag_settings.get(tag, default_size))
            except (ImportError, ModuleNotFoundError):
                self._data[tag] = {}
        self._data[tag][key] = value

    def __delitem__(self, key):
        for tag_data in self._data.values():
            if key in tag_data:
                del tag_data[key]
                return
        raise KeyError(f'Key `{key}` does not exist')

    def __contains__(self, key):
        return any(key in tag_data for tag_data in self._data.values())

    def items(self):
        yield from itertools.chain(*map(lambda x: x.items(), self._data.values()))

    def get(self, key, default=None):
        for tag_data in self._data.values():
            if key in tag_data:
                return tag_data[key]
        return default

    def clear(self):
        self._data = {}

# Global cache instances
cache_settings = {}
boyo_cache = TaggedCache(cache_settings)
boyo_cache_count = {}

def boyo_update_cache(k, tag, v):
    boyo_cache[k] = (tag, v)
    cnt = boyo_cache_count.get(k)
    if cnt is None:
        cnt = 0
        boyo_cache_count[k] = cnt
    else:
        boyo_cache_count[k] += 1

def boyo_remove_cache(key):
    global boyo_cache
    if key == '*':
        boyo_cache = TaggedCache(cache_settings)
        boyo_cache_count.clear()
    elif key in boyo_cache:
        del boyo_cache[key]
        if key in boyo_cache_count:
            del boyo_cache_count[key]
    else:
        print(f"boyo_remove_cache: invalid key {key}")

# Logging functions
def boyo_log_node_info(node_name, message=None):
    """Logs an info message."""
    print(f"[Boyo] {node_name}: {message}" if message else f"[Boyo] {node_name}")

def boyo_log_node_warn(node_name, message=None):
    """Logs a warning message."""
    print(f"[Boyo WARNING] {node_name}: {message}" if message else f"[Boyo WARNING] {node_name}")

def boyo_log_node_error(node_name, message=None):
    """Logs an error message."""
    print(f"[Boyo ERROR] {node_name}: {message}" if message else f"[Boyo ERROR] {node_name}")

# Revision checking
def boyo_compare_revision(num):
    """Always return True for compatibility"""
    return True

# Utility functions from EasyUse
def validate_list_args(args: Dict[str, List[Any]]) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Checks that if there are multiple arguments, they are all the same length or 1
    """
    if len(args) == 1:
        return True, None, None

    len_to_match = None
    matched_arg_name = None
    for arg_name, arg in args.items():
        if arg_name == 'self':
            continue

        if len(arg) != 1:
            if len_to_match is None:
                len_to_match = len(arg)
                matched_arg_name = arg_name
            elif len(arg) != len_to_match:
                return False, arg_name, matched_arg_name

    return True, None, None

def error_if_mismatched_list_args(args: Dict[str, List[Any]]) -> None:
    is_valid, failed_key1, failed_key2 = validate_list_args(args)
    if not is_valid:
        assert failed_key1 is not None
        assert failed_key2 is not None
        raise ValueError(
            f"Mismatched list inputs received. {failed_key1}({len(args[failed_key1])}) !== {failed_key2}({len(args[failed_key2])})"
        )

def zip_with_fill(*lists: Union[List[Any], None]) -> Iterator[Tuple[Any, ...]]:
    """
    Zips lists together, but if a list has 1 element, it will be repeated for each element in the other lists.
    """
    max_len = max(len(lst) if lst is not None else 0 for lst in lists)
    for i in range(max_len):
        yield tuple(None if lst is None else (lst[0] if len(lst) == 1 else lst[i]) for lst in lists)

# GraphBuilder fallback handling
try:
    from comfy_execution.graph_utils import GraphBuilder, is_link
except:
    GraphBuilder = None
    def is_link(value):
        """Fallback is_link implementation"""
        return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)

# Constants
DEFAULT_FLOW_NUM = 2
MAX_FLOW_NUM = 10
any_type = AlwaysEqualProxy("*")

# Lazy options for ComfyUI compatibility
lazy_options = {"lazy": True} if boyo_compare_revision(2543) else {}

# ==================== RESET FUNCTIONALITY ====================

# Global reset tracking
_boyo_reset_signals = {}

def boyo_get_reset_signal(reset_key="default"):
    """Get current reset signal value"""
    return _boyo_reset_signals.get(reset_key, 0)

def boyo_set_reset_signal(reset_key="default", value=1):
    """Set reset signal value"""
    _boyo_reset_signals[reset_key] = value
    boyo_log_node_info("BoyoReset", f"Reset signal '{reset_key}' set to {value}")

def boyo_check_reset(cache_key, reset_key="default"):
    """Check if reset is needed and clear cache if so"""
    reset_cache_key = f"{cache_key}_reset_signal"
    current_signal = boyo_get_reset_signal(reset_key)
    last_signal = boyo_cache.get(reset_cache_key, (None, -1))
    
    if last_signal[1] != current_signal:
        # Reset needed
        boyo_remove_cache(cache_key)
        boyo_update_cache(reset_cache_key, "reset", current_signal)
        boyo_log_node_info("BoyoReset", f"Cache '{cache_key}' reset due to signal change")
        return True
    return False

# ==================== MAIN LOOP NODES ====================

class BoyoWhileLoopStart:
    """Exact replication of EasyUse whileLoopStart with reset capability"""
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "condition": ("BOOLEAN", {"default": True}),
                "loop_id": ("INT", {"default": 1, "min": 1, "max": 999, "step": 1}),
            },
            "optional": {
                "reset_signal": ("INT", {"default": 0}),
            },
        }
        for i in range(MAX_FLOW_NUM):
            inputs["optional"]["initial_value%d" % i] = (any_type,)
        return inputs

    RETURN_TYPES = ByPassTypeTuple(tuple(["FLOW_CONTROL"] + [any_type] * MAX_FLOW_NUM))
    RETURN_NAMES = ByPassTypeTuple(tuple(["flow"] + ["value%d" % i for i in range(MAX_FLOW_NUM)]))
    FUNCTION = "while_loop_open"
    CATEGORY = "Boyo/Logic/While Loop"

    def while_loop_open(self, condition, loop_id, reset_signal=0, **kwargs):
        # Use loop_id for cache isolation, check reset_signal for resets
        cache_key = f"boyo_loop_{loop_id}"
        reset_cache_key = f"{cache_key}_last_reset_signal"
        
        # Check if reset signal changed
        last_signal = boyo_cache.get(reset_cache_key, (None, -1))
        if last_signal[1] != reset_signal:
            # Reset signal changed, clear the loop cache
            boyo_remove_cache(cache_key)
            boyo_update_cache(reset_cache_key, "reset", reset_signal)
            boyo_log_node_info("BoyoWhileLoopStart", f"Loop {loop_id} reset due to signal change")
        
        values = []
        for i in range(MAX_FLOW_NUM):
            values.append(kwargs.get("initial_value%d" % i, None))
        return tuple([cache_key] + values)


class BoyoWhileLoopEnd:
    """Exact replication of EasyUse whileLoopEnd with reset capability"""
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "flow": ("FLOW_CONTROL", {"rawLink": True}),
                "condition": ("BOOLEAN", {}),
                "loop_id": ("INT", {"default": 1, "min": 1, "max": 999, "step": 1}),
            },
            "optional": {
                "reset_signal": ("INT", {"default": 0}),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }
        for i in range(MAX_FLOW_NUM):
            inputs["optional"]["initial_value%d" % i] = (any_type,)
        return inputs

    RETURN_TYPES = ByPassTypeTuple(tuple([any_type] * MAX_FLOW_NUM))
    RETURN_NAMES = ByPassTypeTuple(tuple(["value%d" % i for i in range(MAX_FLOW_NUM)]))
    FUNCTION = "while_loop_close"
    CATEGORY = "Boyo/Logic/While Loop"

    def explore_dependencies(self, node_id, dynprompt, upstream, parent_ids):
        """Exact copy from EasyUse logic.py"""
        node_info = dynprompt.get_node(node_id)
        if "inputs" not in node_info:
            return

        for k, v in node_info["inputs"].items():
            if is_link(v):
                parent_id = v[0]
                display_id = dynprompt.get_display_node_id(parent_id)
                display_node = dynprompt.get_node(display_id)
                class_type = display_node["class_type"]
                if class_type not in ['BoyoForLoopEnd', 'BoyoWhileLoopEnd', 'easy forLoopEnd', 'easy whileLoopEnd']:
                    parent_ids.append(display_id)
                if parent_id not in upstream:
                    upstream[parent_id] = []
                    self.explore_dependencies(parent_id, dynprompt, upstream, parent_ids)

                upstream[parent_id].append(node_id)

    def explore_output_nodes(self, dynprompt, upstream, output_nodes, parent_ids):
        """Exact copy from EasyUse logic.py"""
        for parent_id in upstream:
            display_id = dynprompt.get_display_node_id(parent_id)
            for output_id in output_nodes:
                id = output_nodes[output_id][0]
                if id in parent_ids and display_id == id and output_id not in upstream[parent_id]:
                    if '.' in parent_id:
                        arr = parent_id.split('.')
                        arr[len(arr)-1] = output_id
                        upstream[parent_id].append('.'.join(arr))
                    else:
                        upstream[parent_id].append(output_id)

    def collect_contained(self, node_id, upstream, contained):
        """Exact copy from EasyUse logic.py"""
        if node_id not in upstream:
            return
        for child_id in upstream[node_id]:
            if child_id not in contained:
                contained[child_id] = True
                self.collect_contained(child_id, upstream, contained)

    def while_loop_close(self, flow, condition, loop_id, dynprompt=None, unique_id=None, reset_signal=0, **kwargs):
        # Use loop_id for cache isolation, check reset_signal for resets
        cache_key = f"boyo_loop_{loop_id}"
        reset_cache_key = f"{cache_key}_last_reset_signal"
        
        # Check if reset signal changed
        last_signal = boyo_cache.get(reset_cache_key, (None, -1))
        if last_signal[1] != reset_signal:
            # Reset signal changed, clear the loop cache
            boyo_remove_cache(cache_key)
            boyo_update_cache(reset_cache_key, "reset", reset_signal)
            boyo_log_node_info("BoyoWhileLoopEnd", f"Loop {loop_id} reset due to signal change")
        
        if not condition:
            # We're done with the loop
            values = []
            for i in range(MAX_FLOW_NUM):
                values.append(kwargs.get("initial_value%d" % i, None))
            return tuple(values)

        # If GraphBuilder is not available, fall back to simple loop
        if GraphBuilder is None:
            boyo_log_node_warn("BoyoWhileLoopEnd", "GraphBuilder not available, using simple fallback")
            # Simple increment and continue
            values = []
            for i in range(MAX_FLOW_NUM):
                values.append(kwargs.get("initial_value%d" % i, None))
            return tuple(values)

        # Original EasyUse implementation with GraphBuilder
        # Get all available node class mappings
        try:
            from nodes import NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS
        except:
            ALL_NODE_CLASS_MAPPINGS = {}
        
        this_node = dynprompt.get_node(unique_id)
        upstream = {}
        parent_ids = []
        self.explore_dependencies(unique_id, dynprompt, upstream, parent_ids)
        parent_ids = list(set(parent_ids))
        
        prompts = dynprompt.get_original_prompt()
        output_nodes = {}
        for id in prompts:
            node = prompts[id]
            if "inputs" not in node:
                continue
            class_type = node["class_type"]
            class_def = ALL_NODE_CLASS_MAPPINGS.get(class_type)
            if class_def and hasattr(class_def, 'OUTPUT_NODE') and class_def.OUTPUT_NODE == True:
                for k, v in node['inputs'].items():
                    if is_link(v):
                        output_nodes[id] = v

        graph = GraphBuilder()
        self.explore_output_nodes(dynprompt, upstream, output_nodes, parent_ids)
        contained = {}
        open_node = flow[0]
        self.collect_contained(open_node, upstream, contained)
        contained[unique_id] = True
        contained[open_node] = True

        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.node(original_node["class_type"], "Recurse" if node_id == unique_id else node_id)
            node.set_override_display_id(node_id)
        
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.lookup_node("Recurse" if node_id == unique_id else node_id)
            for k, v in original_node["inputs"].items():
                if is_link(v) and v[0] in contained:
                    parent = graph.lookup_node(v[0])
                    node.set_input(k, parent.out(v[1]))
                else:
                    node.set_input(k, v)

        new_open = graph.lookup_node(open_node)
        for i in range(MAX_FLOW_NUM):
            key = "initial_value%d" % i
            new_open.set_input(key, kwargs.get(key, None))
        
        my_clone = graph.lookup_node("Recurse")
        result = map(lambda x: my_clone.out(x), range(MAX_FLOW_NUM))
        
        return {
            "result": tuple(result),
            "expand": graph.finalize(),
        }


class BoyoForLoopStart:
    """Exact replication of EasyUse forLoopStart with reset capability"""
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total": ("INT", {"default": 1, "min": 1, "max": 100000, "step": 1}),
                "loop_id": ("INT", {"default": 1, "min": 1, "max": 999, "step": 1}),
            },
            "optional": {
                "reset_signal": ("INT", {"default": 0}),
                **{f"initial_value{i}": (any_type,) for i in range(1, MAX_FLOW_NUM)}
            },
            "hidden": {
                "initial_value0": (any_type,),
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ByPassTypeTuple(tuple(["FLOW_CONTROL", "INT"] + [any_type] * (MAX_FLOW_NUM - 1)))
    RETURN_NAMES = ByPassTypeTuple(tuple(["flow", "index"] + [f"value{i}" for i in range(1, MAX_FLOW_NUM)]))
    FUNCTION = "for_loop_start"
    CATEGORY = "Boyo/Logic/For Loop"

    def for_loop_start(self, total, loop_id, prompt=None, extra_pnginfo=None, unique_id=None, reset_signal=0, **kwargs):
        # Use loop_id for cache isolation, check reset_signal for resets
        cache_key = f"boyo_loop_{loop_id}"
        reset_cache_key = f"{cache_key}_last_reset_signal"
        
        # Check if reset signal changed
        last_signal = boyo_cache.get(reset_cache_key, (None, -1))
        if last_signal[1] != reset_signal:
            # Reset signal changed, clear the loop cache
            boyo_remove_cache(cache_key)
            boyo_update_cache(reset_cache_key, "reset", reset_signal)
            boyo_log_node_info("BoyoForLoopStart", f"Loop {loop_id} reset due to signal change: {last_signal[1]} → {reset_signal}")
        
        if GraphBuilder is None:
            boyo_log_node_warn("BoyoForLoopStart", "GraphBuilder not available, using simple fallback")
            outputs = [kwargs.get(f"initial_value{num}", None) for num in range(1, MAX_FLOW_NUM)]
            return tuple([cache_key, 0] + outputs)

        # Original EasyUse implementation
        graph = GraphBuilder()
        i = 0
        if "initial_value0" in kwargs:
            i = kwargs["initial_value0"]

        initial_values = {f"initial_value{num}": kwargs.get(f"initial_value{num}", None) for num in range(1, MAX_FLOW_NUM)}
        while_open = graph.node("BoyoWhileLoopStart", condition=total, initial_value0=i, loop_id=loop_id, reset_signal=reset_signal, **initial_values)
        outputs = [kwargs.get(f"initial_value{num}", None) for num in range(1, MAX_FLOW_NUM)]
        
        return {
            "result": tuple([cache_key, i] + outputs),
            "expand": graph.finalize(),
        }


class BoyoForLoopEnd:
    """Exact replication of EasyUse forLoopEnd with reset capability"""
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": ("FLOW_CONTROL", {"rawLink": True}),
                "loop_id": ("INT", {"default": 1, "min": 1, "max": 999, "step": 1}),
            },
            "optional": {
                "reset_signal": ("INT", {"default": 0}),
                **{f"initial_value{i}": (any_type, {"rawLink": True}) for i in range(1, MAX_FLOW_NUM)}
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            },
        }

    RETURN_TYPES = ByPassTypeTuple(tuple([any_type] * (MAX_FLOW_NUM - 1)))
    RETURN_NAMES = ByPassTypeTuple(tuple([f"value{i}" for i in range(1, MAX_FLOW_NUM)]))
    FUNCTION = "for_loop_end"
    CATEGORY = "Boyo/Logic/For Loop"

    def for_loop_end(self, flow, loop_id, dynprompt=None, extra_pnginfo=None, unique_id=None, reset_signal=0, **kwargs):
        # Use loop_id for cache isolation, check reset_signal for resets
        cache_key = f"boyo_loop_{loop_id}"
        reset_cache_key = f"{cache_key}_last_reset_signal"
        
        # Check if reset signal changed
        last_signal = boyo_cache.get(reset_cache_key, (None, -1))
        if last_signal[1] != reset_signal:
            # Reset signal changed, clear the loop cache
            boyo_remove_cache(cache_key)
            boyo_update_cache(reset_cache_key, "reset", reset_signal)
            boyo_log_node_info("BoyoForLoopEnd", f"Loop {loop_id} reset due to signal change")
        
        if GraphBuilder is None:
            boyo_log_node_warn("BoyoForLoopEnd", "GraphBuilder not available, using simple fallback")
            return tuple([kwargs.get(f"initial_value{i}", None) for i in range(1, MAX_FLOW_NUM)])

        # Original EasyUse implementation
        graph = GraphBuilder()
        while_open = flow[0]
        total = None

        forstart_node = dynprompt.get_node(while_open)
        if forstart_node['class_type'] in ['BoyoForLoopStart', 'easy forLoopStart']:
            inputs = forstart_node['inputs']
            total = inputs['total']
        elif forstart_node['class_type'] == 'easy loadImagesForLoop':
            inputs = forstart_node['inputs']
            limit = inputs['limit']
            start_index = inputs['start_index']
            directory = inputs['directory']
            total = graph.node('easy imagesCountInDirectory', directory=directory, limit=limit, start_index=start_index, extension='*').out(0)

        sub = graph.node("BoyoMathInt", operation="add", a=[while_open, 1], b=1)
        cond = graph.node("BoyoCompare", a=sub.out(0), b=total, comparison='a < b')
        input_values = {f"initial_value{i}": kwargs.get(f"initial_value{i}", None) for i in range(1, MAX_FLOW_NUM)}
        while_close = graph.node("BoyoWhileLoopEnd",
                                flow=flow,
                                condition=cond.out(0),
                                initial_value0=sub.out(0),
                                loop_id=loop_id,
                                reset_signal=reset_signal,
                                **input_values)
        
        return {
            "result": tuple([while_close.out(i) for i in range(1, MAX_FLOW_NUM)]),
            "expand": graph.finalize(),
        }


class BoyoLoopReset:
    """Enhanced reset node that works with the loop system"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": (any_type, {}),
                "reset_key": ("STRING", {"default": "default"}),
            },
            "optional": {
                "increment": ("INT", {"default": 1, "min": 1, "max": 100}),
            }
        }
    
    RETURN_TYPES = (any_type, "INT", "STRING")
    RETURN_NAMES = ("trigger_out", "reset_signal", "status")
    FUNCTION = "reset_loops"
    CATEGORY = "Boyo/Logic"
    
    def reset_loops(self, trigger, reset_key="default", increment=1):
        """Generate reset signal for connected loops"""
        current_signal = boyo_get_reset_signal(reset_key)
        new_signal = current_signal + increment
        boyo_set_reset_signal(reset_key, new_signal)
        
        status = f"Reset signal '{reset_key}' incremented to {new_signal}"
        boyo_log_node_info("BoyoLoopReset", status)
        
        return (trigger, new_signal, status)


# ==================== SUPPORTING NODES ====================

class BoyoMathInt:
    """Simple math operations for integers"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("INT", {"default": 0}),
                "b": ("INT", {"default": 0}),
                "operation": (["add", "subtract", "multiply", "divide", "modulo", "power"],),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "int_math_operation"
    CATEGORY = "Boyo/Logic/Math"

    def int_math_operation(self, a, b, operation):
        if operation == "add":
            return (a + b,)
        elif operation == "subtract":
            return (a - b,)
        elif operation == "multiply":
            return (a * b,)
        elif operation == "divide":
            return (a // b,)
        elif operation == "modulo":
            return (a % b,)
        elif operation == "power":
            return (a ** b,)


COMPARE_FUNCTIONS = {
    "a == b": lambda a, b: a == b,
    "a != b": lambda a, b: a != b,
    "a < b": lambda a, b: a < b,
    "a > b": lambda a, b: a > b,
    "a <= b": lambda a, b: a <= b,
    "a >= b": lambda a, b: a >= b,
}

class BoyoCompare:
    """Comparison operations"""
    
    @classmethod
    def INPUT_TYPES(s):
        compare_functions = list(COMPARE_FUNCTIONS.keys())
        return {
            "required": {
                "a": (any_type, {"default": 0}),
                "b": (any_type, {"default": 0}),
                "comparison": (compare_functions, {"default": "a == b"}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "compare"
    CATEGORY = "Boyo/Logic/Math"

    def compare(self, a, b, comparison):
        return (COMPARE_FUNCTIONS[comparison](a, b),)


# ==================== NODE MAPPINGS ====================

NODE_CLASS_MAPPINGS = {
    "BoyoWhileLoopStart": BoyoWhileLoopStart,
    "BoyoWhileLoopEnd": BoyoWhileLoopEnd,
    "BoyoForLoopStart": BoyoForLoopStart,
    "BoyoForLoopEnd": BoyoForLoopEnd,
    "BoyoLoopReset": BoyoLoopReset,
    "BoyoMathInt": BoyoMathInt,
    "BoyoCompare": BoyoCompare,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoWhileLoopStart": "Boyo While Loop Start",
    "BoyoWhileLoopEnd": "Boyo While Loop End", 
    "BoyoForLoopStart": "Boyo For Loop Start",
    "BoyoForLoopEnd": "Boyo For Loop End",
    "BoyoLoopReset": "Boyo Loop Reset",
    "BoyoMathInt": "Boyo Math Int",
    "BoyoCompare": "Boyo Compare",
}
