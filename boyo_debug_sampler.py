"""
BoyoNodes — Debug utility
Drop in, wire a SAMPLER into it, run, check the console output.
Tells us the exact attribute name to use when unwrapping a SAMPLER object.
"""

import comfy.samplers


class BoyoDebugSampler:

    CATEGORY     = "BoyoNodes/Debug"
    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION     = "inspect"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": ("SAMPLER", {}),
            }
        }

    def inspect(self, sampler):
        print("\n" + "=" * 60)
        print("BOYO SAMPLER DEBUG")
        print("=" * 60)
        print(f"Type          : {type(sampler)}")
        print(f"Dir           : {[a for a in dir(sampler) if not a.startswith('__')]}")
        print(f"__dict__      : {getattr(sampler, '__dict__', 'N/A')}")

        # Try common attribute names
        for attr in ["sampler_function", "sample", "sampler", "func", "function", "fn", "call"]:
            val = getattr(sampler, attr, "NOT FOUND")
            print(f"  .{attr:20s} : {val}")

        print("=" * 60 + "\n")

        # Pass it through unchanged
        return (sampler,)


NODE_CLASS_MAPPINGS = {
    "BoyoDebugSampler": BoyoDebugSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoDebugSampler": "🔧 Sampler Debug (Boyo)",
}
