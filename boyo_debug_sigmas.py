"""
BoyoNodes — Debug utility
BoyoDebugSigmas: takes any SIGMAS input, prints values to console,
passes them through unchanged. Wire between any sigma source and
SamplerCustomAdvanced to inspect what's actually going in.
"""

import torch


class BoyoDebugSigmas:

    CATEGORY     = "BoyoNodes/Debug"
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION     = "inspect"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": ("SIGMAS", {}),
            }
        }

    def inspect(self, sigmas):
        vals = [round(float(s), 6) for s in sigmas.tolist()]
        print("\n" + "=" * 60)
        print("BOYO SIGMA DEBUG")
        print("=" * 60)
        print(f"  Count  : {len(sigmas)} (including terminal zero)")
        print(f"  Max    : {float(sigmas[0]):.6f}")
        print(f"  Min    : {float(sigmas[-2]):.6f} (before terminal)")
        print(f"  Terminal: {float(sigmas[-1]):.6f}")
        print(f"  Values : {vals}")
        print(f"  dtype  : {sigmas.dtype}")
        print(f"  device : {sigmas.device}")
        print("=" * 60 + "\n")
        return (sigmas,)


NODE_CLASS_MAPPINGS = {
    "BoyoDebugSigmas": BoyoDebugSigmas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoDebugSigmas": "🔧 Sigma Debug (Boyo)",
}
