"""
BoyoPiDAspectRatio
------------------
Pick a PiD checkpoint profile + aspect ratio, get back the matching
LDM latent-stage resolution and the final PiD decode resolution.

All numbers are on NVIDIA's documented grid: multiples of 64 on the
PiD side, multiples of 16 on the LDM side. Avoids the "non-square
seam" issue that comes from feeding PiD an off-grid resolution.

2k profile   -> 4x decode, LDM ~512px base
2kto4k       -> 4x decode, LDM ~1024px base (verified against NVIDIA's
                own official demo: --resolution 4096,3072 --pid_ckpt_type
                2kto4k, LDM run at 1024,768)
"""

PROFILES = {
    "2k (512 -> 2048, 4x)": {
        "1:1":  ((512, 512),  (2048, 2048)),
        "4:3":  ((576, 432),  (2304, 1728)),
        "3:4":  ((432, 576),  (1728, 2304)),
        "16:9": ((672, 384),  (2688, 1536)),
        "9:16": ((384, 672),  (1536, 2688)),
    },
    "2kto4k (1024 -> 4096, 4x)": {
        "1:1":  ((1024, 1024), (4096, 4096)),
        "4:3":  ((1024, 768),  (4096, 3072)),
        "3:4":  ((768, 1024),  (3072, 4096)),
        "16:9": ((1024, 576),  (4096, 2304)),
        "9:16": ((576, 1024),  (2304, 4096)),
    },
}

ASPECT_RATIOS = ["1:1", "4:3", "3:4", "16:9", "9:16"]


class BoyoPiDAspectRatio:
    """Outputs LDM width/height + PiD width/height for a chosen profile+ratio."""

    CATEGORY = "BoyoNodes/PiD"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "profile": (list(PROFILES.keys()), {"default": "2kto4k (1024 -> 4096, 4x)"}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "16:9"}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("ldm_width", "ldm_height", "pid_width", "pid_height", "label")
    FUNCTION = "get_resolution"

    def get_resolution(self, profile, aspect_ratio):
        (ldm_w, ldm_h), (pid_w, pid_h) = PROFILES[profile][aspect_ratio]
        label = f"{profile} | {aspect_ratio} | LDM {ldm_w}x{ldm_h} -> PiD {pid_w}x{pid_h}"
        return (ldm_w, ldm_h, pid_w, pid_h, label)


NODE_CLASS_MAPPINGS = {
    "BoyoPiDAspectRatio": BoyoPiDAspectRatio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoPiDAspectRatio": "Boyo PiD Aspect Ratio",
}
