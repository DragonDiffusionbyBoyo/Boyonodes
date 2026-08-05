"""
BoyoNodes — Experimental Pack
Node: BoyoExperimentalRichardson

X0 Prediction Accumulator meta-sampler.

During diffusion sampling, the model produces a denoised x0 prediction
at every step — its best guess at the clean image given current noise level.
Standard samplers throw all intermediate predictions away and return only
the final one.

This node captures every x0 prediction via callback and blends them
into the final output using a weighting curve that emphasises late steps
(where detail resolves) over early steps (where composition is set).

Result: the final latent carries accumulated detail information from the
entire trajectory rather than just the last step. Character depends on
the weighting curve and blend strength.

Weighting modes
---------------
exponential : late steps dominate hard — maximum detail enhancement
linear      : smooth ramp from early (low) to late (high)
uniform     : equal weight to all steps — subtle, safe starting point
late_only   : only the final N steps contribute (N = tail_steps parameter)

Author : Boyo / Dragon Diffusion UK Ltd
Pack   : BoyoNodes (experimental prefix)
"""

import torch
import comfy.samplers


WEIGHTING_MODES = ["exponential", "linear", "uniform", "late_only"]


# ---------------------------------------------------------------------------
# Weighting curves — all return a 1D tensor of length n, summing to 1
# ---------------------------------------------------------------------------

def _weights_exponential(n, sharpness=3.0):
    """Exponential ramp — late steps weighted heavily."""
    w = torch.exp(torch.linspace(0, sharpness, n))
    return w / w.sum()


def _weights_linear(n):
    """Linear ramp from near-zero to 1."""
    w = torch.linspace(0.01, 1.0, n)
    return w / w.sum()


def _weights_uniform(n):
    """Equal weight to all steps."""
    return torch.ones(n) / n


def _weights_late_only(n, tail_steps):
    """Only the last tail_steps steps contribute, equally weighted."""
    tail = min(tail_steps, n)
    w = torch.zeros(n)
    w[-tail:] = 1.0 / tail
    return w


def _get_weights(mode, n, tail_steps):
    if mode == "exponential":
        return _weights_exponential(n)
    elif mode == "linear":
        return _weights_linear(n)
    elif mode == "uniform":
        return _weights_uniform(n)
    elif mode == "late_only":
        return _weights_late_only(n, tail_steps)
    return _weights_exponential(n)


# ---------------------------------------------------------------------------
# Core accumulator sampler
# ---------------------------------------------------------------------------

class X0AccumulatorSampler:
    """
    Runs the base sampler and captures x0 predictions via callback.
    Blends weighted accumulated predictions into the final output.
    """

    def __init__(self, base_sampler, weighting, blend_strength, tail_steps):
        self.base_sampler   = base_sampler
        self.weighting      = weighting
        self.blend_strength = blend_strength
        self.tail_steps     = tail_steps

    def __call__(self, model, x, sigmas, extra_args=None, callback=None, disable=False):
        extra_args   = extra_args or {}
        n_steps      = len(sigmas) - 1
        x0_preds     = []

        print(
            f"[BoyoExperimentalRichardson] x0 accumulator | "
            f"weighting={self.weighting} | "
            f"blend={self.blend_strength:.2f} | "
            f"{n_steps} steps"
        )

        # --- Internal callback to capture x0 predictions ---
        def _capture_callback(d):
            denoised = d.get("denoised", None)
            if denoised is not None:
                x0_preds.append(denoised.clone().detach())
            # Fire the outer callback if present
            if callback is not None:
                callback(d)

        # --- Run the base sampler ---
        x_final = self.base_sampler(
            model,
            x.clone(),
            sigmas,
            extra_args=extra_args,
            callback=_capture_callback,
            disable=disable,
        )

        if not x0_preds:
            print("[BoyoExperimentalRichardson] WARNING: no x0 predictions captured — "
                  "base sampler may not fire callbacks. Returning unmodified output.")
            return x_final

        n_captured = len(x0_preds)
        print(f"[BoyoExperimentalRichardson] captured {n_captured} x0 predictions")

        # --- Build weighted accumulation ---
        weights = _get_weights(self.weighting, n_captured, self.tail_steps)
        weights = weights.to(x_final.device, dtype=x_final.dtype)

        x0_stack    = torch.stack(x0_preds, dim=0)       # [n_steps, B, C, H, W]
        w_expanded  = weights.view(n_captured, *([1] * (x0_stack.dim() - 1)))
        x0_weighted = (x0_stack * w_expanded).sum(dim=0)  # [B, C, H, W]

        # --- Blend accumulated x0 with final output ---
        x_out = (1.0 - self.blend_strength) * x_final + self.blend_strength * x0_weighted

        print(f"[BoyoExperimentalRichardson] blend complete: "
              f"{1.0-self.blend_strength:.2f}×final + {self.blend_strength:.2f}×accumulated")

        return x_out


# ---------------------------------------------------------------------------
# ComfyUI node class
# ---------------------------------------------------------------------------

class BoyoExperimentalRichardson:
    """
    X0 Prediction Accumulator meta-sampler.

    Captures the model's denoised x0 prediction at every sampling step
    and blends a weighted accumulation of them into the final output.
    Late steps (detail) are weighted more heavily than early steps
    (composition) by default.

    This uses information every sampler already produces but discards.

    weighting     : how to weight predictions across steps
    blend_strength: how much accumulated prediction to mix into output
    tail_steps    : steps to use for late_only weighting mode

    Wire:
      KSamplerSelect → [this node] → SamplerCustomAdvanced
    """

    CATEGORY     = "BoyoNodes/Experimental"
    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION     = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": (
                    "SAMPLER",
                    {
                        "tooltip": "Any SAMPLER — KSamplerSelect or another BoyoExperimental node.",
                    },
                ),
                "weighting": (
                    WEIGHTING_MODES,
                    {
                        "default": "exponential",
                        "tooltip": (
                            "How to weight x0 predictions across steps.\n"
                            "exponential : late steps dominate hard — max detail.\n"
                            "linear      : smooth ramp early→late.\n"
                            "uniform     : equal weight — subtle, safe start.\n"
                            "late_only   : only last N steps (set tail_steps)."
                        ),
                    },
                ),
                "blend_strength": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min":     0.0,
                        "max":     1.0,
                        "step":    0.05,
                        "round":   False,
                        "tooltip": (
                            "How much of the accumulated x0 prediction to blend in.\n"
                            "0.0  = bypass, standard sampler output.\n"
                            "0.35 = recommended starting point.\n"
                            "1.0  = full accumulation, no standard output."
                        ),
                    },
                ),
                "tail_steps": (
                    "INT",
                    {
                        "default": 3,
                        "min":     1,
                        "max":     8,
                        "step":    1,
                        "tooltip": "Steps to use for late_only weighting. Ignored for other modes.",
                    },
                ),
            },
        }

    def build(self, sampler, weighting, blend_strength, tail_steps):
        base_callable = sampler.sampler_function

        print(
            f"[BoyoExperimentalRichardson] Wrapping: {base_callable}\n"
            f"  weighting      : {weighting}\n"
            f"  blend_strength : {blend_strength:.2f}\n"
            f"  tail_steps     : {tail_steps}"
        )

        accumulator = X0AccumulatorSampler(
            base_sampler   = base_callable,
            weighting      = weighting,
            blend_strength = blend_strength,
            tail_steps     = tail_steps,
        )

        return (comfy.samplers.KSAMPLER(accumulator),)


# ---------------------------------------------------------------------------
# Node registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "BoyoExperimentalRichardson": BoyoExperimentalRichardson,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoExperimentalRichardson": "🧪 X0 Accumulator (Experimental)",
}
