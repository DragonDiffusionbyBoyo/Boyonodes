"""
BoyoNodes — Experimental Pack
Node: BoyoExperimentalSolverSwitch

Wraps a set of existing k-diffusion samplers and switches between them on a
per-step basis according to a chosen strategy. No new arithmetic — every
solver called here is already stable and tested. This node just routes.

Compatible with: SamplerCustomAdvanced (preferred), KSamplerSelect
Output: SAMPLER object

Strategy options
----------------
round_robin      : cycles through solver list in order, step 0 → solver[0], step 1 → solver[1], etc.
random_seeded    : seeded RNG selection per step — reproducible given same seed and solver list
weighted_random  : biased random using per-solver weights (comma-separated, must match solver count)

Author : Boyo / Dragon Diffusion UK Ltd
Pack   : BoyoNodes (experimental prefix)
"""

import torch
import numpy as np
import comfy.samplers
from comfy.k_diffusion import sampling as k_sampling


# ---------------------------------------------------------------------------
# Solvers available for switching
# Keys must match the strings exposed in the UI multiline/dropdown.
# Values are callables with signature: (model, x, sigmas, **kwargs) → tensor
# ---------------------------------------------------------------------------
AVAILABLE_SOLVERS = {
    "euler":        k_sampling.sample_euler,
    "euler_a":      k_sampling.sample_euler_ancestral,
    "heun":         k_sampling.sample_heun,
    "dpm_2":        k_sampling.sample_dpm_2,
    "dpm_2_a":      k_sampling.sample_dpm_2_ancestral,
    "dpmpp_2s_a":   k_sampling.sample_dpmpp_2s_ancestral,
    "dpmpp_2m":     k_sampling.sample_dpmpp_2m,
    "dpmpp_sde":    k_sampling.sample_dpmpp_sde,
    "dpmpp_3m_sde": k_sampling.sample_dpmpp_3m_sde,
    "lms":          k_sampling.sample_lms,
    "lcm":          k_sampling.sample_lcm,
}

SOLVER_NAMES = list(AVAILABLE_SOLVER for AVAILABLE_SOLVER in AVAILABLE_SOLVERS)

DEFAULT_SOLVERS = "euler\nheun\ndpmpp_2m"


# ---------------------------------------------------------------------------
# Core switching sampler — this is the callable handed to comfy's KSAMPLER
# ---------------------------------------------------------------------------

class SolverSwitchSampler:
    """
    Wraps the per-step switching logic. Instantiated once per node execution,
    called by ComfyUI's sampler infrastructure with (model_wrap, sigmas, ...).
    """

    def __init__(self, solver_list, strategy, seed, weights=None):
        self.solver_list = solver_list          # list of solver name strings, validated
        self.strategy    = strategy
        self.seed        = seed
        self.weights     = weights              # normalised float list or None
        self._rng        = None                 # initialised on first call
        self._step_idx   = 0

    # Called by comfy internals: sample(model, x, sigmas, extra_args, callback, disable)
    def __call__(self, model, x, sigmas, extra_args=None, callback=None, disable=False):
        extra_args = extra_args or {}
        self._rng = np.random.default_rng(self.seed)
        self._step_idx = 0

        n_steps = len(sigmas) - 1
        step_log = []

        for i in range(n_steps):
            solver_name = self._pick_solver(i, n_steps)
            solver_fn   = AVAILABLE_SOLVERS[solver_name]
            step_log.append(f"  step {i:>3d}/{n_steps}: {solver_name}")

            # Slice sigmas to a two-element window for this step, then call solver.
            # Each k-diffusion solver expects the full sigma sequence and advances
            # from the first element — so we give it [sigma_i, sigma_{i+1}].
            sigma_pair = sigmas[i : i + 2]

            x = solver_fn(
                model,
                x,
                sigma_pair,
                extra_args=extra_args,
                callback=callback,
                disable=disable,
            )

        if not disable:
            print("[BoyoExperimentalSolverSwitch] Step → solver assignments:")
            print("\n".join(step_log))

        return x

    def _pick_solver(self, step_idx, total_steps):
        n = len(self.solver_list)

        if self.strategy == "round_robin":
            return self.solver_list[step_idx % n]

        elif self.strategy == "random_seeded":
            # Advance RNG state per step so each step gets its own draw
            # but the sequence is fully reproducible given the same seed
            return self.solver_list[int(self._rng.integers(0, n))]

        elif self.strategy == "weighted_random":
            w = self.weights if self.weights else [1.0 / n] * n
            return self.solver_list[
                int(self._rng.choice(len(self.solver_list), p=w))
            ]

        # Fallback — should never reach here given input validation
        return self.solver_list[0]


# ---------------------------------------------------------------------------
# Utility: parse and validate the solver list from the multiline text input
# ---------------------------------------------------------------------------

def _parse_solver_list(raw_text):
    """
    Accepts newline- or comma-separated solver names.
    Returns validated list of names, or raises ValueError with a helpful message.
    """
    raw = raw_text.replace(",", "\n")
    names = [s.strip().lower() for s in raw.splitlines() if s.strip()]

    if not names:
        raise ValueError("BoyoExperimentalSolverSwitch: solver list is empty.")

    unknown = [n for n in names if n not in AVAILABLE_SOLVERS]
    if unknown:
        raise ValueError(
            f"BoyoExperimentalSolverSwitch: unknown solver(s): {unknown}\n"
            f"Available: {list(AVAILABLE_SOLVERS.keys())}"
        )

    return names


def _parse_weights(raw_text, n_solvers):
    """
    Parse comma-separated weight string. Normalises to sum=1.
    Returns list of floats, or None if input is blank.
    """
    raw = raw_text.strip()
    if not raw:
        return None

    try:
        parts = [float(x.strip()) for x in raw.split(",")]
    except ValueError:
        raise ValueError(
            "BoyoExperimentalSolverSwitch: weights must be comma-separated numbers, "
            f"e.g. '1,2,1' for three solvers."
        )

    if len(parts) != n_solvers:
        raise ValueError(
            f"BoyoExperimentalSolverSwitch: {len(parts)} weight(s) provided "
            f"but {n_solvers} solver(s) in list. Counts must match."
        )

    total = sum(parts)
    if total <= 0:
        raise ValueError("BoyoExperimentalSolverSwitch: weights must sum to a positive value.")

    return [w / total for w in parts]


# ---------------------------------------------------------------------------
# ComfyUI node class
# ---------------------------------------------------------------------------

class BoyoExperimentalSolverSwitch:
    """
    Experimental sampler that switches between k-diffusion solvers on a
    per-step basis. Drop the SAMPLER output into SamplerCustomAdvanced or
    KSamplerSelect.

    No new arithmetic is introduced — each solver called is existing,
    validated k-diffusion code. Safety risk is therefore the same as using
    those solvers individually.
    """

    CATEGORY    = "BoyoNodes/Experimental"
    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION    = "build_sampler"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solvers": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": DEFAULT_SOLVERS,
                        "tooltip": (
                            "One solver name per line (or comma-separated).\n"
                            "Available: " + ", ".join(AVAILABLE_SOLVERS.keys())
                        ),
                    },
                ),
                "strategy": (
                    ["round_robin", "random_seeded", "weighted_random"],
                    {
                        "default": "round_robin",
                        "tooltip": (
                            "round_robin: cycles through solvers in order.\n"
                            "random_seeded: reproducible random selection per step.\n"
                            "weighted_random: biased random — set weights below."
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Seed for random/weighted strategies. Ignored for round_robin.",
                    },
                ),
            },
            "optional": {
                "weights": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": (
                            "Comma-separated weights for weighted_random strategy.\n"
                            "Must have the same count as the solver list.\n"
                            "E.g. '1,2,1' favours the second solver 2× over the others.\n"
                            "Ignored for other strategies."
                        ),
                    },
                ),
            },
        }

    def build_sampler(self, solvers, strategy, seed, weights=""):
        # --- Validate inputs ---
        solver_list = _parse_solver_list(solvers)

        parsed_weights = None
        if strategy == "weighted_random":
            parsed_weights = _parse_weights(weights, len(solver_list))

        print(
            f"[BoyoExperimentalSolverSwitch] Configured:\n"
            f"  solvers  : {solver_list}\n"
            f"  strategy : {strategy}\n"
            f"  seed     : {seed}"
        )
        if parsed_weights:
            print(f"  weights  : {[round(w, 4) for w in parsed_weights]}")

        sampler_instance = SolverSwitchSampler(
            solver_list=solver_list,
            strategy=strategy,
            seed=seed,
            weights=parsed_weights,
        )

        # Wrap in ComfyUI's KSAMPLER container
        return (comfy.samplers.KSAMPLER(sampler_instance),)


# ---------------------------------------------------------------------------
# Node registration — add to your __init__.py mappings
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "BoyoExperimentalSolverSwitch": BoyoExperimentalSolverSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoExperimentalSolverSwitch": "🧪 Solver Switch (Experimental)",
}
