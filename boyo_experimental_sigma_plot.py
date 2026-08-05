"""
BoyoNodes — Experimental Pack
Node: BoyoExperimentalSigmaPlot

Generates a sigma schedule for flow models (Wan, LTX, Krea2).
Sigmas live in [0, 1] — this is non-negotiable for flow models.
sigma_max=1.0, sigma_min is a small positive value near zero.

Outputs:
  - SIGMAS  → wire into SamplerCustomAdvanced
  - IMAGE   → wire into PreviewImage

Strategies
----------
simple          : Linear [1.0 → sigma_min]. Matches BasicScheduler/simple exactly.
linear          : Same as simple — explicit alias.
beta            : Beta distribution spacing (alpha=0.6, beta=0.6). Middle-weighted.
beta57          : Beta (alpha=5, beta=7). Skewed toward low-noise / detail end.
karras          : Karras curve in [0,1] space.
exponential     : Exponential decay in [0,1] space.
chaotic_logistic: Logistic map drives spacing. Seeded, deterministic chaos.
chaotic_henon   : Hénon map drives spacing. Different chaotic texture.
non_monotonic   : Simple base with seeded upward reinjection events.

Author : Boyo / Dragon Diffusion UK Ltd
Pack   : BoyoNodes (experimental prefix)
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from comfy.k_diffusion import sampling as k_diff_sampling


# ---------------------------------------------------------------------------
# Sigma generation — all return float32 tensor length (steps+1), last = 0
# All operate in [0, 1] space — flow models only.
# ---------------------------------------------------------------------------

def _sigmas_simple(steps, sigma_min):
    """Linear descent 1.0 → sigma_min. Matches BasicScheduler simple exactly."""
    sigmas = torch.linspace(1.0, sigma_min, steps)
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def _sigmas_linear(steps, sigma_min):
    return _sigmas_simple(steps, sigma_min)


def _beta_quantiles(steps, alpha, beta_v):
    from scipy.special import betaincinv
    p = np.linspace(0.01, 0.99, steps)
    q = betaincinv(alpha, beta_v, p)
    return torch.tensor(q, dtype=torch.float32)


def _sigmas_beta(steps, sigma_min):
    quantiles = _beta_quantiles(steps, 0.6, 0.6)
    quantiles = (quantiles - quantiles.min()) / (quantiles.max() - quantiles.min())
    sigmas = sigma_min + (1.0 - quantiles) * (1.0 - sigma_min)
    sigmas, _ = sigmas.sort(descending=True)
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def _sigmas_beta57(steps, sigma_min):
    quantiles = _beta_quantiles(steps, 5.0, 7.0)
    quantiles = (quantiles - quantiles.min()) / (quantiles.max() - quantiles.min())
    sigmas = sigma_min + (1.0 - quantiles) * (1.0 - sigma_min)
    sigmas, _ = sigmas.sort(descending=True)
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def _sigmas_karras(steps, sigma_min):
    return k_diff_sampling.get_sigmas_karras(
        steps, sigma_min, 1.0, rho=7.0
    )


def _sigmas_exponential(steps, sigma_min):
    sigmas = torch.linspace(np.log(1.0), np.log(sigma_min), steps).exp()
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def _sigmas_chaotic_logistic(steps, sigma_min, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.01, 0.99)
    r = 3.99
    raw = []
    for _ in range(steps):
        x = r * x * (1 - x)
        raw.append(x)
    raw = np.array(raw)
    rescaled = raw * (1.0 - sigma_min) + sigma_min
    rescaled = np.sort(rescaled)[::-1].copy()
    sigmas = torch.tensor(rescaled, dtype=torch.float32)
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def _sigmas_chaotic_henon(steps, sigma_min, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-0.5, 0.5)
    y = rng.uniform(-0.5, 0.5)
    a, b = 1.4, 0.3
    raw = []
    for _ in range(steps):
        x, y = 1 - a * x**2 + y, b * x
        raw.append(x)
    raw = np.array(raw)
    r_min, r_max = raw.min(), raw.max()
    if r_max - r_min < 1e-8:
        raw = np.linspace(0.99, 0.01, steps)
    else:
        raw = (raw - r_min) / (r_max - r_min)
    rescaled = raw * (1.0 - sigma_min) + sigma_min
    rescaled = np.sort(rescaled)[::-1].copy()
    sigmas = torch.tensor(rescaled, dtype=torch.float32)
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def _sigmas_non_monotonic(steps, sigma_min, seed):
    base = _sigmas_simple(steps, sigma_min)
    sigmas = base.clone()
    rng = np.random.default_rng(seed)
    max_events = max(1, steps // 3)
    n_events = int(rng.integers(1, max_events + 1))
    eligible = list(range(2, steps - 1))
    if not eligible:
        return sigmas
    event_steps = rng.choice(eligible, size=min(n_events, len(eligible)), replace=False)
    for idx in event_steps:
        bump = float(sigmas[idx]) * rng.uniform(0.05, 0.20)
        sigmas[idx] = min(float(sigmas[idx]) + bump, 1.0)
    return sigmas


# ---------------------------------------------------------------------------
# Strategy dispatcher
# ---------------------------------------------------------------------------

STRATEGIES = [
    "simple",
    "beta",
    "beta57",
    "linear",
    "karras",
    "exponential",
    "chaotic_logistic",
    "chaotic_henon",
    "non_monotonic",
]


def _build_sigmas(steps, sigma_min, strategy, seed):
    if strategy in ("simple", "linear"):
        return _sigmas_simple(steps, sigma_min)
    elif strategy == "beta":
        return _sigmas_beta(steps, sigma_min)
    elif strategy == "beta57":
        return _sigmas_beta57(steps, sigma_min)
    elif strategy == "karras":
        return _sigmas_karras(steps, sigma_min)
    elif strategy == "exponential":
        return _sigmas_exponential(steps, sigma_min)
    elif strategy == "chaotic_logistic":
        return _sigmas_chaotic_logistic(steps, sigma_min, seed)
    elif strategy == "chaotic_henon":
        return _sigmas_chaotic_henon(steps, sigma_min, seed)
    elif strategy == "non_monotonic":
        return _sigmas_non_monotonic(steps, sigma_min, seed)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

W, H = 512, 320
PAD_L, PAD_R, PAD_T, PAD_B = 60, 20, 20, 50
BG_COLOUR    = (18, 18, 22)
GRID_COLOUR  = (45, 45, 55)
LINE_COLOUR  = (120, 220, 180)
DOT_COLOUR   = (220, 160, 80)
TEXT_COLOUR  = (180, 180, 190)
LABEL_COLOUR = (120, 120, 140)


def _sigma_to_px(sigma, sigma_max, sigma_min):
    frac = (sigma - sigma_min) / max(sigma_max - sigma_min, 1e-8)
    return int(PAD_T + (1.0 - frac) * (H - PAD_T - PAD_B))


def _step_to_px(step_idx, n_points):
    plot_w = W - PAD_L - PAD_R
    frac = step_idx / max(n_points - 1, 1)
    return int(PAD_L + frac * plot_w)


def _render_plot(sigmas_tensor, strategy, steps, sigma_min):
    sigma_max = 1.0
    img  = Image.new("RGB", (W, H), BG_COLOUR)
    draw = ImageDraw.Draw(img)

    try:
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        font_md = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except Exception:
        font_sm = ImageFont.load_default()
        font_md = font_sm

    for frac in [0.0, 0.33, 0.66, 1.0]:
        sigma_val = sigma_min + frac * (sigma_max - sigma_min)
        y = _sigma_to_px(sigma_val, sigma_max, sigma_min)
        draw.line([(PAD_L, y), (W - PAD_R, y)], fill=GRID_COLOUR, width=1)
        draw.text((2, y - 7), f"{sigma_val:.3f}", fill=LABEL_COLOUR, font=font_sm)

    draw.line([(PAD_L, PAD_T), (PAD_L, H - PAD_B)], fill=TEXT_COLOUR, width=1)
    draw.line([(PAD_L, H - PAD_B), (W - PAD_R, H - PAD_B)], fill=TEXT_COLOUR, width=1)

    vals = sigmas_tensor[:-1].tolist()
    n    = len(vals)
    points = [(_step_to_px(i, n), _sigma_to_px(v, sigma_max, sigma_min)) for i, v in enumerate(vals)]

    if len(points) > 1:
        draw.line(points, fill=LINE_COLOUR, width=2)

    for px, py in points:
        r = 4
        draw.ellipse([(px - r, py - r), (px + r, py + r)], fill=DOT_COLOUR)

    for i in range(n):
        px = _step_to_px(i, n)
        draw.text((px - 3, H - PAD_B + 6), str(i + 1), fill=LABEL_COLOUR, font=font_sm)

    title = f"{strategy}  |  {steps} steps  |  σ 1.000→{sigma_min:.4f}  [flow]"
    draw.text((PAD_L, 4), title, fill=TEXT_COLOUR, font=font_md)

    for i, (px, py) in enumerate(points):
        label    = f"{vals[i]:.3f}"
        offset_y = -18 if py > PAD_T + 20 else 10
        draw.text((px - 14, py + offset_y), label, fill=LINE_COLOUR, font=font_sm)

    return img


def _pil_to_comfy_image(pil_img):
    arr    = np.array(pil_img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0)
    return tensor


# ---------------------------------------------------------------------------
# ComfyUI node class
# ---------------------------------------------------------------------------

class BoyoExperimentalSigmaPlot:
    """
    Sigma schedule generator for flow models (Wan, LTX, Krea2).
    Sigmas always in [0, 1] — the correct range for flow/rectified models.

    Outputs SIGMAS for SamplerCustomAdvanced and IMAGE for PreviewImage.
    """

    CATEGORY     = "BoyoNodes/Experimental"
    RETURN_TYPES = ("SIGMAS", "IMAGE")
    RETURN_NAMES = ("sigmas", "plot")
    FUNCTION     = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": (
                    "INT",
                    {
                        "default": 8,
                        "min":     4,
                        "max":     12,
                        "step":    1,
                        "tooltip": "Sampling steps. Flow model turbo range: 4–12.",
                    },
                ),
                "sigma_min": (
                    "FLOAT",
                    {
                        "default": 0.03,
                        "min":     0.001,
                        "max":     0.5,
                        "step":    0.001,
                        "round":   False,
                        "tooltip": (
                            "Lower sigma bound. Flow models: typically 0.03–0.125.\n"
                            "BasicScheduler/simple at 8 steps ends at 0.125.\n"
                            "Lower values spend more time in fine detail."
                        ),
                    },
                ),
                "strategy": (
                    STRATEGIES,
                    {
                        "default": "simple",
                        "tooltip": (
                            "simple/linear : matches BasicScheduler — use as baseline.\n"
                            "beta          : middle-weighted spacing.\n"
                            "beta57        : skewed toward detail end.\n"
                            "karras        : karras curve in [0,1] space.\n"
                            "exponential   : exponential decay.\n"
                            "chaotic_*     : chaos-map driven spacing (seeded).\n"
                            "non_monotonic : linear base with reinjection events."
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min":     0,
                        "max":     0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Seed for chaotic and non_monotonic strategies.",
                    },
                ),
            },
        }

    def generate(self, steps, sigma_min, strategy, seed):
        if sigma_min >= 1.0:
            raise ValueError(
                f"BoyoExperimentalSigmaPlot: sigma_min ({sigma_min}) must be less than 1.0"
            )

        sigmas = _build_sigmas(steps, sigma_min, strategy, seed)

        print(
            f"[BoyoExperimentalSigmaPlot] {strategy} | {steps} steps\n"
            f"  sigmas: {[round(float(s), 4) for s in sigmas.tolist()]}"
        )

        plot_pil    = _render_plot(sigmas, strategy, steps, sigma_min)
        plot_tensor = _pil_to_comfy_image(plot_pil)

        return (sigmas, plot_tensor)


# ---------------------------------------------------------------------------
# Node registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "BoyoExperimentalSigmaPlot": BoyoExperimentalSigmaPlot,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoExperimentalSigmaPlot": "🧪 Sigma Plot (Experimental)",
}
