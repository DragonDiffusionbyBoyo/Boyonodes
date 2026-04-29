"""
boyo_prompt_relay_nodes.py

Based on ComfyUI-PromptRelay by kijai (Jukka Seppänen)
https://github.com/kijai/ComfyUI-PromptRelay
Ported into BoyoNodes with attribution and extended with BoyoPromptRelayLoraGate.

Original node classes renamed to avoid clashing with the upstream pack:
  PromptRelayEncode         → BoyoPromptRelayEncode
  PromptRelayEncodeTimeline → BoyoPromptRelayEncodeTimeline

New node:
  BoyoPromptRelayLoraGate — applies a LoRA to a single temporal segment
                             using the same Gaussian gating as Prompt Relay.
"""

import logging
import math

import torch
import torch.nn as nn

import comfy.utils
import folder_paths

from comfy_api.latest import io

from .prompt_relay import (
    get_raw_tokenizer,
    map_token_indices,
    build_segments,
    create_mask_fn,
    distribute_segment_lengths,
)
from .patches import detect_model_type, apply_patches

log = logging.getLogger(__name__)


# ── Copied verbatim from kijai/ComfyUI-PromptRelay nodes.py ──────────────────

def _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames):
    """Convert pixel-space segment lengths to integer latent-space lengths using the
    largest-remainder method. Targets the full `latent_frames` when the pixel sum looks
    like full coverage (within one stride of latent_frames * stride). Otherwise targets
    round(total_pixel / temporal_stride) so partial-coverage timelines stay partial.
    """
    if not pixel_lengths:
        return []
    total_pixel = sum(pixel_lengths)
    if total_pixel <= 0:
        return [1] * len(pixel_lengths)

    naive_total = max(1, round(total_pixel / temporal_stride))
    target_total = min(latent_frames, naive_total)
    if target_total >= latent_frames - 1:
        target_total = latent_frames

    exact = [p * target_total / total_pixel for p in pixel_lengths]
    result = [int(e) for e in exact]
    diff = target_total - sum(result)
    if diff > 0:
        order = sorted(range(len(exact)), key=lambda i: -(exact[i] - int(exact[i])))
        for k in range(diff):
            result[order[k % len(order)]] += 1

    for i in range(len(result)):
        if result[i] < 1:
            max_idx = max(range(len(result)), key=lambda j: result[j])
            if result[max_idx] > 1:
                result[max_idx] -= 1
                result[i] = 1

    return result


def _encode_relay(model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon):
    locals_list = [p.strip() for p in local_prompts.split("|") if p.strip()]
    if not locals_list:
        raise ValueError("At least one local prompt is required (separate with |)")

    arch, patch_size, temporal_stride = detect_model_type(model)

    samples = latent["samples"]
    latent_frames = samples.shape[2]
    tokens_per_frame = (samples.shape[3] // patch_size[1]) * (samples.shape[4] // patch_size[2])

    parsed_lengths = None
    if segment_lengths.strip():
        pixel_lengths = [int(x.strip()) for x in segment_lengths.split(",") if x.strip()]
        parsed_lengths = _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames)

    raw_tokenizer = get_raw_tokenizer(clip)
    full_prompt, token_ranges = map_token_indices(raw_tokenizer, global_prompt, locals_list)

    log.info("[BoyoPromptRelay] Global: tokens [0:%d] (%d tokens)", token_ranges[0][0], token_ranges[0][0])
    for i, (s, e) in enumerate(token_ranges):
        log.info("[BoyoPromptRelay] Segment %d: tokens [%d:%d] (%d tokens)", i, s, e, e - s)

    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))

    effective_lengths = distribute_segment_lengths(len(locals_list), latent_frames, parsed_lengths)

    log.info(
        "[BoyoPromptRelay] Latent: %d frames, %d tokens/frame, segments: %s",
        latent_frames, tokens_per_frame, effective_lengths,
    )

    q_token_idx = build_segments(token_ranges, effective_lengths, epsilon)
    mask_fn = create_mask_fn(q_token_idx, tokens_per_frame, latent_frames)

    patched = model.clone()
    apply_patches(patched, arch, mask_fn)

    # Store segment metadata so BoyoPromptRelayLoraGate nodes can gate LoRAs temporally
    patched.model_options["boyo_pr_segments"] = q_token_idx
    patched.model_options["boyo_pr_latent_frames"] = latent_frames
    patched.model_options["boyo_pr_tokens_per_frame"] = tokens_per_frame

    return patched, conditioning


# ── Renamed node classes (kijai originals, prefixed Boyo) ────────────────────

class BoyoPromptRelayEncode(io.ComfyNode):
    """Encodes temporal local prompts and patches the model for Prompt Relay.
    Original: kijai/ComfyUI-PromptRelay — PromptRelayEncode
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BoyoPromptRelayEncode",
            display_name="Boyo Prompt Relay Encode",
            category="conditioning/boyo_prompt_relay",
            description=(
                "Encodes a global prompt combined with temporal local prompts and patches the model "
                "for Prompt Relay temporal control. Local prompts are separated by |. "
                "Use a standard CLIPTextEncode for the negative prompt. "
                "(Based on kijai/ComfyUI-PromptRelay)"
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Latent.Input("latent", tooltip="Empty latent video — dimensions are read from its shape."),
                io.String.Input(
                    "global_prompt", multiline=True, default="",
                    tooltip="Conditions the entire video. Anchors persistent characters, objects, and scene context.",
                ),
                io.String.Input(
                    "local_prompts", multiline=True, default="",
                    tooltip="Ordered prompts for each temporal segment, separated by |",
                ),
                io.String.Input(
                    "segment_lengths", default="",
                    tooltip="Comma-separated pixel space frame counts per segment. Leave empty to auto-distribute evenly.",
                ),
                io.Float.Input(
                    "epsilon", default=1e-3, min=1e-6, max=0.99, step=1e-4,
                    tooltip="Penalty decay parameter. Values below ~0.1 all produce sharp boundaries (paper default 0.001). For softer transitions, try 0.5 or higher.",
                ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon) -> io.NodeOutput:
        patched, conditioning = _encode_relay(
            model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon,
        )
        return io.NodeOutput(patched, conditioning)


class BoyoPromptRelayEncodeTimeline(io.ComfyNode):
    """WYSIWYG timeline variant — segments and lengths come from a visual editor in the node UI.
    Original: kijai/ComfyUI-PromptRelay — PromptRelayEncodeTimeline
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BoyoPromptRelayEncodeTimeline",
            display_name="Boyo Prompt Relay Encode (Timeline)",
            category="conditioning/boyo_prompt_relay",
            description=(
                "Same as Boyo Prompt Relay Encode, but local prompts and segment lengths are edited "
                "visually as draggable blocks on a timeline. The max_frames input only sets the "
                "timeline scale (pixel space) — actual frame count is still read from the latent. "
                "(Based on kijai/ComfyUI-PromptRelay)"
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Latent.Input("latent", tooltip="Empty latent video — dimensions are read from its shape."),
                io.String.Input(
                    "global_prompt", multiline=True, default="",
                    tooltip="Conditions the entire video. Anchors persistent characters, objects, and scene context.",
                ),
                io.Int.Input(
                    "max_frames", default=129, min=1, max=10000, step=1,
                    tooltip="Total timeline length in pixel-space frames. Used by the editor for visual scale only.",
                ),
                io.String.Input(
                    "timeline_data", default="",
                    tooltip="JSON state of the timeline editor (auto-managed; do not edit by hand).",
                ),
                io.String.Input(
                    "local_prompts", multiline=True, default="",
                    tooltip="Auto-populated from the timeline editor.",
                ),
                io.String.Input(
                    "segment_lengths", default="",
                    tooltip="Auto-populated from the timeline editor (pixel-space frame counts).",
                ),
                io.Float.Input(
                    "epsilon", default=1e-3, min=1e-6, max=0.99, step=1e-4,
                    tooltip="Penalty decay parameter. Values below ~0.1 all produce sharp boundaries (paper default 0.001). For softer transitions, try 0.5 or higher.",
                ),
                io.Float.Input(
                    "fps", default=24.0, min=0.1, max=240.0, step=0.1, optional=True,
                    tooltip="Frames per second — only affects how time is displayed in the timeline editor when time_units is set to 'seconds'.",
                ),
                io.Combo.Input(
                    "time_units", options=["frames", "seconds"], default="frames", optional=True,
                    tooltip="Display the ruler, segment ranges, length input, and total in frames or seconds. Internal storage is always pixel-space frames.",
                ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, latent, global_prompt, max_frames, timeline_data, local_prompts, segment_lengths, epsilon, fps=24.0, time_units="frames") -> io.NodeOutput:
        patched, conditioning = _encode_relay(
            model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon,
        )
        return io.NodeOutput(patched, conditioning)


# ── BoyoPromptRelayLoraGate ───────────────────────────────────────────────────

class _TemporalGatedLoRA(nn.Module):
    """Wraps a linear layer and adds a temporally-gated LoRA delta.

    For video tokens, each token's contribution from the LoRA is scaled by a
    per-frame gate weight derived from the same Gaussian window that Prompt Relay
    uses for attention masking. Tokens outside the segment window contribute
    progressively less LoRA influence down to effectively zero.

    For non-video sequences (text context keys/values etc.), the gate is not
    applied — those receive the LoRA at full requested strength, which is the
    correct behaviour for cross-attention style/content keys.
    """

    def __init__(self, original, lora_down, lora_up, scale, strength, gate_weights, tokens_per_frame):
        super().__init__()
        self.original = original
        # Store as plain tensors — device/dtype are reconciled in forward
        self._lora_down = lora_down.cpu()
        self._lora_up = lora_up.cpu()
        self.scale = scale
        self.strength = strength
        self._gate_weights = gate_weights.cpu()   # [latent_frames]
        self.tokens_per_frame = tokens_per_frame

    def forward(self, x, *args, **kwargs):
        base = self.original(x, *args, **kwargs)

        lora_down = self._lora_down.to(device=x.device, dtype=x.dtype)
        lora_up = self._lora_up.to(device=x.device, dtype=x.dtype)

        # Full LoRA delta for all tokens
        delta = (x @ lora_down.T) @ lora_up.T
        delta = delta * (self.scale * self.strength)

        if x.dim() < 2:
            return base + delta

        seq_len = x.shape[-2]
        n_frames = len(self._gate_weights)
        n_video_tokens = n_frames * self.tokens_per_frame

        if seq_len >= n_video_tokens and n_video_tokens > 0:
            gate_weights = self._gate_weights.to(device=x.device, dtype=x.dtype)

            # Per-token gate for the video portion
            frame_idx = torch.arange(n_video_tokens, device=x.device) // self.tokens_per_frame
            frame_idx = frame_idx.clamp(0, n_frames - 1)
            video_gates = gate_weights[frame_idx]  # [n_video_tokens]

            if seq_len > n_video_tokens:
                # Non-video tokens (audio etc.) — zero gate, LoRA not meaningful here
                extra = torch.zeros(seq_len - n_video_tokens, device=x.device, dtype=x.dtype)
                token_gates = torch.cat([video_gates, extra])
            else:
                token_gates = video_gates

            # Reshape to broadcast over all batch dims and the feature dim
            view_shape = (1,) * (x.dim() - 2) + (seq_len, 1)
            delta = delta * token_gates.view(view_shape)

        return base + delta


def _build_gate_weights(seg, latent_frames):
    """Build a [latent_frames] float32 tensor of Gaussian gate weights for one segment.

    The gate mirrors the Prompt Relay penalty in reverse:
      gate = 1.0  at the segment centre
      gate → 0.0  beyond the segment window (same sigma as the attention mask)
    """
    midpoint = float(seg["midpoint"])
    window = float(seg["window"])
    sigma = float(seg["sigma"])

    weights = []
    for frame in range(latent_frames):
        d = abs(frame - midpoint)
        cost = (max(d - window, 0.0) ** 2) / (2.0 * sigma ** 2)
        weights.append(math.exp(-cost))

    return torch.tensor(weights, dtype=torch.float32)


def _get_module(diffusion_model, local_path):
    """Navigate a module tree by dot-separated path relative to diffusion_model."""
    m = diffusion_model
    for part in local_path.split("."):
        m = getattr(m, part)
    return m


def _apply_gated_lora(model_clone, lora_path, segment, strength, tokens_per_frame, latent_frames):
    """Load a LoRA file and install temporally-gated wrappers on model_clone.

    Uses ComfyUI's own key-normalisation (comfy.loras) to map LoRA file keys to
    model parameter paths, so any LoRA format that ComfyUI understands is handled.
    Skips conv weights, bias terms, and anything that isn't a plain linear LoRA pair.
    """
    try:
        import comfy.loras
        import comfy.utils
    except ImportError:
        log.error("[BoyoLoraGate] comfy.loras not available in this ComfyUI build — cannot apply LoRA gate.")
        return

    lora_file = comfy.utils.load_torch_file(lora_path, safe_load=True)

    key_map = {}
    try:
        comfy.loras.model_lora_keys_unet(model_clone.model, key_map)
    except Exception as e:
        log.error("[BoyoLoraGate] Failed to build LoRA key map: %s", e)
        return

    try:
        patches = comfy.loras.load_lora(lora_file, key_map)
    except Exception as e:
        log.error("[BoyoLoraGate] Failed to parse LoRA file: %s", e)
        return

    gate_weights = _build_gate_weights(segment, latent_frames)
    diffusion_model = model_clone.get_model_object("diffusion_model")

    applied = 0
    skipped = 0

    for param_key, patch_data in patches.items():
        # param_key e.g. "diffusion_model.transformer_blocks.0.attn1.to_q.weight"
        if not param_key.endswith(".weight"):
            skipped += 1
            continue

        module_path = param_key[:-7]  # strip ".weight" → module path

        if not module_path.startswith("diffusion_model."):
            skipped += 1
            continue

        local_path = module_path[len("diffusion_model."):]

        # Unpack (alpha, lora_up, lora_down [, optional extras])
        try:
            alpha = patch_data[0]
            lora_up = patch_data[1]
            lora_down = patch_data[2]
        except (TypeError, IndexError):
            skipped += 1
            continue

        if lora_up is None or lora_down is None:
            skipped += 1
            continue

        # Only handle plain linear LoRA — skip conv layers etc.
        if lora_down.dim() != 2 or lora_up.dim() != 2:
            skipped += 1
            continue

        rank = lora_down.shape[0]
        scale = float(alpha) / rank if alpha is not None else 1.0

        # If a previous LoraGate already patched this module path, wrap that
        # wrapper so both gates stack correctly rather than the second replacing the first
        if module_path in model_clone.object_patches:
            target = model_clone.object_patches[module_path]
        else:
            try:
                target = _get_module(diffusion_model, local_path)
            except AttributeError:
                skipped += 1
                continue

        wrapped = _TemporalGatedLoRA(
            original=target,
            lora_down=lora_down,
            lora_up=lora_up,
            scale=scale,
            strength=strength,
            gate_weights=gate_weights,
            tokens_per_frame=tokens_per_frame,
        )

        model_clone.add_object_patch(module_path, wrapped)
        applied += 1

    log.info(
        "[BoyoLoraGate] Segment midpoint=%.1f window=%.1f — applied %d patches, skipped %d",
        float(segment["midpoint"]), float(segment["window"]), applied, skipped,
    )


class BoyoPromptRelayLoraGate(io.ComfyNode):
    """Applies a LoRA to one temporal segment defined by a Boyo Prompt Relay node.

    Chain up to five of these after BoyoPromptRelayEncode / BoyoPromptRelayEncodeTimeline,
    one per segment. Bypass any gate you are not using. Strength 0.0 is a no-op passthrough.
    The LoRA influence follows the same Gaussian window used by Prompt Relay — full strength
    at the segment centre, fading to zero at the boundaries.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BoyoPromptRelayLoraGate",
            display_name="Boyo Prompt Relay LoRA Gate",
            category="conditioning/boyo_prompt_relay",
            description=(
                "Applies a LoRA with Gaussian temporal gating to one Prompt Relay segment. "
                "Chain multiple gates — one per segment. Bypass unused gates. "
                "segment_index is zero-based and must match the segment order in the Prompt Relay node."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "lora_name",
                    folder_paths.get_filename_list("loras"),
                    tooltip="LoRA to apply to this segment.",
                ),
                io.Int.Input(
                    "segment_index", default=0, min=0, max=99, step=1,
                    tooltip="Zero-based index of the Prompt Relay segment this LoRA targets. Segment 0 is the first local prompt.",
                ),
                io.Float.Input(
                    "strength", default=0.8, min=0.0, max=2.0, step=0.01,
                    tooltip="LoRA strength at the segment centre. Fades to zero outside the segment window.",
                ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, model, lora_name, segment_index, strength) -> io.NodeOutput:
        # Passthrough if strength is zero — no need to touch the model
        if strength == 0.0:
            return io.NodeOutput(model)

        segments = model.model_options.get("boyo_pr_segments")
        latent_frames = model.model_options.get("boyo_pr_latent_frames")
        tokens_per_frame = model.model_options.get("boyo_pr_tokens_per_frame")

        if segments is None:
            log.warning(
                "[BoyoLoraGate] No Prompt Relay segment metadata on this model. "
                "Ensure the model comes from a Boyo Prompt Relay Encode node. Passing through."
            )
            return io.NodeOutput(model)

        if segment_index >= len(segments):
            log.warning(
                "[BoyoLoraGate] segment_index %d is out of range — only %d segment(s) defined. "
                "Passing through unchanged.",
                segment_index, len(segments),
            )
            return io.NodeOutput(model)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None:
            raise ValueError(f"[BoyoLoraGate] LoRA file not found: {lora_name}")

        patched = model.clone()
        _apply_gated_lora(
            patched,
            lora_path,
            segments[segment_index],
            strength,
            tokens_per_frame,
            latent_frames,
        )

        # Forward segment metadata so further gates in the chain can read it
        patched.model_options["boyo_pr_segments"] = segments
        patched.model_options["boyo_pr_latent_frames"] = latent_frames
        patched.model_options["boyo_pr_tokens_per_frame"] = tokens_per_frame

        return io.NodeOutput(patched)


# ── Mappings ──────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "BoyoPromptRelayEncode": BoyoPromptRelayEncode,
    "BoyoPromptRelayEncodeTimeline": BoyoPromptRelayEncodeTimeline,
    "BoyoPromptRelayLoraGate": BoyoPromptRelayLoraGate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoPromptRelayEncode": "Boyo Prompt Relay Encode",
    "BoyoPromptRelayEncodeTimeline": "Boyo Prompt Relay Encode (Timeline)",
    "BoyoPromptRelayLoraGate": "Boyo Prompt Relay LoRA Gate",
}
