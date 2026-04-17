"""Custom model provider for Gemma4.

Installs Gemma4-specific behaviors that sit outside the transformer layer:
- embedding scaling (multiply embeddings by sqrt(hidden_size))
- logit softcapping (`final_logit_softcapping`)
- dual-RoPE (different rope_theta + partial-rotary for global vs sliding layers)
- layer_scalar buffers loaded from the HF checkpoint
"""
import json
import os

import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args

from slime_plugins.models.gemma4 import _load_hf_text_config


def model_provider(pre_process=True, post_process=True, vp_stage=None):
    args = get_args()
    config = core_transformer_config_from_args(args)

    transformer_layer_spec = import_module(args.spec)
    if callable(transformer_layer_spec):
        transformer_layer_spec = transformer_layer_spec(args, config, vp_stage)

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
    )

    _install_hooks(model, args, config, pre_process, post_process)
    return model


class DualRotaryEmbedding(torch.nn.Module):
    """Wraps a (global, local) pair of Megatron RotaryEmbedding modules.

    Returns a single concatenated `[seq, 1, 1, global_dim + local_dim]` tensor
    from forward(), with the global embedding first. `Gemma4TransformerLayer`
    splits this back per-layer based on whether the layer is sliding or global.

    The concat format is chosen so that downstream code that expects a single
    tensor (e.g. distributed checkpointing, CP sharding) continues to work.
    """

    def __init__(self, local_rope, global_rope, global_dim: int):
        super().__init__()
        self.local_rope = local_rope
        self.global_rope = global_rope
        self.global_dim = global_dim

    def get_rotary_seq_len(self, *args, **kwargs):
        # Both ropes share the same sequence-length logic (they only differ in
        # theta and partial-rotary); delegate to the local one.
        return self.local_rope.get_rotary_seq_len(*args, **kwargs)

    def forward(self, seq_len, **kwargs):
        global_emb = self.global_rope(seq_len, **kwargs)
        local_emb = self.local_rope(seq_len, **kwargs)
        return torch.cat([global_emb, local_emb], dim=-1)


def _install_hooks(model, args, config, pre_process, post_process):
    hf_text = _load_hf_text_config(args.hf_checkpoint)
    hidden_size = config.hidden_size
    # Match HF's `Gemma4TextScaledWordEmbedding` which casts the scale to the
    # model dtype. We store the bf16-rounded scalar to keep numerics identical.
    embed_scale = torch.tensor(hidden_size ** 0.5, dtype=torch.bfloat16).item()

    inner = model.module if hasattr(model, "module") else model

    # Embedding scaling — HF applies this inside the embedding module.
    if pre_process and hasattr(inner, "embedding"):
        def _embed_hook(module, inp, output):
            return output * embed_scale
        inner.embedding.register_forward_hook(_embed_hook)

    # Final logit softcapping — HF applies tanh(logits / cap) * cap.
    softcap = getattr(hf_text, "final_logit_softcapping", None)
    if post_process and softcap and hasattr(inner, "output_layer"):
        def _softcap_hook(module, inp, output):
            if isinstance(output, tuple):
                return (torch.tanh(output[0] / softcap) * softcap,) + output[1:]
            return torch.tanh(output / softcap) * softcap
        inner.output_layer.register_forward_hook(_softcap_hook)

    # Dual RoPE: replace Megatron's single rotary_pos_emb with a wrapper that
    # produces (global, local) RoPE side-by-side. Gemma4 uses partial-rotary
    # on global layers (implemented here by zeroing the tail of inv_freq).
    if hasattr(inner, "rotary_pos_emb"):
        from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

        rope_params = getattr(hf_text, "rope_parameters", {}) or {}
        full = rope_params.get("full_attention", {}) or {}
        sliding = rope_params.get("sliding_attention", {}) or {}
        global_theta = full.get("rope_theta", 1_000_000.0)
        local_theta = sliding.get("rope_theta", 10_000.0)
        global_head_dim = hf_text.global_head_dim
        global_partial = full.get("partial_rotary_factor", 0.25)

        local_rope = inner.rotary_pos_emb  # already built with args.rotary_base

        global_rope = RotaryEmbedding(
            kv_channels=global_head_dim,
            rotary_percent=1.0,
            rotary_base=global_theta,
        )
        # HF "proportional" RoPE: first (partial * head_dim // 2) inv_freq
        # entries are live, the rest are zero (no rotation on those dims).
        # Writing this to the existing buffer keeps device/dtype correct.
        rope_angles = int(global_partial * global_head_dim // 2)
        inv_freq_live = 1.0 / (
            global_theta ** (
                torch.arange(0, 2 * rope_angles, 2, dtype=torch.float) / global_head_dim
            )
        )
        nope = global_head_dim // 2 - rope_angles
        inv_freq = torch.cat([inv_freq_live, torch.zeros(nope)]) if nope > 0 else inv_freq_live
        global_rope.inv_freq.copy_(inv_freq.to(global_rope.inv_freq.device))

        inner.rotary_pos_emb = DualRotaryEmbedding(local_rope, global_rope, global_head_dim)
        # Layers split the concatenated tensor by this dim.
        config.dual_rope_global_dim = global_head_dim
        print(
            f"[Gemma4] DualRotaryEmbedding: local_theta={local_theta}, "
            f"global_theta={global_theta}, global_dim={global_head_dim}, "
            f"rope_angles={rope_angles} (nope={nope})"
        )

    # Load layer scalars from the HF checkpoint. These are buffers, not
    # parameters, and are applied once per layer after the MoE/MLP block.
    if hasattr(inner, "decoder") and args.hf_checkpoint:
        _load_layer_scalars(inner, args.hf_checkpoint, config)


def _load_layer_scalars(inner, hf_checkpoint, config):
    index_path = os.path.join(hf_checkpoint, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print(f"[Gemma4] No safetensors index at {index_path}; skipping layer scalars")
        return
    try:
        from safetensors import safe_open

        with open(index_path) as f:
            index = json.load(f)

        scalars: dict[int, float] = {}
        for key, filename in index["weight_map"].items():
            if "layer_scalar" not in key:
                continue
            layer_idx = int(key.split(".layers.")[1].split(".")[0])
            with safe_open(os.path.join(hf_checkpoint, filename), framework="pt", device="cpu") as sf:
                scalars[layer_idx] = sf.get_tensor(key).item()

        if not scalars:
            print("[Gemma4] No layer_scalar weights found in checkpoint")
            return

        # Under pipeline-parallelism, inner.decoder.layers holds only this
        # rank's local subset. Translate the local index back to the global
        # (HF 0-indexed) layer index so we apply the right scalar per layer.
        from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
        pp_offset = get_transformer_layer_offset(config)

        loaded = 0
        for i, layer in enumerate(inner.decoder.layers):
            if hasattr(layer, "layer_scalar"):
                layer.layer_scalar.fill_(scalars.get(i + pp_offset, 1.0))
                loaded += 1
        print(
            f"[Gemma4] Applied {loaded}/{len(inner.decoder.layers)} layer scalars "
            f"(pp_offset={pp_offset}, range={min(scalars.values()):.4f}..{max(scalars.values()):.4f})"
        )
    except Exception as e:
        # Don't crash training on a scalar-loading glitch; the default (all
        # ones) produces a functional model, just not numerically identical
        # to HF. Warn loudly.
        print(f"[Gemma4] WARNING: failed to load layer scalars: {type(e).__name__}: {e}")
