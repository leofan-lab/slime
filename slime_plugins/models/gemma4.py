import copy
import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.spec_utils import ModuleSpec

try:
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4TextAttention,
        Gemma4TextRotaryEmbedding,
        Gemma4RMSNorm,
    )
except ImportError:
    Gemma4TextAttention = None
    Gemma4TextRotaryEmbedding = None
    Gemma4RMSNorm = None

from .hf_attention import HuggingfaceAttention


class Gemma4EmbeddingScaling(nn.Module):
    """Post-embedding hook that scales by sqrt(hidden_size), matching HF Gemma4."""

    def __init__(self, hidden_size):
        super().__init__()
        self.scale = hidden_size ** 0.5

    def forward(self, embeddings):
        return embeddings * self.scale


class Gemma4Attention(HuggingfaceAttention):
    """Wraps HuggingFace Gemma4TextAttention for use inside Megatron's pipeline."""

    # Shared across all Gemma4Attention instances
    _shared_kv_states = {}
    _shared_rotary_emb = None
    _shared_rotary_config = None

    def __init__(self, args, config, layer_number, cp_comm_type="p2p", pg_collection=None):
        super().__init__(args, config, layer_number, cp_comm_type, pg_collection)
        if Gemma4TextAttention is None:
            raise ImportError("Please install transformers with Gemma 4 support.")

        text_config = self.hf_config.text_config if hasattr(self.hf_config, "text_config") else self.hf_config
        text_config._attn_implementation = "sdpa"  # flash_attention_2 has head_dim=256 limit issues
        self.attn = Gemma4TextAttention(text_config, self.hf_layer_idx)
        self.input_layernorm = Gemma4RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)

        self.layer_type = text_config.layer_types[self.hf_layer_idx]

        # Share a single RoPE instance across all layers, matching HF's Gemma4TextModel.
        # Defer creation to first forward call so inv_freq is computed on GPU.
        if Gemma4Attention._shared_rotary_emb is None:
            Gemma4Attention._shared_rotary_config = text_config

    def hf_forward(self, hidden_states, packed_seq_params):
        batch_size, seq_len, _ = hidden_states.shape
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)

        if Gemma4Attention._shared_rotary_emb is None:
            Gemma4Attention._shared_rotary_emb = Gemma4TextRotaryEmbedding(
                Gemma4Attention._shared_rotary_config, device=hidden_states.device
            )

        position_embeddings = Gemma4Attention._shared_rotary_emb(hidden_states, position_ids, layer_type=self.layer_type)

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=None,
            shared_kv_states=Gemma4Attention._shared_kv_states,
        )
        return hidden_states

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Override to save HF wrapper params without Megatron sharding validation."""
        from megatron.core.dist_checkpointing.mapping import ShardedTensor
        sharded_state_dict = {}
        for name, param in self.state_dict().items():
            full_key = f"{prefix}{name}"
            sharded_state_dict[full_key] = ShardedTensor.from_rank_offsets(
                full_key, param, replica_id=0, allow_shape_mismatch=True,
            )
        return sharded_state_dict


def get_gemma4_spec(args, config, vp_stage):
    """Return a single layer spec with HF Gemma4TextAttention replacing Megatron attention."""
    # Gemma uses GeGLU (GELU-tanh + GLU), not SwiGLU (SiLU + GLU).
    # Override here since Megatron's --swiglu flag incorrectly sets F.silu.
    config.activation_func = functools.partial(F.gelu, approximate="tanh")
    config.bias_activation_fusion = False

    # Gemma 4 has heterogeneous layers (sliding vs full attention with different shapes)
    config.hetereogenous_dist_checkpoint = True
    base_spec = get_gpt_layer_with_transformer_engine_spec(
        qk_layernorm=False,
        post_self_attn_layernorm=True,
        post_mlp_layernorm=True,
    )
    spec = copy.deepcopy(base_spec)
    spec.submodules.self_attention = ModuleSpec(
        module=Gemma4Attention,
        params={"args": args},
    )
    # Remove q/k layernorm — handled inside the HF wrapper
    spec.submodules.self_attention_q_layernorm = None
    spec.submodules.self_attention_k_layernorm = None

    # Use non-fused LayerNorm + Linear for MLP to match HF numerics.
    # The fused TELayerNormColumnParallelLinear produces numerically different
    # results from HF's separate RMSNorm + nn.Linear due to kernel-level
    # differences in accumulation order and intermediate precision.
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    spec.submodules.mlp.submodules.linear_fc1 = TEColumnParallelLinear
    spec.submodules.mlp.metainfo["fuse_pre_mlp_layernorm"] = False
    spec.submodules.pre_mlp_layernorm = TESpecProvider().layer_norm()

    return spec


# ---------------------------------------------------------------------------
# Gemma4-specific hooks: embedding scaling + logit softcapping
# These are installed on the GPTModel after construction.
# ---------------------------------------------------------------------------

_EMBED_SCALE = None  # set from config at hook install time
_SOFTCAP = None


def _embedding_post_hook(module, input, output):
    """Scale embeddings by sqrt(hidden_size) after the embedding layer.
    
    Uses bf16 precision to match HF's Gemma4TextScaledWordEmbedding which casts
    the scale to the weight dtype (bf16) before multiplying.
    """
    return output * _EMBED_SCALE


def _output_post_hook(module, input, output):
    """Apply logit softcapping: tanh(logits / cap) * cap."""
    if isinstance(output, tuple):
        logits = output[0]
        return (torch.tanh(logits / _SOFTCAP) * _SOFTCAP,) + output[1:]
    return torch.tanh(output / _SOFTCAP) * _SOFTCAP


def install_gemma4_hooks(model, hidden_size, final_logit_softcapping=None, hf_checkpoint=None):
    """Install Gemma4-specific forward hooks on a GPTModel.

    Call this after model construction. Adds:
      1. Embedding scaling by sqrt(hidden_size)
      2. Logit softcapping with tanh (if softcapping value provided)
      3. Clears KV sharing state before each forward pass
      4. Per-layer scalar from HF checkpoint
    """
    global _EMBED_SCALE, _SOFTCAP
    _EMBED_SCALE = torch.tensor(hidden_size ** 0.5, dtype=torch.bfloat16).item()
    _SOFTCAP = final_logit_softcapping

    # Unwrap if wrapped (e.g. Float16Module)
    inner = model.module if hasattr(model, 'module') else model

    # Hook 1: embedding scaling
    if hasattr(inner, 'embedding'):
        inner.embedding.register_forward_hook(_embedding_post_hook)

    # Hook 2: logit softcapping
    if _SOFTCAP and hasattr(inner, 'output_layer'):
        inner.output_layer.register_forward_hook(_output_post_hook)

    # Hook 3: clear KV sharing state before each forward
    def _clear_kv_pre_hook(module, args):
        Gemma4Attention._shared_kv_states.clear()

    model.register_forward_pre_hook(_clear_kv_pre_hook)

    # Hook 4: per-layer scalar
    if hf_checkpoint and hasattr(inner, 'decoder'):
        import json
        from safetensors import safe_open

        # Load layer_scalar values from safetensors (avoid loading full model)
        with open(f"{hf_checkpoint}/model.safetensors.index.json") as f:
            index = json.load(f)

        scalars = {}
        for key, filename in index["weight_map"].items():
            if "layer_scalar" in key:
                layer_idx = int(key.split(".layers.")[1].split(".")[0])
                with safe_open(f"{hf_checkpoint}/{filename}", framework="pt", device="cpu") as sf:
                    scalars[layer_idx] = sf.get_tensor(key).item()

        for i, layer in enumerate(inner.decoder.layers):
            scalar = scalars.get(i, 1.0)

            def _make_scalar_hook(s):
                def _hook(module, input, output):
                    hidden_states, context = output
                    return hidden_states * s, context
                return _hook

            layer.register_forward_hook(_make_scalar_hook(scalar))
            if i < 3 or i >= 57:
                print(f"[Gemma4] Layer {i}: scalar={scalar:.6f}")
