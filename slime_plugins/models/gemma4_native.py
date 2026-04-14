"""Native Megatron Gemma4 transformer layer and config.

Extends the Gemma3 implementation from mbridge with Gemma4-specific features:
- Heterogeneous attention: global layers use head_dim=512, num_kv_heads=4;
  sliding layers use head_dim=256, num_kv_heads=16.
- attention_k_eq_v: global layers reuse K output as V (no v_proj).
- v_norm: RMSNorm without learnable scale applied to V states.
- layer_scalar: buffer multiplied after residual (not learned).
- final_logit_softcapping: applied to output logits in the model wrapper.
"""

import functools
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import make_viewless_tensor

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TENorm,
        TERowParallelLinear,
    )
    HAVE_TE = True
except ImportError:
    HAVE_TE = False

from mbridge.models.gemma3.transformer_config import Gemma3TransformerConfig


# Gemma uses GeGLU, not SwiGLU.
_gelu_tanh = functools.partial(F.gelu, approximate="tanh")


@dataclass
class Gemma4TransformerConfig(Gemma3TransformerConfig):
    """Gemma4-specific config extending Gemma3."""
    # Heterogeneous attention: global layers use different head_dim and num_kv_heads
    global_kv_channels: int = 512
    global_num_query_groups: int = 4
    attention_k_eq_v: bool = True  # global layers: V = K (no v_proj)
    final_logit_softcapping: float = 30.0


class VNorm(nn.Module):
    """RMSNorm without learnable scale, matching Gemma4's v_norm."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        return (x * torch.pow(x.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)).to(dtype)


@dataclass
class Gemma4TransformerLayerSubmodules(TransformerLayerSubmodules):
    post_attention_layernorm: ModuleSpec | type = IdentityOp
    post_feedforward_layernorm: ModuleSpec | type = IdentityOp


class Gemma4TransformerLayer(TransformerLayer):
    """Gemma4 transformer layer with heterogeneous attention and layer_scalar."""

    def __init__(
        self,
        config: Gemma4TransformerConfig,
        submodules: Gemma4TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
        **kwargs,
    ):
        self.is_sliding = bool(layer_number % config.sliding_window_pattern)
        self._is_global = not self.is_sliding

        # For global layers, temporarily override config to use different head_dim/kv_heads
        if self._is_global:
            orig_kv_channels = config.kv_channels
            orig_num_query_groups = config.num_query_groups
            config.kv_channels = config.global_kv_channels
            config.num_query_groups = config.global_num_query_groups

        super().__init__(
            config=config, submodules=submodules,
            layer_number=layer_number, hidden_dropout=hidden_dropout,
            **kwargs,
        )

        # Restore config after super().__init__ built the attention
        if self._is_global:
            config.kv_channels = orig_kv_channels
            config.num_query_groups = orig_num_query_groups

        # Tell the attention module whether this is a global layer
        self.self_attention._is_global = self._is_global

        # Replace TE core attention with PyTorch SDPA for all layers
        # (TE flash attention produces different results for Gemma4's partial RoPE + GQA)
        self.self_attention.core_attention = SDPACoreAttention(
            config=config,
            layer_number=self.layer_number,
            attn_mask_type=AttnMaskType.causal,
            softmax_scale=config.softmax_scale,
        )

        # Post-attention and post-feedforward layernorms (Gemma-specific)
        self.post_attention_layernorm = build_module(
            submodules.post_attention_layernorm,
            config=self.config, hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.post_feedforward_layernorm = build_module(
            submodules.post_feedforward_layernorm,
            config=self.config, hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # V norm (no learnable scale) — applied to value states
        kv_channels = config.global_kv_channels if self._is_global else config.kv_channels
        self.v_norm = VNorm(kv_channels, eps=config.layernorm_epsilon)

        # Layer scalar (buffer, not learned)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        inference_context=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        **kwargs,
    ):
        # Select per-layer rotary embeddings and attention mask
        if isinstance(rotary_pos_emb, tuple) and isinstance(attention_mask, tuple):
            if self.is_sliding:
                rotary_pos_emb = rotary_pos_emb[1]
                attention_mask = attention_mask[1]
            else:
                rotary_pos_emb = rotary_pos_emb[0]
                attention_mask = attention_mask[0]

        # Global layers use partial RoPE (25% of head_dim=512 = 128 dims)
        # Local layers use full RoPE (100% of head_dim=256 = 256 dims)
        # Megatron generates RoPE with rotary_percent=1.0 for local head_dim=256
        # For global layers, truncate to partial_rotary_factor * global_head_dim / 2
        if not self.is_sliding and rotary_pos_emb is not None:
            global_rope_dim = int(self.config.global_kv_channels * 0.25 / 2)
            if isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = tuple(r[..., :global_rope_dim] for r in rotary_pos_emb)
            else:
                rotary_pos_emb = rotary_pos_emb[..., :global_rope_dim]

        residual = hidden_states

        extra_kwargs = {}
        try:
            from megatron.core import __version__
            from packaging import version
            if version.parse(__version__) >= version.parse("0.12.0"):
                extra_kwargs["inference_context"] = inference_context
            else:
                extra_kwargs["inference_params"] = inference_params
        except Exception:
            extra_kwargs["inference_params"] = inference_params

        # Input layernorm
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention
        hidden_states, hidden_states_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            **extra_kwargs,
        )

        if hidden_states_bias is not None:
            hidden_states = hidden_states + hidden_states_bias
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)
        hidden_states, hidden_states_bias = self.mlp(pre_mlp_layernorm_output)
        if hidden_states_bias is not None:
            hidden_states = hidden_states + hidden_states_bias
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Layer scalar
        hidden_states = hidden_states * self.layer_scalar

        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True,
        )

        if self.config.external_cuda_graph and self.training:
            return output
        return output, context


class SDPACoreAttention(nn.Module):
    """Simple core attention using PyTorch SDPA. Supports head_dim > 256."""
    def __init__(self, config, layer_number, attn_mask_type, attention_type="self",
                 attention_dropout=None, softmax_scale=None, **kwargs):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = config.attention_dropout if attention_dropout is None else attention_dropout

    @staticmethod
    def _make_varlen_causal_mask(cu_seqlens, total_len, device, dtype):
        """Build a [1, 1, T, T] causal mask from cu_seqlens for packed sequences."""
        mask = torch.full((total_len, total_len), float("-inf"), device=device, dtype=dtype)
        for i in range(len(cu_seqlens) - 1):
            s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            # causal within each sequence
            seq_mask = torch.triu(torch.zeros(e - s, e - s, device=device, dtype=dtype), diagonal=1).fill_(float("-inf"))
            seq_mask = torch.where(torch.triu(torch.ones(e - s, e - s, device=device, dtype=torch.bool), diagonal=1), float("-inf"), 0.0)
            mask[s:e, s:e] = seq_mask
        return mask.unsqueeze(0).unsqueeze(0)

    def _expand_kv_for_gqa(self, q, k, v):
        """Expand K/V heads to match Q heads for GQA (needed when SDPA can't broadcast)."""
        nq, nk = q.shape[1], k.shape[1]
        if nq != nk:
            repeat = nq // nk
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        return k, v

    def forward(self, query, key, value, attention_mask=None, attn_mask_type=None,
                packed_seq_params=None, **kwargs):
        is_thd = query.dim() == 3
        if is_thd:
            # Packed path: [t, np, hn] -> [1, np, t, hn]
            q = query.unsqueeze(0).permute(0, 2, 1, 3)
            k = key.unsqueeze(0).permute(0, 2, 1, 3)
            v = value.unsqueeze(0).permute(0, 2, 1, 3)
            k, v = self._expand_kv_for_gqa(q, k, v)

            if packed_seq_params is not None:
                attn_mask = self._make_varlen_causal_mask(
                    packed_seq_params.cu_seqlens_q, query.shape[0], query.device, query.dtype
                )
            else:
                attn_mask = None

            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p if self.training else 0.0,
                scale=self.softmax_scale, is_causal=(attn_mask is None),
            )
            # [1, np, t, hn] -> [t, np*hn]
            return out.permute(0, 2, 1, 3).reshape(query.shape[0], -1)
        else:
            # Standard path: [sq, b, np, hn] -> [b, np, sq, hn]
            q = query.permute(1, 2, 0, 3)
            k = key.permute(1, 2, 0, 3)
            v = value.permute(1, 2, 0, 3)
            k, v = self._expand_kv_for_gqa(q, k, v)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout_p if self.training else 0.0,
                scale=self.softmax_scale, is_causal=True,
            )
            # [b, np, sq, hn] -> [sq, b, np*hn]
            return out.permute(2, 0, 1, 3).reshape(out.size(2), out.size(0), -1)


class Gemma4SelfAttention(SelfAttention):
    """SelfAttention with Gemma4-specific modifications:
    - v_norm: RMSNorm without learnable scale applied to value states
    - attention_k_eq_v: for global layers, V = K (after k_norm, before v_norm)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_global = False  # set by Gemma4TransformerLayer after construction
        self.v_norm = VNorm(self.hidden_size_per_attention_head, eps=self.config.layernorm_epsilon)

    def get_query_key_value_tensors(
        self, hidden_states, key_value_states=None, output_gate=False, split_qkv=True
    ):
        # For K=V global layers, we need the raw K before k_norm/rope.
        # Save it by temporarily disabling k_layernorm, splitting, then applying manually.
        if self._is_global and self.config.attention_k_eq_v and split_qkv:
            saved_k_ln = self.k_layernorm
            self.k_layernorm = None  # skip k_norm in parent
            result = super().get_query_key_value_tensors(
                hidden_states, key_value_states, output_gate=output_gate, split_qkv=True
            )
            self.k_layernorm = saved_k_ln  # restore

            if output_gate:
                query, key, value, gate = result
            else:
                query, key, value = result

            # V = v_norm(raw_k_proj), K = k_norm(raw_k_proj)
            # At this point key = raw k_proj output (k_norm was skipped)
            value = self.v_norm(key.clone())
            if saved_k_ln is not None:
                key = saved_k_ln(key)

            if output_gate:
                return query, key, value, gate
            return query, key, value

        result = super().get_query_key_value_tensors(
            hidden_states, key_value_states, output_gate=output_gate, split_qkv=split_qkv
        )
        if not split_qkv:
            return result

        if output_gate:
            query, key, value, gate = result
        else:
            query, key, value = result

        # Apply v_norm to all layers (local layers have separate V from QKV split)
        value = self.v_norm(value)

        if output_gate:
            return query, key, value, gate
        return query, key, value


def get_gemma4_layer_spec_te() -> ModuleSpec:
    """Layer spec for Gemma4 using native Megatron attention with TE."""
    return ModuleSpec(
        module=Gemma4TransformerLayer,
        submodules=Gemma4TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=Gemma4SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=TENorm,
                    k_layernorm=TENorm,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=IdentityOp,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TELayerNormColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
            post_attention_layernorm=TENorm,
            post_feedforward_layernorm=TENorm,
        ),
    )


def get_gemma4_spec(args, config, vp_stage):
    """Return the native Gemma4 layer spec with proper config overrides."""
    # Gemma uses GeGLU, not SwiGLU
    config.activation_func = _gelu_tanh
    config.bias_activation_fusion = False

    # Heterogeneous layers need special checkpoint handling
    config.hetereogenous_dist_checkpoint = True

    # Read Gemma4-specific config from HF
    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(args.hf_checkpoint)
    hf_text = hf_config.text_config if hasattr(hf_config, "text_config") else hf_config

    # Set Gemma4TransformerConfig fields
    config.__class__ = Gemma4TransformerConfig
    config.global_kv_channels = getattr(hf_text, "global_head_dim", 512)
    config.global_num_query_groups = getattr(hf_text, "num_global_key_value_heads", 4)
    config.attention_k_eq_v = getattr(hf_text, "attention_k_eq_v", True)
    config.final_logit_softcapping = getattr(hf_text, "final_logit_softcapping", 30.0)
    config.sliding_window = getattr(hf_text, "sliding_window", 1024)
    config.sliding_window_pattern = 6  # every 6th layer is global
    config.query_pre_attn_scalar = getattr(hf_text, "query_pre_attn_scalar", 256)
    config.softmax_scale = 1.0  # Gemma4 uses scaling=1.0, Q/K norms handle scaling
    config.apply_rope_fusion = False  # Use unfused RoPE for parity with HF

    # RoPE config
    rope_params = getattr(hf_text, "rope_parameters", {})
    config.rope_local_base_freq = (
        rope_params.get("sliding_attention", {}).get("rope_theta", 10000.0)
    )

    # Embedding scaling: Gemma4 multiplies embeddings by sqrt(hidden_size)
    config.embedding_scaling_factor = hf_text.hidden_size ** 0.5

    spec = get_gemma4_layer_spec_te()

    # Use unfused layernorm + linear for MLP (matches HF numerics)
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    spec.submodules.mlp.submodules.linear_fc1 = TEColumnParallelLinear
    spec.submodules.mlp.metainfo = {"fuse_pre_mlp_layernorm": False}
    spec.submodules.pre_mlp_layernorm = TESpecProvider().layer_norm()

    return spec
