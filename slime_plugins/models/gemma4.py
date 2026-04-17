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
    global_partial_rotary_factor: float = 0.25  # fraction of global head_dim that gets RoPE
    attention_k_eq_v: bool = True  # global layers: V = K (no v_proj)
    final_logit_softcapping: float = 30.0
    enable_moe_block: bool = False  # 26B-A4B MoE variant


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


class Gemma4Router(nn.Module):
    """Gemma4 MoE router: RMSNorm(no scale) → learnable scale → proj → softmax → topk → per_expert_scale."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_moe_experts
        self.top_k = config.moe_router_topk
        self.scalar_root_size = self.hidden_size ** -0.5
        self.norm = VNorm(self.hidden_size, eps=config.layernorm_epsilon)
        self.proj = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.scale = nn.Parameter(torch.ones(self.hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(self.num_experts))

    def forward(self, hidden_states):
        # hidden_states: [tokens, hidden_size]
        h = self.norm(hidden_states)
        h = h * self.scale * self.scalar_root_size
        logits = self.proj(h)
        probs = torch.softmax(logits, dim=-1)
        top_k_weights, top_k_index = torch.topk(probs, k=self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
        return top_k_weights, top_k_index


class Gemma4Experts(nn.Module):
    """Gemma4 MoE experts with fused 3D gate_up_proj/down_proj tensors."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_moe_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_ffn_hidden_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_size, self.hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, self.intermediate_size)
        )
        self.act_fn = _gelu_tanh

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # hidden_states: [tokens, hidden_size]
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)  # [E, K, T]
        for expert_idx in expert_mask.sum(dim=(-1, -2)).nonzero(as_tuple=True)[0]:
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            x = hidden_states[token_idx]
            gate, up = F.linear(x, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            out = F.linear(self.act_fn(gate) * up, self.down_proj[expert_idx])
            final.index_add_(0, token_idx, out * top_k_weights[token_idx, top_k_pos, None])
        return final


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
        from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
        global_layer_number = layer_number + get_transformer_layer_offset(config)
        self.is_sliding = bool(global_layer_number % config.sliding_window_pattern)
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

        # Replace TE core attention with PyTorch SDPA for all layers.
        # Global layers require this because head_dim=512 exceeds flash attention's limit (256).
        # Local layers also use SDPA for consistency.
        self.self_attention.core_attention = SDPACoreAttention(
            config=config,
            layer_number=self.layer_number,
            attn_mask_type=AttnMaskType.causal,
            softmax_scale=config.softmax_scale,
        )
        self.self_attention.core_attention._is_sliding = self.is_sliding

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

        # Layer scalar (buffer, not learned)
        self.register_buffer("layer_scalar", torch.ones(1))

        # MoE block (26B-A4B): router + experts + extra layernorms
        self.enable_moe_block = getattr(config, 'enable_moe_block', False)
        if self.enable_moe_block:
            self.router = Gemma4Router(config)
            self.experts = Gemma4Experts(config)
            self.post_feedforward_layernorm_1 = TENorm(
                config=config, hidden_size=config.hidden_size, eps=config.layernorm_epsilon,
            )
            self.pre_feedforward_layernorm_2 = TENorm(
                config=config, hidden_size=config.hidden_size, eps=config.layernorm_epsilon,
            )
            self.post_feedforward_layernorm_2 = TENorm(
                config=config, hidden_size=config.hidden_size, eps=config.layernorm_epsilon,
            )

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
        # DualRotaryEmbedding returns concatenated [seq, 1, global_dim + local_dim] tensor.
        # Split and select based on layer type.
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            global_dim = getattr(self.config, 'dual_rope_global_dim', 0)
            if global_dim > 0 and rotary_pos_emb.shape[-1] > global_dim:
                if self.is_sliding:
                    rotary_pos_emb = rotary_pos_emb[..., global_dim:]  # local part
                else:
                    rotary_pos_emb = rotary_pos_emb[..., :global_dim]  # global part
        elif isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = rotary_pos_emb[1] if self.is_sliding else rotary_pos_emb[0]
        if isinstance(attention_mask, tuple):
            attention_mask = attention_mask[1] if self.is_sliding else attention_mask[0]

        # Global layers use partial RoPE (25% of head_dim=512 = 128 dims)
        # Local layers use full RoPE (100% of head_dim=256 = 256 dims)
        # With DualRotaryEmbedding, global RoPE is full-size (512 dims) with zero-padded
        # non-rotated dims, so no truncation needed.
        # With single RoPE (local only, 256 dims), truncate for global layers.
        if not self.is_sliding and rotary_pos_emb is not None:
            global_rope_dim = int(self.config.global_kv_channels * self.config.global_partial_rotary_factor)
            if rotary_pos_emb.shape[-1] != self.config.global_kv_channels and rotary_pos_emb.shape[-1] > global_rope_dim:
                rotary_pos_emb = rotary_pos_emb[..., :global_rope_dim]

        residual = hidden_states

        extra_kwargs = {}
        if inference_context is not None:
            extra_kwargs["inference_context"] = inference_context
        elif inference_params is not None:
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

        # MLP (dense path)
        residual = hidden_states
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)
        hidden_states, hidden_states_bias = self.mlp(pre_mlp_layernorm_output)
        if hidden_states_bias is not None:
            hidden_states = hidden_states + hidden_states_bias

        # MoE path: dense MLP + experts are summed, then post_feedforward_layernorm, then residual
        if self.enable_moe_block:
            mlp_output = self.post_feedforward_layernorm_1(hidden_states)

            flat = residual.view(-1, residual.shape[-1])
            top_k_weights, top_k_index = self.router(flat)
            moe_input = self.pre_feedforward_layernorm_2(flat)
            moe_output = self.experts(moe_input, top_k_index, top_k_weights)
            moe_output = moe_output.view_as(residual)
            moe_output = self.post_feedforward_layernorm_2(moe_output)

            hidden_states = mlp_output + moe_output

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
    """Simple core attention using PyTorch SDPA. Supports head_dim > 256 and CP."""
    def __init__(self, config, layer_number, attn_mask_type, attention_type="self",
                 attention_dropout=None, softmax_scale=None, **kwargs):
        super().__init__()
        self.config = config
        self.softmax_scale = softmax_scale
        self.dropout_p = config.attention_dropout if attention_dropout is None else attention_dropout
        self._is_sliding = False  # set by Gemma4TransformerLayer

    @staticmethod
    def _make_varlen_causal_mask(cu_seqlens, total_len, device, dtype):
        """Build a [1, 1, T, T] causal mask from cu_seqlens for packed sequences."""
        mask = torch.full((total_len, total_len), float("-inf"), device=device, dtype=dtype)
        for i in range(len(cu_seqlens) - 1):
            s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            seq_len = e - s
            seq_mask = torch.where(
                torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1),
                float("-inf"), 0.0,
            )
            mask[s:e, s:e] = seq_mask
        return mask.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def _adjust_cu_seqlens_for_cp(cu_seqlens, cp_rank, cp_size):
        """Adjust cu_seqlens for a CP rank's local chunk."""
        local_cu = [0]
        for i in range(len(cu_seqlens) - 1):
            seq_len = (cu_seqlens[i + 1] - cu_seqlens[i]).item()
            chunk = seq_len // cp_size
            local_cu.append(local_cu[-1] + chunk)
        return torch.tensor(local_cu, dtype=cu_seqlens.dtype, device=cu_seqlens.device)

    def _expand_kv_for_gqa(self, q, k, v):
        """Expand K/V heads to match Q heads for GQA (needed when SDPA can't broadcast)."""
        nq, nk = q.shape[1], k.shape[1]
        if nq != nk:
            repeat = nq // nk
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        return k, v

    def _forward_cp_global(self, query, key, value, attention_mask, packed_seq_params):
        """Global attention with CP: all-gather KV, manual matmul attention."""
        from megatron.core import parallel_state
        import torch.distributed as dist
        import time as _time

        cp_group = parallel_state.get_context_parallel_group()
        cp_size = parallel_state.get_context_parallel_world_size()
        cp_rank = parallel_state.get_context_parallel_rank()

        # One-shot timing on first call, rank 0
        _do_time = (cp_rank == 0 and not getattr(SDPACoreAttention, '_cp_timed', False))
        if _do_time:
            torch.cuda.synchronize()
            _t0 = _time.time()

        t_local = query.shape[0]
        np_q, hn = query.shape[1], query.shape[2]
        nk = key.shape[1]
        scale = self.softmax_scale or (hn ** -0.5)

        # All-gather K and V across CP ranks
        k_full = torch.empty(t_local * cp_size, *key.shape[1:], dtype=key.dtype, device=key.device)
        v_full = torch.empty(t_local * cp_size, *value.shape[1:], dtype=value.dtype, device=value.device)
        dist.all_gather_into_tensor(k_full, key.contiguous(), group=cp_group)
        dist.all_gather_into_tensor(v_full, value.contiguous(), group=cp_group)

        cu_seqlens = packed_seq_params.cu_seqlens_q if packed_seq_params is not None else None
        out = torch.empty(t_local, np_q * hn, dtype=query.dtype, device=query.device)

        if cu_seqlens is not None:
            local_offset = 0
            for s_idx in range(len(cu_seqlens) - 1):
                seq_start = cu_seqlens[s_idx].item()
                seq_len = (cu_seqlens[s_idx + 1] - cu_seqlens[s_idx]).item()
                chunk_size = seq_len // cp_size

                # Local Q [chunk, np, hn], full KV [seq_len, nk, hn]
                q_seq = query[local_offset:local_offset + chunk_size]
                k_seq = k_full[seq_start:seq_start + seq_len]
                v_seq = v_full[seq_start:seq_start + seq_len]

                # [b, h, s, d] format for matmul
                q4 = q_seq.unsqueeze(0).permute(0, 2, 1, 3)  # [1, np, chunk, hn]
                k4 = k_seq.unsqueeze(0).permute(0, 2, 1, 3)  # [1, nk, seq_len, hn]
                v4 = v_seq.unsqueeze(0).permute(0, 2, 1, 3)

                # Expand KV for GQA
                if np_q != nk:
                    repeat = np_q // nk
                    k4 = k4.repeat_interleave(repeat, dim=1)
                    v4 = v4.repeat_interleave(repeat, dim=1)

                # Q @ K^T * scale -> [1, np, chunk, seq_len]
                attn_w = torch.matmul(q4, k4.transpose(-2, -1)) * scale

                # Causal bias
                q_global_start = cp_rank * chunk_size
                row_idx = torch.arange(chunk_size, device=query.device) + q_global_start
                col_idx = torch.arange(seq_len, device=query.device)
                causal_mask = col_idx.unsqueeze(0) > row_idx.unsqueeze(1)  # [chunk, seq_len]
                attn_w.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

                # Softmax + @ V
                attn_w = F.softmax(attn_w, dim=-1, dtype=attn_w.dtype)
                o4 = torch.matmul(attn_w, v4)  # [1, np, chunk, hn]

                out[local_offset:local_offset + chunk_size] = o4.permute(0, 2, 1, 3).reshape(chunk_size, -1)
                local_offset += chunk_size
        else:
            q4 = query.unsqueeze(0).permute(0, 2, 1, 3)
            k4 = k_full.unsqueeze(0).permute(0, 2, 1, 3)
            v4 = v_full.unsqueeze(0).permute(0, 2, 1, 3)
            if np_q != nk:
                repeat = np_q // nk
                k4 = k4.repeat_interleave(repeat, dim=1)
                v4 = v4.repeat_interleave(repeat, dim=1)
            attn_w = torch.matmul(q4, k4.transpose(-2, -1)) * scale
            offset = cp_rank * t_local
            row_idx = torch.arange(t_local, device=query.device) + offset
            col_idx = torch.arange(t_local * cp_size, device=query.device)
            causal_mask = col_idx.unsqueeze(0) > row_idx.unsqueeze(1)
            attn_w.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            attn_w = F.softmax(attn_w, dim=-1, dtype=attn_w.dtype)
            o4 = torch.matmul(attn_w, v4)
            out = o4.permute(0, 2, 1, 3).reshape(t_local, -1)

        if _do_time:
            torch.cuda.synchronize()
            _elapsed = _time.time() - _t0
            cu = packed_seq_params.cu_seqlens_q if packed_seq_params is not None else None
            _nseq = len(cu) - 1 if cu is not None else 1
            print(f"[CP_GLOBAL] t_local={t_local} np={np_q} hn={hn} nseq={_nseq} "
                  f"time={_elapsed:.3f}s", flush=True)
            # Run a profiled replay of one sequence to get kernel breakdown
            try:
                import os
                _prof_dir = os.environ.get("SAVE_DIR", "/tmp") + "/cp_profile"
                os.makedirs(_prof_dir, exist_ok=True)
                if cu is not None and len(cu) > 1:
                    ss, sl = cu[0].item(), (cu[1] - cu[0]).item()
                    ch = sl // cp_size
                    _qp = query[:ch].unsqueeze(0).permute(0, 2, 1, 3)
                    _kp = k_full[ss:ss+sl].unsqueeze(0).permute(0, 2, 1, 3)
                    _vp = v_full[ss:ss+sl].unsqueeze(0).permute(0, 2, 1, 3)
                    if np_q != nk:
                        _kp = _kp.repeat_interleave(np_q // nk, dim=1)
                        _vp = _vp.repeat_interleave(np_q // nk, dim=1)
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=True, with_flops=True,
                    ) as _prof:
                        _aw = torch.matmul(_qp, _kp.transpose(-2, -1)) * scale
                        _ri = torch.arange(ch, device=query.device) + cp_rank * ch
                        _ci = torch.arange(sl, device=query.device)
                        _aw.masked_fill_((_ci.unsqueeze(0) > _ri.unsqueeze(1)).unsqueeze(0).unsqueeze(0), float("-inf"))
                        _aw = F.softmax(_aw, dim=-1, dtype=_aw.dtype)
                        _ = torch.matmul(_aw, _vp)
                        torch.cuda.synchronize()
                    print(_prof.key_averages().table(sort_by="cuda_time_total", row_limit=15), flush=True)
                    _prof.export_chrome_trace(f"{_prof_dir}/cp_global_trace.json")
                    print(f"[CP_GLOBAL] trace saved to {_prof_dir}/cp_global_trace.json", flush=True)
            except Exception as e:
                print(f"[CP_GLOBAL] profiler error: {e}", flush=True)
            SDPACoreAttention._cp_timed = True

        return out

    def forward(self, query, key, value, attention_mask=None, attn_mask_type=None,
                packed_seq_params=None, **kwargs):
        cp_size = self.config.context_parallel_size if hasattr(self.config, 'context_parallel_size') else 1

        is_thd = query.dim() == 3
        if is_thd:
            # For global layers with CP > 1, use manual matmul attention
            if cp_size > 1 and not self._is_sliding:
                return self._forward_cp_global(query, key, value, attention_mask, packed_seq_params)

            # Determine cu_seqlens (adjust for CP on sliding window layers)
            if packed_seq_params is not None:
                cu_seqlens = packed_seq_params.cu_seqlens_q
                if cp_size > 1:
                    from megatron.core import parallel_state
                    cp_rank = parallel_state.get_context_parallel_rank()
                    cu_seqlens = self._adjust_cu_seqlens_for_cp(cu_seqlens, cp_rank, cp_size)
            else:
                cu_seqlens = None

            # Use flash_attn_varlen_func for head_dim <= 256 with packed sequences
            hn = query.shape[2]
            if cu_seqlens is not None and hn <= 256:
                from flash_attn import flash_attn_varlen_func
                # flash_attn expects [t, np, hn] — already in that format
                # Expand KV for GQA not needed — flash_attn handles GQA natively
                max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
                out = flash_attn_varlen_func(
                    query.contiguous(), key.contiguous(), value.contiguous(),
                    cu_seqlens_q=cu_seqlens.to(torch.int32),
                    cu_seqlens_k=cu_seqlens.to(torch.int32),
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale,
                    causal=True,
                )
                # flash_attn returns [t, np, hn] -> [t, np*hn]
                return out.reshape(query.shape[0], -1)

            # Fallback: SDPA with explicit mask (head_dim > 256 or no packed_seq_params)
            q = query.unsqueeze(0).permute(0, 2, 1, 3)
            k = key.unsqueeze(0).permute(0, 2, 1, 3)
            v = value.unsqueeze(0).permute(0, 2, 1, 3)
            k, v = self._expand_kv_for_gqa(q, k, v)

            if cu_seqlens is not None:
                attn_mask = self._make_varlen_causal_mask(
                    cu_seqlens, query.shape[0], query.device, query.dtype
                )
            else:
                attn_mask = None

            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p if self.training else 0.0,
                scale=self.softmax_scale, is_causal=(attn_mask is None),
            )
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

    # MoE: disable Megatron's built-in MoE (we use custom Gemma4 MoE in the layer)
    config.moe_layer_freq = [0] * config.num_layers

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
    config.sliding_window_pattern = getattr(hf_text, "sliding_window_pattern", 6)
    config.query_pre_attn_scalar = getattr(hf_text, "query_pre_attn_scalar", 256)
    config.softmax_scale = 1.0  # Gemma4 uses scaling=1.0, Q/K norms handle scaling
    config.apply_rope_fusion = False  # Unfused RoPE needed for correct partial-rotary on global layers

    # MoE config (26B-A4B)
    config.enable_moe_block = getattr(hf_text, 'enable_moe_block', False)

    # RoPE config
    rope_params = getattr(hf_text, "rope_parameters", {})
    config.rope_local_base_freq = (
        rope_params.get("sliding_attention", {}).get("rope_theta", 10000.0)
    )
    config.global_partial_rotary_factor = (
        rope_params.get("full_attention", {}).get("partial_rotary_factor", 0.25)
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
