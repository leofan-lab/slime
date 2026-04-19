"""Native Megatron Gemma4 transformer layer and config.

Extends the Gemma3 implementation from mbridge with Gemma4-specific features:
- Heterogeneous attention: global layers use head_dim=512, num_kv_heads=4;
  sliding layers use head_dim=256, num_kv_heads=16.
- attention_k_eq_v: global layers reuse K output as V (no v_proj).
- v_norm: RMSNorm without learnable scale applied to V states.
- layer_scalar: buffer multiplied after residual (not learned).
- final_logit_softcapping: applied to output logits in the model wrapper.
- MoE block (26B-A4B): Gemma4's custom router (with per-expert scale) plugged
  into Megatron's MoE infrastructure for proper expert-parallel sharding.
  The router is still custom (see Gemma4Router); dispatching + grouped-GEMM
  come from Megatron's MoELayer + TEGroupedMLP.
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
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules, BaseMoELayer
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
    # For MoE-enabled variants (26B-A4B), the primary `mlp` submodule is swapped
    # to a Gemma4MoELayer and the original dense MLP moves to `dense_mlp`. This
    # keeps the `.mlp.experts.linear_fc...` naming that mbridge's EP auto-handling
    # expects while preserving Gemma4's dense+MoE-in-parallel structure.
    dense_mlp: ModuleSpec | type = IdentityOp


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

    def set_layer_number(self, layer_number):
        # MoELayer.set_layer_number delegates to router.set_layer_number;
        # Megatron's TopKRouter stores it for logging/aux-loss scoping. We
        # don't use it, so accept and ignore.
        self.layer_number = layer_number


class Gemma4MoELayer(MoELayer):
    """Gemma4 MoE block: Megatron's MoELayer with Gemma4's custom router.

    Megatron's MoELayer hardcodes its own ``TopKRouter`` which uses a
    softmax-with-expert-bias scheme. Gemma4 has its own router semantics
    (no-scale RMSNorm → learnable per-hidden scale → proj → softmax → topk →
    per-expert scale multiplier). We reuse all of Megatron's infrastructure
    for dispatching (alltoall), expert parallelism, and grouped-GEMM expert
    computation — but swap in our ``Gemma4Router`` and convert its compact
    (top_k_weights [T, K], top_k_index [T, K]) output into Megatron's
    expected (probs [T, E], routing_map [T, E]) format inside ``route()``.
    """

    def __init__(self, config, submodules=None, layer_number=None, pg_collection=None):
        # Bypass MoELayer.__init__ so we can avoid building Megatron's TopKRouter,
        # then run the rest of MoELayer's setup ourselves. The parts we need:
        # - self.ep_group / num_local_experts / local_expert_indices (from BaseMoELayer)
        # - token_dispatcher, experts (alltoall path, GroupedMLP)
        # Anything shared-expert related is disabled: Gemma4 has no shared experts in
        # this sense — its "dense MLP" lives outside the MoE block in the parent layer.
        #
        # Fall back to Megatron's global parallel_state when pg_collection isn't
        # explicitly passed. TransformerLayer only forwards pg_collection when
        # submodules.mlp.module is *exactly* one of
        # (MoELayer, GroupedMLP, TEGroupedMLP, SequentialMLP) — an identity check
        # via `in`, so Gemma4MoELayer (a MoELayer subclass) slips through and
        # receives None. BaseMoELayer.__init__ then crashes on `pg_collection.ep`.
        # Same fallback MoELayer.__init__ uses when invoked directly.
        if pg_collection is None:
            from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
            pg_collection = get_default_pg_collection()
        BaseMoELayer.__init__(
            self, config=config, layer_number=layer_number, pg_collection=pg_collection
        )
        # Disable Megatron-checkpoint paths that don't apply here.
        self.moe_layer_recompute = False
        self.shared_experts_recompute = False
        self.submodules = submodules

        # --- Router: Gemma4's custom router, not TopKRouter. ---
        self.router = Gemma4Router(config)

        # --- Token dispatcher (identical to MoELayer.__init__). ---
        from megatron.core.transformer.moe.token_dispatcher import (
            MoEAllGatherTokenDispatcher,
            MoEAlltoAllTokenDispatcher,
            MoEFlexTokenDispatcher,
        )
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices,
                config=self.config, pg_collection=pg_collection,
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices,
                config=self.config, pg_collection=pg_collection,
            )
        elif config.moe_token_dispatcher_type == "flex":
            self.token_dispatcher = MoEFlexTokenDispatcher(
                self.num_local_experts, self.local_expert_indices,
                config=self.config, pg_collection=pg_collection,
            )
        else:
            raise ValueError(f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}")

        # --- Experts: Megatron's GroupedMLP / TEGroupedMLP. ---
        self.experts = build_module(
            self.submodules.experts,
            self.num_local_experts,
            self.config,
            pg_collection=pg_collection,
        )

        # Gemma4 doesn't use shared experts in Megatron's sense.
        self.shared_experts = None

        # cudagraph tensor store (required by MoELayer.forward decorators)
        from megatron.core.transformer.moe.moe_utils import MoECudaGraphTensorStore
        self.cudagraph_tensor_store = MoECudaGraphTensorStore()

    def route(self, hidden_states: torch.Tensor):
        """Call Gemma4Router and pack its output into Megatron's (probs, routing_map).

        Gemma4Router expects a flattened [T, H] tensor and returns compact top-k:
            top_k_weights: [T, K] — routing weights (already scaled by per_expert_scale)
            top_k_index:   [T, K] — which experts each token routes to
        Megatron's dispatcher wants:
            probs:       [T, E] — weight per (token, expert), 0 where not routed
            routing_map: [T, E] — boolean mask
        """
        orig_shape = hidden_states.shape
        flat = hidden_states.view(-1, orig_shape[-1])
        top_k_weights, top_k_index = self.router(flat)

        num_tokens = flat.shape[0]
        num_experts = self.config.num_moe_experts
        probs = torch.zeros(
            num_tokens, num_experts,
            dtype=top_k_weights.dtype, device=top_k_weights.device,
        )
        probs.scatter_(1, top_k_index, top_k_weights)
        routing_map = probs != 0
        return probs, routing_map

    # BaseMoELayer.forward is abstract; MoELayer defines a concrete forward
    # that we inherit. It calls self.route → self.preprocess → self.dispatch →
    # self.routed_experts_compute → self.combine. All of those work with
    # Megatron's (probs, routing_map) format that our route() now produces.


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
        # Megatron passes `layer_number` as 1-indexed (default 1), so in 0-indexed
        # HF space a global layer is `(i+1) % pattern == 0` → `i % pattern == pattern-1`.
        # Equivalently: `is_sliding` when `global_layer_number % pattern != 0`.
        self.is_sliding = bool(global_layer_number % config.sliding_window_pattern)
        self._is_global = not self.is_sliding

        # Global layers have different head_dim and num_kv_heads. Swap those
        # fields into `config` just long enough for super().__init__ to build
        # the attention, then restore. Using try/finally keeps the shared
        # config clean even if init raises. (Still not reentrant-safe under
        # concurrent construction — Megatron layer init is single-threaded per
        # rank today.)
        if self._is_global:
            orig_kv_channels = config.kv_channels
            orig_num_query_groups = config.num_query_groups
            config.kv_channels = config.global_kv_channels
            config.num_query_groups = config.global_num_query_groups
        try:
            super().__init__(
                config=config, submodules=submodules,
                layer_number=layer_number, hidden_dropout=hidden_dropout,
                **kwargs,
            )
        finally:
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

        # MoE block (26B-A4B): super().__init__ already built self.mlp from the
        # layer spec, which when enable_moe_block=True is a Gemma4MoELayer (not
        # a dense MLP). We also build a parallel `dense_mlp` for Gemma4's
        # dense + MoE combined-FFN pattern. The two outputs are summed in
        # forward().
        self.enable_moe_block = getattr(config, 'enable_moe_block', False)
        if self.enable_moe_block:
            # Parallel dense MLP branch (sibling to self.mlp, which is the MoE).
            self.dense_mlp = build_module(
                submodules.dense_mlp,
                config=config,
            )
            # Gemma4 pre/post layernorms that wrap the two FFN paths.
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

        # FFN path. Two shapes depending on whether this is a MoE-enabled layer:
        #   - dense (31B):  `self.mlp` IS the dense MLP; nothing else to do.
        #   - MoE (26B-A4B): `self.mlp` is a Gemma4MoELayer (so mcore state-dict
        #     paths look like `.mlp.experts.linear_fc*.weight*` and match the
        #     mbridge EP handling). The parallel dense path lives in
        #     `self.dense_mlp` and is summed with the MoE output, matching
        #     Gemma4's HF reference which runs both feed-forwards per layer.
        residual = hidden_states
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        if self.enable_moe_block:
            # Dense MLP is the "main" feed-forward in HF terminology; it
            # passes through `post_feedforward_layernorm_1` before summing.
            dense_out, dense_bias = self.dense_mlp(pre_mlp_layernorm_output)
            if dense_bias is not None:
                dense_out = dense_out + dense_bias
            mlp_output = self.post_feedforward_layernorm_1(dense_out)

            # MoE input uses a dedicated pre-norm (HF's `pre_feedforward_layernorm_2`).
            moe_input = self.pre_feedforward_layernorm_2(residual)
            moe_output, _ = self.mlp(moe_input)  # Gemma4MoELayer returns (output, mlp_bias=None)
            moe_output = self.post_feedforward_layernorm_2(moe_output)

            hidden_states = mlp_output + moe_output
        else:
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
    """Gemma4 core attention.

    Replaces TE's DotProductAttention because:
    - Global layers have head_dim=512, which flash-attn 2.x doesn't support.
    - Sliding-window layers need an explicit left-window mask (HF behavior).
    - Context-parallelism on the global layers needs an all-gather+full-attn
      path with a differentiable K/V gather.

    Dispatch at call time (packed / thd shape):
      - global  + CP>1   : `_forward_cp_global` (manual attention, per sub-seq)
      - global  + CP==1  : sub-sequence causal SDPA (no O(T²) mask alloc)
      - sliding + any    : flash_attn_varlen_func with (sw-1, 0) window
    """

    def __init__(self, config, layer_number, attn_mask_type, attention_type="self",
                 attention_dropout=None, softmax_scale=None, **kwargs):
        super().__init__()
        self.config = config
        self.softmax_scale = softmax_scale
        self.dropout_p = config.attention_dropout if attention_dropout is None else attention_dropout
        self._is_sliding = False  # set by Gemma4TransformerLayer

    def _resolve_scale(self, hn: int) -> float:
        # `0.0 or fallback` would silently mask a misconfigured scale; be explicit.
        return self.softmax_scale if self.softmax_scale is not None else (hn ** -0.5)

    @staticmethod
    def _adjust_cu_seqlens_for_cp(cu_seqlens, cp_rank, cp_size):
        """Remap packed-sequence boundaries to a single CP rank's local chunk."""
        local_cu = [0]
        for i in range(len(cu_seqlens) - 1):
            seq_len = (cu_seqlens[i + 1] - cu_seqlens[i]).item()
            chunk = seq_len // cp_size
            local_cu.append(local_cu[-1] + chunk)
        return torch.tensor(local_cu, dtype=cu_seqlens.dtype, device=cu_seqlens.device)

    def _forward_cp_global(self, query, key, value, packed_seq_params):
        """Global attention under CP: differentiable all-gather of K/V, then
        per-sub-sequence causal SDPA using local Q against full-length K/V.
        """
        from megatron.core import parallel_state
        from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

        cp_group = parallel_state.get_context_parallel_group()
        cp_size = parallel_state.get_context_parallel_world_size()
        cp_rank = parallel_state.get_context_parallel_rank()

        t_local = query.shape[0]
        np_q, hn = query.shape[1], query.shape[2]
        nk = key.shape[1]
        scale = self._resolve_scale(hn)

        # Differentiable all-gather along the token dim. forward: AG,
        # backward: RS — so K/V grads on non-owning ranks flow back to the
        # originating rank. The raw `dist.all_gather_into_tensor` has no
        # autograd rule and PyTorch prints a "silently incorrect behavior"
        # warning + drops those grads.
        k_full = gather_from_sequence_parallel_region(key.contiguous(), group=cp_group)
        v_full = gather_from_sequence_parallel_region(value.contiguous(), group=cp_group)

        cu_seqlens = packed_seq_params.cu_seqlens_q if packed_seq_params is not None else None
        out = torch.empty(t_local, np_q * hn, dtype=query.dtype, device=query.device)

        if cu_seqlens is None:
            # Single-sequence fallback.
            q4 = query.unsqueeze(0).transpose(1, 2)   # [1, np, t_local, hn]
            k4 = k_full.unsqueeze(0).transpose(1, 2)  # [1, nk, t_full, hn]
            v4 = v_full.unsqueeze(0).transpose(1, 2)
            t_full = k_full.shape[0]
            q_offset = cp_rank * t_local
            row_idx = torch.arange(t_local, device=query.device) + q_offset
            col_idx = torch.arange(t_full, device=query.device)
            # SDPA attn_mask: additive; -inf where we must NOT attend.
            mask = torch.where(
                col_idx[None, :] > row_idx[:, None],
                torch.finfo(query.dtype).min, 0.0,
            ).to(dtype=query.dtype)
            o = F.scaled_dot_product_attention(
                q4, k4, v4, attn_mask=mask[None, None, :, :],
                dropout_p=self.dropout_p if self.training else 0.0,
                scale=scale, enable_gqa=(np_q != nk),
            )
            return o.transpose(1, 2).reshape(t_local, -1)

        # Packed sub-sequences: loop; each sub-seq emits a [chunk, T] causal
        # block directly to `out`. Per-sub-seq masks stay small even for
        # max_tokens_per_gpu = many thousands.
        local_offset = 0
        for s_idx in range(len(cu_seqlens) - 1):
            seq_start = cu_seqlens[s_idx].item()
            seq_len = (cu_seqlens[s_idx + 1] - cu_seqlens[s_idx]).item()
            chunk = seq_len // cp_size

            q_seq = query[local_offset:local_offset + chunk]
            k_seq = k_full[seq_start:seq_start + seq_len]
            v_seq = v_full[seq_start:seq_start + seq_len]

            q4 = q_seq.unsqueeze(0).transpose(1, 2)   # [1, np, chunk, hn]
            k4 = k_seq.unsqueeze(0).transpose(1, 2)   # [1, nk, seq_len, hn]
            v4 = v_seq.unsqueeze(0).transpose(1, 2)

            q_global_start = cp_rank * chunk
            row_idx = torch.arange(chunk, device=query.device) + q_global_start
            col_idx = torch.arange(seq_len, device=query.device)
            mask = torch.where(
                col_idx[None, :] > row_idx[:, None],
                torch.finfo(query.dtype).min, 0.0,
            ).to(dtype=query.dtype)

            o = F.scaled_dot_product_attention(
                q4, k4, v4, attn_mask=mask[None, None, :, :],
                dropout_p=self.dropout_p if self.training else 0.0,
                scale=scale, enable_gqa=(np_q != nk),
            )
            out[local_offset:local_offset + chunk] = o.transpose(1, 2).reshape(chunk, -1)
            local_offset += chunk

        return out

    def _forward_thd_flash(self, query, key, value, cu_seqlens):
        """Sliding-window or head_dim<=256 path via flash_attn_varlen_func.

        Sliding-window layers must pass `window_size=(sliding_window-1, 0)` so
        only tokens within `sliding_window` positions back are attended to —
        this matches HF's `sliding_window_mask_function`. Global layers and
        dense-attention sliding layers use the default full-causal window.
        """
        from flash_attn import flash_attn_varlen_func

        window_size = (-1, -1)  # full causal when causal=True
        if self._is_sliding:
            sw = getattr(self.config, "sliding_window", None)
            if sw and sw > 0:
                window_size = (int(sw) - 1, 0)

        cu = cu_seqlens.to(torch.int32)
        max_seqlen = (cu[1:] - cu[:-1]).max().item()
        out = flash_attn_varlen_func(
            query.contiguous(), key.contiguous(), value.contiguous(),
            cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            dropout_p=self.dropout_p if self.training else 0.0,
            softmax_scale=self._resolve_scale(query.shape[2]),
            causal=True,
            window_size=window_size,
        )
        return out.reshape(query.shape[0], -1)

    def _forward_thd_sdpa_per_subseq(self, query, key, value, cu_seqlens):
        """Per-sub-sequence causal SDPA — used when flash-attn can't handle
        head_dim (global layer w/o CP). Avoids materializing a [T, T] mask.
        """
        np_q, hn = query.shape[1], query.shape[2]
        nk = key.shape[1]
        scale = self._resolve_scale(hn)
        out = torch.empty(query.shape[0], np_q * hn, dtype=query.dtype, device=query.device)
        for i in range(len(cu_seqlens) - 1):
            s = cu_seqlens[i].item()
            e = cu_seqlens[i + 1].item()
            q4 = query[s:e].unsqueeze(0).transpose(1, 2)  # [1, np, L, hn]
            k4 = key[s:e].unsqueeze(0).transpose(1, 2)
            v4 = value[s:e].unsqueeze(0).transpose(1, 2)
            o = F.scaled_dot_product_attention(
                q4, k4, v4,
                dropout_p=self.dropout_p if self.training else 0.0,
                scale=scale, is_causal=True, enable_gqa=(np_q != nk),
            )
            out[s:e] = o.transpose(1, 2).reshape(e - s, -1)
        return out

    def forward(self, query, key, value, attention_mask=None, attn_mask_type=None,
                packed_seq_params=None, **kwargs):
        cp_size = getattr(self.config, "context_parallel_size", 1) or 1
        is_thd = query.dim() == 3

        if is_thd:
            # Global layer + CP > 1: all-gather KV, manual attention.
            if cp_size > 1 and not self._is_sliding:
                return self._forward_cp_global(query, key, value, packed_seq_params)

            # Remap cu_seqlens to the local CP chunk (sliding path only; global
            # path above handles its own CP geometry).
            cu_seqlens = None
            if packed_seq_params is not None:
                cu_seqlens = packed_seq_params.cu_seqlens_q
                if cp_size > 1:
                    from megatron.core import parallel_state
                    cp_rank = parallel_state.get_context_parallel_rank()
                    cu_seqlens = self._adjust_cu_seqlens_for_cp(cu_seqlens, cp_rank, cp_size)

            hn = query.shape[2]
            if cu_seqlens is not None:
                if hn <= 256:
                    return self._forward_thd_flash(query, key, value, cu_seqlens)
                # Global layer, no CP, packed — flash-attn won't take head_dim>256.
                return self._forward_thd_sdpa_per_subseq(query, key, value, cu_seqlens)

            # Un-packed thd (rare, e.g. eval with batch=1 seq only): plain SDPA.
            q = query.unsqueeze(0).transpose(1, 2)
            k = key.unsqueeze(0).transpose(1, 2)
            v = value.unsqueeze(0).transpose(1, 2)
            nq, nk = q.shape[1], k.shape[1]
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout_p if self.training else 0.0,
                scale=self._resolve_scale(hn), is_causal=True,
                enable_gqa=(nq != nk),
            )
            return out.transpose(1, 2).reshape(query.shape[0], -1)

        # bshd path (unused in training but kept for inference/eval parity).
        q = query.permute(1, 2, 0, 3)
        k = key.permute(1, 2, 0, 3)
        v = value.permute(1, 2, 0, 3)
        nq, nk = q.shape[1], k.shape[1]
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            scale=self._resolve_scale(query.shape[3]), is_causal=True,
            enable_gqa=(nq != nk),
        )
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


def _build_moe_submodule_spec(config):
    """Build the MoE submodule spec (Gemma4MoELayer + TE GroupedMLP experts)."""
    from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

    # Reuse Megatron's canonical TE-backed MoE spec factory to get
    # TEColumnParallelGroupedLinear / TERowParallelGroupedLinear etc. wired up
    # properly for GroupedMLP experts. Then swap the top-level module from
    # Megatron's MoELayer to our Gemma4MoELayer, which keeps all that wiring
    # but plugs in Gemma4Router.
    base_spec = get_moe_module_spec_for_backend(
        backend=TESpecProvider(),
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        use_te_activation_func=False,  # use plain F.gelu(approximate='tanh') from config.activation_func
    )
    return ModuleSpec(
        module=Gemma4MoELayer,
        submodules=base_spec.submodules,
        metainfo=base_spec.metainfo,
    )


def get_gemma4_layer_spec_te(config=None) -> ModuleSpec:
    """Layer spec for Gemma4 using native Megatron attention with TE.

    If ``config.enable_moe_block`` is set, the main ``mlp`` submodule is a
    :class:`Gemma4MoELayer` (so that the state-dict path
    ``.mlp.experts.linear_fc*.weight*`` matches mbridge's EP auto-handling),
    and the original dense MLP moves to a sibling ``dense_mlp`` submodule that
    the layer forward sums with the MoE output. For the 31B dense variant,
    ``enable_moe_block=False`` and ``mlp`` stays as the normal Megatron MLP.
    """
    dense_mlp_spec = ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TELayerNormColumnParallelLinear,
            linear_fc2=TERowParallelLinear,
        ),
    )
    if config is not None and getattr(config, "enable_moe_block", False):
        mlp_spec = _build_moe_submodule_spec(config)
        dense_spec = dense_mlp_spec
    else:
        mlp_spec = dense_mlp_spec
        dense_spec = IdentityOp

    submods = Gemma4TransformerLayerSubmodules(
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
        mlp=mlp_spec,
        mlp_bda=get_bias_dropout_add,
        post_attention_layernorm=TENorm,
        post_feedforward_layernorm=TENorm,
        dense_mlp=dense_spec,
    )
    return ModuleSpec(module=Gemma4TransformerLayer, submodules=submods)


def _load_hf_text_config(hf_checkpoint):
    """Load HF config and unwrap `text_config` if it's a multimodal wrapper.

    Cached via lru_cache so repeated callers (model provider, mbridge, weight
    converter) all share the same parsed object.
    """
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(hf_checkpoint, trust_remote_code=True)
    return cfg.text_config if hasattr(cfg, "text_config") else cfg


def get_gemma4_spec(args, config, vp_stage):
    """Return the native Gemma4 layer spec with proper config overrides."""
    hf_text = _load_hf_text_config(args.hf_checkpoint)

    # Gemma4 features that this plugin does NOT implement — fail loudly rather
    # than silently training a degraded model.
    if getattr(hf_text, "hidden_size_per_layer_input", 0):
        raise NotImplementedError(
            "Gemma4 per-layer input mechanism "
            f"(hidden_size_per_layer_input={hf_text.hidden_size_per_layer_input}) "
            "is not implemented. See Gemma4TextDecoderLayer.per_layer_input_gate in HF."
        )
    if getattr(hf_text, "num_kv_shared_layers", 0):
        raise NotImplementedError(
            "Gemma4 KV-sharing across the last N layers "
            f"(num_kv_shared_layers={hf_text.num_kv_shared_layers}) is not implemented."
        )
    if getattr(hf_text, "use_double_wide_mlp", False):
        raise NotImplementedError("Gemma4 use_double_wide_mlp is not implemented.")
    # Text-only training assumes causal attention for all layers. HF attention
    # turns off causality when use_bidirectional_attention == "all".
    if getattr(hf_text, "use_bidirectional_attention", "vision") == "all":
        raise NotImplementedError(
            "Gemma4 use_bidirectional_attention='all' disables causal masking; not supported."
        )

    # Gemma uses GeGLU (gated gelu-tanh), not SwiGLU. The shell scripts omit
    # --swiglu and we set the gated-linear-unit flag + activation explicitly
    # here so no downstream code is misled by stale Megatron defaults.
    config.gated_linear_unit = True
    config.activation_func = _gelu_tanh
    config.bias_activation_fusion = False

    # MoE: our custom Gemma4MoELayer handles MoE inside the layer body. We
    # set moe_layer_freq to all-zero so Megatron's built-in MoE dispatch (in
    # the stock layer spec) doesn't fire — Gemma4TransformerLayer always uses
    # the dense MLP slot for its dense feed-forward and calls the MoE block
    # explicitly in its forward() when enable_moe_block is True.
    config.moe_layer_freq = [0] * config.num_layers

    # Heterogeneous layers (different head_dim / num_kv_heads on global vs
    # sliding layers) need special checkpoint handling.
    config.hetereogenous_dist_checkpoint = True

    # Promote the config to Gemma4TransformerConfig in place so the dataclass
    # fields below are reachable by the rest of Megatron.
    config.__class__ = Gemma4TransformerConfig
    config.global_kv_channels = hf_text.global_head_dim
    config.global_num_query_groups = hf_text.num_global_key_value_heads
    config.attention_k_eq_v = getattr(hf_text, "attention_k_eq_v", True)
    config.final_logit_softcapping = getattr(hf_text, "final_logit_softcapping", 30.0)
    config.sliding_window = hf_text.sliding_window
    # `sliding_window_pattern` is not present in Gemma4 HF configs — infer from
    # layer_types (find the first full_attention layer, its 1-indexed position
    # is the pattern). Fall back to 6 for safety.
    layer_types = list(getattr(hf_text, "layer_types", []))
    try:
        first_full = layer_types.index("full_attention")
        config.sliding_window_pattern = first_full + 1
    except ValueError:
        config.sliding_window_pattern = 6
    config.softmax_scale = 1.0  # Gemma4 uses scaling=1.0, Q/K norms handle scaling
    # Unfused RoPE is required because we zero out `inv_freq` tail entries to
    # implement partial-rotary on global layers — fused kernels ignore the
    # zeroed freqs and rotate the full head_dim unconditionally.
    config.apply_rope_fusion = False

    # MoE block (26B-A4B). Gemma4MoELayer reads these fields and delegates
    # to Megatron's MoE dispatch for expert parallelism (EP), so setting
    # --expert-model-parallel-size > 1 actually shards expert weights.
    config.enable_moe_block = getattr(hf_text, "enable_moe_block", False)
    if config.enable_moe_block:
        config.num_moe_experts = hf_text.num_experts
        config.moe_router_topk = hf_text.top_k_experts
        config.moe_ffn_hidden_size = hf_text.moe_intermediate_size
        # Megatron MoE infrastructure needs these fields set even though we
        # use a custom router. Defaults mirror Qwen3.5-A3B's working config.
        config.moe_token_dispatcher_type = getattr(config, "moe_token_dispatcher_type", None) or "alltoall"
        config.moe_grouped_gemm = getattr(config, "moe_grouped_gemm", None) or True
        # No aux loss: Gemma4's router doesn't use one. Megatron requires the
        # field to be set; 0.0 disables it.
        config.moe_aux_loss_coeff = 0.0
        config.moe_router_load_balancing_type = getattr(config, "moe_router_load_balancing_type", None) or "none"
        # Route scoring: Gemma4 does softmax itself in the router; tell
        # Megatron to not re-compute anything — the only consumer of these
        # fields on our path is TopKRouter, which we override with
        # Gemma4Router. Set reasonable defaults so any internal validator
        # passes.
        config.moe_router_score_function = getattr(config, "moe_router_score_function", None) or "softmax"
        config.moe_router_topk_scaling_factor = getattr(config, "moe_router_topk_scaling_factor", None) or 1.0
        config.moe_router_pre_softmax = False

    # RoPE config
    rope_params = getattr(hf_text, "rope_parameters", {}) or {}
    config.rope_local_base_freq = rope_params.get("sliding_attention", {}).get("rope_theta", 10000.0)
    config.global_partial_rotary_factor = rope_params.get("full_attention", {}).get(
        "partial_rotary_factor", 0.25
    )

    # CP sliding-window layer correctness: each CP rank processes a local chunk
    # of each sequence and attends only within that chunk. This is only valid
    # if the chunk is at least `sliding_window` tokens; otherwise tokens within
    # the window that live on other ranks are silently dropped from attention.
    cp_size = getattr(args, "context_parallel_size", 1) or 1
    if cp_size > 1:
        max_tokens = getattr(args, "max_tokens_per_gpu", None)
        if max_tokens is not None and max_tokens < config.sliding_window:
            raise ValueError(
                f"context_parallel_size={cp_size} with max_tokens_per_gpu={max_tokens} "
                f"< sliding_window={config.sliding_window}: sliding-window layers would "
                "silently miss in-window tokens from other CP ranks. Reduce CP or raise "
                "max_tokens_per_gpu."
            )

    # Build layer spec AFTER setting MoE fields so the MoE submodule spec
    # is attached when enable_moe_block is True.
    spec = get_gemma4_layer_spec_te(config)

    # Use unfused layernorm + linear for MLP (matches HF numerics)
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    spec.submodules.mlp.submodules.linear_fc1 = TEColumnParallelLinear
    spec.submodules.mlp.metainfo = {"fuse_pre_mlp_layernorm": False}
    spec.submodules.pre_mlp_layernorm = TESpecProvider().layer_norm()

    return spec
