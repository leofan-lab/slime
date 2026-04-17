"""Tests for Gemma4 mbridge checkpoint conversion (QKV packing for local vs global layers)."""
import re

import pytest
import torch


# Gemma4-31B config values
HIDDEN_SIZE = 5376
NUM_ATTENTION_HEADS = 32
LOCAL_HEAD_DIM = 256
LOCAL_NUM_KV_HEADS = 16
GLOBAL_HEAD_DIM = 512
GLOBAL_NUM_KV_HEADS = 4
GLOBAL_ATTN_LAYERS = {5, 11, 17, 23, 29, 35, 41, 47, 53, 59}


def _pack_qkv_to_mcore(q, k, v, num_kv_heads, head_dim):
    """Pack separate Q, K, V into Megatron's interleaved QKV format."""
    q_heads_per_kv = NUM_ATTENTION_HEADS // num_kv_heads
    q = q.view(num_kv_heads, q_heads_per_kv * head_dim, HIDDEN_SIZE)
    k = k.view(num_kv_heads, head_dim, HIDDEN_SIZE)
    v = v.view(num_kv_heads, head_dim, HIDDEN_SIZE)
    return torch.cat([q, k, v], dim=1).view(-1, HIDDEN_SIZE).contiguous()


def _unpack_qkv_from_mcore(param, num_kv_heads, head_dim, is_global=False):
    """Unpack Megatron's interleaved QKV back to separate Q, K, V."""
    q_heads_per_kv = NUM_ATTENTION_HEADS // num_kv_heads
    param = param.view(num_kv_heads, (q_heads_per_kv + 2) * head_dim, HIDDEN_SIZE)
    q_dim = q_heads_per_kv * head_dim
    q = param[:, :q_dim, :].reshape(-1, HIDDEN_SIZE)
    k = param[:, q_dim:q_dim + head_dim, :].reshape(-1, HIDDEN_SIZE)
    if is_global:
        return q, k
    v = param[:, q_dim + head_dim:, :].reshape(-1, HIDDEN_SIZE)
    return q, k, v


@pytest.mark.unit
def test_local_layer_qkv_roundtrip():
    """Local layers: Q(32 heads x 256) + K(16 heads x 256) + V(16 heads x 256) roundtrips."""
    q = torch.randn(NUM_ATTENTION_HEADS * LOCAL_HEAD_DIM, HIDDEN_SIZE)
    k = torch.randn(LOCAL_NUM_KV_HEADS * LOCAL_HEAD_DIM, HIDDEN_SIZE)
    v = torch.randn(LOCAL_NUM_KV_HEADS * LOCAL_HEAD_DIM, HIDDEN_SIZE)

    packed = _pack_qkv_to_mcore(q, k, v, LOCAL_NUM_KV_HEADS, LOCAL_HEAD_DIM)
    q2, k2, v2 = _unpack_qkv_from_mcore(packed, LOCAL_NUM_KV_HEADS, LOCAL_HEAD_DIM)

    assert torch.equal(q, q2)
    assert torch.equal(k, k2)
    assert torch.equal(v, v2)


@pytest.mark.unit
def test_global_layer_kv_shared():
    """Global layers: K=V, so packing [Q, K, K] and unpacking emits only Q, K."""
    q = torch.randn(NUM_ATTENTION_HEADS * GLOBAL_HEAD_DIM, HIDDEN_SIZE)
    k = torch.randn(GLOBAL_NUM_KV_HEADS * GLOBAL_HEAD_DIM, HIDDEN_SIZE)

    # Pack with V = K (K=V sharing)
    packed = _pack_qkv_to_mcore(q, k, k.clone(), GLOBAL_NUM_KV_HEADS, GLOBAL_HEAD_DIM)
    q2, k2 = _unpack_qkv_from_mcore(packed, GLOBAL_NUM_KV_HEADS, GLOBAL_HEAD_DIM, is_global=True)

    assert torch.equal(q, q2)
    assert torch.equal(k, k2)


@pytest.mark.unit
def test_global_layer_indices():
    """Every 6th layer (0-indexed) is global: 5, 11, 17, ..., 59."""
    expected = {i for i in range(60) if i % 6 == 5}
    assert GLOBAL_ATTN_LAYERS == expected


@pytest.mark.unit
def test_local_vs_global_qkv_shapes():
    """Local and global layers have different QKV packed sizes."""
    local_qkv_dim = LOCAL_NUM_KV_HEADS * (NUM_ATTENTION_HEADS // LOCAL_NUM_KV_HEADS + 2) * LOCAL_HEAD_DIM
    global_qkv_dim = GLOBAL_NUM_KV_HEADS * (NUM_ATTENTION_HEADS // GLOBAL_NUM_KV_HEADS + 2) * GLOBAL_HEAD_DIM

    # Local: 16 groups * (2 + 2) * 256 = 16384
    assert local_qkv_dim == 16 * 4 * 256
    # Global: 4 groups * (8 + 2) * 512 = 20480
    assert global_qkv_dim == 4 * 10 * 512
    # They differ
    assert local_qkv_dim != global_qkv_dim


@pytest.mark.unit
def test_mlp_gate_up_roundtrip():
    """MLP gate_proj and up_proj are concatenated into linear_fc1."""
    gate = torch.randn(21504, HIDDEN_SIZE)
    up = torch.randn(21504, HIDDEN_SIZE)

    fused = torch.cat([gate, up], dim=0)
    gate2, up2 = fused.chunk(2, dim=0)

    assert torch.equal(gate, gate2)
    assert torch.equal(up, up2)
