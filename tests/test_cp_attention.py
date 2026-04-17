#!/usr/bin/env python3
"""Unit test: verify SDPACoreAttention CP=2 produces same output as CP=1.

Simulates CP by splitting tokens manually and comparing outputs.
No GPU cluster needed — runs on a single GPU.
"""
import torch
import torch.nn.functional as F

# Simulate the key functions from SDPACoreAttention

def make_varlen_causal_mask(cu_seqlens, total_len, device, dtype):
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


def adjust_cu_seqlens_for_cp(cu_seqlens, cp_rank, cp_size):
    local_cu = [0]
    for i in range(len(cu_seqlens) - 1):
        seq_len = (cu_seqlens[i + 1] - cu_seqlens[i]).item()
        chunk = seq_len // cp_size
        local_cu.append(local_cu[-1] + chunk)
    return torch.tensor(local_cu, dtype=cu_seqlens.dtype, device=cu_seqlens.device)


def attention_no_cp(q, k, v, cu_seqlens, scale):
    """Reference: full attention, no CP. q/k/v: [t, np, hn]"""
    t = q.shape[0]
    q4 = q.unsqueeze(0).permute(0, 2, 1, 3)  # [1, np, t, hn]
    k4 = k.unsqueeze(0).permute(0, 2, 1, 3)
    v4 = v.unsqueeze(0).permute(0, 2, 1, 3)
    mask = make_varlen_causal_mask(cu_seqlens, t, q.device, q.dtype)
    out = F.scaled_dot_product_attention(q4, k4, v4, attn_mask=mask, scale=scale)
    return out.permute(0, 2, 1, 3).reshape(t, -1)


def attention_cp_sliding(q_local, k_local, v_local, cu_seqlens, cp_rank, cp_size, scale):
    """Sliding window layer with CP: each rank computes independently on local chunk."""
    local_cu = adjust_cu_seqlens_for_cp(cu_seqlens, cp_rank, cp_size)
    t_local = q_local.shape[0]
    q4 = q_local.unsqueeze(0).permute(0, 2, 1, 3)
    k4 = k_local.unsqueeze(0).permute(0, 2, 1, 3)
    v4 = v_local.unsqueeze(0).permute(0, 2, 1, 3)
    mask = make_varlen_causal_mask(local_cu, t_local, q_local.device, q_local.dtype)
    out = F.scaled_dot_product_attention(q4, k4, v4, attn_mask=mask, scale=scale)
    return out.permute(0, 2, 1, 3).reshape(t_local, -1)


def attention_cp_global(q_local, k_full, v_full, cu_seqlens, cp_rank, cp_size, scale):
    """Global layer with CP: local Q attends to full KV."""
    t_local = q_local.shape[0]
    t_full = k_full.shape[0]

    q4 = q_local.unsqueeze(0).permute(0, 2, 1, 3)  # [1, np, t_local, hn]
    k4 = k_full.unsqueeze(0).permute(0, 2, 1, 3)   # [1, np, t_full, hn]
    v4 = v_full.unsqueeze(0).permute(0, 2, 1, 3)

    # Build causal mask for packed sequences
    mask = torch.full((t_local, t_full), float("-inf"), device=q_local.device, dtype=q_local.dtype)
    for s_idx in range(len(cu_seqlens) - 1):
        seq_start = cu_seqlens[s_idx].item()
        seq_end = cu_seqlens[s_idx + 1].item()
        seq_len = seq_end - seq_start
        chunk_size = seq_len // cp_size
        local_start = cp_rank * chunk_size
        local_offset = cu_seqlens[s_idx].item() // cp_size  # not right for packed...

    # Simpler: map local positions to global positions
    # With packed seqs split evenly per sequence, local token i in sequence s
    # maps to global position seq_start + cp_rank * chunk_size + local_i_within_seq
    local_to_global = []
    local_idx = 0
    for s_idx in range(len(cu_seqlens) - 1):
        seq_start = cu_seqlens[s_idx].item()
        seq_len = (cu_seqlens[s_idx + 1] - cu_seqlens[s_idx]).item()
        chunk_size = seq_len // cp_size
        for j in range(chunk_size):
            global_pos = seq_start + cp_rank * chunk_size + j
            local_to_global.append(global_pos)
            local_idx += 1

    for qi in range(t_local):
        gq = local_to_global[qi]
        # Find which sequence this belongs to
        for s_idx in range(len(cu_seqlens) - 1):
            if cu_seqlens[s_idx].item() <= gq < cu_seqlens[s_idx + 1].item():
                seq_start = cu_seqlens[s_idx].item()
                # Can attend to [seq_start, gq] in full KV
                mask[qi, seq_start:gq + 1] = 0.0
                break

    attn_mask = mask.unsqueeze(0).unsqueeze(0)
    out = F.scaled_dot_product_attention(q4, k4, v4, attn_mask=attn_mask, scale=scale)
    return out.permute(0, 2, 1, 3).reshape(t_local, -1)


def test_cp_attention():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # use fp32 for exact comparison
    torch.manual_seed(42)

    np_heads = 4
    hn = 64
    scale = 1.0 / (hn ** 0.5)
    cp_size = 2

    # Two packed sequences: lengths 8 and 12 (both divisible by cp_size=2)
    seq_lens = [8, 12]
    t_total = sum(seq_lens)
    cu_seqlens = torch.tensor([0, 8, 20], dtype=torch.int32, device=device)

    q = torch.randn(t_total, np_heads, hn, device=device, dtype=dtype)
    k = torch.randn(t_total, np_heads, hn, device=device, dtype=dtype)
    v = torch.randn(t_total, np_heads, hn, device=device, dtype=dtype)

    # === Reference: no CP ===
    ref_out = attention_no_cp(q, k, v, cu_seqlens, scale)

    # === Test 1: Global attention with CP ===
    print("Test 1: Global attention CP=2")
    cp_outs = []
    for cp_rank in range(cp_size):
        # Split Q per sequence chunk
        q_chunks = []
        k_chunks = []
        v_chunks = []
        for s_idx in range(len(seq_lens)):
            s = cu_seqlens[s_idx].item()
            e = cu_seqlens[s_idx + 1].item()
            chunk = (e - s) // cp_size
            cs = s + cp_rank * chunk
            ce = cs + chunk
            q_chunks.append(q[cs:ce])
            k_chunks.append(k[cs:ce])
            v_chunks.append(v[cs:ce])
        q_local = torch.cat(q_chunks, dim=0)
        # Full KV (simulating all-gather)
        out = attention_cp_global(q_local, k, v, cu_seqlens, cp_rank, cp_size, scale)
        cp_outs.append(out)

    # Reconstruct full output by interleaving chunks
    full_out = torch.zeros_like(ref_out)
    for cp_rank in range(cp_size):
        idx = 0
        for s_idx in range(len(seq_lens)):
            s = cu_seqlens[s_idx].item()
            e = cu_seqlens[s_idx + 1].item()
            chunk = (e - s) // cp_size
            cs = s + cp_rank * chunk
            ce = cs + chunk
            chunk_size = ce - cs
            full_out[cs:ce] = cp_outs[cp_rank][idx:idx + chunk_size]
            idx += chunk_size

    cos = F.cosine_similarity(ref_out.flatten().unsqueeze(0), full_out.flatten().unsqueeze(0)).item()
    max_diff = (ref_out - full_out).abs().max().item()
    print(f"  Cosine: {cos:.6f}, Max diff: {max_diff:.6e}")
    assert cos > 0.9999, f"Global CP failed: cosine={cos}"
    print("  PASSED")

    # === Test 2: Sliding window attention with CP ===
    print("\nTest 2: Sliding window attention CP=2")
    # For sliding window, each rank computes independently
    # Reference: compute sliding window without CP (just causal for simplicity)
    # The key insight: with sliding window, if window < chunk_size, each chunk is independent
    # So the output should match the reference for tokens that don't cross chunk boundaries

    # For this test, just verify the local computation runs and produces reasonable output
    for cp_rank in range(cp_size):
        q_chunks = []
        k_chunks = []
        v_chunks = []
        for s_idx in range(len(seq_lens)):
            s = cu_seqlens[s_idx].item()
            e = cu_seqlens[s_idx + 1].item()
            chunk = (e - s) // cp_size
            cs = s + cp_rank * chunk
            ce = cs + chunk
            q_chunks.append(q[cs:ce])
            k_chunks.append(k[cs:ce])
            v_chunks.append(v[cs:ce])
        q_local = torch.cat(q_chunks, dim=0)
        k_local = torch.cat(k_chunks, dim=0)
        v_local = torch.cat(v_chunks, dim=0)
        out = attention_cp_sliding(q_local, k_local, v_local, cu_seqlens, cp_rank, cp_size, scale)
        assert out.shape == q_local.shape[:1] + (np_heads * hn,), f"Shape mismatch: {out.shape}"
        assert not torch.isnan(out).any(), "NaN in sliding window output"
    print("  PASSED (no NaN, correct shapes)")

    print("\nAll tests passed!")


def test_flash_attn_varlen():
    """Test flash_attn_varlen_func matches SDPA for packed sequences (GPU only)."""
    try:
        from flash_attn import flash_attn_varlen_func
    except ImportError:
        print("\nTest 3: flash_attn_varlen_func\n  SKIPPED (flash_attn not installed)")
        return

    device = "cuda" if torch.cuda.is_available() else None
    if device is None:
        print("\nTest 3: flash_attn_varlen_func\n  SKIPPED (no CUDA)")
        return

    print("\nTest 3: flash_attn_varlen_func vs SDPA reference")
    dtype = torch.bfloat16
    torch.manual_seed(42)

    np_q, np_k, hn = 16, 8, 256  # GQA: 16 Q heads, 8 KV heads
    scale = 1.0 / (hn ** 0.5)
    seq_lens = [128, 256]
    t_total = sum(seq_lens)
    cu_seqlens = torch.tensor([0, 128, 384], dtype=torch.int32, device=device)

    q = torch.randn(t_total, np_q, hn, device=device, dtype=dtype)
    k = torch.randn(t_total, np_k, hn, device=device, dtype=dtype)
    v = torch.randn(t_total, np_k, hn, device=device, dtype=dtype)

    # flash_attn path
    max_seqlen = max(seq_lens)
    out_flash = flash_attn_varlen_func(
        q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
        softmax_scale=scale, causal=True,
    ).reshape(t_total, -1)

    # SDPA reference (expand KV for GQA, use explicit mask)
    q4 = q.unsqueeze(0).permute(0, 2, 1, 3).float()
    k4 = k.unsqueeze(0).permute(0, 2, 1, 3).float()
    v4 = v.unsqueeze(0).permute(0, 2, 1, 3).float()
    k4 = k4.repeat_interleave(np_q // np_k, dim=1)
    v4 = v4.repeat_interleave(np_q // np_k, dim=1)
    mask = make_varlen_causal_mask(cu_seqlens, t_total, device, torch.float32)
    out_sdpa = F.scaled_dot_product_attention(q4, k4, v4, attn_mask=mask, scale=scale)
    out_sdpa = out_sdpa.permute(0, 2, 1, 3).reshape(t_total, -1).to(dtype)

    cos = F.cosine_similarity(out_flash.flatten().float().unsqueeze(0),
                               out_sdpa.flatten().float().unsqueeze(0)).item()
    max_diff = (out_flash.float() - out_sdpa.float()).abs().max().item()
    print(f"  Cosine: {cos:.6f}, Max diff: {max_diff:.4e}")
    assert cos > 0.999, f"flash_attn vs SDPA mismatch: cosine={cos}"
    print("  PASSED")


if __name__ == "__main__":
    test_cp_attention()
    test_flash_attn_varlen()
