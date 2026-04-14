import re
import torch


_GLOBAL_ATTN_LAYERS = {5, 11, 17, 23, 29, 35, 41, 47, 53, 59}

# Gemma4-31B HF config values
_LOCAL_HEAD_DIM = 256
_GLOBAL_HEAD_DIM = 512
_NUM_ATTENTION_HEADS = 32
_LOCAL_NUM_KV_HEADS = 16
_GLOBAL_NUM_KV_HEADS = 4
_HIDDEN_SIZE = 5376


def convert_gemma4_to_hf(args, name, param):
    prefix = "model.language_model."

    if name == "module.module.embedding.word_embeddings.weight":
        return [(f"{prefix}embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [(f"{prefix}embed_tokens.weight", param)]  # tied embeddings
    if name == "module.module.decoder.final_layernorm.weight":
        return [(f"{prefix}norm.weight", param)]

    match = re.match(r"module\.module\.decoder\.layers\.(\d+)\.(.+)", name)
    if match:
        layer_idx = int(match.group(1))
        rest = match.group(2)
        L = f"{prefix}layers.{layer_idx}"
        is_global = layer_idx in _GLOBAL_ATTN_LAYERS

        if rest == "self_attention.linear_proj.weight":
            return [(f"{L}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.weight":
            if is_global:
                head_dim = _GLOBAL_HEAD_DIM
                num_kv_heads = _GLOBAL_NUM_KV_HEADS
            else:
                head_dim = _LOCAL_HEAD_DIM
                num_kv_heads = _LOCAL_NUM_KV_HEADS

            q_heads_per_kv = _NUM_ATTENTION_HEADS // num_kv_heads
            # Megatron packs QKV as [num_kv_heads, (q_heads_per_kv + 2) * head_dim, hidden]
            param = param.view(num_kv_heads, (q_heads_per_kv + 2) * head_dim, _HIDDEN_SIZE)
            q_dim = q_heads_per_kv * head_dim
            k_dim = head_dim
            v_dim = head_dim
            q_param = param[:, :q_dim, :].reshape(-1, _HIDDEN_SIZE)
            k_param = param[:, q_dim:q_dim + k_dim, :].reshape(-1, _HIDDEN_SIZE)

            if is_global:
                # Global layers: K=V shared, only emit q and k
                return [
                    (f"{L}.self_attn.q_proj.weight", q_param),
                    (f"{L}.self_attn.k_proj.weight", k_param),
                ]
            else:
                v_param = param[:, q_dim + k_dim:, :].reshape(-1, _HIDDEN_SIZE)
                return [
                    (f"{L}.self_attn.q_proj.weight", q_param),
                    (f"{L}.self_attn.k_proj.weight", k_param),
                    (f"{L}.self_attn.v_proj.weight", v_param),
                ]
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"{L}.input_layernorm.weight", param)]
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"{L}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"{L}.self_attn.k_norm.weight", param)]
        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"{L}.mlp.gate_proj.weight", gate_weight),
                (f"{L}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"{L}.mlp.down_proj.weight", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"{L}.pre_feedforward_layernorm.weight", param)]
        elif rest == "pre_mlp_layernorm.weight":
            return [(f"{L}.pre_feedforward_layernorm.weight", param)]
        elif rest == "post_attention_layernorm.weight":
            return [(f"{L}.post_attention_layernorm.weight", param)]
        elif rest == "post_feedforward_layernorm.weight":
            return [(f"{L}.post_feedforward_layernorm.weight", param)]

    raise ValueError(f"Unknown Gemma4 parameter name: {name}")
