import functools
import re

import torch
import torch.nn.functional as F
from mbridge.core import register_model
from mbridge.models import Gemma3Bridge

# Gemma uses GeGLU (GELU with tanh approximation + gated linear unit), not SwiGLU.
# See: https://developers.googleblog.com/en/gemma-explained-new-in-gemma-2/
_gelu_tanh = functools.partial(F.gelu, approximate="tanh")


@register_model("gemma4")
class Gemma4Bridge(Gemma3Bridge):
    """
    Bridge for Gemma 4 dense 31B.

    Megatron-side keys have NO language_model. prefix (text-only model).
    HF-side values have model.language_model. prefix (Gemma4ForConditionalGeneration).
    """

    _ATTENTION_MAPPING = {
        "decoder.layers.{layer_number}.self_attention.linear_qkv.weight": [
            "model.language_model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.language_model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.language_model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "decoder.layers.{layer_number}.self_attention.linear_proj.weight": [
            "model.language_model.layers.{layer_number}.self_attn.o_proj.weight",
        ],
        "decoder.layers.{layer_number}.self_attention.linear_qkv.layer_norm_weight": [
            "model.language_model.layers.{layer_number}.input_layernorm.weight",
        ],
        "decoder.layers.{layer_number}.self_attention.q_layernorm.weight": [
            "model.language_model.layers.{layer_number}.self_attn.q_norm.weight",
        ],
        "decoder.layers.{layer_number}.self_attention.k_layernorm.weight": [
            "model.language_model.layers.{layer_number}.self_attn.k_norm.weight",
        ],
    }

    _MLP_MAPPING = {
        "decoder.layers.{layer_number}.mlp.linear_fc1.weight": [
            "model.language_model.layers.{layer_number}.mlp.gate_proj.weight",
            "model.language_model.layers.{layer_number}.mlp.up_proj.weight",
        ],
        "decoder.layers.{layer_number}.mlp.linear_fc2.weight": [
            "model.language_model.layers.{layer_number}.mlp.down_proj.weight",
        ],
        "decoder.layers.{layer_number}.mlp.linear_fc1.layer_norm_weight": [
            "model.language_model.layers.{layer_number}.pre_feedforward_layernorm.weight",
        ],
        "decoder.layers.{layer_number}.pre_mlp_layernorm.weight": [
            "model.language_model.layers.{layer_number}.pre_feedforward_layernorm.weight",
        ],
    }

    _OTHER_MAPPING = {
        "decoder.layers.{layer_number}.post_attention_layernorm.weight": [
            "model.language_model.layers.{layer_number}.post_attention_layernorm.weight",
        ],
        "decoder.layers.{layer_number}.post_feedforward_layernorm.weight": [
            "model.language_model.layers.{layer_number}.post_feedforward_layernorm.weight",
        ],
        "decoder.layers.{layer_number}.layer_scalar": [
            "model.language_model.layers.{layer_number}.layer_scalar",
        ],
        # MoE (26B-A4B)
        "decoder.layers.{layer_number}.router.proj.weight": [
            "model.language_model.layers.{layer_number}.router.proj.weight",
        ],
        "decoder.layers.{layer_number}.router.scale": [
            "model.language_model.layers.{layer_number}.router.scale",
        ],
        "decoder.layers.{layer_number}.router.per_expert_scale": [
            "model.language_model.layers.{layer_number}.router.per_expert_scale",
        ],
        "decoder.layers.{layer_number}.experts.gate_up_proj": [
            "model.language_model.layers.{layer_number}.experts.gate_up_proj",
        ],
        "decoder.layers.{layer_number}.experts.down_proj": [
            "model.language_model.layers.{layer_number}.experts.down_proj",
        ],
        "decoder.layers.{layer_number}.pre_feedforward_layernorm_2.weight": [
            "model.language_model.layers.{layer_number}.pre_feedforward_layernorm_2.weight",
        ],
        "decoder.layers.{layer_number}.post_feedforward_layernorm_2.weight": [
            "model.language_model.layers.{layer_number}.post_feedforward_layernorm_2.weight",
        ],
        "decoder.layers.{layer_number}.post_feedforward_layernorm_1.weight": [
            "model.language_model.layers.{layer_number}.post_feedforward_layernorm_1.weight",
        ],
    }

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.language_model.norm.weight",
        "output_layer.weight": "model.language_model.embed_tokens.weight",  # tied embeddings
    }

    _BUFFER_NAMES = [
        "model.language_model.layers.{layer_number}.layer_scalar",
    ]

    _GLOBAL_ATTN_LAYERS = None  # derived from HF config in __init__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hf_text = self.hf_config.text_config if hasattr(self.hf_config, "text_config") else self.hf_config
        layer_types = getattr(hf_text, "layer_types", [])
        self._GLOBAL_ATTN_LAYERS = {i for i, t in enumerate(layer_types) if t == "full_attention"}

    def _weight_name_mapping_attention(self, name: str) -> list[str]:
        split_name = name.split(".")
        layer_number = int(split_name[2])
        split_name[2] = "{layer_number}"
        key = ".".join(split_name)

        # For global layers with K=V, linear_qkv maps to only [q_proj, k_proj]
        if key == "decoder.layers.{layer_number}.self_attention.linear_qkv.weight":
            if layer_number in self._GLOBAL_ATTN_LAYERS:
                return [
                    f"model.language_model.layers.{layer_number}.self_attn.q_proj.weight",
                    f"model.language_model.layers.{layer_number}.self_attn.k_proj.weight",
                ]

        return [x.format(layer_number=layer_number) for x in self._ATTENTION_MAPPING[key]]

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        split_name = name.split(".")
        layer_number = split_name[2]
        split_name[2] = "{layer_number}"
        key = ".".join(split_name)
        return [x.format(layer_number=layer_number) for x in self._MLP_MAPPING[key]]

    def _weight_name_mapping_other(self, name: str) -> list[str]:
        split_name = name.split(".")
        layer_number = split_name[2]
        split_name[2] = "{layer_number}"
        key = ".".join(split_name)
        return [x.format(layer_number=layer_number) for x in self._OTHER_MAPPING[key]]

    def _weight_to_mcore_format(self, mcore_weights_name, hf_weights):
        if len(hf_weights) == 1:
            return hf_weights[0]

        if (
            "self_attention.linear_qkv." in mcore_weights_name
            and "layer_norm" not in mcore_weights_name
        ):
            m = re.search(r"layers\.(\d+)\.", mcore_weights_name)
            layer_num = int(m.group(1)) if m else -1
            is_global = layer_num in self._GLOBAL_ATTN_LAYERS

            hf_text = self.hf_config.text_config if hasattr(self.hf_config, "text_config") else self.hf_config
            num_attention_heads = hf_text.num_attention_heads
            if is_global:
                head_dim = hf_text.global_head_dim
                num_kv_heads = hf_text.num_global_key_value_heads
            else:
                head_dim = hf_text.head_dim
                num_kv_heads = hf_text.num_key_value_heads

            # For K=V global layers: [q, k] -> [q, k, k]
            if len(hf_weights) == 2:
                q, k = hf_weights
                hf_weights = [q, k, k.clone()]

            q, k, v = hf_weights
            group_dim = head_dim * num_attention_heads // num_kv_heads
            real_num_kv_heads = q.shape[0] // group_dim
            q = q.view(real_num_kv_heads, group_dim, -1)
            k = k.view(real_num_kv_heads, head_dim, -1)
            v = v.view(real_num_kv_heads, head_dim, -1)
            return torch.cat([q, k, v], dim=1).view(-1, hf_text.hidden_size).contiguous()

        if "linear_fc1.weight" in mcore_weights_name:
            assert len(hf_weights) == 2
            gate, up = hf_weights
            return torch.cat([gate, up], dim=0)

        raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")

    def _build_config(self):
        text_config_key = "text_config" if hasattr(self.hf_config, "text_config") else None
        hf_text = self.hf_config.text_config if text_config_key else self.hf_config

        return self._build_base_config(
            text_config_key=text_config_key,
            use_cpu_initialization=False,
            add_qkv_bias=False,
            qk_layernorm=True,
            layernorm_zero_centered_gamma=False,
            normalization="RMSNorm",
            persist_layer_norm=True,
            activation_func=_gelu_tanh,
            bias_activation_fusion=False,
            bias_dropout_fusion=True,
            rope_local_base_freq=hf_text.rope_parameters.get(
                "sliding_attention", {}
            ).get("rope_theta", 10000.0),
        )
