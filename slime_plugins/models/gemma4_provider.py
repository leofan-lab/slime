"""Custom model provider for Gemma4 that installs embedding scaling and logit softcapping hooks."""
import json
import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.transformer.spec_utils import import_module


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

    # Install Gemma4 hooks
    _install_hooks(model, args, config, pre_process, post_process)
    print(f"[Gemma4Provider] Hooks installed: pre_process={pre_process}, post_process={post_process}")

    return model


def _install_hooks(model, args, config, pre_process, post_process):
    hidden_size = config.hidden_size
    embed_scale = torch.tensor(hidden_size ** 0.5, dtype=torch.bfloat16).item()

    # Read softcapping from HF config
    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    hf_text = hf_config.text_config if hasattr(hf_config, "text_config") else hf_config
    softcap = getattr(hf_text, "final_logit_softcapping", None)

    inner = model.module if hasattr(model, "module") else model
    print(f"[Gemma4Hooks] inner type={type(inner).__name__}, has embedding={hasattr(inner, 'embedding')}, has decoder={hasattr(inner, 'decoder')}, has output_layer={hasattr(inner, 'output_layer')}")
    print(f"[Gemma4Hooks] embed_scale={embed_scale}, softcap={softcap}")

    # Embedding scaling
    if pre_process and hasattr(inner, "embedding"):
        def _embed_hook(module, input, output):
            return output * embed_scale
        inner.embedding.register_forward_hook(_embed_hook)

    # Logit softcapping
    if post_process and softcap and hasattr(inner, "output_layer"):
        def _softcap_hook(module, input, output):
            if isinstance(output, tuple):
                logits = output[0]
                return (torch.tanh(logits / softcap) * softcap,) + output[1:]
            return torch.tanh(output / softcap) * softcap
        inner.output_layer.register_forward_hook(_softcap_hook)

    # Layer scalars from HF checkpoint
    if hasattr(inner, "decoder") and args.hf_checkpoint:
        try:
            with open(f"{args.hf_checkpoint}/model.safetensors.index.json") as f:
                index = json.load(f)
            from safetensors import safe_open
            scalars = {}
            for key, filename in index["weight_map"].items():
                if "layer_scalar" in key:
                    layer_idx = int(key.split(".layers.")[1].split(".")[0])
                    with safe_open(f"{args.hf_checkpoint}/{filename}", framework="pt", device="cpu") as sf:
                        scalars[layer_idx] = sf.get_tensor(key).item()
            print(f"[Gemma4] Loaded {len(scalars)} layer scalars, range: {min(scalars.values()):.4f} - {max(scalars.values()):.4f}")
            loaded = 0
            for i, layer in enumerate(inner.decoder.layers):
                if hasattr(layer, "layer_scalar"):
                    layer.layer_scalar.fill_(scalars.get(i, 1.0))
                    loaded += 1
            print(f"[Gemma4] Applied scalars to {loaded}/{len(inner.decoder.layers)} layers")
        except Exception as e:
            print(f"[Gemma4] Warning: could not load layer scalars: {e}")
            import traceback; traceback.print_exc()
