"""Qwen3-0.6B weight views and packing for the task-graph Megakernel."""

import struct

import torch

from nanovllm.rope_config import (
    QWEN3_DEFAULT_ROPE_THETA,
    assert_supported_rope_type,
    resolve_rope_theta,
)

NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
Q_SIZE = 16 * HEAD_DIM  # 2048
KV_SIZE = 8 * HEAD_DIM  # 1024
MAX_SEQ_LEN = 2048
VOCAB_SIZE = 151936


def _resolve_rope_theta(config, default: float = QWEN3_DEFAULT_ROPE_THETA) -> float:
    """Resolve RoPE exactly as the nano-vLLM Qwen3 model does."""

    return resolve_rope_theta(config, default)


def _rope_tables(rope_theta: float = QWEN3_DEFAULT_ROPE_THETA):
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    positions = torch.arange(MAX_SEQ_LEN, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos_table = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    sin_table = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    return cos_table, sin_table


def load_weights(model_name="Qwen/Qwen3-0.6B", verbose: bool = True):
    """Load Qwen3-0.6B weights from Hugging Face for standalone TG tools."""

    if not verbose:
        import os

        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    if not verbose:
        hf_logging.set_verbosity_error()
        try:
            hf_logging.disable_progress_bar()
        except AttributeError:
            pass
        try:
            from huggingface_hub import logging as hf_hub_logging

            hf_hub_logging.set_verbosity_error()
        except Exception:
            pass

    if verbose:
        print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    state = model.state_dict()

    assert_supported_rope_type(model.config)
    cos_table, sin_table = _rope_tables(_resolve_rope_theta(model.config))

    layer_weights = []
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}."
        layer_weights.extend(
            [
                state[p + "input_layernorm.weight"].contiguous(),
                state[p + "self_attn.q_proj.weight"].contiguous(),
                state[p + "self_attn.k_proj.weight"].contiguous(),
                state[p + "self_attn.v_proj.weight"].contiguous(),
                state[p + "self_attn.q_norm.weight"].contiguous(),
                state[p + "self_attn.k_norm.weight"].contiguous(),
                state[p + "self_attn.o_proj.weight"].contiguous(),
                state[p + "post_attention_layernorm.weight"].contiguous(),
                state[p + "mlp.gate_proj.weight"].contiguous(),
                state[p + "mlp.up_proj.weight"].contiguous(),
                state[p + "mlp.down_proj.weight"].contiguous(),
            ]
        )

    embed_weight = state["model.embed_tokens.weight"].contiguous()
    weights = dict(
        embed_weight=embed_weight,
        layer_weights=layer_weights,
        final_norm_weight=state["model.norm.weight"].contiguous(),
        lm_head_weight=embed_weight,
        cos_table=cos_table,
        sin_table=sin_table,
    )
    del model
    torch.cuda.empty_cache()
    return weights, tokenizer


def weights_from_nano_model(model, hf_config):
    """Create task-graph weight views from a tensor-parallel nano-vLLM model."""

    qwen = model.model
    if qwen.embed_tokens.weight.dtype != torch.bfloat16:
        raise RuntimeError("Qwen Megakernel requires BF16 model weights")
    if getattr(hf_config, "attention_bias", False):
        raise RuntimeError("Qwen Megakernel requires Qwen QK normalization")
    if model.lm_head.weight.data_ptr() != qwen.embed_tokens.weight.data_ptr():
        raise RuntimeError(
            "Qwen Megakernel requires tied lm_head and embed_tokens storage"
        )

    layer_weights = []
    for layer in qwen.layers:
        attn = layer.self_attn
        qkv = attn.qkv_proj.weight
        q_weight, k_weight, v_weight = qkv.split(
            [attn.q_size, attn.kv_size, attn.kv_size], dim=0
        )
        gate_weight, up_weight = layer.mlp.gate_up_proj.weight.split(
            [INTERMEDIATE_SIZE, INTERMEDIATE_SIZE], dim=0
        )
        layer_weights.extend(
            [
                layer.input_layernorm.weight,
                q_weight,
                k_weight,
                v_weight,
                attn.q_norm.weight,
                attn.k_norm.weight,
                attn.o_proj.weight,
                layer.post_attention_layernorm.weight,
                gate_weight,
                up_weight,
                layer.mlp.down_proj.weight,
            ]
        )

    assert_supported_rope_type(hf_config)
    cos_table, sin_table = _rope_tables(_resolve_rope_theta(hf_config))
    embed_weight = qwen.embed_tokens.weight
    return dict(
        embed_weight=embed_weight,
        layer_weights=layer_weights,
        final_norm_weight=qwen.norm.weight,
        lm_head_weight=model.lm_head.weight,
        cos_table=cos_table,
        sin_table=sin_table,
    )


def _pack_layer_weights(layer_weights: list[torch.Tensor]) -> torch.Tensor:
    """Pack 11 tensors per layer into the CUDA ABI's pointer blob."""

    ptr_size = 8
    n_ptrs = 11
    struct_bytes = n_ptrs * ptr_size
    buf = bytearray(NUM_LAYERS * struct_bytes)
    for i in range(NUM_LAYERS):
        for j in range(n_ptrs):
            ptr = layer_weights[i * n_ptrs + j].data_ptr()
            struct.pack_into("Q", buf, (i * n_ptrs + j) * ptr_size, ptr)
    return torch.frombuffer(buf, dtype=torch.uint8).cuda()
