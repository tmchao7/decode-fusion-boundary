"""Debug NaN: compare task-graph kernel with reference kernel.cu and
PyTorch reference for layer 0.

Strategy:
1. Run reference kernel.cu for token 42, get output token.
2. Run task-graph kernel for token 42, check x_resid / lm_logits.
3. Compute PyTorch reference for layer 0 QKV and compare with kernel q_out.
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import sys
import math
import glob
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nanovllm.backends.megakernel.build import get_extension
get_extension()

from nanovllm.backends.megakernel.model import (
    load_weights, _pack_layer_weights, _rope_tables,
    NUM_LAYERS, HIDDEN_SIZE, INTERMEDIATE_SIZE,
    NUM_KV_HEADS, HEAD_DIM, Q_SIZE, KV_SIZE, MAX_SEQ_LEN,
)
from nanovllm.backends.megakernel import schedule as tg_schedule
from nanovllm.backends.megakernel.model import Decoder

NUM_Q_HEADS = Q_SIZE // HEAD_DIM  # 16
_tg_ops = torch.ops.nanovllm_megakernel_C


def _resolve_model_path() -> str:
    env = os.environ.get("QWEN3_MODEL_PATH")
    if env and os.path.isfile(os.path.join(env, "config.json")):
        return env
    if os.path.isfile("/models/Qwen3-0.6B/config.json"):
        return "/models/Qwen3-0.6B"
    for pattern in (
        "/home/*/edge_llm/*/models/Qwen3-0.6B",
        "/home/*/models/Qwen3-0.6B",
        "/models/Qwen3-0.6B",
        "/root/models/Qwen3-0.6B",
    ):
        for cand in glob.glob(pattern):
            if os.path.isfile(os.path.join(cand, "model.safetensors")) and \
               os.path.isfile(os.path.join(cand, "config.json")):
                return cand
    return "Qwen/Qwen3-0.6B"


def main():
    model_path = _resolve_model_path()
    print(f"Loading model from {model_path}...")
    weights, tokenizer = load_weights(model_path, verbose=False)
    cos_table, sin_table = _rope_tables()
    layer_weights_packed = _pack_layer_weights(weights["layer_weights"])
    embed_weight = weights["embed_weight"]
    final_norm_weight = weights["final_norm_weight"]

    token_id = 42

    # ==================================================================
    # Part 1: Run reference kernel.cu (Decoder.decode_paged_batch_wmma)
    # ==================================================================
    print("\n=== Part 1: Reference kernel.cu ===")
    try:
        dec = Decoder(
            weights=weights,
            tokenizer=tokenizer,
            max_batch_size=1,
        )
        # Set up for paged decode
        block_size = 16
        num_cache_blocks = 256
        dec._paged = True
        dec._block_size = block_size
        dec._num_cache_blocks = num_cache_blocks
        dec._k_cache = torch.zeros(
            NUM_LAYERS, num_cache_blocks, block_size,
            NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda",
        )
        dec._v_cache = torch.zeros_like(dec._k_cache)

        input_ids = torch.tensor([token_id], dtype=torch.int64, device="cuda")
        positions = torch.tensor([0], dtype=torch.int64, device="cuda")
        block_tables = torch.zeros(1, 64, dtype=torch.int32, device="cuda")
        block_tables[0, 0] = 0

        ref_token = dec.decode_paged_batch_wmma(input_ids, positions, block_tables)
        print(f"Reference kernel output token: {ref_token}")
    except Exception as e:
        print(f"Reference kernel failed: {e}")
        import traceback; traceback.print_exc()
        ref_token = None

    # ==================================================================
    # Part 2: Run task-graph kernel and check intermediate buffers
    # ==================================================================
    print("\n=== Part 2: Task-graph kernel ===")
    BS = 1
    block_size = 16
    num_cache_blocks = 256
    kv_layer_stride = num_cache_blocks * block_size * KV_SIZE

    k_cache = torch.zeros(
        NUM_LAYERS, num_cache_blocks, block_size,
        NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda",
    )
    v_cache = torch.zeros_like(k_cache)

    geo = tg_schedule.default_geometry(BS=1, attn_splits=1)
    sched = tg_schedule.build_schedule(geo, num_sms=1)

    f32 = dict(dtype=torch.float32, device="cuda")
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    i32 = dict(dtype=torch.int32, device="cuda")
    u32 = dict(dtype=torch.uint32, device="cuda")

    WMMA_PAD = 16
    s = {}
    s["inst_buf"] = torch.tensor(sched.flat_inst_buf, dtype=torch.int32).cuda()
    s["num_inst_per_sm"] = torch.tensor(sched.num_inst_per_sm, dtype=torch.int32).cuda()
    s["norm_out"] = torch.empty(BS, HIDDEN_SIZE, **f32)
    s["q_out"] = torch.empty(WMMA_PAD, Q_SIZE, **f32)
    s["k_out"] = torch.empty(WMMA_PAD, KV_SIZE, **f32)
    s["v_out"] = torch.empty(WMMA_PAD, KV_SIZE, **f32)
    s["attn_out"] = torch.empty(BS, Q_SIZE, **f32)
    s["o_proj_out"] = torch.empty(WMMA_PAD, HIDDEN_SIZE, **f32)
    s["x_attn"] = torch.empty(BS, HIDDEN_SIZE, **f32)
    s["post_attn_norm"] = torch.empty(BS, HIDDEN_SIZE, **f32)
    s["silu_out"] = torch.empty(WMMA_PAD, INTERMEDIATE_SIZE, **f32)
    s["ffn_down_out"] = torch.empty(WMMA_PAD, HIDDEN_SIZE, **f32)
    s["up_out"] = torch.empty(WMMA_PAD, INTERMEDIATE_SIZE, **f32)
    s["x_ffn"] = torch.empty(BS, HIDDEN_SIZE, **f32)
    s["x_resid"] = torch.empty(BS, HIDDEN_SIZE, **bf16)
    s["lm_logits"] = torch.empty(WMMA_PAD, 151936, **f32)
    s["o_partial"] = torch.empty(BS, NUM_KV_HEADS, 1, HIDDEN_SIZE, **f32)
    s["lse_partial"] = torch.empty(BS, NUM_KV_HEADS, 1, **f32)

    d = sched.bar_dims
    s["bar_qkv"] = torch.zeros(d.bar_qkv, **u32)
    s["bar_attn"] = torch.zeros(d.bar_attn, **u32)
    s["bar_oproj"] = torch.zeros(d.bar_oproj, **u32)
    s["bar_silu"] = torch.zeros(d.bar_silu, **u32)
    s["bar_ffn_down"] = torch.zeros(d.bar_ffn_down, **u32)
    s["bar_layer"] = torch.zeros(d.bar_layer, **u32)
    s["bar_combine"] = torch.zeros(max(d.bar_combine, 1), **u32)

    s["page_table"] = torch.zeros(BS, 64, dtype=torch.int32, device="cuda")
    s["cache_seqlens"] = torch.zeros(BS, dtype=torch.int32, device="cuda")
    s["row_active"] = torch.ones(BS, dtype=torch.int32, device="cuda")

    # Zero everything except inputs
    _DO_NOT_ZERO = {"inst_buf", "num_inst_per_sm", "page_table", "cache_seqlens", "row_active"}
    for k2, v2 in s.items():
        if k2 in _DO_NOT_ZERO:
            continue
        if v2.dtype in (torch.uint32, torch.float32, torch.bfloat16, torch.int32):
            v2.zero_()

    # Set up input
    s["x_resid"][0] = embed_weight[token_id]
    s["cache_seqlens"][0] = 0
    s["page_table"][0, 0] = 0
    s["row_active"][0] = 1

    # Print input stats
    print(f"Input x_resid absmax: {s['x_resid'].abs().max().item():.6f}")
    print(f"Input x_resid norm: {s['x_resid'].float().norm().item():.6f}")

    _tg_ops.tg_decode_step(
        s["inst_buf"], s["num_inst_per_sm"],
        sched.max_inst, 1,
        layer_weights_packed, embed_weight,
        final_norm_weight, cos_table, sin_table,
        k_cache, v_cache, s["page_table"],
        s["cache_seqlens"], s["row_active"],
        num_cache_blocks, 64, block_size,
        kv_layer_stride,
        s["norm_out"], s["q_out"], s["k_out"], s["v_out"],
        s["attn_out"], s["o_proj_out"], s["x_attn"],
        s["post_attn_norm"], s["silu_out"], s["ffn_down_out"],
        s["up_out"],
        s["x_ffn"], s["x_resid"], s["lm_logits"],
        s["o_partial"], s["lse_partial"],
        s["bar_qkv"], s["bar_attn"], s["bar_oproj"],
        s["bar_silu"], s["bar_ffn_down"], s["bar_layer"],
        s["bar_combine"],
        NUM_LAYERS, BS, HIDDEN_SIZE, INTERMEDIATE_SIZE,
        Q_SIZE, KV_SIZE, NUM_Q_HEADS, NUM_KV_HEADS,
        HEAD_DIM, 151936, 1, 1, 1,
    )

    torch.cuda.synchronize()

    # Check final buffer states (these contain LAST LAYER's values)
    print(f"\n--- Final buffer states (last layer values) ---")
    for name in ["q_out", "k_out", "v_out", "attn_out", "o_proj_out",
                 "x_attn", "silu_out", "ffn_down_out", "up_out",
                 "x_ffn", "x_resid", "norm_out", "lm_logits"]:
        t = s[name]
        has_nan = torch.isnan(t).any().item()
        has_inf = torch.isinf(t).any().item()
        absmax = t.abs().max().item() if not has_nan else float('nan')
        print(f"  {name:20s}: shape={list(t.shape)}, absmax={absmax:.4e}, nan={has_nan}, inf={has_inf}")

    # Check lm_logits
    logits = s["lm_logits"][0]
    has_nan = torch.isnan(logits).any().item()
    print(f"\nlm_logits has NaN: {has_nan}")
    if not has_nan:
        print(f"lm_logits argmax: {logits.argmax().item()}")
        print(f"lm_logits max: {logits.max().item():.4f}")

    # Check sentinel
    s0 = s["lm_logits"][0, 0].view(torch.int32).item()
    s1 = s["lm_logits"][0, 1].view(torch.int32).item()
    print(f"\nSentinel [0]: {s0:#x} (expect 0xdeadbeef)")
    print(f"Worker count [1]: {s1}")

    # ==================================================================
    # Part 3: PyTorch reference for layer 0 QKV
    # ==================================================================
    print("\n=== Part 3: PyTorch reference for layer 0 ===")
    # Input: embedding[token_id]
    x = embed_weight[token_id].float().cuda()  # [hidden]
    print(f"Input embedding absmax: {x.abs().max().item():.6f}")

    # RMSNorm with layer 0's input_layernorm
    layer_weights = weights["layer_weights"]
    input_ln_w = layer_weights[0].float().cuda()  # [hidden]
    rms = x.pow(2).mean().sqrt() + 1e-6
    x_normed = (x / rms) * input_ln_w
    print(f"RMSNorm output absmax: {x_normed.abs().max().item():.6f}")

    # QKV projection: q_weight = [q_size, hidden], output = x_normed @ q_weight^T
    q_weight = layer_weights[1].float().cuda()  # [q_size, hidden]
    k_weight = layer_weights[2].float().cuda()  # [kv_size, hidden]
    v_weight = layer_weights[3].float().cuda()  # [kv_size, hidden]

    q_ref = x_normed @ q_weight.T  # [q_size]
    k_ref = x_normed @ k_weight.T  # [kv_size]
    v_ref = x_normed @ v_weight.T  # [kv_size]

    print(f"PyTorch q_ref absmax: {q_ref.abs().max().item():.6f}")
    print(f"PyTorch k_ref absmax: {k_ref.abs().max().item():.6f}")
    print(f"PyTorch v_ref absmax: {v_ref.abs().max().item():.6f}")

    # Now we need to compare with the kernel's q_out. But the kernel overwrites
    # q_out at each layer, so after completion, q_out has the LAST layer's values.
    # We can't directly compare layer 0's q_out.

    # Instead, let's check if the kernel's FINAL x_resid matches a PyTorch
    # reference for all 28 layers.

    # ==================================================================
    # Part 4: Full PyTorch reference (all 28 layers)
    # ==================================================================
    print("\n=== Part 4: Full PyTorch reference (all 28 layers) ===")
    x_resid_pt = embed_weight[token_id].float().cuda()  # [hidden]

    for layer in range(NUM_LAYERS):
        # Input RMSNorm
        ln_w = layer_weights[layer * 11 + 0].float().cuda()
        rms = x_resid_pt.pow(2).mean().sqrt() + 1e-6
        x_normed = (x_resid_pt / rms) * ln_w

        # QKV
        qw = layer_weights[layer * 11 + 1].float().cuda()
        kw = layer_weights[layer * 11 + 2].float().cuda()
        vw = layer_weights[layer * 11 + 3].float().cuda()
        q = x_normed @ qw.T  # [q_size]
        k = x_normed @ kw.T  # [kv_size]
        v = x_normed @ vw.T  # [kv_size]

        # QK-norm + RoPE (simplified: just apply norm, skip RoPE for debug)
        q_norm_w = layer_weights[layer * 11 + 4].float().cuda()
        k_norm_w = layer_weights[layer * 11 + 5].float().cuda()

        # Reshape Q to [num_q_heads, head_dim]
        q = q.view(NUM_Q_HEADS, HEAD_DIM)
        k = k.view(NUM_KV_HEADS, HEAD_DIM)
        v = v.view(NUM_KV_HEADS, HEAD_DIM)

        # QK-norm (per head)
        for h in range(NUM_Q_HEADS):
            rms_h = q[h].pow(2).mean().sqrt() + 1e-6
            q[h] = (q[h] / rms_h) * q_norm_w
        for h in range(NUM_KV_HEADS):
            rms_h = k[h].pow(2).mean().sqrt() + 1e-6
            k[h] = (k[h] / rms_h) * k_norm_w

        # RoPE (skip for now — just note it)
        # For position 0, cos=1, sin=0, so RoPE is identity
        if layer == 0:
            pass  # position 0, RoPE is identity

        # Attention (for first token, just V since QK=1 position)
        # For cache_len=1, attention output = V[0] (only 1 position)
        attn_out = torch.zeros(Q_SIZE, device="cuda", dtype=torch.float32)
        for qh in range(NUM_Q_HEADS):
            kv_head = qh // (NUM_Q_HEADS // NUM_KV_HEADS)
            attn_out[qh * HEAD_DIM:(qh + 1) * HEAD_DIM] = v[kv_head]

        # O_PROJ
        o_proj_w = layer_weights[layer * 11 + 6].float().cuda()  # [hidden, q_size]
        o_proj_out = attn_out @ o_proj_w.T  # [hidden]

        # Residual add
        x_attn = x_resid_pt + o_proj_out

        # Post-attn RMSNorm
        post_ln_w = layer_weights[layer * 11 + 7].float().cuda()
        rms = x_attn.pow(2).mean().sqrt() + 1e-6
        x_normed2 = (x_attn / rms) * post_ln_w

        # Gate/Up
        gate_w = layer_weights[layer * 11 + 8].float().cuda()  # [intermediate, hidden]
        up_w = layer_weights[layer * 11 + 9].float().cuda()    # [intermediate, hidden]
        gate = x_normed2 @ gate_w.T  # [intermediate]
        up = x_normed2 @ up_w.T      # [intermediate]

        # SiLU
        silu = (gate * torch.sigmoid(gate)) * up  # [intermediate]

        # Down
        down_w = layer_weights[layer * 11 + 10].float().cuda()  # [hidden, intermediate]
        ffn_down = silu @ down_w.T  # [hidden]

        # FFN residual
        x_ffn = x_attn + ffn_down
        x_resid_pt = x_ffn

        if layer < 5 or layer >= NUM_LAYERS - 2:
            print(f"  Layer {layer:2d}: x_resid absmax={x_resid_pt.abs().max().item():.4e}, "
                  f"x_attn absmax={x_attn.abs().max().item():.4e}, "
                  f"o_proj absmax={o_proj_out.abs().max().item():.4e}, "
                  f"silu absmax={silu.abs().max().item():.4e}, "
                  f"ffn_down absmax={ffn_down.abs().max().item():.4e}")

    # Final RMSNorm
    rms = x_resid_pt.pow(2).mean().sqrt() + 1e-6
    x_final = (x_resid_pt / rms) * final_norm_weight.float().cuda()

    # LM head
    lm_head_w = embed_weight.float().cuda()  # [vocab, hidden] (tied)
    logits_pt = x_final @ lm_head_w.T  # [vocab]
    print(f"\nPyTorch final x_resid absmax: {x_resid_pt.abs().max().item():.4e}")
    print(f"PyTorch logits argmax: {logits_pt.argmax().item()}")
    print(f"PyTorch logits max: {logits_pt.max().item():.4f}")

    # Compare with reference kernel
    if ref_token is not None:
        print(f"\nReference kernel token: {ref_token}")
        print(f"PyTorch token: {logits_pt.argmax().item()}")
        print(f"Match: {ref_token[0] == logits_pt.argmax().item()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
