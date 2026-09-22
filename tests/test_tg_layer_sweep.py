"""Debug NaN: build schedules with increasing num_layers to find the
divergence point.

Strategy: build a schedule with N layers (N=1,2,3,...) and check x_resid
absmax after the kernel completes. If x_resid is reasonable for N layers
but explodes for N+1, the bug is in layer N+1.
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import sys
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


def run_with_n_layers(n_layers, layer_weights_packed, embed_weight,
                      final_norm_weight, cos_table, sin_table):
    """Build a schedule with n_layers layers and run the kernel."""
    BS = 1
    block_size = 16
    num_cache_blocks = 256
    kv_layer_stride = num_cache_blocks * block_size * KV_SIZE

    k_cache = torch.zeros(
        NUM_LAYERS, num_cache_blocks, block_size,
        NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda",
    )
    v_cache = torch.zeros_like(k_cache)

    geo = tg_schedule.TgGeometry(
        num_layers=n_layers,
        hidden=HIDDEN_SIZE,
        intermediate=INTERMEDIATE_SIZE,
        q_size=Q_SIZE,
        kv_size=KV_SIZE,
        num_q_heads=NUM_Q_HEADS,
        hkv=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        vocab=151936,
        BS=BS,
        tiling=tg_schedule.TgTiling(),
        attn_splits=1,
        max_attn_splits=4,
    )
    sched = tg_schedule.build_schedule(geo, num_sms=1)

    f32 = dict(dtype=torch.float32, device="cuda")
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
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

    _DO_NOT_ZERO = {"inst_buf", "num_inst_per_sm", "page_table", "cache_seqlens", "row_active"}
    for k, v in s.items():
        if k in _DO_NOT_ZERO:
            continue
        if v.dtype in (torch.uint32, torch.float32, torch.bfloat16, torch.int32):
            v.zero_()

    token_id = 42
    s["x_resid"][0] = embed_weight[token_id]
    s["cache_seqlens"][0] = 0
    s["page_table"][0, 0] = 0
    s["row_active"][0] = 1

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

    x_resid_absmax = s["x_resid"].abs().max().item()
    silu_absmax = s["silu_out"][:BS].abs().max().item()
    o_proj_absmax = s["o_proj_out"][:BS].abs().max().item()
    attn_absmax = s["attn_out"][:BS].abs().max().item()

    for v in s.values():
        del v
    del k_cache, v_cache, sched
    torch.cuda.empty_cache()

    return {
        "x_resid_absmax": x_resid_absmax,
        "silu_absmax": silu_absmax,
        "o_proj_absmax": o_proj_absmax,
        "attn_absmax": attn_absmax,
    }


def pytorch_ref_n_layers(n_layers, layer_weights, embed_weight, final_norm_weight):
    """PyTorch reference for n_layers."""
    x_resid = embed_weight[42].float().cuda()

    for layer in range(n_layers):
        ln_w = layer_weights[layer * 11 + 0].float().cuda()
        rms = x_resid.pow(2).mean().sqrt() + 1e-6
        x_normed = (x_resid / rms) * ln_w

        qw = layer_weights[layer * 11 + 1].float().cuda()
        kw = layer_weights[layer * 11 + 2].float().cuda()
        vw = layer_weights[layer * 11 + 3].float().cuda()
        q = x_normed @ qw.T
        k = x_normed @ kw.T
        v = x_normed @ vw.T

        q_norm_w = layer_weights[layer * 11 + 4].float().cuda()
        k_norm_w = layer_weights[layer * 11 + 5].float().cuda()
        q = q.view(NUM_Q_HEADS, HEAD_DIM)
        k = k.view(NUM_KV_HEADS, HEAD_DIM)
        v = v.view(NUM_KV_HEADS, HEAD_DIM)
        for h in range(NUM_Q_HEADS):
            rms_h = q[h].pow(2).mean().sqrt() + 1e-6
            q[h] = (q[h] / rms_h) * q_norm_w
        for h in range(NUM_KV_HEADS):
            rms_h = k[h].pow(2).mean().sqrt() + 1e-6
            k[h] = (k[h] / rms_h) * k_norm_w

        attn_out = torch.zeros(Q_SIZE, device="cuda", dtype=torch.float32)
        for qh in range(NUM_Q_HEADS):
            kv_head = qh // (NUM_Q_HEADS // NUM_KV_HEADS)
            attn_out[qh * HEAD_DIM:(qh + 1) * HEAD_DIM] = v[kv_head]

        o_proj_w = layer_weights[layer * 11 + 6].float().cuda()
        o_proj_out = attn_out @ o_proj_w.T
        x_attn = x_resid + o_proj_out

        post_ln_w = layer_weights[layer * 11 + 7].float().cuda()
        rms = x_attn.pow(2).mean().sqrt() + 1e-6
        x_normed2 = (x_attn / rms) * post_ln_w

        gate_w = layer_weights[layer * 11 + 8].float().cuda()
        up_w = layer_weights[layer * 11 + 9].float().cuda()
        gate = x_normed2 @ gate_w.T
        up = x_normed2 @ up_w.T
        silu = (gate * torch.sigmoid(gate)) * up

        down_w = layer_weights[layer * 11 + 10].float().cuda()
        ffn_down = silu @ down_w.T
        x_resid = x_attn + ffn_down

    return x_resid.abs().max().item()


def main():
    model_path = _resolve_model_path()
    print(f"Loading model from {model_path}...")
    weights, tokenizer = load_weights(model_path, verbose=False)
    cos_table, sin_table = _rope_tables()
    layer_weights_packed = _pack_layer_weights(weights["layer_weights"])
    embed_weight = weights["embed_weight"]
    final_norm_weight = weights["final_norm_weight"]

    print(f"\n{'Layers':>6}  {'TG x_resid':>14}  {'PT x_resid':>14}  {'Ratio':>8}  {'TG silu':>14}")
    print("-" * 70)

    all_ok = True
    for n in [1, 2, 3, 5, 10, 28]:
        try:
            tg = run_with_n_layers(n, layer_weights_packed, embed_weight,
                                   final_norm_weight, cos_table, sin_table)
            pt_absmax = pytorch_ref_n_layers(n, weights["layer_weights"], embed_weight,
                                             final_norm_weight)
            ratio = tg["x_resid_absmax"] / pt_absmax if pt_absmax > 0 else 0
            ok = 0.5 < ratio < 2.0
            all_ok = all_ok and ok
            status = "OK" if ok else "MISMATCH"
            print(f"{n:6d}  {tg['x_resid_absmax']:14.4e}  {pt_absmax:14.4e}  "
                  f"{ratio:8.4f}  {tg['silu_absmax']:14.4e}  {status}")
        except Exception as e:
            print(f"{n:6d}  ERROR: {e}")
            import traceback; traceback.print_exc()
            all_ok = False

    if all_ok:
        print("\nAll layer counts match PyTorch reference (within 2x).")
    else:
        print("\nSome layer counts mismatch — see above.")


if __name__ == "__main__":
    main()
