"""Compare TG kernel logits against a full PyTorch reference.

The layer-sweep test only compares x_resid absmax, which can hide element-wise
divergence.  This test builds a full PyTorch reference (RMSNorm → QKV → QK-norm
→ RoPE → attention → O-proj → FFN → final norm → LM head) and compares the
actual logits against the TG kernel output.

Usage:
    CUDA_VISIBLE_DEVICES=0 \
    LD_PRELOAD=.nccl_shim.so \
    HF_HUB_OFFLINE=1 \
    python tests/test_tg_logits_compare.py
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import sys
import glob
import math
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
GROUP_SIZE = NUM_Q_HEADS // NUM_KV_HEADS  # 2
VOCAB = 151936
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


# ---------------------------------------------------------------------------
# Full PyTorch reference (matches kernel.cu / TG kernel computation order)
# ---------------------------------------------------------------------------

def pytorch_full_reference(layer_weights, embed_weight, final_norm_weight,
                           cos_table, sin_table, token_id=42, num_layers=NUM_LAYERS):
    """Full single-token forward pass in PyTorch, mirroring the kernel.

    Uses bf16 intermediate rounding (tg_round_bf16) to match the kernel's
    numerical path as closely as possible.
    """
    bf16 = torch.bfloat16
    f32 = torch.float32

    def round_bf16(x):
        return x.to(bf16).to(f32)

    # x_resid starts as embedding (bf16)
    x_resid = embed_weight[token_id].to(f32).cuda()  # [hidden]

    for layer in range(num_layers):
        lw = layer_weights[layer * 11: (layer + 1) * 11]
        ln_w = lw[0].to(f32).cuda()
        q_w = lw[1].to(f32).cuda()
        k_w = lw[2].to(f32).cuda()
        v_w = lw[3].to(f32).cuda()
        q_norm_w = lw[4].to(f32).cuda()
        k_norm_w = lw[5].to(f32).cuda()
        o_proj_w = lw[6].to(f32).cuda()
        post_ln_w = lw[7].to(f32).cuda()
        gate_w = lw[8].to(f32).cuda()
        up_w = lw[9].to(f32).cuda()
        down_w = lw[10].to(f32).cuda()

        # --- Input RMSNorm ---
        rms = x_resid.pow(2).mean().sqrt() + 1e-6
        rstd = 1.0 / rms
        x_normed = round_bf16(round_bf16(x_resid * rstd) * ln_w)

        # --- QKV projection ---
        q = x_normed @ q_w.T  # [q_size]
        k = x_normed @ k_w.T  # [kv_size]
        v = x_normed @ v_w.T  # [kv_size]

        # --- QK-norm ---
        q = q.view(NUM_Q_HEADS, HEAD_DIM)
        k = k.view(NUM_KV_HEADS, HEAD_DIM)
        v = v.view(NUM_KV_HEADS, HEAD_DIM)

        for h in range(NUM_KV_HEADS):
            # K-norm
            k_rms = k[h].pow(2).mean().sqrt() + 1e-6
            k_rstd = 1.0 / k_rms
            k[h] = round_bf16(round_bf16(k[h] * k_rstd) * k_norm_w)

        for h in range(NUM_Q_HEADS):
            # Q-norm
            q_rms = q[h].pow(2).mean().sqrt() + 1e-6
            q_rstd = 1.0 / q_rms
            q[h] = round_bf16(round_bf16(q[h] * q_rstd) * q_norm_w)

        # --- RoPE at position 0 ---
        # cos(0)=1, sin(0)=0 → RoPE is identity at position 0.
        # Skip explicitly (identity).
        # But apply to match numerical path (cos/sin loaded as bf16):
        cos_pos = cos_table[0].to(f32)  # [head_dim]
        sin_pos = sin_table[0].to(f32)  # [head_dim]
        for h in range(NUM_KV_HEADS):
            kh = k[h]
            half = HEAD_DIM // 2
            k_new = torch.zeros_like(kh)
            k_new[:half] = kh[:half] * cos_pos[:half] - kh[half:] * sin_pos[:half]
            k_new[half:] = kh[half:] * sin_pos[half:] + kh[:half] * cos_pos[half:]
            # Wait, the kernel RoPE uses a different pairing:
            # For i < hd/2: out[i] = x[i]*cos - x[i+hd/2]*sin
            # For i >= hd/2: out[i] = x[i-hd/2]*sin + x[i]*cos
            # Let me use the kernel's exact formula:
            k_rot = torch.zeros_like(kh)
            for i in range(HEAD_DIM):
                cv = cos_pos[i].item()
                sv = sin_pos[i].item()
                if i < half:
                    partner = i + half
                    k_rot[i] = round_bf16(kh[i] * cv - kh[partner] * sv)
                else:
                    partner = i - half
                    k_rot[i] = round_bf16(kh[partner] * sv + kh[i] * cv)
            k[h] = k_rot

        for h in range(NUM_Q_HEADS):
            qh = q[h]
            q_rot = torch.zeros_like(qh)
            for i in range(HEAD_DIM):
                cv = cos_pos[i].item()
                sv = sin_pos[i].item()
                if i < half:
                    partner = i + half
                    q_rot[i] = round_bf16(qh[i] * cv - qh[partner] * sv)
                else:
                    partner = i - half
                    q_rot[i] = round_bf16(qh[partner] * sv + qh[i] * cv)
            q[h] = q_rot

        # --- Attention (single token, position 0) ---
        # For seq_len=1: softmax(q@k.T) = 1.0, attn_out = v
        attn_out = torch.zeros(Q_SIZE, device="cuda", dtype=f32)
        for qh in range(NUM_Q_HEADS):
            kv_head = qh // GROUP_SIZE
            attn_out[qh * HEAD_DIM:(qh + 1) * HEAD_DIM] = v[kv_head]

        # --- O-proj + residual ---
        o_proj = attn_out @ o_proj_w.T  # [hidden]
        x_attn = round_bf16(o_proj) + x_resid

        # --- Post-attn RMSNorm ---
        rms2 = x_attn.pow(2).mean().sqrt() + 1e-6
        rstd2 = 1.0 / rms2
        x_normed2 = round_bf16(round_bf16(x_attn * rstd2) * post_ln_w)

        # --- FFN: gate, up, SiLU, down ---
        gate = x_normed2 @ gate_w.T  # [intermediate]
        up = x_normed2 @ up_w.T      # [intermediate]
        silu = round_bf16(torch.sigmoid(gate) * gate)
        silu_out = round_bf16(silu * up)
        ffn_down = silu_out @ down_w.T  # [hidden]

        # --- Residual ---
        x_ffn = round_bf16(ffn_down) + x_attn
        x_resid = round_bf16(x_ffn)

    # --- Final RMSNorm ---
    rms_f = x_resid.pow(2).mean().sqrt() + 1e-6
    rstd_f = 1.0 / rms_f
    x_final = round_bf16(round_bf16(x_resid * rstd_f) * final_norm_weight.to(f32).cuda())

    # --- LM head (tied embeddings) ---
    logits = x_final @ embed_weight.to(f32).cuda().T  # [vocab]

    return logits, x_resid


# ---------------------------------------------------------------------------
# TG kernel runner
# ---------------------------------------------------------------------------

def run_tg_kernel(layer_weights_packed, embed_weight, final_norm_weight,
                  cos_table, sin_table, token_id=42, num_layers=NUM_LAYERS):
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
        num_layers=num_layers,
        hidden=HIDDEN_SIZE,
        intermediate=INTERMEDIATE_SIZE,
        q_size=Q_SIZE,
        kv_size=KV_SIZE,
        num_q_heads=NUM_Q_HEADS,
        hkv=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        vocab=VOCAB,
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
    s["lm_logits"] = torch.empty(WMMA_PAD, VOCAB, **f32)
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
        HEAD_DIM, VOCAB, 1, 1, 1,
    )

    torch.cuda.synchronize()

    # The argmax token is stored as int32 at lm_logits[0, 0]
    tg_token = s["lm_logits"][0, 0].view(torch.int32).item()

    # The actual logits are in lm_logits[0, :] (but [0] is overwritten with token id)
    # We need to re-read from a clean copy. Actually, the argmax overwrites
    # lm_logits[batch] as int, which is lm_logits[0, 0] (4 bytes).
    # The float logits start at lm_logits[0, 0] but the first float is clobbered.
    # Let's just read from index 1 onwards for comparison, and note that
    # the PT reference's logits[0] is the true value.
    tg_logits = s["lm_logits"][0].clone()
    # Restore the clobbered first element from the PT reference later

    tg_x_resid = s["x_resid"][0].to(torch.float32)

    # Clean up
    for v in s.values():
        del v
    del k_cache, v_cache, sched
    torch.cuda.empty_cache()

    return tg_logits, tg_x_resid, tg_token


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    model_path = _resolve_model_path()
    print(f"Loading model from {model_path}...")
    weights, tokenizer = load_weights(model_path, verbose=False)
    cos_table, sin_table = _rope_tables()
    layer_weights_packed = _pack_layer_weights(weights["layer_weights"])
    embed_weight = weights["embed_weight"]
    final_norm_weight = weights["final_norm_weight"]

    token_id = 42

    # Run TG kernel
    print(f"\nRunning TG kernel (token={token_id})...")
    tg_logits, tg_x_resid, tg_token = run_tg_kernel(
        layer_weights_packed, embed_weight, final_norm_weight,
        cos_table, sin_table, token_id=token_id,
    )
    print(f"  TG token: {tg_token}")
    print(f"  TG x_resid absmax: {tg_x_resid.abs().max().item():.4f}")
    print(f"  TG logits[1:10]: {tg_logits[1:10].tolist()}")

    # Run PyTorch reference
    print(f"\nRunning PyTorch reference (token={token_id})...")
    pt_logits, pt_x_resid = pytorch_full_reference(
        weights["layer_weights"], embed_weight, final_norm_weight,
        cos_table, sin_table, token_id=token_id,
    )
    pt_token = pt_logits.argmax().item()
    print(f"  PT token: {pt_token}")
    print(f"  PT x_resid absmax: {pt_x_resid.abs().max().item():.4f}")
    print(f"  PT logits[1:10]: {pt_logits[1:10].tolist()}")

    # Compare x_resid
    x_resid_diff = (tg_x_resid - pt_x_resid).abs()
    print(f"\n--- x_resid comparison ---")
    print(f"  Max abs diff:  {x_resid_diff.max().item():.6f}")
    print(f"  Mean abs diff: {x_resid_diff.mean().item():.6f}")
    print(f"  TG absmax:     {tg_x_resid.abs().max().item():.4f}")
    print(f"  PT absmax:     {pt_x_resid.abs().max().item():.4f}")

    # Compare logits (skip index 0 which is clobbered by argmax in TG)
    # Actually, the argmax overwrites lm_logits[0,0] (4 bytes as int).
    # The float at index 0 is clobbered. Compare from index 1.
    # But we can restore index 0 from PT.
    tg_logits_f = tg_logits.to(torch.float32)
    pt_logits_f = pt_logits.to(torch.float32)

    # Compare top-10 logits
    pt_top10 = pt_logits_f.topk(10)
    tg_top10 = tg_logits_f.topk(10)
    print(f"\n--- Top-10 logits ---")
    print(f"  PT: tokens={pt_top10.indices.tolist()}, vals={[f'{v:.4f}' for v in pt_top10.values.tolist()]}")
    print(f"  TG: tokens={tg_top10.indices.tolist()}, vals={[f'{v:.4f}' for v in tg_top10.values.tolist()]}")

    # Logit comparison (skip index 0 which is clobbered)
    logits_diff = (tg_logits_f[1:] - pt_logits_f[1:]).abs()
    print(f"\n--- Logits comparison (excluding index 0) ---")
    print(f"  Max abs diff:  {logits_diff.max().item():.6f}")
    print(f"  Mean abs diff: {logits_diff.mean().item():.6f}")
    print(f"  PT logit range: [{pt_logits_f[1:].min().item():.4f}, {pt_logits_f[1:].max().item():.4f}]")
    print(f"  TG logit range: [{tg_logits_f[1:].min().item():.4f}, {tg_logits_f[1:].max().item():.4f}]")

    # Check if the top-1 matches
    if tg_token == pt_token:
        print(f"\n*** TOKENS MATCH: {tg_token} ***")
    else:
        print(f"\n*** TOKEN MISMATCH: TG={tg_token}, PT={pt_token} ***")
        # Check how far apart they are in logit space
        pt_logit_at_tg = pt_logits_f[tg_token].item()
        tg_logit_at_pt = tg_logits_f[pt_token].item()
        pt_logit_at_pt = pt_logits_f[pt_token].item()
        tg_logit_at_tg = tg_logits_f[tg_token].item()
        print(f"  PT logit at PT token ({pt_token}): {pt_logit_at_pt:.4f}")
        print(f"  TG logit at TG token ({tg_token}): {tg_logit_at_tg:.4f}")
        print(f"  PT logit at TG token ({tg_token}): {pt_logit_at_tg:.4f}")
        print(f"  TG logit at PT token ({pt_token}): {tg_logit_at_pt:.4f}")
        print(f"  Gap (PT): {pt_logit_at_pt - pt_logit_at_tg:.4f}")
        print(f"  Gap (TG): {tg_logit_at_tg - tg_logit_at_pt:.4f}")


if __name__ == "__main__":
    main()
