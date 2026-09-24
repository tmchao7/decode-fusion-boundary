"""Debug: check if the FFN_UPGATE GEMM weight pointer is correct.

The gate/up GEMM output is ~77000x too large. This script checks:
1. Are the weight pointers in layer_weights_packed correct?
2. Are the weight values reasonable (absmax ~0.01)?
3. Is the tile_n offset correct?
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import sys
import struct
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nanovllm.backends.megakernel.build import get_extension
get_extension()

from nanovllm.backends.megakernel.weights import (
    load_weights, _pack_layer_weights, _rope_tables,
    NUM_LAYERS, HIDDEN_SIZE, INTERMEDIATE_SIZE,
    NUM_KV_HEADS, HEAD_DIM, Q_SIZE, KV_SIZE,
)
from nanovllm.backends.megakernel import schedule as tg_schedule

NUM_Q_HEADS = Q_SIZE // HEAD_DIM


def _resolve_model_path():
    import glob
    env = os.environ.get("QWEN3_MODEL_PATH")
    if env and os.path.isfile(os.path.join(env, "config.json")):
        return env
    if os.path.isfile("/models/Qwen3-0.6B/config.json"):
        return "/models/Qwen3-0.6B"
    for pattern in ("/home/*/edge_llm/*/models/Qwen3-0.6B",
                    "/home/*/models/Qwen3-0.6B",
                    "/models/Qwen3-0.6B",
                    "/root/models/Qwen3-0.6B"):
        for cand in glob.glob(pattern):
            if os.path.isfile(os.path.join(cand, "model.safetensors")) and \
               os.path.isfile(os.path.join(cand, "config.json")):
                return cand
    return "Qwen/Qwen3-0.6B"


def main():
    model_path = _resolve_model_path()
    print(f"Loading model from {model_path}...")
    weights, tokenizer = load_weights(model_path, verbose=False)
    layer_weights_packed = _pack_layer_weights(weights["layer_weights"])
    embed_weight = weights["embed_weight"]

    layer_weights = weights["layer_weights"]
    n_ptrs = 11

    # Check layer 0 weights
    print(f"\n=== Layer 0 weight check ===")
    for j, name in enumerate(["input_ln", "q_proj", "k_proj", "v_proj",
                              "q_norm", "k_norm", "o_proj", "post_attn_ln",
                              "gate_proj", "up_proj", "down_proj"]):
        w = layer_weights[j]
        print(f"  [{j:2d}] {name:15s}: shape={list(w.shape)}, "
              f"absmax={w.abs().max().item():.6f}, "
              f"mean={w.float().mean().item():.6f}, "
              f"data_ptr=0x{w.data_ptr():x}")

    # Check packed pointers
    print(f"\n=== Packed weight pointers (layer 0) ===")
    packed_buf = layer_weights_packed.cpu().numpy().tobytes()
    for j in range(n_ptrs):
        offset = j * 8  # 8 bytes per pointer
        ptr = struct.unpack_from("Q", packed_buf, offset)[0]
        w = layer_weights[j]
        expected_ptr = w.data_ptr()
        match = "✓" if ptr == expected_ptr else "✗ MISMATCH"
        print(f"  [{j:2d}]: packed=0x{ptr:x}, expected=0x{expected_ptr:x} {match}")

    # Now check what the kernel sees via tg_layer_weight
    # The kernel does: layer_weights[layer * TG_LW_COUNT + weight_idx]
    # where layer_weights is the packed pointer array.
    # Let's verify layer 0, gate_proj (index 8)
    print(f"\n=== Gate/Up weight verification ===")
    gate_w = layer_weights[8]  # [3072, 1024]
    up_w = layer_weights[9]    # [3072, 1024]
    down_w = layer_weights[10] # [1024, 3072]

    print(f"gate_w: shape={list(gate_w.shape)}, absmax={gate_w.abs().max().item():.6f}")
    print(f"up_w:   shape={list(up_w.shape)}, absmax={up_w.abs().max().item():.6f}")
    print(f"down_w: shape={list(down_w.shape)}, absmax={down_w.abs().max().item():.6f}")

    # Check tile offset: gate_w + tile_n * 16 * hidden
    # For tile_n=0: gate_w[0:16, :]
    # For tile_n=1: gate_w[16:32, :]
    print(f"\nGate weight tile check:")
    for tile_n in [0, 1, 2]:
        ptr_offset = tile_n * 16 * HIDDEN_SIZE
        tile_data = gate_w.flatten()[ptr_offset:ptr_offset + 16 * HIDDEN_SIZE]
        print(f"  tile_n={tile_n}: offset={ptr_offset}, "
              f"absmax={tile_data.abs().max().item():.6f}, "
              f"first 4 vals={tile_data[:4].tolist()}")

    # PyTorch reference: compute gate GEMM for layer 0
    print(f"\n=== PyTorch gate GEMM reference (layer 0) ===")
    x = embed_weight[42].float().cuda()  # [hidden]
    input_ln_w = layer_weights[0].float().cuda()
    rms = x.pow(2).mean().sqrt() + 1e-6
    x_normed = (x / rms) * input_ln_w
    print(f"  RMSNorm output absmax: {x_normed.abs().max().item():.6f}")

    # QKV
    q_weight = layer_weights[1].float().cuda()
    q_out = x_normed @ q_weight.T
    print(f"  Q output absmax: {q_out.abs().max().item():.6f}")

    # Attention (first token = V)
    v_weight = layer_weights[3].float().cuda()
    v_out = x_normed @ v_weight.T  # [kv_size]
    v_out = v_out.view(NUM_KV_HEADS, HEAD_DIM)
    attn_out = torch.zeros(Q_SIZE, device="cuda", dtype=torch.float32)
    for qh in range(NUM_Q_HEADS):
        kv_head = qh // (NUM_Q_HEADS // NUM_KV_HEADS)
        attn_out[qh * HEAD_DIM:(qh + 1) * HEAD_DIM] = v_out[kv_head]

    # O_PROJ
    o_proj_w = layer_weights[6].float().cuda()
    o_proj_out = attn_out @ o_proj_w.T
    print(f"  O_PROJ output absmax: {o_proj_out.abs().max().item():.6f}")

    # Residual add
    x_attn = x + o_proj_out
    print(f"  x_attn absmax: {x_attn.abs().max().item():.6f}")

    # Post-attn RMSNorm
    post_ln_w = layer_weights[7].float().cuda()
    rms2 = x_attn.pow(2).mean().sqrt() + 1e-6
    x_normed2 = (x_attn / rms2) * post_ln_w
    print(f"  Post-attn RMSNorm output absmax: {x_normed2.abs().max().item():.6f}")

    # Gate GEMM
    gate_out = x_normed2 @ gate_w.T  # [intermediate]
    print(f"  Gate GEMM output absmax: {gate_out.abs().max().item():.6f}")

    # Up GEMM
    up_out = x_normed2 @ up_w.T  # [intermediate]
    print(f"  Up GEMM output absmax: {up_out.abs().max().item():.6f}")

    # SiLU
    silu_out = (gate_out * torch.sigmoid(gate_out)) * up_out
    print(f"  SiLU output absmax: {silu_out.abs().max().item():.6f}")

    # Down
    ffn_down = silu_out @ down_w.T
    print(f"  FFN down output absmax: {ffn_down.abs().max().item():.6f}")

    # Expected: task graph gives silu = 3.09e+05, PyTorch gives silu = ?
    # If PyTorch silu is ~4, the task graph is ~77000x too large

    # Now let's check: what if the RMSNorm is NOT working in the kernel?
    # If s_norm = x_attn (unnormalized, absmax ~3), then:
    # gate_out = x_attn @ gate_w.T ≈ 3 * 0.01 * sqrt(1024) ≈ 1
    # That's not 77000x too large.

    # What if s_norm has the wrong data? Like, what if it's reading from
    # the wrong memory location?
    # Let's check: what would the GEMM output be if the input was
    # the raw weights instead of the normalized data?
    print(f"\n=== Hypothesis: what if s_norm contains weight data? ===")
    # If s_norm = gate_w[0, :1024] (a row of the weight matrix)
    fake_input = gate_w[0].float().cuda()  # [hidden]
    fake_gate_out = fake_input @ gate_w.T
    print(f"  If s_norm = gate_w[0]: gate_out absmax = {fake_gate_out.abs().max().item():.6f}")

    # What if s_norm = o_proj_out (wrong buffer)?
    fake_input2 = o_proj_out  # [hidden]
    fake_gate_out2 = fake_input2 @ gate_w.T
    print(f"  If s_norm = o_proj_out: gate_out absmax = {fake_gate_out2.abs().max().item():.6f}")

    # What if the GEMM is reading the weight as the input and vice versa?
    # C = W @ s_norm^T instead of C = s_norm @ W^T
    # This would give C[j] = sum_k W[j][k] * s_norm[k], same as correct.
    # So transposing input/weight wouldn't change the result.

    # What if the GEMM is doing C = W @ W^T (reading weight for both A and B)?
    fake_gate_out3 = gate_w.float() @ gate_w.T.float()
    print(f"  If C = W @ W^T: absmax = {fake_gate_out3.abs().max().item():.6f}")

    # What if K is wrong? K = 3072 instead of 1024?
    # That would mean reading beyond s_norm into other shared memory.
    # Let's check what's in shared memory after s_norm:
    # s_norm[1024], s_reduce[7], s_max[7], s_sum[7], s_out_acc[7*128=896]
    # Total after s_norm: 7 + 7 + 7 + 896 = 917 floats
    # If K = 3072, the GEMM reads s_norm[0..3067], which extends into
    # s_reduce, s_max, s_sum, s_out_acc.
    # s_out_acc contains attention accumulator values (~1.0).
    # This would add ~2048 extra terms of ~1.0 * weight ≈ 0.01,
    # adding ~2048*0.01 = ~20 to the output. Not 77000x.

    # What if the weight stride is wrong?
    # Weight is accessed as W + n0 * K + k, with K = g.hidden = 1024
    # If K is actually 3072 (INTERMEDIATE_SIZE), the weight load would
    # be at wrong offsets, potentially reading garbage.
    # Let's check: W + n0 * 3072 + k instead of W + n0 * 1024 + k
    # For n0=0, k=0: same (W+0). But for n0=1, k=0: W+3072 vs W+1024.
    # W+3072 would be gate_w[3, 0] (3rd row, element 0) instead of
    # gate_w[1, 0] (1st row, element 0).
    # This would scramble the weight matrix but not amplify by 77000x.

    print(f"\n=== Check WMMA store: does it write MORE than 16x16? ===")
    # wmma::store_matrix_sync writes a full 16x16 tile.
    # For M=1, only row 0 has valid data. Rows 1-15 are zeros (from the
    # zero-padded A fragment).
    # But the STORE writes all 16 rows. If the output buffer is not
    # padded, this would overwrite other data.
    # The output is silu_out + tile_n * 16, with stride 3072.
    # So the store writes to silu_out[0..15, tile_n*16..tile_n*16+15].
    # For tile_n=0: silu_out[0..15, 0..15]. Rows 1-15 are zeros. ✓
    # But then Phase 3 (SiLU) reads silu_out[i] for i = 0..BS*3072-1.
    # For BS=1: i = 0..3071. This reads silu_out[0, 0..3071].
    # But the GEMM only wrote to silu_out[0, 0..15] for tile_n=0.
    # The other tiles (tile_n=1..191) write to silu_out[0, 16..3071].
    # So silu_out[0, 0..3071] should be fully written by all tiles. ✓

    # But wait — the GEMM also writes to rows 1-15 (zeros).
    # For tile_n=0: silu_out[1, 0..15] = 0
    # For tile_n=1: silu_out[1, 16..31] = 0
    # So silu_out[1, 0..3071] is all zeros.
    # Phase 3 reads silu_out[i] for i = 0..3071, which is silu_out[0, 0..3071].
    # This is correct — it reads row 0 only. ✓

    print(f"\nDone.")


if __name__ == "__main__":
    main()
