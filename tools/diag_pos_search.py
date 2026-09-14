"""Search for which position's RoPE matches the mega cache K."""
from __future__ import annotations

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

HEAD_DIM = 128
NUM_KV_HEADS = 8
RMS_EPS = 1e-6
ROPE_THETA = 1_000_000.0


def main():
    from nanovllm import LLM
    from nanovllm.backends.megakernel.build import get_extension
    get_extension()
    from nanovllm.backends.megakernel.model import _rope_tables

    model_path = os.environ.get(
        "NANOVLLM_MODEL_PATH", os.path.expanduser("~/models/Qwen3-0.6B")
    )

    diag_dir = os.environ.get(
        "NANOVLLM_DIAG_DIR", os.path.expanduser("~/.cache/nanovllm/diag")
    )
    mega_data = torch.load(os.path.join(diag_dir, "layer0_megakernel.pt"), weights_only=True)
    torch_data = torch.load(os.path.join(diag_dir, "layer0_torch.pt"), weights_only=True)

    decode_input_token = torch_data["out"][0]  # 198
    k_cache_mega = mega_data["k"][8].cuda()  # [kv_heads, head_dim] at pos 8
    k_cache_torch = torch_data["k"][8].cuda()

    llm = LLM(model_path, decode_backend="torch", enforce_eager=True,
              max_num_seqs=1, max_model_len=2048)
    model = llm.model_runner.model
    qwen = model.model
    layer0 = qwen.layers[0]
    attn = layer0.self_attn

    embed_weight = qwen.embed_tokens.weight
    input_norm_weight = layer0.input_layernorm.weight
    qkv_weight = attn.qkv_proj.weight
    q_size = attn.q_size
    kv_size = attn.kv_size
    k_weight = qkv_weight[q_size:q_size+kv_size]
    k_norm_weight = attn.k_norm.weight

    # Compute K matvec
    hidden = embed_weight[decode_input_token:decode_input_token+1]
    x = hidden.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + RMS_EPS))
    x_normed = x.to(torch.bfloat16).mul_(input_norm_weight)
    k_matvec = torch.nn.functional.linear(x_normed, k_weight)  # [1, 1024] bf16
    k_reshaped = k_matvec.view(NUM_KV_HEADS, HEAD_DIM).float()  # [8, 128]

    # K RMSNorm (per-head)
    k_var = k_reshaped.pow(2).mean(dim=-1, keepdim=True)
    k_normed = k_reshaped * torch.rsqrt(k_var + RMS_EPS)
    k_normed = k_normed.to(torch.bfloat16).mul_(k_norm_weight).float()  # [8, 128]

    cos_table, sin_table = _rope_tables(ROPE_THETA)

    # Search positions 0-20 to find best match
    print("=== Position search: which position's RoPE matches mega cache K? ===")
    for test_pos in range(0, 21):
        cos_64 = cos_table[test_pos, :64].float()
        sin_64 = sin_table[test_pos, :64].float()
        k_x1 = k_normed[:, :64]
        k_x2 = k_normed[:, 64:]
        k_y1 = k_x1 * cos_64 - k_x2 * sin_64
        k_y2 = k_x2 * cos_64 + k_x1 * sin_64
        k_rope = torch.cat([k_y1, k_y2], dim=-1).to(torch.bfloat16)

        d_mega = (k_rope.float() - k_cache_mega.float()).abs().max().item()
        d_torch = (k_rope.float() - k_cache_torch.float()).abs().max().item()
        marker = " <==" if d_mega < 0.01 else ""
        print(f"  pos={test_pos:3d}: vs_mega={d_mega:.6f}  vs_torch={d_torch:.6f}{marker}")

    # Also check: what if the K matvec output is wrong?
    # Try using the torch cache K (pre-RoPE) to see if the RMSNorm input is different
    # We can't directly get the pre-RoPE K from the cache, but we can check
    # if the mega K at pos 8 matches ANY position with ANY input

    # Check: is the mega K at pos 8 consistent with ANY position's RoPE
    # applied to the torch K at pos 8 (un-rope the torch K, re-rope with different pos)?
    # Un-RoPE is: x1 = y1*cos + y2*sin, x2 = y2*cos - y1*sin
    k_torch_f = k_cache_torch.float()  # [8, 128]
    k_torch_x1 = k_torch_f[:, :64]
    k_torch_x2 = k_torch_f[:, 64:]
    # Un-rope with pos 8
    cos8 = cos_table[8, :64].float()
    sin8 = sin_table[8, :64].float()
    unrope_x1 = k_torch_x1 * cos8 + k_torch_x2 * sin8  # = original x1
    unrope_x2 = k_torch_x2 * cos8 - k_torch_x1 * sin8  # = original x2
    k_unrotated = torch.cat([unrope_x1, unrope_x2], dim=-1)  # [8, 128] pre-RoPE

    print("\n=== Search: re-apply RoPE with different positions to torch's pre-RoPE K ===")
    for test_pos in range(0, 21):
        cos_64 = cos_table[test_pos, :64].float()
        sin_64 = sin_table[test_pos, :64].float()
        x1 = k_unrotated[:, :64]
        x2 = k_unrotated[:, 64:]
        k_y1 = x1 * cos_64 - x2 * sin_64
        k_y2 = x2 * cos_64 + x1 * sin_64
        k_reroped = torch.cat([k_y1, k_y2], dim=-1).to(torch.bfloat16)

        d_mega = (k_reroped.float() - k_cache_mega.float()).abs().max().item()
        marker = " <==" if d_mega < 0.01 else ""
        print(f"  pos={test_pos:3d}: vs_mega={d_mega:.6f}{marker}")

    # Check: is the mega K using the WRONG cos/sin table (e.g., rope_theta=10000)?
    print("\n=== Check: wrong rope_theta (10000 instead of 1e6)? ===")
    inv_freq_10k = 1.0 / (10000.0 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    positions = torch.arange(2048, dtype=torch.float32)
    freqs_10k = torch.outer(positions, inv_freq_10k)
    cos_10k = torch.cos(freqs_10k).repeat(1, 2).to(torch.bfloat16).cuda()
    sin_10k = torch.sin(freqs_10k).repeat(1, 2).to(torch.bfloat16).cuda()

    for test_pos in [8]:
        cos_64 = cos_10k[test_pos, :64].float()
        sin_64 = sin_10k[test_pos, :64].float()
        k_x1 = k_normed[:, :64]
        k_x2 = k_normed[:, 64:]
        k_y1 = k_x1 * cos_64 - k_x2 * sin_64
        k_y2 = k_x2 * cos_64 + k_x1 * sin_64
        k_rope_10k = torch.cat([k_y1, k_y2], dim=-1).to(torch.bfloat16)

        d_mega = (k_rope_10k.float() - k_cache_mega.float()).abs().max().item()
        print(f"  pos={test_pos}: theta=10000 vs_mega={d_mega:.6f}")

    llm.exit()


if __name__ == "__main__":
    main()
