"""Isolate RoPE: compare megakernel cos/sin tables and formula against torch."""
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

HEAD_DIM = 128
ROPE_THETA = 1_000_000.0


def torch_rope_tables():
    """Build cos/sin exactly like nanovllm/layers/rotary_embedding.py."""
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float) / HEAD_DIM))
    max_pos = 2048
    t = torch.arange(max_pos, dtype=torch.float)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)  # [max_pos, 64]
    cos = freqs.cos()  # [max_pos, 64]
    sin = freqs.sin()  # [max_pos, 64]
    return cos.cuda(), sin.cuda()


def mega_rope_tables():
    """Build cos/sin exactly like megakernel/model.py _rope_tables."""
    from nanovllm.backends.megakernel.build import get_extension
    get_extension()  # ensure JIT op is registered
    from nanovllm.backends.megakernel.model import _rope_tables
    cos_table, sin_table = _rope_tables(ROPE_THETA)
    # cos_table is [max_seq_len, 128] with repeat(1,2)
    # First 64 cols are the unique cos values
    return cos_table[:, :64], sin_table[:, :64]


def apply_rope_torch(x, cos, sin):
    """Apply RoPE like nanovllm/layers/rotary_embedding.py apply_rotary_emb."""
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1)


def main():
    torch_cos, torch_sin = torch_rope_tables()
    mega_cos, mega_sin = mega_rope_tables()

    print("=== Cos/Sin table comparison ===")
    cos_diff = (torch_cos - mega_cos.float()).abs()
    sin_diff = (torch_sin - mega_sin.float()).abs()
    print(f"cos: max diff = {cos_diff.max().item():.6e}, mean diff = {cos_diff.mean().item():.6e}")
    print(f"sin: max diff = {sin_diff.max().item():.6e}, mean diff = {sin_diff.mean().item():.6e}")
    print(f"cos[0:4, 0:4] torch:\n{torch_cos[:4, :4]}")
    print(f"cos[0:4, 0:4] mega:\n{mega_cos[:4, :4]}")
    print(f"sin[0:4, 0:4] torch:\n{torch_sin[:4, :4]}")
    print(f"sin[0:4, 0:4] mega:\n{mega_sin[:4, :4]}")

    # Check inv_freq values
    print("\n=== inv_freq ===")
    torch_inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float) / HEAD_DIM))
    mega_inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    print(f"torch inv_freq[0:4]: {torch_inv_freq[:4]}")
    print(f"mega  inv_freq[0:4]: {mega_inv_freq[:4]}")
    print(f"inv_freq max diff: {(torch_inv_freq - mega_inv_freq).abs().max().item():.6e}")

    # Now test RoPE application at various positions
    print("\n=== RoPE application test (single head) ===")
    torch.manual_seed(42)
    x = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device="cuda")

    for pos in [0, 1, 2, 7, 8, 15, 16, 31, 32]:
        # Torch reference: uses [64] cos/sin
        cos_p = torch_cos[pos]  # [64]
        sin_p = torch_sin[pos]  # [64]
        x1 = x.float()[:64]
        x2 = x.float()[64:]
        y_torch = torch.cat([x1 * cos_p - x2 * sin_p, x2 * cos_p + x1 * sin_p], dim=-1)

        # Kernel-style: uses [128] repeated cos/sin
        cos_full = torch.cat([mega_cos[pos], mega_cos[pos]], dim=-1)  # [128]
        sin_full = torch.cat([mega_sin[pos], mega_sin[pos]], dim=-1)  # [128]
        x_f = x.float()
        y_kernel = torch.empty(HEAD_DIM, dtype=torch.float32, device="cuda")
        for i in range(HEAD_DIM):
            cv = cos_full[i].item()
            sv = sin_full[i].item()
            if i < HEAD_DIM // 2:
                pv = x_f[i + HEAD_DIM // 2].item()
                y_kernel[i] = x_f[i] * cv - pv * sv
            else:
                pv = x_f[i - HEAD_DIM // 2].item()
                y_kernel[i] = pv * sv + x_f[i] * cv

        diff = (y_torch - y_kernel).abs().max().item()
        print(f"pos={pos:3d}: torch vs kernel-formula max diff = {diff:.6e}")

    # Check if there's a position offset
    print("\n=== Position offset search ===")
    x_flat = x[0, 0]  # [128]
    for pos in [8, 16]:
        cos_p = torch_cos[pos].unsqueeze(0)
        sin_p = torch_sin[pos].unsqueeze(0)
        y_correct = apply_rope_torch(x_flat.unsqueeze(0), cos_p, sin_p)

        best_offset = 0
        best_diff = float("inf")
        for offset in range(-5, 6):
            p = pos + offset
            if p < 0:
                continue
            cos_p2 = mega_cos[p].unsqueeze(0)
            sin_p2 = mega_sin[p].unsqueeze(0)
            y_test = apply_rope_torch(x_flat.unsqueeze(0), cos_p2, sin_p2)
            d = (y_correct.float() - y_test.float()).abs().max().item()
            if d < best_diff:
                best_diff = d
                best_offset = offset
        print(f"pos={pos}: best offset = {best_offset:+d}, diff = {best_diff:.6e}")


if __name__ == "__main__":
    main()
