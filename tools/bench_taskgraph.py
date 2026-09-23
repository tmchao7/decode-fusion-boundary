"""Task-graph megakernel decode-step benchmark.

Times one decode step (the whole 28-layer forward + LM head) for a given
batch size, context length and grid size, and reports ms/step plus the
effective weight bandwidth. Decode is weight-bandwidth bound: ~1.2 GB of
weights are streamed per step regardless of batch size, so GB/s is the number
to watch.

The KV cache is pre-filled to ``--context-len`` so the attention cost reflects
a realistic context. Positions beyond ``context_len`` are not exercised.

Usage:
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
    python tools/bench_taskgraph.py --batch-size 1 --context-len 512
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DEFAULT_MODEL = os.environ.get(
    "NANOVLLM_MODEL_PATH", os.path.expanduser("~/models/Qwen3-0.6B")
)
VOCAB = 151936
NUM_Q_HEADS = 16


def parse_batches(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=parse_batches, default=[1])
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--num-sms", type=int, default=0,
                        help="grid size; 0 = the device's SM count")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args(argv)

    import json

    import torch

    if args.num_sms <= 0:
        args.num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    from nanovllm.backends.megakernel.build import get_extension
    get_extension()

    from nanovllm.backends.megakernel.model import (
        load_weights, _pack_layer_weights, _rope_tables,
        NUM_LAYERS, HIDDEN_SIZE, INTERMEDIATE_SIZE,
        NUM_KV_HEADS, HEAD_DIM, Q_SIZE, KV_SIZE,
    )
    from nanovllm.backends.megakernel import schedule as tg_schedule

    _tg_ops = torch.ops.nanovllm_megakernel_C
    weights, _ = load_weights(args.model, verbose=False)
    cos_table, sin_table = _rope_tables()
    layer_weights_packed = _pack_layer_weights(weights["layer_weights"])
    embed_weight = weights["embed_weight"]
    final_norm_weight = weights["final_norm_weight"]

    block_size = args.block_size
    # The kernel writes the new token at position cache_seqlens, so with
    # cache_seqlens = context_len - 1 the highest logical block is
    # (context_len-1)//block_size and max_pages of them are enough.
    max_pages = max(1, args.context_len // block_size)
    ctx_position = max(0, args.context_len - 1)

    results = []
    for BS in args.batch_size:
        num_cache_blocks = max_pages * BS + 1
        kv_layer_stride = num_cache_blocks * block_size * KV_SIZE
        k_cache = torch.zeros(
            NUM_LAYERS, num_cache_blocks, block_size, NUM_KV_HEADS, HEAD_DIM,
            dtype=torch.bfloat16, device="cuda",
        )
        v_cache = torch.zeros_like(k_cache)

        geo = tg_schedule.default_geometry(BS=BS, attn_splits=1)
        sched = tg_schedule.build_schedule(geo, num_sms=args.num_sms)

        f32 = dict(dtype=torch.float32, device="cuda")
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        u32 = dict(dtype=torch.uint32, device="cuda")
        WMMA_PAD = 16
        s = {
            "inst_buf": torch.tensor(sched.flat_inst_buf, dtype=torch.uint32).cuda().view(torch.int32),
            "num_inst_per_sm": torch.tensor(sched.num_inst_per_sm, dtype=torch.int32).cuda(),
            "norm_out": torch.empty(BS, HIDDEN_SIZE, **f32),
            "q_out": torch.empty(WMMA_PAD, Q_SIZE, **f32),
            "k_out": torch.empty(WMMA_PAD, KV_SIZE, **f32),
            "v_out": torch.empty(WMMA_PAD, KV_SIZE, **f32),
            "attn_out": torch.empty(BS, Q_SIZE, **f32),
            "o_proj_out": torch.empty(WMMA_PAD, HIDDEN_SIZE, **f32),
            "x_attn": torch.empty(BS, HIDDEN_SIZE, **f32),
            "post_attn_norm": torch.empty(BS, HIDDEN_SIZE, **f32),
            "silu_out": torch.empty(WMMA_PAD, INTERMEDIATE_SIZE, **f32),
            "ffn_down_out": torch.empty(WMMA_PAD, HIDDEN_SIZE, **f32),
            "up_out": torch.empty(WMMA_PAD, INTERMEDIATE_SIZE, **f32),
            "x_ffn": torch.empty(BS, HIDDEN_SIZE, **f32),
            "x_resid": torch.empty(BS, HIDDEN_SIZE, **bf16),
            "lm_logits": torch.empty(WMMA_PAD, VOCAB, **f32),
            "o_partial": torch.empty(BS, NUM_KV_HEADS, 1, HIDDEN_SIZE, **f32),
            "lse_partial": torch.empty(BS, NUM_KV_HEADS, 1, **f32),
            "page_table": torch.zeros(BS, max_pages, dtype=torch.int32, device="cuda"),
            "cache_seqlens": torch.full((BS,), ctx_position, dtype=torch.int32, device="cuda"),
            "row_active": torch.ones(BS, dtype=torch.int32, device="cuda"),
        }
        d = sched.bar_dims
        for name, size in (
            ("bar_qkv", d.bar_qkv), ("bar_attn", d.bar_attn),
            ("bar_oproj", d.bar_oproj), ("bar_silu", d.bar_silu),
            ("bar_ffn_down", d.bar_ffn_down), ("bar_layer", d.bar_layer),
            ("bar_combine", max(d.bar_combine, 1)), ("bar_lm_head", d.bar_lm_head),
        ):
            s[name] = torch.zeros(size, **u32)
        # Each batch row gets its own physical page range.
        for b in range(BS):
            for p in range(max_pages):
                s["page_table"][b, p] = (b * max_pages + p) % num_cache_blocks
        for b in range(BS):
            s["x_resid"][b] = embed_weight[42 + b]

        args_scratch = (
            s["norm_out"], s["q_out"], s["k_out"], s["v_out"], s["attn_out"],
            s["o_proj_out"], s["x_attn"], s["post_attn_norm"], s["silu_out"],
            s["ffn_down_out"], s["up_out"], s["x_ffn"], s["x_resid"],
            s["lm_logits"], s["o_partial"], s["lse_partial"],
        )
        bars = (s["bar_qkv"], s["bar_attn"], s["bar_oproj"], s["bar_silu"],
                s["bar_ffn_down"], s["bar_layer"], s["bar_combine"],
                s["bar_lm_head"])
        # One zero-regions launch per step (the intended runtime path), not one
        # .zero_() per buffer. The KV cache and the inputs are excluded.
        # Keep the descriptor on the HOST: the binding copies it to a cached
        # device buffer, and a CUDA tensor would force a sync D2H copy.
        regions = torch.tensor(
            [(t.data_ptr(), t.numel() * t.element_size())
             for t in (*args_scratch, *bars)],
            dtype=torch.int64,
        )

        def step():
            _tg_ops.tg_zero_regions(regions, regions.shape[0])
            _tg_ops.tg_decode_step(
                s["inst_buf"], s["num_inst_per_sm"], sched.max_inst, args.num_sms,
                layer_weights_packed, embed_weight, final_norm_weight,
                cos_table, sin_table, k_cache, v_cache, s["page_table"],
                s["cache_seqlens"], s["row_active"],
                num_cache_blocks, max_pages, block_size, kv_layer_stride,
                *args_scratch, *bars,
                NUM_LAYERS, BS, HIDDEN_SIZE, INTERMEDIATE_SIZE,
                Q_SIZE, KV_SIZE, NUM_Q_HEADS, NUM_KV_HEADS,
                HEAD_DIM, VOCAB, 1, 1, 1,
            )

        for _ in range(args.warmup):
            step()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.num_steps):
            step()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / args.num_steps

        # Every weight is read once per step: ~0.6B params * 2 bytes = 1.2 GB.
        weight_gb = 1.2
        bw = weight_gb / (ms / 1000.0)
        results.append({"batch_size": BS, "ms_per_step": ms, "weight_gbps": bw})
        print(f"BS={BS} num_sms={args.num_sms} ctx={args.context_len}: "
              f"{ms:.3f} ms/step  ({bw:.0f} GB/s effective weight bw)")

        del s, k_cache, v_cache
        torch.cuda.empty_cache()

    if args.json_path:
        with open(args.json_path, "w") as fh:
            json.dump(results, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
