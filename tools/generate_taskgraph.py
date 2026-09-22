"""Greedy text generation with the task-graph megakernel.

Runs the persistent task-graph kernel end to end on real prompts: teacher-force
each prompt to fill the paged KV cache, then feed each sampled token back as
the next step's input (the same feedback loop tests/test_taskgraph_positions.py
checks against a reference). Prints the decoded text so the output can be
eyeballed, which the numeric tests cannot do.

Rows may have different prompt lengths; each row advances one position per
step and the kernel tracks positions per row via cache_seqlens. Rows that
finish early keep re-feeding their last token (row_active is not implemented
yet, so the kernel still computes them; their output is ignored).

Usage:
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
    python tools/generate_taskgraph.py --prompt "The capital of France is" \
        --max-tokens 32
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DEFAULT_MODEL = os.environ.get(
    "NANOVLLM_MODEL_PATH", os.path.expanduser("~/models/Qwen3-0.6B")
)
VOCAB = 151936
NUM_Q_HEADS = 16


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", action="append", default=None,
                        help="repeatable; one decode row per prompt")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--num-sms", type=int, default=0,
                        help="grid size; 0 = the device's SM count")
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args(argv)

    prompts = args.prompt or ["The capital of France is"]
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")

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
    weights, tokenizer = load_weights(args.model, verbose=False)
    cos_table, sin_table = _rope_tables()
    layer_weights_packed = _pack_layer_weights(weights["layer_weights"])
    embed_weight = weights["embed_weight"]
    final_norm_weight = weights["final_norm_weight"]

    # Qwen3-0.6B has no pad token; tokenize each prompt on its own.
    prompts_ids = [
        tokenizer.encode(p, add_special_tokens=False) or [tokenizer.eos_token_id]
        for p in prompts
    ]
    BS = len(prompts_ids)
    prompt_lens = [len(x) for x in prompts_ids]
    total_steps = max(prompt_lens) + args.max_tokens

    block_size = args.block_size
    max_pages = max(1, (total_steps + block_size - 1) // block_size)
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
        "inst_buf": torch.tensor(sched.flat_inst_buf, dtype=torch.int32).cuda(),
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
        "cache_seqlens": torch.zeros(BS, dtype=torch.int32, device="cuda"),
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
    # Row b owns physical pages [b*max_pages, (b+1)*max_pages).
    for b in range(BS):
        for p in range(max_pages):
            s["page_table"][b, p] = b * max_pages + p

    args_scratch = (
        s["norm_out"], s["q_out"], s["k_out"], s["v_out"], s["attn_out"],
        s["o_proj_out"], s["x_attn"], s["post_attn_norm"], s["silu_out"],
        s["ffn_down_out"], s["up_out"], s["x_ffn"], s["x_resid"],
        s["lm_logits"], s["o_partial"], s["lse_partial"],
    )
    bars = (s["bar_qkv"], s["bar_attn"], s["bar_oproj"], s["bar_silu"],
            s["bar_ffn_down"], s["bar_layer"], s["bar_combine"],
            s["bar_lm_head"])
    regions = torch.tensor(
        [(t.data_ptr(), t.numel() * t.element_size())
         for t in (*args_scratch, *bars)],
        dtype=torch.int64,
    )

    def step(input_tokens, positions):
        _tg_ops.tg_zero_regions(regions, regions.shape[0])
        for b in range(BS):
            s["x_resid"][b] = embed_weight[input_tokens[b]]
            s["cache_seqlens"][b] = positions[b]
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
        torch.cuda.synchronize()
        return [s["lm_logits"][b, 0].view(torch.int32).item() for b in range(BS)]

    # Per-row state: next input token, current position, and generated tokens.
    next_tok = [prompts_ids[b][0] for b in range(BS)]
    pos = [0] * BS
    generated = [[] for _ in range(BS)]
    done = [False] * BS

    for _ in range(total_steps):
        out = step(next_tok, pos)
        for b in range(BS):
            if done[b]:
                continue
            p = pos[b]
            if p + 1 < prompt_lens[b]:
                # Still teacher-forcing this row's prompt.
                next_tok[b] = prompts_ids[b][p + 1]
            else:
                generated[b].append(out[b])
                next_tok[b] = out[b]
                if len(generated[b]) >= args.max_tokens:
                    done[b] = True
                    continue
            pos[b] = p + 1
        if all(done):
            break

    for b, prompt in enumerate(prompts):
        text = tokenizer.decode(generated[b], skip_special_tokens=True)
        print(f"\n=== row {b} (len(prompt)={prompt_lens[b]}) ===")
        print(f"prompt: {prompt}")
        print(f"output: {prompt}{text}")
        print(f"tokens: {generated[b]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
