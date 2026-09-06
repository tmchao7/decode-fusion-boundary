# Qwen Megakernel Design Notes

## Current integration boundary

The current backend is intentionally conservative:

- prefill stays on the standard nano-vLLM PyTorch path;
- Megakernel is limited to Qwen3-0.6B, BF16, greedy decode, one sequence, and `max_model_len <= 2048`;
- the first decode path owns a private contiguous KV cache and replays the prompt once;
- unsupported requests or V100 execution fall back to the standard backend.

This is a correctness and launch-overhead baseline, not yet the final paged-KV implementation.

## Target decode interface

The current development path receives the scheduler's existing metadata
through the single-sequence `decode_paged` operator; the eventual batch API
will use the same contract directly:

```text
decode_batch(
    input_ids, positions, context_lens, block_tables,
    kv_cache, model_weights, output_token_ids
)
```

The cache contract is implemented and unit-tested in
`src/nanovllm/backends/megakernel/cache.py`. The kernel must write and read the
nano-vLLM cache without a temporary contiguous copy. The cache is laid out as:

```text
[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
```

For sequence `s` and logical position `p`:

```text
block_id     = block_tables[s, p // block_size]
block_offset = p % block_size
physical_slot = block_id * block_size + block_offset
```

The per-layer K/V address then indexes `physical_slot`, KV head, and head dimension. GQA must map each Q head to `q_head // (num_q_heads / num_kv_heads)`.

## Fusion boundaries

The intended decode pipeline is:

1. embedding lookup;
2. each transformer layer: RMSNorm, QKV, QK-Norm, RoPE, paged attention, output projection, residual, MLP;
3. final RMSNorm;
4. fused vocabulary reduction and argmax.

The first correctness version may keep the LM head as a second kernel. Any kernel count claim must report the actual number observed in Nsight, not an assumed count.

## Validation milestones

1. Single sequence, contiguous reference KV cache, exact greedy token match.
2. Single sequence, direct paged KV read/write, exact token match on the server.
3. Batch decode with block tables and per-sequence context lengths.
4. Sampling/logits support only after greedy correctness and profiling are stable.

A100 (`sm_80`) is the primary performance target. V100 (`sm_70`) is currently a compatibility/fallback target until a validated FP16 kernel path exists.
