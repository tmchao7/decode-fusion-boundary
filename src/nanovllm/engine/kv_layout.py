"""Position and paged-cache slot contract for prefill and decode.

This module is the single source of truth for the three numbers that used to be
duplicated (and drifted) between ``ModelRunner.prepare_prefill``,
``ModelRunner.prepare_decode``, ``TaskGraphDecoder`` and the CUDA kernel:

* **positions** — the RoPE position of every token in a step;
* **cache length** — how many KV entries a step writes and attends over;
* **slot mapping** — which flat paged-cache element a token's K/V goes to.

The contract is deliberately small and torch-free so it can be locked by
CPU-only tests (``tests/test_kv_block_boundary.py``) before any GPU exists.

Locked identities
-----------------

Prefill of a fresh prompt of ``L`` tokens::

    positions        = [0, L)                      # prefill_positions(0, L)
    cache length     = L                           # prefill_cache_len(0, L)
    slot_mapping     = one entry per token

Decode step ``k`` (0-based) for a prompt of ``L`` tokens::

    num_tokens       = L + k + 1                   # the sampled token was appended
    position         = num_tokens - 1 = L + k      # decode_position(num_tokens)
    cache_len        = position + 1                # decode_cache_len(num_tokens)
    pre-decode len   = position                    # decode_cache_seqlens(num_tokens)

``cache_len`` is the number of KV entries the kernel loops over, hence
``cache_len = position + 1``: the new token's K/V is written at ``position``
*before* attention reads the cache, so attention must include it.

The CUDA kernel agrees with the identity above (``taskgraph_kernel.cu``)::

    int cache_len = g.cache_seqlens[batch] + 1;   // +1 for the new token
    int position  = cache_len - 1;                // == g.cache_seqlens[batch]
    pos_end       = cache_len;                    // attention loop bound

so the host passes the **pre-decode** length (``position``) as
``cache_seqlens``, while the torch path passes ``len(seq)`` as
``context_lens`` (which is already ``position + 1``). Both are the same
contract expressed for two consumers; do not "unify" them without changing
both sides.

The two boundary cases these helpers exist to protect:

* a prompt of exactly ``block_size`` tokens (256) ends on a block boundary, so
  the *next* decode position (256) needs a newly appended block;
* a prompt of ``block_size + 1`` (257) tokens already spans two blocks during
  prefill, and its slot mapping must cross the boundary contiguously.
"""

from __future__ import annotations

from typing import Iterable, Sequence


def prefill_positions(num_cached_tokens: int, num_scheduled_tokens: int) -> range:
    """RoPE positions for one prefill chunk: ``[cached, cached + scheduled)``.

    A prefix-cache hit starts the range at the first uncached token, so the
    chunk's positions continue the cached prefix instead of restarting at 0.
    """

    if num_cached_tokens < 0:
        raise ValueError("num_cached_tokens must be non-negative")
    if num_scheduled_tokens < 0:
        raise ValueError("num_scheduled_tokens must be non-negative")
    return range(num_cached_tokens, num_cached_tokens + num_scheduled_tokens)


def prefill_cache_len(num_cached_tokens: int, num_scheduled_tokens: int) -> int:
    """KV length visible to a prefill chunk (``cu_seqlens_k`` delta).

    For a chunked prefill this is the *end* position, not the whole prompt:
    FlashAttention varlen attends over the cached prefix plus this chunk.
    """

    return num_cached_tokens + num_scheduled_tokens


def prefill_slot_mapping(
    block_table: Sequence[int],
    start: int,
    end: int,
    block_size: int,
) -> list[int]:
    """Flat paged-cache slots for the prefill tokens in ``[start, end)``.

    Consecutive positions inside one physical block map to consecutive slots;
    the mapping is contiguous across a block boundary by construction (the loop
    walks one logical block at a time and each block contributes a full row of
    ``block_size`` slots, trimmed at both ends).
    """

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if start < 0 or end < start:
        raise ValueError(f"invalid prefill range [{start}, {end})")
    if end == start:
        return []
    if not block_table:
        raise RuntimeError(
            "prefill slot mapping requires an allocated block table; the "
            "scheduler must call BlockManager.allocate() before the model runs"
        )
    start_block = start // block_size
    end_block = (end + block_size - 1) // block_size
    if len(block_table) < end_block:
        raise RuntimeError(
            f"prefill tokens [{start}, {end}) need {end_block} blocks but the "
            f"sequence only owns {len(block_table)}"
        )
    slots: list[int] = []
    for i in range(start_block, end_block):
        slot_start = block_table[i] * block_size
        if i == start_block:
            slot_start += start % block_size
        if i != end_block - 1:
            slot_end = block_table[i] * block_size + block_size
        else:
            slot_end = block_table[i] * block_size + end - i * block_size
        slots.extend(range(slot_start, slot_end))
    return slots


def decode_position(num_tokens: int) -> int:
    """RoPE position of the token a decode step is about to compute."""

    if num_tokens < 1:
        raise ValueError(f"num_tokens must be at least 1; got {num_tokens}")
    return num_tokens - 1


def decode_cache_seqlens(num_tokens: int) -> int:
    """Pre-decode KV length: the value the task-graph kernel reads as input."""

    return decode_position(num_tokens)


def decode_cache_len(num_tokens: int) -> int:
    """KV length after the step: ``position + 1``, including the new token."""

    return num_tokens


def decode_slot_mapping(
    block_table: Sequence[int],
    num_tokens: int,
    block_size: int,
) -> int:
    """Flat paged-cache slot for the single token a decode step writes.

    ``num_tokens`` is the sequence length *after* the sampled token has been
    appended, so its slot is the last position of the last block.
    """

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if num_tokens < 1:
        raise ValueError(f"num_tokens must be at least 1; got {num_tokens}")
    num_blocks = (num_tokens + block_size - 1) // block_size
    if len(block_table) != num_blocks:
        # A shorter table indexes before the cache (flash-attn would read
        # garbage); a longer one means a block was never freed. Either way the
        # K/V write would land in the wrong physical block and corrupt another
        # sequence's cache, so fail instead of writing.
        raise RuntimeError(
            f"decode of {num_tokens} tokens needs {num_blocks} blocks but the "
            f"block table has {len(block_table)}; BlockManager.may_append() "
            "must run before the model call"
        )
    last_block_num_tokens = num_tokens - (num_blocks - 1) * block_size
    return block_table[num_blocks - 1] * block_size + last_block_num_tokens - 1


def pad_block_table(block_table: Iterable[int], max_pages: int) -> list[int]:
    """Right-pad a host block table to ``max_pages`` for a rectangular batch.

    Padding value is 0 (not -1): the megakernel addresses KV slots as
    ``block_table[position // block_size] * ...``, so a negative padding value
    would index before the cache allocation.
    """

    table = list(block_table)
    if len(table) > max_pages:
        raise ValueError(
            f"block table has {len(table)} pages but the stride is {max_pages}"
        )
    return table + [0] * (max_pages - len(table))
