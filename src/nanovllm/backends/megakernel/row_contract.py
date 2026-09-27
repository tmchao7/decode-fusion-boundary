"""The host <-> kernel contract for per-row activation.

``row_active`` is a **correctness contract, not a performance knob**. The kernel
gates the paged-KV write on it:

    // csrc/taskgraph_kernel.cu, Phase A
    const bool row_on = (g.row_active[batch] != 0);
    if (wwarp == 0 && row_on) { ...write K/V... }

The reason is continuous batching: once a sequence finishes, the scheduler frees
its pages and may hand them to a new sequence, so a stale row writing "its" old
pages would corrupt the new owner. Everything *else* an inactive row computes is
row-private scratch whose output the host ignores, and every bar still arrives, so
the wait targets stay valid whether a row is on or off. That last property is what
keeps the schedule independent of the mask -- the same instruction stream, the
same arrivals, one masked store.

Today the runtime launches exactly the live rows (one cached geometry per batch
size), so the launcher always activates every row. ``active_rows`` exists for a
fixed-slot runtime, and it is validated here rather than trusted: a mask of the
wrong length, or one that activates nothing, is a launch bug that would otherwise
surface as corrupted KV on the GPU.

This module is torch-free so the contract is testable without a GPU.
"""

from __future__ import annotations

from typing import Iterable


def row_active_mask(active_rows: Iterable[object] | None, batch_size: int) -> list[int]:
    """Normalize the per-row activation mask for a launch of ``batch_size`` rows.

    ``None`` means "every row active", which is what the current runtime always
    passes. Any other iterable must have exactly ``batch_size`` entries and
    activate at least one row.
    """

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1; got {batch_size}")
    if active_rows is None:
        return [1] * batch_size

    mask = [1 if value else 0 for value in active_rows]
    if len(mask) != batch_size:
        raise ValueError(
            f"active_rows has {len(mask)} entries for a {batch_size}-row launch"
        )
    if not any(mask):
        raise ValueError("active_rows activates no row; there is nothing to launch")
    return mask
