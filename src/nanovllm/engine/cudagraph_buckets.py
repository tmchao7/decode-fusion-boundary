"""CUDA-graph batch buckets, and the padding a replay implies.

``ModelRunner.capture_cudagraph`` captures one fixed-shape graph per batch bucket
and replays the smallest bucket ``>=`` the live batch. When ``max_num_seqs``
exceeds the batch actually running, the replay executes the whole bucket while
only the real rows are read back, so the measured per-token time carries that
padding.

This module is the single source of truth for the bucket list so the runner and
the benchmarks cannot drift apart, and it makes the padding factor explicit
instead of something a reader has to re-derive from the code:

* the current benchmark protocol sets ``max_num_seqs == batch_size``
  (``benchmarks/compare.py``), which makes ``batch_size`` itself a captured
  bucket -- cudagraph is therefore measured with **no** padding;
* the archived crossover scan used ``max_num_seqs = max(batch_size, 32)``, so its
  batch=3/5/7 rows replayed buckets 4/8/8 and batch=10 replayed 16: up to 1.6x
  more GEMM work, and up to 1.88x for batch=17..31 (bucket 32). Those rows are
  penalised against the megakernel.

Both facts are locked by ``tests/test_cudagraph_buckets.py``.
"""

from __future__ import annotations

GRAPH_BATCH_CAP = 512


def graph_buckets(max_num_seqs: int) -> list[int]:
    """Batch buckets a runner with this ``max_num_seqs`` captures.

    Mirrors ``ModelRunner.capture_cudagraph``: ``[1, 2, 4, 8]`` plus every
    multiple of 16 up to ``min(max_num_seqs, 512)``, and ``max_bs`` itself is
    appended when it is not already a bucket. That last rule is what gives a
    non-bucket batch (e.g. 10) an exact graph when ``max_num_seqs`` equals it.
    """

    if max_num_seqs < 1:
        raise ValueError(f"max_num_seqs must be >= 1; got {max_num_seqs}")
    max_bs = min(max_num_seqs, GRAPH_BATCH_CAP)
    buckets = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    if max_bs > buckets[-1]:
        buckets.append(max_bs)
    return buckets


def select_graph_bucket(batch_size: int, buckets: list[int]) -> int | None:
    """Smallest captured bucket ``>= batch_size``, or ``None`` if there is none.

    ``None`` matches the runner's eager fallback for batches above the largest
    captured shape.
    """

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1; got {batch_size}")
    return next((bucket for bucket in buckets if bucket >= batch_size), None)


def batch_padding_factor(batch_size: int, buckets: list[int]) -> float | None:
    """``replayed_bucket / batch_size``; 1.0 means the replay is exact."""

    bucket = select_graph_bucket(batch_size, buckets)
    if bucket is None:
        return None
    return bucket / batch_size


def padding_report(
    batch_size: int, max_num_seqs: int, *, uses_cuda_graph: bool
) -> dict[str, object]:
    """The recorded fairness block for one measured batch size."""

    buckets = graph_buckets(max_num_seqs)
    return {
        "uses_cuda_graph": uses_cuda_graph,
        "max_num_seqs": max_num_seqs,
        "buckets": buckets,
        "bucket": select_graph_bucket(batch_size, buckets),
        "padding_factor": batch_padding_factor(batch_size, buckets),
    }
