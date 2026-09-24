"""Task-graph megakernel decode backend (``decode_backend="megakernel-tg"``).

Wraps the persistent task-graph kernel (``csrc/taskgraph_kernel.cu``) behind the
same ``decode`` / ``decode_batch`` interface the frozen ``kernel.cu`` decoder
exposes, so :class:`MegakernelBackend` can dispatch to either path and
``benchmarks/compare.py`` can A/B them in separate engines.

Differences from the frozen path that matter for integration (docs/13 §2):

* The kernel is **decode-only and whole-batch**: one cooperative launch of
  ``num_sms`` CTAs runs the entire 28-layer forward for all rows. There is no
  per-sequence grid partitioning, so the ``LDG_ATTN_BLOCKS`` batch ceiling that
  caps the WMMA path does not apply; the limit here is ``TG_MAX_BATCH`` (8),
  the compile-time scratch/bar sizing ceiling.
* The batch shape is baked into the instruction stream, so one schedule is
  cached per batch size and the kernel is launched once per decode step.
* The kernel does **not** do the embedding lookup: the host stages
  ``x_resid[b] = embed_weight[token_b]`` before the launch (same contract as
  ``tools/bench_taskgraph.py`` / ``tools/generate_taskgraph.py``).

This module deliberately imports ``torch`` lazily so the pure compatibility
policy below stays unit-testable on a CUDA-less checkout (the same reason
``engine/burst_policy.py`` exists).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

# Compile-time ceiling for scratch/bar sizing (``-DTG_MAX_BATCH`` in build.py).
# Keep in sync with ``taskgraph_kernel.cu``; ``test_taskgraph_abi`` locks the
# kernel/build.py defaults and ``test_megakernel_backend`` locks this value
# against the batch policy.
TG_MAX_BATCH = 8

# Qwen3-0.6B geometry the task-graph kernel is compiled for.
NUM_LAYERS = 28
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
Q_SIZE = NUM_Q_HEADS * HEAD_DIM   # 2048
KV_SIZE = NUM_KV_HEADS * HEAD_DIM  # 1024
VOCAB_SIZE = 151936

# Capacity of ``o_partial`` / ``lse_partial``; must equal the
# ``max_attn_splits`` that ``schedule.default_geometry`` bakes into the
# schedule, because the kernel indexes the partials with it.
MAX_ATTN_SPLITS = 4

# WMMA outputs are written with ``wmma::store_matrix_sync``, which always
# stores a full 16-row tile even when BS < 16 (see the binding's layout guards).
WMMA_PAD_ROWS = 16


def max_pages_per_seq(max_model_len: int, block_size: int) -> int:
    """Row stride of the ``page_table`` buffer for a fixed decode geometry."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return max(1, (max_model_len + block_size - 1) // block_size)


def resolve_num_sms(device_sm_count: int) -> int:
    """Grid size for the cooperative launch.

    The kernel is launched with one CTA per SM and every CTA must be resident
    (the ``tg_wait`` spin needs all producers running), so the grid is the
    device's SM count. ``TG_NUM_SMS`` overrides it for A/B experiments (e.g.
    measuring how much of the 25x-at-108-SM scaling is left on the table).
    """

    override = os.getenv("TG_NUM_SMS")
    if override:
        value = int(override)
        if value <= 0:
            raise ValueError(f"TG_NUM_SMS must be positive; got {value}")
        return value
    return max(1, int(device_sm_count))


@dataclass(frozen=True)
class TaskGraphCompatibility:
    supported: bool
    reason: str = ""


def taskgraph_compatibility(config: Any, seqs: list[Any], is_prefill: bool) -> TaskGraphCompatibility:
    """Torch-free admission policy for the task-graph decode path.

    Mirrors the shape of :meth:`MegakernelBackend.compatibility` but with the
    task-graph limits: whole-batch cooperative launch (so no
    ``LDG_ATTN_BLOCKS`` grid-partition ceiling) and ``TG_MAX_BATCH`` as the
    scratch ceiling.
    """

    if is_prefill:
        return TaskGraphCompatibility(
            False, "Task-graph megakernel is decode-only; use the prefill backend"
        )
    if not seqs:
        return TaskGraphCompatibility(False, "Task-graph megakernel needs at least one sequence")
    if len(seqs) > TG_MAX_BATCH:
        return TaskGraphCompatibility(
            False,
            f"Task-graph megakernel supports batch sizes 1..{TG_MAX_BATCH} "
            "(TG_MAX_BATCH, the compile-time scratch/bar ceiling)",
        )
    max_num_seqs = getattr(config, "max_num_seqs", None)
    if isinstance(max_num_seqs, int) and max_num_seqs < len(seqs):
        return TaskGraphCompatibility(
            False,
            f"batch {len(seqs)} exceeds max_num_seqs={max_num_seqs}",
        )
    if getattr(config, "tensor_parallel_size", 1) != 1:
        return TaskGraphCompatibility(
            False, "Task-graph megakernel requires tensor_parallel_size=1"
        )
    for seq in seqs:
        if bool(getattr(seq, "is_greedy", False)):
            continue
        temperature = getattr(seq, "temperature", None)
        if temperature is None or temperature > 1e-10:
            return TaskGraphCompatibility(
                False, "Task-graph megakernel supports greedy sampling only"
            )
    return TaskGraphCompatibility(True)


@dataclass
class _Geometry:
    """One cached schedule plus the launch metadata the binding needs."""

    BS: int
    attn_splits: int
    schedule: Any
    inst_buf: Any
    num_inst_per_sm: Any


class TaskGraphDecoder:
    """Owns the schedule cache, scratch, bars and one decode step."""

    def __init__(
        self,
        weights,
        *,
        k_cache,
        v_cache,
        block_size: int,
        num_cache_blocks: int,
        max_model_len: int,
        attn_splits: int = 1,
        verbose: bool = False,
    ):
        import torch

        from nanovllm.backends.megakernel import schedule as tg_schedule
        from nanovllm.backends.megakernel.model import _pack_layer_weights

        self._torch = torch
        self._tg_schedule = tg_schedule

        self._weights = weights
        self._embed_weight = weights["embed_weight"]
        self._final_norm_weight = weights["final_norm_weight"]
        self._cos_table = weights["cos_table"]
        self._sin_table = weights["sin_table"]
        self._layer_weights_packed = _pack_layer_weights(weights["layer_weights"])

        self._k_cache = k_cache
        self._v_cache = v_cache
        self._block_size = int(block_size)
        self._num_cache_blocks = int(num_cache_blocks)
        self._max_pages = max_pages_per_seq(int(max_model_len), self._block_size)
        # KV layer stride in *elements* of one cache view.
        self._kv_layer_stride = self._num_cache_blocks * self._block_size * KV_SIZE
        self._attn_splits = int(attn_splits)
        if not 1 <= self._attn_splits <= MAX_ATTN_SPLITS:
            # o_partial/lse_partial are sized MAX_ATTN_SPLITS and the kernel
            # indexes them with that stride while the schedule emits
            # ``attn_splits`` partials. More splits than capacity writes past
            # the buffers (and the combine then reads garbage).
            raise ValueError(
                f"attn_splits must be 1..{MAX_ATTN_SPLITS} "
                f"(o_partial/lse_partial capacity); got {self._attn_splits}"
            )
        self._verbose = verbose

        self._num_sms = resolve_num_sms(
            torch.cuda.get_device_properties(0).multi_processor_count
        )

        # ---- scratch (allocated once, sized for the largest batch) ----------
        f32 = dict(dtype=torch.float32, device="cuda")
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        bs = TG_MAX_BATCH
        self._norm_out = torch.empty(bs, HIDDEN_SIZE, **f32)
        self._q_out = torch.empty(WMMA_PAD_ROWS, Q_SIZE, **f32)
        self._k_out = torch.empty(WMMA_PAD_ROWS, KV_SIZE, **f32)
        self._v_out = torch.empty(WMMA_PAD_ROWS, KV_SIZE, **f32)
        self._attn_out = torch.empty(bs, Q_SIZE, **f32)
        self._o_proj_out = torch.empty(WMMA_PAD_ROWS, HIDDEN_SIZE, **f32)
        self._x_attn = torch.empty(bs, HIDDEN_SIZE, **f32)
        self._post_attn_norm = torch.empty(bs, HIDDEN_SIZE, **f32)
        self._silu_out = torch.empty(WMMA_PAD_ROWS, INTERMEDIATE_SIZE, **f32)
        self._ffn_down_out = torch.empty(WMMA_PAD_ROWS, HIDDEN_SIZE, **f32)
        self._up_out = torch.empty(WMMA_PAD_ROWS, INTERMEDIATE_SIZE, **f32)
        self._x_ffn = torch.empty(bs, HIDDEN_SIZE, **f32)
        self._x_resid = torch.empty(bs, HIDDEN_SIZE, **bf16)
        self._lm_logits = torch.empty(WMMA_PAD_ROWS, VOCAB_SIZE, **f32)
        # int32 alias of the same storage: the kernel writes the sampled token
        # id into logits[0] of each row, and the runtime reads it back as int32.
        self._lm_logits_i32 = self._lm_logits.view(torch.int32)
        self._o_partial = torch.empty(
            bs, NUM_KV_HEADS, MAX_ATTN_SPLITS, HIDDEN_SIZE, **f32
        )
        self._lse_partial = torch.empty(bs, NUM_KV_HEADS, MAX_ATTN_SPLITS, **f32)

        self._scratch = (
            self._norm_out, self._q_out, self._k_out, self._v_out,
            self._attn_out, self._o_proj_out, self._x_attn,
            self._post_attn_norm, self._silu_out, self._ffn_down_out,
            self._up_out, self._x_ffn, self._x_resid, self._lm_logits,
            self._o_partial, self._lse_partial,
        )

        # ---- per-step inputs the host owns ---------------------------------
        u32 = dict(dtype=torch.uint32, device="cuda")
        i32 = dict(dtype=torch.int32, device="cuda")
        self._page_table = torch.zeros(bs, self._max_pages, **i32)
        self._cache_seqlens = torch.zeros(bs, **i32)
        self._row_active = torch.ones(bs, **i32)
        # Lazily-allocated 1-row staging for decode_row() (see its docstring).
        self._row_input = None
        self._row_pos = None
        self._row_table = None
        # ---- schedules (built for the bucket batches; others lazily) -------
        self._bar_names = (
            "bar_qkv", "bar_attn", "bar_oproj", "bar_silu",
            "bar_ffn_down", "bar_layer", "bar_combine", "bar_lm_head",
        )
        self._schedules: dict[tuple[int, int], _Geometry] = {}
        for bucket_bs in (1, 2, 4, 8):
            self._build_geometry(bucket_bs, self._attn_splits)

        # ---- bars: allocate the element-wise max over the cached geometries
        dims = {
            name: max(
                getattr(geo.schedule.bar_dims, name) for geo in self._schedules.values()
            )
            for name in self._bar_names
        }
        self._bar_dims = dims
        self._bars = {
            name: torch.zeros(max(1, size), **u32) for name, size in dims.items()
        }
        # ---- one fixed zero-region list (scratch + bars) -------------------
        # Keeping the pointers stable is what lets the binding cache its device
        # staging buffer; rebuilding the list per step would add a H2D copy to
        # the critical path (see tg_zero_regions_binding).
        self._regions = torch.tensor(
            [
                (t.data_ptr(), t.numel() * t.element_size())
                for t in (*self._scratch, *self._bars.values())
            ],
            dtype=torch.int64,
        )

        self._ops = torch.ops.nanovllm_megakernel_C
        if self._verbose:
            print(
                f"taskgraph decoder: num_sms={self._num_sms} "
                f"attn_splits={self._attn_splits} max_pages={self._max_pages} "
                f"max_inst={max(g.schedule.max_inst for g in self._schedules.values())}"
            )

    # -- schedule cache ----------------------------------------------------

    def _build_geometry(self, batch_size: int, attn_splits: int) -> _Geometry:
        if batch_size < 1 or batch_size > TG_MAX_BATCH:
            raise ValueError(
                f"task-graph batch must be 1..{TG_MAX_BATCH}; got {batch_size}"
            )
        key = (batch_size, attn_splits)
        cached = self._schedules.get(key)
        if cached is not None:
            return cached

        geo = self._tg_schedule.default_geometry(BS=batch_size, attn_splits=attn_splits)
        sched = self._tg_schedule.build_schedule(geo, num_sms=self._num_sms)
        # Bars were allocated from the bucket geometries. Anything wider would
        # index past the allocation and silently corrupt a neighbouring bar.
        for name in self._bar_names:
            allocated = self._bars[name].numel() if hasattr(self, "_bars") else None
            needed = getattr(sched.bar_dims, name)
            if allocated is not None and needed > allocated:
                raise RuntimeError(
                    f"bar {name} needs {needed} slots but only {allocated} were "
                    f"allocated for BS={batch_size}, attn_splits={attn_splits}"
                )
        entry = _Geometry(
            BS=batch_size,
            attn_splits=attn_splits,
            schedule=sched,
            inst_buf=self._torch.tensor(
                sched.flat_inst_buf, dtype=self._torch.uint32
            ).cuda().view(self._torch.int32),
            num_inst_per_sm=self._torch.tensor(
                sched.num_inst_per_sm, dtype=self._torch.int32
            ).cuda(),
        )
        self._schedules[key] = entry
        return entry

    # -- decode ------------------------------------------------------------

    def decode_rows(self, input_ids, positions, block_tables) -> list[int]:
        """One decode step for every row.

        ``input_ids``/``positions`` are 1-D int64 CUDA tensors; ``block_tables``
        is the padded ``[rows, <= max_pages]`` int32 CUDA tensor from
        ``ModelRunner.prepare_block_tables``. Returns one token id per row.
        """

        import torch

        rows = int(input_ids.numel())
        if rows == 0:
            return []
        geo = self._build_geometry(rows, self._attn_splits)
        sched = geo.schedule

        # Zero every scratch buffer and bar counter before the launch. This must
        # happen BEFORE the embedding is staged: x_resid is part of the scratch
        # set (the tools' step() does the same), and clearing it afterwards
        # would wipe the input.
        self._ops.tg_zero_regions(self._regions, self._regions.shape[0])

        # Stage the embedding of each row's current token into x_resid. The
        # kernel starts from x_resid, not from a token id.
        self._x_resid[:rows] = self._embed_weight[input_ids]
        # cache_seqlens is the *pre-decode* length; the kernel writes the new
        # token at position cache_seqlens[row]. ``positions`` is int64 (the
        # ModelRunner convention) and this is a copy_, which casts in place --
        # no per-step allocation.
        self._cache_seqlens[:rows] = positions
        cols = int(block_tables.size(1))
        if cols > self._max_pages:
            raise ValueError(
                f"block table has {cols} columns but the task-graph page_table "
                f"stride is {self._max_pages} (max_model_len // block_size)"
            )
        self._page_table[:rows, :cols] = block_tables

        self._ops.tg_decode_step(
            geo.inst_buf, geo.num_inst_per_sm, sched.max_inst, self._num_sms,
            self._layer_weights_packed, self._embed_weight, self._final_norm_weight,
            self._cos_table, self._sin_table,
            self._k_cache, self._v_cache, self._page_table,
            self._cache_seqlens, self._row_active,
            self._num_cache_blocks, self._max_pages, self._block_size,
            self._kv_layer_stride,
            *self._scratch, *self._bars.values(),
            NUM_LAYERS, rows, HIDDEN_SIZE, INTERMEDIATE_SIZE,
            Q_SIZE, KV_SIZE, NUM_Q_HEADS, NUM_KV_HEADS,
            HEAD_DIM, VOCAB_SIZE, 1, 1, MAX_ATTN_SPLITS,
        )
        # One sync for the whole batch. ``.item()`` per row would serialize a
        # D2H per row; the sampled ids are contiguous in the int32 alias.
        return self._lm_logits_i32[:rows, 0].cpu().tolist()

    def decode_row(self, token_id: int, position: int, block_table) -> int:
        """Single-row convenience wrapper (the batch=1 ModelRunner path).

        Uses cached 1-row staging tensors: the batch=1 path is the one that is
        supposed to win by removing per-step host work, so allocating three
        small CUDA tensors per step (as the obvious implementation would) would
        tax exactly the case being optimized.
        """

        import torch

        if self._row_input is None:
            self._row_input = torch.zeros(1, dtype=torch.int64, device="cuda")
            self._row_pos = torch.zeros(1, dtype=torch.int64, device="cuda")
            self._row_table = torch.zeros(
                1, self._max_pages, dtype=torch.int32, device="cuda"
            )
        self._row_input[0] = int(token_id)
        self._row_pos[0] = int(position)
        table = block_table
        if not torch.is_tensor(table):
            table = torch.as_tensor(table, dtype=torch.int32, device="cuda")
        # Only the first ``cols`` entries are ever read: the kernel maps a
        # position to ``block_table[position // block_size]`` and every position
        # it touches is < the sequence length, whose block count is ``cols``
        # (prepare_block_tables pads to exactly that). Columns past ``cols`` may
        # therefore keep a stale value from an earlier, longer sequence without
        # being read -- so no per-step tail clear is needed.
        cols = min(int(table.numel()), self._max_pages)
        self._row_table[0, :cols] = table[:cols]
        return int(
            self.decode_rows(self._row_input, self._row_pos, self._row_table)[0]
        )

    # -- frozen-Decoder-compatible aliases --------------------------------
    # ``MegakernelBackend.decode``/``decode_batch`` call these names, so the
    # task-graph decoder can be swapped in without branching at every call.

    def step(self, token_id: int, position: int | None = None, block_table=None) -> int:
        if position is None or block_table is None:
            raise ValueError("task-graph decode requires position and block_table")
        return self.decode_row(token_id, position, block_table)

    def step_batch(self, input_token_ids, positions, block_tables) -> list[int]:
        return self.decode_rows(input_token_ids, positions, block_tables)
