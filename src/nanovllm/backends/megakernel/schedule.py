"""Task-graph schedule construction for the Qwen3-0.6B megakernel.

Builds the per-SM instruction stream (``inst_buf``) and bar-array dimensions
for one decode step. Pure Python + torch (only for tensor creation), so the
core logic is unit-testable without a GPU.

See ``docs/11_task_graph重写设计.md`` §6 for the design. This module is the
host-side counterpart of ``csrc/taskgraph_kernel.cu``: every instruction it
emits must match an opcode + field layout the kernel knows how to dispatch.

Layout-locked contract: ``TG_INST_WIDTH = 16`` and the opcode enum live in
``csrc/tg_abi.h``. ``test_taskgraph_abi.py`` parses that header and enforces
this module stays in sync with it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# =============================================================================
# Instruction width + opcode enum. Mirrors csrc/tg_abi.h. The ABI test parses
# the header and asserts these match; do not edit one without the other.
# =============================================================================

TG_INST_WIDTH = 16


class TgOp:
    NOP = 0
    QKV_PROJ = 1
    ATTN_DECODE = 2
    ATTN_COMBINE = 3
    O_PROJ = 4
    FFN_UPGATE_ACT = 5
    FFN_DOWN = 6
    ADD_RMSNORM = 7
    LM_HEAD = 8
    LM_ARGMAX = 9


# =============================================================================
# Bar dimension bookkeeping. See docs/11 §4.1.
# =============================================================================


@dataclass(frozen=True)
class TgBarDims:
    """Lengths of the 7 dense-model bar arrays.

    Each bar is a device ``uint32`` array zeroed per step. The index formulas
    in :class:`TgBarIndex` map a (layer, batch, kv_head, split) onto a slot in
    these arrays. Tests lock these formulas against the kernel's expectations.
    """

    bar_qkv: int      # layers * hkv
    bar_attn: int      # layers * oproj_splits
    bar_oproj: int     # layers
    bar_silu: int      # layers * down_splits
    bar_ffn_down: int  # layers
    bar_layer: int     # layers + 1
    bar_combine: int   # layers * BS * hkv (0 when split==1 always)
    bar_lm_head: int   # 1 (LM_HEAD tiles -> LM_ARGMAX)

    @classmethod
    def for_geometry(
        cls,
        num_layers: int,
        hkv: int,
        BS: int,
        oproj_splits: int,
        down_splits: int,
        attn_splits: int,
    ) -> "TgBarDims":
        """Compute bar lengths for one (BS, tiling) geometry.

        ``bar_combine`` is only needed when ``attn_splits > 1``; we always
        allocate it (sized to the max splits) so a bucket can switch on split
        without re-allocating. When ``attn_splits == 1`` it stays allocated at
        ``layers * BS * hkv`` but is never produced/consumed.

        ``bar_layer`` has ``num_layers + 1`` slots: layer L's ADD_RMSNORM
        produces slot ``L+1``, and the next layer's QKV/FFN wait on slot ``L``
        (or LM_HEAD waits on slot ``num_layers`` for the final layer). This
        avoids the final layer colliding with its own input slot.
        """
        return cls(
            bar_qkv=num_layers * hkv,
            bar_attn=num_layers * oproj_splits,
            bar_oproj=num_layers,
            bar_silu=num_layers * down_splits,
            bar_ffn_down=num_layers,
            bar_layer=num_layers + 1,
            bar_combine=num_layers * BS * hkv if attn_splits > 1 else 0,
            bar_lm_head=1,
        )


class TgBarIndex:
    """Index formulas for each bar. Mirrors docs/11 §6.3.

    Every method is a pure function of (layer, batch, kv_head, split); tests
    lock them against the kernel's decoder. Keep these in one place so the
    schedule builder and the tests share a single source of truth.
    """

    @staticmethod
    def qkv(layer: int, hkv: int, kv_head: int) -> int:
        return layer * hkv + kv_head

    @staticmethod
    def attn(layer: int, oproj_splits: int, split: int) -> int:
        return layer * oproj_splits + split

    @staticmethod
    def oproj(layer: int) -> int:
        return layer

    @staticmethod
    def silu(layer: int, down_splits: int, split: int) -> int:
        return layer * down_splits + split

    @staticmethod
    def ffn_down(layer: int) -> int:
        return layer

    @staticmethod
    def layer(layer: int) -> int:
        return layer

    @staticmethod
    def combine(layer: int, BS: int, hkv: int, batch: int, kv_head: int) -> int:
        return layer * (BS * hkv) + batch * hkv + kv_head


# =============================================================================
# Instruction encoding. Each builder returns a list of 16 ints.
# =============================================================================


def _inst(opcode: int, *fields: int) -> list[int]:
    """Pad ``fields`` to TG_INST_WIDTH with zeros."""
    out = [0] * TG_INST_WIDTH
    out[0] = opcode
    for i, v in enumerate(fields, start=1):
        if i >= TG_INST_WIDTH:
            raise ValueError(f"too many fields for instruction: {fields}")
        out[i] = int(v)
    return out


def _pack_tile(tile_m: int, tile_n: int) -> int:
    return (tile_m << 16) | (tile_n & 0xFFFF)


def make_nop() -> list[int]:
    return _inst(TgOp.NOP)


def make_qkv_proj(
    layer: int,
    tile_n: int,
    gemm_idx: int,
    needs_rope: bool,
    wait_bar_idx: int,
    wait_target: int,
    produce_bar_idx: int,
    batch_row: int,
    num_tiles: int = 1,
) -> list[int]:
    """TG_QKV_PROJ. See docs/11 §3.3.

    ``gemm_idx``: 0=Q, 1=K, 2=V. ``needs_rope``: True for Q and K (RoPE + KV
    cache write); False for V (V is written to KV cache without RoPE). The
    schedule builder decides per gemm_idx.

    ``num_tiles`` packs the ``num_tiles`` CONSECUTIVE ``tile_n`` starting at
    ``tile_n`` into one instruction (data[9]=count, data[10]=first tile_n). The
    kernel issues ONE GEMM with N = count*16 based at that tile, splitting the N
    across the worker warps. All packed tiles must share the same
    (gemm_idx, produce_bar_idx) so they read the same weight matrix and signal
    the same bar slot.
    """
    if num_tiles < 1:
        raise ValueError(f"num_tiles must be >= 1, got {num_tiles}")
    return _inst(
        TgOp.QKV_PROJ,
        layer,
        tile_n,
        gemm_idx,
        1 if needs_rope else 0,
        wait_bar_idx,
        wait_target,
        produce_bar_idx,
        batch_row,
        num_tiles,
        tile_n,   # inst[10] = range start; the kernel reads tile_n0 from here
    )


def make_attn_decode(
    layer: int,
    batch_idx: int,
    kv_head_idx: int,
    split_idx: int,
    num_splits: int,
    window_size: int,
    wait_bar_idx: int,
    wait_target: int,
    produce_bar_idx: int,
    produce_is_combine: bool,
) -> list[int]:
    """TG_ATTN_DECODE. ``produce_is_combine`` selects bar_combine vs bar_attn.

    Encoded in bit 31 of ``produce_bar_idx`` (sign bit, since indices are
    non-negative): if set, the kernel reads ``bar_combine[produce_bar_idx]``;
    otherwise ``bar_attn[produce_bar_idx]``. This avoids a separate flag word.
    """
    produce = produce_bar_idx if not produce_is_combine else (produce_bar_idx | (1 << 31))
    return _inst(
        TgOp.ATTN_DECODE,
        layer,
        batch_idx,
        kv_head_idx,
        split_idx,
        num_splits,
        window_size,
        wait_bar_idx,
        wait_target,
        produce,
    )


def make_attn_combine(
    layer: int,
    batch_idx: int,
    kv_head_idx: int,
    num_splits: int,
    wait_bar_idx: int,
    wait_target: int,
    produce_bar_idx: int,
) -> list[int]:
    return _inst(
        TgOp.ATTN_COMBINE,
        layer,
        batch_idx,
        kv_head_idx,
        num_splits,
        wait_bar_idx,
        wait_target,
        produce_bar_idx,
    )


def make_gemm(
    opcode: int,
    layer: int,
    tile_m: int,
    tile_n: int,
    split: int,
    k_tiles: int,
    wait_bar_idx: int,
    wait_target: int,
    produce_bar_idx: int,
    num_tiles: int = 1,
) -> list[int]:
    """Generic GEMM instruction (O_PROJ / FFN_UPGATE_ACT / FFN_DOWN / LM_HEAD).

    Covers the ``num_tiles`` CONSECUTIVE N tiles starting at ``tile_n``
    (data[9]=count, data[10]=first tile_n). The kernel derives N = count*16 and
    the weight/output base from the first tile, so one instruction is one GEMM
    whose N is split across the worker warps — which is why the packer must
    keep the tiles consecutive.

    ``tile_m`` is always 0 (batch fits in one WMMA M tile, M<=16) and is kept
    for layout compatibility.
    """
    if num_tiles < 1:
        raise ValueError(f"num_tiles must be >= 1, got {num_tiles}")
    return _inst(
        opcode,
        layer,
        tile_m,
        tile_n,
        split,
        k_tiles,
        wait_bar_idx,
        wait_target,
        produce_bar_idx,
        num_tiles,
        # inst[10] = range start (tile_m<<16 | tile_n, as before). The kernel
        # masks to 16 bits; inst[11..15] stay unused because the range replaces
        # the old tile list.
        _pack_tile(tile_m, tile_n),
    )


def make_add_rmsnorm(
    layer: int,
    batch_idx: int,
    wait_oproj_idx: int,
    wait_oproj_target: int,
    wait_ffn_idx: int,
    wait_ffn_target: int,
    produce_bar_idx: int,
    is_final_layer: bool,
) -> list[int]:
    return _inst(
        TgOp.ADD_RMSNORM,
        layer,
        batch_idx,
        wait_oproj_idx,
        wait_oproj_target,
        wait_ffn_idx,
        wait_ffn_target,
        produce_bar_idx,
        1 if is_final_layer else 0,
    )


def make_lm_head(
    layer: int,
    tile_m: int,
    tile_n: int,
    wait_bar_idx: int,
    wait_target: int,
    num_tiles: int = 1,
) -> list[int]:
    """TG_LM_HEAD tile. Waits on bar_layer, produces bar_lm_head[0].

    The per-batch argmax is a separate terminal TG_LM_ARGMAX instruction so it
    can wait for every LM_HEAD tile on every SM (cross-SM) before scanning the
    logits. Doing the argmax inside each tile instruction races when
    num_sms > 1: a tile's scan would read tiles other SMs have not written.
    """
    return make_gemm(
        TgOp.LM_HEAD,
        layer=layer,
        tile_m=tile_m,
        tile_n=tile_n,
        split=0,
        k_tiles=0,
        wait_bar_idx=wait_bar_idx,
        wait_target=wait_target,
        produce_bar_idx=0,
        num_tiles=num_tiles,
    )


def make_lm_argmax(
    wait_bar_idx: int,
    wait_target: int,
) -> list[int]:
    """TG_LM_ARGMAX. Terminal per-batch argmax; no produce bar.

    Layout: inst[6]=wait_bar_idx (bar_lm_head), inst[7]=wait_target
    (= total LM_HEAD tiles = vocab / lm_head_bn).
    """
    return _inst(TgOp.LM_ARGMAX, 0, 0, 0, 0, 0, wait_bar_idx, wait_target)


# =============================================================================
# Tiling configuration per GEMM. Phase 0 uses conservative defaults; Phase 2
# tunes per batch. Tile sizes are WMMA m16n16k16 multiples.
# =============================================================================


@dataclass(frozen=True)
class TgTiling:
    """Tile sizes (in units of 16, the WMMA m16n16k16 dimension)."""

    # M dimension tiles: batch is small (<=8), so tiles_M = ceil(BS/16) = 1.
    # We don't tile M; one GEMM covers all rows.
    qkv_bn: int = 16   # N tile for QKV (Q: 2048/16=128 tiles; K,V: 1024/16=64)
    qkv_bk: int = 16
    oproj_bn: int = 16  # O: 1024/16=64 tiles
    oproj_bk: int = 16
    ffn_bn: int = 16    # Gate/Up: 3072/16=192 tiles; Down: 1024/16=64 tiles
    ffn_bk: int = 16
    lm_head_bn: int = 16  # vocab 151936/16 = 9496 tiles
    lm_head_bk: int = 16
    oproj_splits: int = 1  # how O_PROJ's N tiles are split across bar_attn slots
    down_splits: int = 1   # how FFN_DOWN's K tiles are split across bar_silu slots


@dataclass(frozen=True)
class TgGeometry:
    """One (BS, tiling, attn_splits) configuration the schedule is built for."""

    BS: int
    num_layers: int
    hkv: int
    num_q_heads: int
    head_dim: int
    hidden: int
    intermediate: int
    vocab: int
    q_size: int
    kv_size: int
    tiling: TgTiling
    attn_splits: int  # static split count for this context bucket
    max_attn_splits: int  # capacity of o_partial/lse_partial buffers


# =============================================================================
# Wave construction. Each wave is a list of instructions that are mutually
# independent (can run in parallel). The placement step distributes them.
# =============================================================================


def _qkv_wait_target(layer: int, BS: int) -> int:
    """Layer 0 doesn't wait (no previous rmsnorm); else wait for BS arrivals."""
    return 0 if layer == 0 else BS


def _qkv_wait_idx(layer: int) -> int:
    """QKV waits on bar_layer[layer] (the previous ADD_RMSNORM produces there)."""
    return TgBarIndex.layer(layer)


_Q_TILES_PER_KV_HEAD = 16   # q_size=2048, bn=16 -> 128 tiles / 8 kv_heads = 16
_KV_TILES_PER_KV_HEAD = 8   # kv_size=1024, bn=16 -> 64 tiles / 8 kv_heads = 8

# Consecutive tiles packed into one GEMM instruction. The kernel derives
# N = num_tiles*16 from the first tile and splits that N across the 7 worker
# warps, so this is also the warp-saturation width: 7 tiles = one per warp.
# inst[11..15] are unused (the tile *list* was replaced by the tile *range*).
_TILE_SLOTS_PER_INST = 7


def _qkv_tile_count(geo: TgGeometry) -> int:
    bn = geo.tiling.qkv_bn
    return geo.q_size // bn + 2 * (geo.kv_size // bn)


def _oproj_tile_count(geo: TgGeometry) -> int:
    return geo.hidden // geo.tiling.oproj_bn


def _ffn_upgate_tile_count(geo: TgGeometry) -> int:
    return (geo.intermediate // geo.tiling.ffn_bn) * geo.tiling.down_splits


def _ffn_down_tile_count(geo: TgGeometry) -> int:
    return (geo.hidden // geo.tiling.ffn_bn) * geo.tiling.down_splits


def _lm_head_tile_count(geo: TgGeometry) -> int:
    return geo.vocab // geo.tiling.lm_head_bn


def _max_per_inst(num_tiles: int, num_sms: int) -> int:
    """How many consecutive tiles to pack into one GEMM instruction.

    Answer: as many as fit, i.e. :data:`_TILE_SLOTS_PER_INST`. The tempting
    alternative — pack only ``ceil(tiles / num_sms)`` so every SM gets one
    instruction — is *not* better, because the tiles inside one instruction run
    in PARALLEL across the 7 worker warps. Per-SM the wave costs

        ceil(ceil(tiles / M) / num_sms) * ceil(M / 7) * (K / 16) K-steps

    and both ``M = 7`` and ``M = 2`` evaluate to the same number of K-steps:
    fewer tiles per instruction means proportionally fewer active warps, so the
    SM streams proportionally less per K-step. What actually sets the wave time
    is ``(K/16) * k_step_time``, which this knob cannot change — only the
    kernel's per-warp pipeline depth can (see TG_B_STAGES in the kernel).

    ``M = 7`` is therefore chosen because it saturates all seven warps (more
    bytes in flight per SM than M < 7) while keeping the instruction count, and
    hence the barrier traffic, minimal. ``num_sms`` is accepted for callers that
    want to reason about grid coverage but does not change the answer.

    Bar accounting is unaffected by the choice: every GEMM arrival carries its
    tile count as ``inc`` (``tg_arrive(bar, idx, num_tiles)``), so the totals
    consumers wait on are identical however the tiles are grouped.
    """
    if num_tiles <= 0:
        return 1
    return max(1, min(_TILE_SLOTS_PER_INST, num_tiles))


def _pack_tiles(instructions: list[list[int]],
                max_per_inst: int = _TILE_SLOTS_PER_INST) -> list[list[int]]:
    """Merge consecutive instructions sharing a dependency into one GEMM.

    Greedy pack: groups runs of instructions that differ only in ``tile_n`` and
    emits one instruction per group covering the whole consecutive range
    (``num_tiles`` + first ``tile_n``). Packing cuts inst_buf size by up to
    ``max_per_inst``x without changing the dependency graph.

    The kernel turns each group into ONE GEMM call with
    ``N = num_tiles * TG_WMMA_TILE`` based at the first tile's ``tile_n``, so
    the packed tiles MUST be consecutive. That lets ``tg_wmma_gemm`` split the N
    tiles across the worker warps instead of serializing them on warp 0; the
    grouping loop below therefore breaks the run on a non-consecutive ``tile_n``
    and on a differing ``tile_m``.

    Applies to the generic GEMM opcodes (O_PROJ / FFN_UPGATE_ACT / FFN_DOWN /
    LM_HEAD). TG_QKV_PROJ is packed by :func:`_pack_qkv` (its own field layout);
    ATTN/ADD_RMSNORM/LM_ARGMAX are not packed.
    """
    if not instructions:
        return []
    out: list[list[int]] = []
    i = 0
    while i < len(instructions):
        inst = instructions[i]
        opcode = inst[0]
        # Only pack generic GEMM opcodes.
        if opcode not in (TgOp.O_PROJ, TgOp.FFN_UPGATE_ACT, TgOp.FFN_DOWN, TgOp.LM_HEAD):
            out.append(inst)
            i += 1
            continue
        # Group key: (layer, tile_m, split, k_tiles, wait_idx, wait_target,
        # produce_idx). tile_m is part of the key because the kernel does not
        # tile M: mixing two tile_m values in one range would be wrong.
        key = (inst[1], inst[2], inst[4], inst[5], inst[6], inst[7], inst[8])
        first_n = inst[3]
        count = 1
        j = i + 1
        while j < len(instructions) and count < max_per_inst:
            nxt = instructions[j]
            if nxt[0] != opcode:
                break
            if (nxt[1], nxt[2], nxt[4], nxt[5], nxt[6], nxt[7], nxt[8]) != key:
                break
            # The kernel derives N = num_tiles*16 from the first tile_n, so the
            # group must stay consecutive in tile_n. A gap would make that one
            # GEMM cover the wrong weight/output range.
            if nxt[3] != first_n + count:
                break
            count += 1
            j += 1
        out.append(
            make_gemm(
                opcode,
                layer=inst[1],
                tile_m=inst[2],
                tile_n=first_n,
                split=inst[4],
                k_tiles=inst[5],
                wait_bar_idx=inst[6],
                wait_target=inst[7],
                produce_bar_idx=inst[8],
                num_tiles=count,
            )
        )
        i = j
    return out


def _pack_qkv(instructions: list[list[int]],
              max_per_inst: int = _TILE_SLOTS_PER_INST) -> list[list[int]]:
    """Pack consecutive TG_QKV_PROJ tiles that share everything but tile_n.

    QKV has its own field layout, so it needs its own packer. The group key
    includes ``gemm_idx`` (Q/K/V use different weights) and ``produce_bar_idx``
    (tiles of different kv_heads must not be merged). The kernel derives
    N = count*16 from the first tile_n, so the group must stay consecutive in
    tile_n.
    """
    if not instructions:
        return []
    out: list[list[int]] = []
    i = 0
    while i < len(instructions):
        inst = instructions[i]
        # (layer, gemm_idx, needs_rope, wait_idx, wait_target, produce, batch)
        key = (inst[1], inst[3], inst[4], inst[5], inst[6], inst[7], inst[8])
        first_n = inst[2]
        count = 1
        j = i + 1
        while j < len(instructions) and count < max_per_inst:
            nxt = instructions[j]
            nxt_key = (nxt[1], nxt[3], nxt[4], nxt[5], nxt[6], nxt[7], nxt[8])
            if nxt_key != key or nxt[2] != first_n + count:
                break
            count += 1
            j += 1
        out.append(
            make_qkv_proj(
                layer=inst[1],
                tile_n=first_n,
                gemm_idx=inst[3],
                needs_rope=bool(inst[4]),
                wait_bar_idx=inst[5],
                wait_target=inst[6],
                produce_bar_idx=inst[7],
                batch_row=inst[8],
                num_tiles=count,
            )
        )
        i = j
    return out


def build_qkv_wave(geo: TgGeometry, layer: int) -> list[list[int]]:
    """Wave 0: QKV projection. One instruction per (gemm_idx, tile_n).

    Each tile belongs to exactly one kv_head:
      - Q: tile_n // 16 = kv_head  (q_size=2048, 128 tiles, 16 tiles/kv_head)
      - K: tile_n // 8  = kv_head  (kv_size=1024, 64 tiles, 8 tiles/kv_head)
      - V: same as K

    Producing into bar_qkv[layer*hkv + kv_head] lets attention for each
    kv_head start as soon as ITS 16 Q + 8 K + 8 V tiles finish, without
    waiting for other kv_heads (backfill). Q/K need RoPE; V does not.
    """
    wait_idx = _qkv_wait_idx(layer)
    wait_target = _qkv_wait_target(layer, geo.BS)
    out: list[list[int]] = []
    bn = geo.tiling.qkv_bn
    # Q: N = q_size = 2048 -> 128 tiles, 16 tiles per kv_head
    for tile_n in range(geo.q_size // bn):
        kv_head = tile_n // _Q_TILES_PER_KV_HEAD
        produce = TgBarIndex.qkv(layer, geo.hkv, kv_head)
        out.append(
            make_qkv_proj(
                layer=layer,
                tile_n=tile_n,
                gemm_idx=0,
                needs_rope=True,
                wait_bar_idx=wait_idx,
                wait_target=wait_target,
                produce_bar_idx=produce,
                batch_row=0,
            )
        )
    # K: N = kv_size = 1024 -> 64 tiles, 8 tiles per kv_head
    for tile_n in range(geo.kv_size // bn):
        kv_head = tile_n // _KV_TILES_PER_KV_HEAD
        produce = TgBarIndex.qkv(layer, geo.hkv, kv_head)
        out.append(
            make_qkv_proj(
                layer=layer,
                tile_n=tile_n,
                gemm_idx=1,
                needs_rope=True,
                wait_bar_idx=wait_idx,
                wait_target=wait_target,
                produce_bar_idx=produce,
                batch_row=0,
            )
        )
    # V: N = kv_size = 1024 -> 64 tiles, 8 tiles per kv_head
    for tile_n in range(geo.kv_size // bn):
        kv_head = tile_n // _KV_TILES_PER_KV_HEAD
        produce = TgBarIndex.qkv(layer, geo.hkv, kv_head)
        out.append(
            make_qkv_proj(
                layer=layer,
                tile_n=tile_n,
                gemm_idx=2,
                needs_rope=False,
                wait_bar_idx=wait_idx,
                wait_target=wait_target,
                produce_bar_idx=produce,
                batch_row=0,
            )
        )
    return out


def _qkv_wait_per_kv(geo: TgGeometry) -> int:
    """Attention for one kv_head waits for its 16 Q + 8 K + 8 V tiles = 32."""
    return _Q_TILES_PER_KV_HEAD + 2 * _KV_TILES_PER_KV_HEAD


def build_attn_wave(geo: TgGeometry, layer: int) -> list[list[int]]:
    """Wave 1: ATTN_DECODE. One instruction per (batch, kv_head, split).

    With static splits, num_splits is fixed for the bucket. split>1 produces
    into bar_combine; split==1 produces into bar_attn directly.
    """
    wait_per_kv = _qkv_wait_per_kv(geo)
    out: list[list[int]] = []
    for batch in range(geo.BS):
        for kv_head in range(geo.hkv):
            wait_idx = TgBarIndex.qkv(layer, geo.hkv, kv_head)
            for split in range(geo.attn_splits):
                if geo.attn_splits > 1:
                    produce_idx = TgBarIndex.combine(
                        layer, geo.BS, geo.hkv, batch, kv_head
                    )
                    produce_is_combine = True
                else:
                    produce_idx = TgBarIndex.attn(
                        layer, geo.tiling.oproj_splits, split
                    )
                    produce_is_combine = False
                out.append(
                    make_attn_decode(
                        layer=layer,
                        batch_idx=batch,
                        kv_head_idx=kv_head,
                        split_idx=split,
                        num_splits=geo.attn_splits,
                        window_size=0,  # Qwen3-0.6B: no sliding window in Phase 0
                        wait_bar_idx=wait_idx,
                        wait_target=wait_per_kv,
                        produce_bar_idx=produce_idx,
                        produce_is_combine=produce_is_combine,
                    )
                )
    return out


def build_attn_combine_wave(geo: TgGeometry, layer: int) -> list[list[int]]:
    """Wave 2: ATTN_COMBINE (only when attn_splits > 1).

    ATTN_DECODE publishes unnormalized per-split partials + LSE, and one
    combine per (batch, kv_head) rescales and normalizes them back into
    attn_out for O_PROJ. Validated against the causal reference by
    tests/test_taskgraph_positions.py::test_attn_splits_matches_causal_reference
    and deadlock-checked for attn_splits=2 by the execution simulator.
    """
    if geo.attn_splits <= 1:
        return []
    out: list[list[int]] = []
    for batch in range(geo.BS):
        for kv_head in range(geo.hkv):
            wait_idx = TgBarIndex.combine(layer, geo.BS, geo.hkv, batch, kv_head)
            # Combine produces into bar_attn for O_PROJ to consume.
            # With oproj_splits=1, all combines produce bar_attn[layer*1 + 0].
            produce_idx = TgBarIndex.attn(layer, geo.tiling.oproj_splits, 0)
            out.append(
                make_attn_combine(
                    layer=layer,
                    batch_idx=batch,
                    kv_head_idx=kv_head,
                    num_splits=geo.attn_splits,
                    wait_bar_idx=wait_idx,
                    wait_target=geo.attn_splits,
                    produce_bar_idx=produce_idx,
                )
            )
    return out


def build_o_proj_wave(geo: TgGeometry, layer: int) -> list[list[int]]:
    """Wave 3: O_PROJ. N = hidden = 1024 -> 64 tiles (bn=16)."""
    wait_idx = TgBarIndex.attn(layer, geo.tiling.oproj_splits, 0)
    # O_PROJ waits for all attention combines to finish: BS * hkv arrivals
    # (when split>1) or BS * hkv attention decodes (when split==1).
    wait_target = geo.BS * geo.hkv
    produce = TgBarIndex.oproj(layer)
    out: list[list[int]] = []
    bn = geo.tiling.oproj_bn
    for tile_n in range(geo.hidden // bn):
        out.append(
            make_gemm(
                TgOp.O_PROJ,
                layer=layer,
                tile_m=0,
                tile_n=tile_n,
                split=0,
                k_tiles=geo.q_size // geo.tiling.oproj_bk,
                wait_bar_idx=wait_idx,
                wait_target=wait_target,
                produce_bar_idx=produce,
            )
        )
    return out


def build_ffn_upgate_wave(geo: TgGeometry, layer: int) -> list[list[int]]:
    """Wave 4: FFN_UPGATE_ACT (gate + up fused). N = intermediate = 3072 -> 192 tiles.

    Waits on bar_oproj (O_PROJ done) — NOT bar_layer — because FFN consumes
    the POST-ATTENTION norm, not the previous layer's norm. The op body fuses:
    residual add (x_resid + o_proj_out → x_attn) + post-attn RMSNorm + gate/up
    GEMM + SiLU. Each tile instruction redundantly computes the per-row RMSNorm
    (cheap relative to the GEMM); this avoids a separate POST_ATTN_NORM wave.
    """
    wait_idx = TgBarIndex.oproj(layer)
    oproj_tiles = geo.hidden // geo.tiling.oproj_bn
    wait_target = oproj_tiles  # all O_PROJ tiles must finish
    out: list[list[int]] = []
    bn = geo.tiling.ffn_bn
    for tile_n in range(geo.intermediate // bn):
        for split in range(geo.tiling.down_splits):
            produce = TgBarIndex.silu(layer, geo.tiling.down_splits, split)
            out.append(
                make_gemm(
                    TgOp.FFN_UPGATE_ACT,
                    layer=layer,
                    tile_m=0,
                    tile_n=tile_n,
                    split=split,
                    k_tiles=geo.hidden // geo.tiling.ffn_bk,
                    wait_bar_idx=wait_idx,
                    wait_target=wait_target,
                    produce_bar_idx=produce,
                )
            )
    return out


def build_ffn_down_wave(geo: TgGeometry, layer: int) -> list[list[int]]:
    """Wave 5: FFN_DOWN. N = hidden = 1024 -> 64 tiles."""
    out: list[list[int]] = []
    bn = geo.tiling.ffn_bn
    silu_tiles_per_split = geo.intermediate // (bn * geo.tiling.down_splits)
    for tile_n in range(geo.hidden // bn):
        for split in range(geo.tiling.down_splits):
            wait_idx = TgBarIndex.silu(layer, geo.tiling.down_splits, split)
            produce = TgBarIndex.ffn_down(layer)
            out.append(
                make_gemm(
                    TgOp.FFN_DOWN,
                    layer=layer,
                    tile_m=0,
                    tile_n=tile_n,
                    split=split,
                    k_tiles=silu_tiles_per_split,
                    wait_bar_idx=wait_idx,
                    wait_target=silu_tiles_per_split,
                    produce_bar_idx=produce,
                )
            )
    return out


def build_add_rmsnorm_wave(geo: TgGeometry, layer: int) -> list[list[int]]:
    """Wave 6: ADD_RMSNORM. One instruction per batch row (per-row rmsnorm).

    Always produces into ``bar_layer[layer + 1]``: the next layer's QKV/FFN
    wait there. The final layer (layer == num_layers - 1) produces into
    ``bar_layer[num_layers]``, which LM_HEAD waits on. ``bar_layer`` has
    ``num_layers + 1`` slots to accommodate this without colliding with the
    final layer's own input slot (``bar_layer[num_layers - 1]``).
    """
    is_final = layer == geo.num_layers - 1
    oproj_target = geo.hidden // geo.tiling.oproj_bn  # oproj tile count
    ffn_target = geo.hidden // geo.tiling.ffn_bn  # ffn_down tile count
    out: list[list[int]] = []
    for batch in range(geo.BS):
        produce = TgBarIndex.layer(layer + 1)  # always layer+1
        out.append(
            make_add_rmsnorm(
                layer=layer,
                batch_idx=batch,
                wait_oproj_idx=TgBarIndex.oproj(layer),
                wait_oproj_target=oproj_target,
                wait_ffn_idx=TgBarIndex.ffn_down(layer),
                wait_ffn_target=ffn_target,
                produce_bar_idx=produce,
                is_final_layer=is_final,
            )
        )
    return out


def build_lm_head_wave(geo: TgGeometry) -> list[list[int]]:
    """Wave 7 (last layer only): LM_HEAD tiles + the terminal LM_ARGMAX.

    N = vocab = 151936 -> 9496 tiles. The tiles wait on
    ``bar_layer[num_layers]`` (produced by the final layer's ADD_RMSNORM) with
    target = BS (all rows' final rmsnorm), and produce ``bar_lm_head[0]`` with
    inc = tile count. The trailing LM_ARGMAX waits for every tile on every SM,
    then scans each row's logits and stores the token id.
    """
    wait_idx = TgBarIndex.layer(geo.num_layers)  # == num_layers (the +1 slot)
    wait_target = geo.BS  # all rows' final rmsnorm
    out: list[list[int]] = []
    bn = geo.tiling.lm_head_bn
    num_tiles = geo.vocab // bn
    for tile_n in range(num_tiles):
        out.append(
            make_lm_head(
                layer=geo.num_layers - 1,
                tile_m=0,
                tile_n=tile_n,
                wait_bar_idx=wait_idx,
                wait_target=wait_target,
            )
        )
    # Terminal argmax: waits for all LM_HEAD tiles (cross-SM) before scanning.
    out.append(make_lm_argmax(wait_bar_idx=0, wait_target=num_tiles))
    return out


# =============================================================================
# Per-layer wave assembly + LM_HEAD.
# =============================================================================


def build_layer_waves(geo: TgGeometry, layer: int,
                      num_sms: int = _TILE_SLOTS_PER_INST) -> list[list[list[int]]]:
    """Return the 7 named waves for one layer (LM_HEAD is separate).

    GEMM waves (O_PROJ, FFN_UPGATE_ACT, FFN_DOWN) are tile-packed to cut
    inst_buf size; QKV/ATTN/ADD_RMSNORM use opcode-specific layouts that
    are not packed. ``num_sms`` sets the tile granularity so each wave covers
    as many SMs as possible (see :func:`_max_per_inst`).
    """
    return [
        _pack_qkv(build_qkv_wave(geo, layer),
                  _max_per_inst(_qkv_tile_count(geo), num_sms)),
        build_attn_wave(geo, layer),
        build_attn_combine_wave(geo, layer),
        _pack_tiles(build_o_proj_wave(geo, layer),
                    _max_per_inst(_oproj_tile_count(geo), num_sms)),
        _pack_tiles(build_ffn_upgate_wave(geo, layer),
                    _max_per_inst(_ffn_upgate_tile_count(geo), num_sms)),
        _pack_tiles(build_ffn_down_wave(geo, layer),
                    _max_per_inst(_ffn_down_tile_count(geo), num_sms)),
        build_add_rmsnorm_wave(geo, layer),
    ]


# =============================================================================
# Placement: round-robin across SMs. Adapted from cohere's _round_robin_waves
# (schedule.py:1244-1267). Cursor persists across waves so waiters don't all
# land on low-numbered SMs.
# =============================================================================


def place_round_robin(
    waves: list[list[list[int]]], num_sms: int
) -> list[list[list[int]]]:
    """Distribute instructions across SMs round-robin.

    Returns ``per_sm[sm]`` = list of instructions for that SM. Short SMs are
    padded with NOP to ``max_inst`` later during serialization. The cursor
    advances across waves (not reset per wave) so dependencies naturally spread.
    """
    per_sm: list[list[list[int]]] = [[] for _ in range(num_sms)]
    cursor = 0
    for wave in waves:
        for inst in wave:
            per_sm[cursor % num_sms].append(inst)
            cursor += 1
    return per_sm


# =============================================================================
# Serialization: per_sm -> flat inst_buf tensor + num_inst_per_sm.
# =============================================================================


def serialize_inst_buf(
    per_sm: list[list[list[int]]], num_sms: int
) -> tuple[list[int], list[int], int]:
    """Pack ``per_sm`` into a flat int32 buffer + per-SM counts + max_inst.

    Returns ``(flat_inst_buf, num_inst_per_sm, max_inst)`` where
    ``flat_inst_buf`` has shape ``[num_sms, max_inst, TG_INST_WIDTH]`` (row-major),
    short SMs padded with NOP.
    """
    max_inst = max((len(sm) for sm in per_sm), default=0)
    if max_inst == 0:
        max_inst = 1  # avoid zero-stride; the kernel still launches but does nothing
    flat: list[int] = []
    counts: list[int] = []
    for sm in per_sm:
        counts.append(len(sm))
        for i in range(max_inst):
            inst = sm[i] if i < len(sm) else make_nop()
            flat.extend(inst)
    return flat, counts, max_inst


# =============================================================================
# Top-level entry: build a complete schedule for one geometry.
# =============================================================================


@dataclass
class TgSchedule:
    """Output of :func:`build_schedule`."""

    flat_inst_buf: list[int]          # [num_sms, max_inst, TG_INST_WIDTH]
    num_inst_per_sm: list[int]        # [num_sms]
    max_inst: int
    bar_dims: TgBarDims
    num_sms: int
    total_instructions: int


def build_schedule(geo: TgGeometry, num_sms: int,
                   layer_range: tuple[int, int] | None = None) -> TgSchedule:
    """Build the per-SM instruction stream for one decode step.

    Assembles 28 layers × 7 waves + LM_HEAD + LM_ARGMAX, places them
    round-robin, and serializes. Pure function of (geo, num_sms, layer_range) —
    no GPU state, fully testable.

    ``layer_range=(start, end)`` builds one **segment** of the same step: the
    identical stream restricted to layers ``[start, end)``, plus the LM head
    when the segment owns the final layer. Launching the segments back-to-back
    with one cooperative launch each is the "partial fusion" arm of the fusion
    granularity experiment; ``layer_range=None`` is the current single
    megakernel and stays behaviourally identical to the previous code.

    Bar dims are always sized for the **full** layer count, because bar slots
    are indexed by global layer and every segment shares the one bar array that
    is zeroed once per step. The only cross-segment dependency is
    ``bar_layer[end]``: this segment's last ADD_RMSNORM produces it and the next
    segment's first QKV waits on it with target ``BS`` — the same slot, already
    satisfied because a kernel launch is an implicit full barrier.
    """
    if geo.tiling.oproj_splits != 1 or geo.tiling.down_splits != 1:
        raise NotImplementedError(
            "oproj_splits / down_splits > 1 are not implemented: the kernel "
            "GEMMs always run the full K/N, and build_ffn_upgate_wave would "
            "emit every tile once per split (duplicated work + double bar "
            "arrivals). Keep both at 1 until split-K accumulation lands."
        )
    start_layer, end_layer = (
        layer_range if layer_range is not None else (0, geo.num_layers)
    )
    if not 0 <= start_layer < end_layer <= geo.num_layers:
        raise ValueError(
            f"invalid layer_range {layer_range} for {geo.num_layers} layers"
        )
    waves: list[list[list[int]]] = []
    for layer in range(start_layer, end_layer):
        waves.extend(build_layer_waves(geo, layer, num_sms))
    if end_layer == geo.num_layers:
        waves.append(_pack_tiles(build_lm_head_wave(geo),
                                 _max_per_inst(_lm_head_tile_count(geo), num_sms)))

    per_sm = place_round_robin(waves, num_sms)
    flat, counts, max_inst = serialize_inst_buf(per_sm, num_sms)

    bar_dims = TgBarDims.for_geometry(
        num_layers=geo.num_layers,
        hkv=geo.hkv,
        BS=geo.BS,
        oproj_splits=geo.tiling.oproj_splits,
        down_splits=geo.tiling.down_splits,
        attn_splits=geo.attn_splits,
    )

    total = sum(counts)
    return TgSchedule(
        flat_inst_buf=flat,
        num_inst_per_sm=counts,
        max_inst=max_inst,
        bar_dims=bar_dims,
        num_sms=num_sms,
        total_instructions=total,
    )


def split_layers(num_layers: int, groups: int) -> list[tuple[int, int]]:
    """Split ``num_layers`` into ``groups`` contiguous, near-equal segments.

    The fusion granularity knob: one segment = one cooperative launch per decode
    step, so ``groups == 1`` is the current single megakernel and
    ``groups == num_layers`` is one launch per layer.

    Layer-contiguous by construction, which is what makes segmentation
    semantically cheap: bar slots are global-layer-indexed (two segments never
    share a slot, so the once-per-step zeroing stays valid) and the only
    cross-segment edge is the ``bar_layer`` handoff described in
    :func:`build_schedule`.
    """
    if groups < 1:
        raise ValueError(f"fusion groups must be >= 1; got {groups}")
    if groups > num_layers:
        raise ValueError(
            f"fusion groups ({groups}) cannot exceed the layer count "
            f"({num_layers})"
        )
    base, extra = divmod(num_layers, groups)
    segments: list[tuple[int, int]] = []
    start = 0
    for index in range(groups):
        size = base + (1 if index < extra else 0)
        segments.append((start, start + size))
        start += size
    return segments


# =============================================================================
# Convenience: default geometry for Qwen3-0.6B.
# =============================================================================


def default_geometry(BS: int, attn_splits: int = 1) -> TgGeometry:
    """Default Qwen3-0.6B geometry for one batch size + context bucket."""
    return TgGeometry(
        BS=BS,
        num_layers=28,
        hkv=8,
        num_q_heads=16,
        head_dim=128,
        hidden=1024,
        intermediate=3072,
        vocab=151936,
        q_size=2048,
        kv_size=1024,
        tiling=TgTiling(),
        attn_splits=attn_splits,
        max_attn_splits=4,  # capacity for o_partial/lse_partial
    )


if __name__ == "__main__":
    # Smoke test: build a BS=1 schedule and print stats.
    geo = default_geometry(BS=1, attn_splits=1)
    sched = build_schedule(geo, num_sms=108)
    print(f"BS=1, attn_splits=1, num_sms=108")
    print(f"  total_instructions = {sched.total_instructions}")
    print(f"  max_inst           = {sched.max_inst}")
    print(f"  bar_dims           = {sched.bar_dims}")
    print(f"  inst_buf size      = {len(sched.flat_inst_buf)} ints "
          f"({len(sched.flat_inst_buf) * 4 / 1024:.1f} KiB)")
