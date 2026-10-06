"""Offline ownership and geometry audits for task-graph instruction streams.

This module deliberately stays torch/CUDA-free.  The scheduler/verifier proves
that dependency counters can make progress; this audit proves that each packed
instruction describes a legal matrix range and that the schedule does not give
more than one instruction ownership of the same output tile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from . import schedule as tg


@dataclass(frozen=True)
class AuditReport:
    """Results of one schedule ownership/geometry audit."""

    ok: bool
    failures: tuple[str, ...]
    checked_writes: int

    def describe(self) -> str:
        if self.ok:
            return f"audit OK ({self.checked_writes} output writes checked)"
        return "; ".join(self.failures)


def _iter_instructions(schedule) -> Iterable[tuple[int, list[int]]]:
    width = tg.TG_INST_WIDTH
    for sm, count in enumerate(schedule.num_inst_per_sm):
        for instruction in range(count):
            start = (sm * schedule.max_inst + instruction) * width
            yield sm, list(schedule.flat_inst_buf[start : start + width])


def _decode_tile_range(inst: list[int]) -> tuple[int, int, int]:
    """Return ``(tile_m, tile_n0, count)`` for a generic GEMM instruction."""

    packed = int(inst[10])
    return packed >> 16, packed & 0xFFFF, int(inst[9])


def audit_schedule(
    schedule,
    *,
    geometry: tg.TgGeometry,
    batch_size: int,
    num_sms: int,
    attn_splits: int,
) -> AuditReport:
    """Audit one flat/per-SM schedule without launching CUDA.

    The checks intentionally mirror the address arithmetic in
    ``taskgraph_kernel.cu``.  They are not a replacement for device-side
    memcheck: they catch malformed ranges, split indices, duplicate tile
    ownership, and invalid cross-head QKV packs before a server session.
    """

    failures: list[str] = []
    writes: dict[tuple, tuple[int, int]] = {}
    checked_writes = 0
    if schedule.num_sms != num_sms:
        failures.append(
            f"schedule num_sms={schedule.num_sms} does not match audit num_sms={num_sms}"
        )
    if geometry.BS != batch_size:
        failures.append(
            f"geometry BS={geometry.BS} does not match audit batch_size={batch_size}"
        )
    if geometry.attn_splits != attn_splits:
        failures.append(
            f"geometry attn_splits={geometry.attn_splits} does not match audit attn_splits={attn_splits}"
        )

    def record(key: tuple, sm: int, op: int) -> None:
        nonlocal checked_writes
        checked_writes += 1
        previous = writes.get(key)
        if previous is not None:
            failures.append(
                f"duplicate output ownership for {key}: SM {previous[0]} and SM {sm}"
            )
        else:
            writes[key] = (sm, op)

    for sm, inst in _iter_instructions(schedule):
        if len(inst) != tg.TG_INST_WIDTH:
            failures.append(f"SM {sm} has truncated instruction ({len(inst)} words)")
            continue
        op = int(inst[0])
        if op == tg.TgOp.NOP:
            continue

        layer = int(inst[1])
        if not 0 <= layer < geometry.num_layers:
            failures.append(f"opcode {op} has invalid layer={layer}")
            continue

        if op == tg.TgOp.QKV_PROJ:
            gemm = int(inst[3])
            count = int(inst[9])
            tile_n0 = int(inst[10])
            if gemm not in (0, 1, 2):
                failures.append(f"QKV has invalid gemm_idx={gemm}")
                continue
            tile_limit = geometry.q_size // geometry.tiling.qkv_bn if gemm == 0 else geometry.kv_size // geometry.tiling.qkv_bn
            if count < 1 or tile_n0 < 0 or tile_n0 + count > tile_limit:
                failures.append(
                    f"QKV range out of bounds: gemm={gemm} start={tile_n0} count={count} limit={tile_limit}"
                )
                continue
            tiles_per_head = _Q_TILES_PER_KV_HEAD if gemm == 0 else _KV_TILES_PER_KV_HEAD
            first_head = tile_n0 // tiles_per_head
            last_head = (tile_n0 + count - 1) // tiles_per_head
            if first_head != last_head:
                failures.append(
                    f"QKV packed range crosses bar owner: start={tile_n0} count={count} heads={first_head}->{last_head}"
                )
            for tile in range(tile_n0, tile_n0 + count):
                record(("qkv", layer, gemm, tile), sm, op)

        elif op in (tg.TgOp.O_PROJ, tg.TgOp.FFN_UPGATE_ACT, tg.TgOp.FFN_DOWN, tg.TgOp.LM_HEAD):
            tile_m, tile_n0, count = _decode_tile_range(inst)
            if tile_m != 0:
                failures.append(f"opcode {op} has unsupported tile_m={tile_m}")
            if op == tg.TgOp.O_PROJ:
                tile_limit = geometry.hidden // geometry.tiling.oproj_bn
                name = "O_PROJ"
            elif op == tg.TgOp.FFN_UPGATE_ACT:
                tile_limit = geometry.intermediate // geometry.tiling.ffn_bn
                name = "FFN_UPGATE_ACT"
            elif op == tg.TgOp.FFN_DOWN:
                tile_limit = geometry.hidden // geometry.tiling.ffn_bn
                name = "FFN_DOWN"
            else:
                tile_limit = geometry.vocab // geometry.tiling.lm_head_bn
                name = "LM_HEAD"
            if count < 1 or tile_n0 < 0 or tile_n0 + count > tile_limit:
                failures.append(
                    f"{name} range out of bounds: start={tile_n0} count={count} limit={tile_limit}"
                )
                continue
            for tile in range(tile_n0, tile_n0 + count):
                record((name, layer, tile), sm, op)

        elif op == tg.TgOp.ATTN_DECODE:
            batch = int(inst[2])
            kv_head = int(inst[3])
            split = int(inst[4])
            num_splits = int(inst[5])
            if not 0 <= batch < batch_size:
                failures.append(f"ATTN_DECODE has invalid batch={batch}")
            if not 0 <= kv_head < geometry.hkv:
                failures.append(f"ATTN_DECODE has invalid kv_head={kv_head}")
            if num_splits != attn_splits:
                failures.append(
                    f"ATTN_DECODE split count={num_splits} does not match geometry={attn_splits}"
                )
            if not 0 <= split < max(1, num_splits):
                failures.append(f"ATTN_DECODE has invalid split={split}/{num_splits}")
            if 0 <= batch < batch_size and 0 <= kv_head < geometry.hkv and 0 <= split < max(1, num_splits):
                target = "attn_partial" if num_splits > 1 else "attn"
                record((target, layer, batch, kv_head, split), sm, op)

        elif op == tg.TgOp.ATTN_COMBINE:
            batch = int(inst[2])
            kv_head = int(inst[3])
            num_splits = int(inst[4])
            if not 0 <= batch < batch_size:
                failures.append(f"ATTN_COMBINE has invalid batch={batch}")
            if not 0 <= kv_head < geometry.hkv:
                failures.append(f"ATTN_COMBINE has invalid kv_head={kv_head}")
            if num_splits != attn_splits or num_splits <= 1:
                failures.append(
                    f"ATTN_COMBINE split count={num_splits} is inconsistent with geometry={attn_splits}"
                )
            if 0 <= batch < batch_size and 0 <= kv_head < geometry.hkv:
                record(("attn", layer, batch, kv_head, 0), sm, op)

        elif op == tg.TgOp.ADD_RMSNORM:
            batch = int(inst[2])
            if not 0 <= batch < batch_size:
                failures.append(f"ADD_RMSNORM has invalid batch={batch}")
            elif layer + 1 < geometry.num_layers + 1:
                record(("layer", layer + 1, batch), sm, op)

        elif op in (tg.TgOp.LM_ARGMAX,):
            pass
        else:
            failures.append(f"unknown opcode={op} on SM {sm}")

    # A segmented launch intentionally contains only a contiguous subset of
    # layers, so completeness is checked by the scheduler verifier. This audit
    # is deliberately limited to local range/ownership safety.

    return AuditReport(not failures, tuple(failures), checked_writes)


_Q_TILES_PER_KV_HEAD = 16
_KV_TILES_PER_KV_HEAD = 8
