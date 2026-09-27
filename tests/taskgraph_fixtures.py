"""Hand-built task-graph schedules for the CPU-side verifier and cost-model tests.

The verifier and the cost model both need to hand-build instruction streams that
a real geometry would never emit -- a two-instruction cycle, a bar nobody
produces, a field with the wrong value. Duplicating that construction in each
test module meant two definitions of ``instruction()`` and ``RawSchedule`` that
could drift from the ABI the same way any other copy does.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from nanovllm.backends.megakernel import schedule as tg  # noqa: E402

WIDTH = tg.TG_INST_WIDTH


def instruction(opcode: int, **fields: int) -> list[int]:
    """Build one raw instruction row; ``f3=1`` sets field 3."""

    row = [0] * WIDTH
    row[0] = opcode
    for name, value in fields.items():
        row[int(name[1:])] = value
    return row


class RawSchedule:
    """Minimal stand-in for ``TgSchedule`` so tests can hand-build streams."""

    def __init__(self, rows: list[list[int]], per_sm: list[int], max_inst: int) -> None:
        self.flat_inst_buf = [value for row in rows for value in row]
        self.num_inst_per_sm = per_sm
        self.max_inst = max_inst
        self.bar_dims = None


#: A dependency cycle whose arrival counts exactly cancel: each bar produces as
#: many arrivals as its largest wait threshold, so only the cycle itself is wrong.
BALANCED_CYCLE: list[list[int]] = [
    instruction(tg.TgOp.QKV_PROJ, f5=0, f6=1, f7=0, f9=1),
    instruction(tg.TgOp.ATTN_DECODE, f7=0, f8=1, f9=0),
    instruction(tg.TgOp.O_PROJ, f6=0, f7=1, f8=0, f9=1),
    instruction(tg.TgOp.ADD_RMSNORM, f3=0, f4=1, f5=0, f6=0, f7=0),
]

#: A consistent chain: every produced count equals the threshold that consumes it.
BALANCED_CHAIN: list[list[int]] = [
    instruction(tg.TgOp.QKV_PROJ, f5=0, f6=1, f7=0, f9=2),
    instruction(tg.TgOp.ATTN_DECODE, f7=0, f8=2, f9=0),
    instruction(tg.TgOp.O_PROJ, f6=0, f7=1, f8=0, f9=1),
    instruction(tg.TgOp.ADD_RMSNORM, f3=0, f4=1, f5=0, f6=0, f7=0),
]
