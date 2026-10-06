"""Byte-identical lock on the generated instruction stream.

Why a hash fixture rather than more property tests
-------------------------------------------------
``tests/test_taskgraph_abi.py`` already checks many *properties* of the schedule:
bar accounting, packed-range coverage, placement balance, cross-segment handoff. None
of those notices a change that is legal in isolation but alters the emitted bytes --
and the emitted bytes are what the kernel executes.

That matters for the ``docs/22`` refactor series. S1 (one model spec) and S2 (derived
ABI counts) both claim "no geometry changed", and the schedule scaffolding is next:
the plan is to replace the four per-wave builders and their three copy-pasted packers
with one ``Wave`` abstraction, then hang the sparse opcodes off it. A refactor of that
shape has exactly one safe acceptance test -- **the output must be identical** -- so
this file pins it.

Reading a failure
-----------------
A diff here is not automatically a bug; it is automatically a *decision*. If the
change is intended, regenerate the fixture deliberately::

    python -m tests.regenerate_schedule_hashes

and say in the commit message which bytes moved and why. If it was not intended, the
sub-case name identifies which (batch, num_sms, attn_splits) combination changed.
"""

import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nanovllm.backends.megakernel import schedule as tg  # noqa: E402
from nanovllm.model_spec import QWEN3_06B  # noqa: E402

NUM_LAYERS = QWEN3_06B.num_layers
NUM_KV_HEADS = QWEN3_06B.num_kv_heads

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "schedule_hashes.json"

#: The grid the fixture covers. Kept here (not only in the fixture) so a shrinking
#: grid is visible in review: a hashes file that quietly lost half its entries would
#: still "pass".
BATCH_SIZES = (1, 2, 4, 8)
NUM_SMS = (1, 4, 108, 170)
ATTN_SPLITS = (1, 2)


def schedule_digest(geo: tg.TgGeometry, num_sms: int) -> tuple[str, tg.TgSchedule]:
    """SHA-256 over the serialized instruction buffer, plus the schedule itself."""

    schedule = tg.build_schedule(geo, num_sms=num_sms)
    blob = bytearray()
    for word in schedule.flat_inst_buf:
        blob += int(word & 0xFFFFFFFF).to_bytes(4, "little", signed=False)
    return hashlib.sha256(bytes(blob)).hexdigest(), schedule


def key(batch_size: int, num_sms: int, attn_splits: int) -> str:
    return f"bs{batch_size}_sms{num_sms}_sp{attn_splits}"


def expected_hashes() -> dict[str, dict[str, object]]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class ScheduleFingerprintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.expected = expected_hashes()

    def test_the_fixture_covers_the_whole_grid(self) -> None:
        """A fixture that lost entries would still pass the per-case checks."""

        wanted = {
            key(bs, sms, splits)
            for bs in BATCH_SIZES
            for sms in NUM_SMS
            for splits in ATTN_SPLITS
        }
        self.assertEqual(set(self.expected), wanted)

    def test_the_instruction_stream_is_byte_identical(self) -> None:
        for bs in BATCH_SIZES:
            for sms in NUM_SMS:
                for splits in ATTN_SPLITS:
                    name = key(bs, sms, splits)
                    with self.subTest(config=name):
                        geo = tg.default_geometry(bs, attn_splits=splits)
                        digest, schedule = schedule_digest(geo, sms)
                        want = self.expected[name]
                        self.assertEqual(
                            schedule.total_instructions,
                            want["total_instructions"],
                            "instruction count changed",
                        )
                        self.assertEqual(schedule.max_inst, want["max_inst"])
                        self.assertEqual(
                            digest,
                            want["sha256"],
                            f"the emitted instruction bytes for {name} changed; if "
                            f"that is intended, regenerate the fixture and say which "
                            f"bytes moved",
                        )

    def test_bar_geometry_is_part_of_the_fingerprint(self) -> None:
        """Bar lengths feed the launch descriptor, so they are not incidental.

        Checked separately from the hash because a bar-dimension change can leave the
        instruction bytes alone while changing how much scratch the host sizes and
        zeroes -- which the hash cannot see.
        """

        for bs in BATCH_SIZES:
            for splits in ATTN_SPLITS:
                with self.subTest(batch=bs, attn_splits=splits):
                    geo = tg.default_geometry(bs, attn_splits=splits)
                    dims = tg.build_schedule(geo, num_sms=108).bar_dims
                    # Every declared bar must be sized, including the ones this
                    # geometry never uses: the kernel's zero-region pass walks the
                    # whole struct, so a missing size is an out-of-bounds zeroing.
                    for name in tg.BAR_FIELDS:
                        with self.subTest(bar=name):
                            self.assertGreaterEqual(getattr(dims, name), 0)
                    self.assertEqual(dims.bar_layer, NUM_LAYERS + 1)
                    self.assertEqual(dims.bar_qkv, NUM_LAYERS * NUM_KV_HEADS)

    def test_g_one_is_the_single_launch_baseline(self) -> None:
        """``fusion_groups = 1`` must be byte-identical to the unsegmented stream."""

        for bs in BATCH_SIZES:
            with self.subTest(batch=bs):
                geo = tg.default_geometry(bs)
                whole = tg.build_schedule(geo, num_sms=108)
                segmented = tg.build_schedule(
                    geo, num_sms=108, layer_range=(0, NUM_LAYERS)
                )
                self.assertEqual(whole.flat_inst_buf, segmented.flat_inst_buf)

    def test_attn_splits_two_adds_the_combine_wave(self) -> None:
        """A documented structural difference, so a hash change is interpretable."""

        geo1 = tg.default_geometry(1, attn_splits=1)
        geo2 = tg.default_geometry(1, attn_splits=2)
        s1 = tg.build_schedule(geo1, num_sms=108)
        s2 = tg.build_schedule(geo2, num_sms=108)
        self.assertEqual(s1.bar_dims.bar_combine, 0)
        self.assertEqual(s2.bar_dims.bar_combine, NUM_LAYERS * 1 * NUM_KV_HEADS)
        self.assertGreater(s2.total_instructions, s1.total_instructions)


if __name__ == "__main__":
    unittest.main()
