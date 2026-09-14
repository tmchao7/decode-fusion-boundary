"""KV-cache index audit for the batch decode paths.

Two independent things are checked here, both without a GPU:

1. the device-side KV pointer arithmetic (mirrored from ``ldg_kv_ptr``) agrees
   with the canonical ``PagedKVCacheSpec.element_offset`` contract; and
2. every physical block the kernel can reach is a valid block, for *unequal*
   sequence lengths in one batch.

(2) is the important one.  ``ModelRunner.prepare_block_tables`` right-pads the
block table of short sequences so the batch is rectangular.  The torch path is
unaffected because it zeroes its graph buffer and bounds every read with
``cache_seqlens``/``context_lens``.  The megakernel path consumes the padded
table directly and bounds its attention loop with ``position + 1`` (the *longest*
sequence is not the issue -- each sequence uses its own position).  Padding with
-1 made ``physical_block`` negative, so the KV address ran backwards out of the
cache allocation.
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nanovllm.backends.megakernel.cache import PagedKVCacheSpec  # noqa: E402

MODEL_RUNNER = REPO_ROOT / "src" / "nanovllm" / "engine" / "model_runner.py"


def block_table_pad_value() -> int:
    """Read the padding value out of ``prepare_block_tables``.

    Importing ``model_runner`` needs torch, so the source is parsed instead.
    """

    import re

    source = MODEL_RUNNER.read_text(encoding="utf-8")
    match = re.search(
        r"seq\.block_table \+ \[(-?\d+)\] \* \(max_len - len\(seq\.block_table\)\)",
        source,
    )
    if match is None:
        raise AssertionError("prepare_block_tables padding rule changed")
    return int(match.group(1))


def ldg_kv_ptr_offset(
    *,
    kv_head: int,
    position: int,
    max_seq_len: int,
    block_table,
    block_size: int,
    num_kv_heads: int = 8,
    head_dim: int = 128,
) -> int:
    """Python mirror of ``ldg_kv_ptr`` in ``kernel.cu`` (element offset).

    Kept structurally identical to the CUDA so the arithmetic can be audited on
    a laptop.  ``block_table=None`` selects the contiguous (non-paged) path.
    """

    kv_size = num_kv_heads * head_dim
    if block_table is None:
        return kv_head * max_seq_len * head_dim + position * head_dim
    logical_block = position // block_size
    block_offset = position % block_size
    physical_block = block_table[logical_block]
    return (
        physical_block * block_size * kv_size
        + block_offset * kv_size
        + kv_head * head_dim
    )


def _cache_spec(num_blocks: int = 64, block_size: int = 16) -> PagedKVCacheSpec:
    return PagedKVCacheSpec(
        num_layers=28,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=8,
        head_dim=128,
        dtype_itemsize=2,
    )


class KvPointerMatchesSpecTests(unittest.TestCase):
    """The CUDA arithmetic must equal the documented cache contract."""

    def test_paged_offset_matches_paged_kv_cache_spec(self) -> None:
        spec = _cache_spec()
        block_table = [3, 7, 1, 0]
        for position in range(0, 4 * spec.block_size):
            for kv_head in (0, 3, 7):
                with self.subTest(position=position, kv_head=kv_head):
                    self.assertEqual(
                        ldg_kv_ptr_offset(
                            kv_head=kv_head,
                            position=position,
                            max_seq_len=spec.num_blocks * spec.block_size,
                            block_table=block_table,
                            block_size=spec.block_size,
                            num_kv_heads=spec.num_kv_heads,
                            head_dim=spec.head_dim,
                        ),
                        spec.element_offset(0, block_table, position, kv_head, 0),
                    )

    def test_contiguous_offset_matches_layer_stride_math(self) -> None:
        # Non-paged: [kv_head, position, head_dim] with max_seq_len as the
        # per-head stride. This is the shape the contiguous reference path uses.
        block_size, num_blocks = 16, 64
        max_seq_len = block_size * num_blocks
        offset = ldg_kv_ptr_offset(
            kv_head=2,
            position=5,
            max_seq_len=max_seq_len,
            block_table=None,
            block_size=block_size,
        )
        self.assertEqual(offset, 2 * max_seq_len * 128 + 5 * 128)


class PaddedBlockTableTests(unittest.TestCase):
    """Unequal sequence lengths must not produce an invalid physical block."""

    BLOCK_SIZE = 16
    NUM_BLOCKS = 64
    SHORT = [0, 1]              # 32 tokens
    LONG = [2, 3, 4, 5, 6]      # 80 tokens

    def _pad(self, pad: int) -> list[list[int]]:
        max_len = max(len(self.SHORT), len(self.LONG))
        return [
            table + [pad] * (max_len - len(table))
            for table in (self.SHORT, self.LONG)
        ]

    def _offsets(self, table, positions: int) -> list[int]:
        return [
            ldg_kv_ptr_offset(
                kv_head=kv_head,
                position=position,
                max_seq_len=self.NUM_BLOCKS * self.BLOCK_SIZE,
                block_table=table,
                block_size=self.BLOCK_SIZE,
            )
            for position in range(positions)
            for kv_head in (0, 7)
        ]

    def test_negative_padding_is_out_of_bounds(self) -> None:
        """Documents the defect: pad=-1 indexes before the cache allocation."""

        tables = self._pad(-1)
        self.assertEqual(tables[0][2], -1)
        offsets = self._offsets(tables[0], len(self.SHORT + [-1, -1, -1]) * self.BLOCK_SIZE)
        self.assertLess(
            min(offsets), 0, "pad=-1 was expected to produce a negative offset"
        )

    def test_zero_padding_keeps_every_offset_inside_the_cache(self) -> None:
        """The required property: any reachable read stays in-bounds."""

        spec = _cache_spec(num_blocks=self.NUM_BLOCKS, block_size=self.BLOCK_SIZE)
        tables = self._pad(0)
        for table in tables:
            offsets = self._offsets(table, len(table) * self.BLOCK_SIZE)
            with self.subTest(table=table):
                self.assertGreaterEqual(min(offsets), 0)
                self.assertLess(max(offsets), spec.total_elements)

    def test_engine_padding_is_a_valid_physical_block(self) -> None:
        """Pin the engine so -1 cannot come back."""

        pad = block_table_pad_value()
        self.assertGreaterEqual(
            pad,
            0,
            "prepare_block_tables pads with a negative physical block; the "
            "megakernel consumes the padded table directly and ldg_kv_ptr would "
            "index before the KV cache",
        )
        spec = _cache_spec(num_blocks=self.NUM_BLOCKS, block_size=self.BLOCK_SIZE)
        tables = self._pad(pad)
        for table in tables:
            offsets = self._offsets(table, len(table) * self.BLOCK_SIZE)
            self.assertGreaterEqual(min(offsets), 0)
            self.assertLess(max(offsets), spec.total_elements)


if __name__ == "__main__":
    unittest.main()
