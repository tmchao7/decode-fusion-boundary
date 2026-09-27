"""Cross-block-boundary position / slot-mapping contract tests.

These lock the three numbers that must not drift when the scheduler or the model
runner changes:

* **positions** -- RoPE positions of a prefill chunk and of a decode step;
* **cache length** -- ``position + 1`` for decode, ``end`` for prefill;
* **slot mapping** -- which flat paged-cache element each token's K/V lands in.

The interesting prompt lengths are 255 / 256 / 257 around
``kvcache_block_size = 256``: below the boundary, exactly on it, and one token
past it. The whole file runs on CPU (no torch, no CUDA) because
``nanovllm.engine.kv_layout`` owns the arithmetic and ``ModelRunner`` delegates
to it; the CUDA side of the same identity is locked separately by reading the
kernel source.
"""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine_fixtures import (  # noqa: E402
    BLOCK_SIZE,
    make_scheduler,
    make_sequence,
)
from nanovllm.engine.kv_layout import (  # noqa: E402
    decode_cache_len,
    decode_cache_seqlens,
    decode_position,
    decode_slot_mapping,
    pad_block_table,
    prefill_cache_len,
    prefill_positions,
    prefill_slot_mapping,
)
from nanovllm.engine.sequence import Sequence  # noqa: E402

PROMPT_LENGTHS = (255, 256, 257)


class EngineTestCase(unittest.TestCase):
    """Temp model dir + a fixed ``Sequence.block_size`` for every test."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        Sequence.block_size = BLOCK_SIZE

    @property
    def model_dir(self) -> str:
        return self._tmp.name

    def prefill(self, prompt_len: int, *, num_blocks: int = 64):
        """Add a fresh prompt and run one scheduler prefill step."""

        scheduler = make_scheduler(self.model_dir, num_blocks=num_blocks)
        seq = make_sequence([7] * prompt_len)
        scheduler.add(seq)
        seqs, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill, "a fresh prompt must be scheduled as prefill")
        self.assertEqual(seqs, [seq])
        return scheduler, seq

    def first_decode(self, prompt_len: int, *, num_blocks: int = 64):
        """Prefill, sample one token (postprocess), then schedule the decode."""

        scheduler, seq = self.prefill(prompt_len, num_blocks=num_blocks)
        scheduler.postprocess([seq], [42], True)
        return scheduler, seq


class PrefillBoundaryTests(EngineTestCase):
    def test_positions_and_block_table_lengths_around_the_boundary(self):
        for prompt_len in PROMPT_LENGTHS:
            expected_blocks = (prompt_len + BLOCK_SIZE - 1) // BLOCK_SIZE
            with self.subTest(prompt_len=prompt_len):
                _, seq = self.prefill(prompt_len)
                self.assertEqual(seq.num_scheduled_tokens, prompt_len)
                self.assertEqual(seq.num_tokens, prompt_len)
                self.assertEqual(seq.num_cached_tokens, 0)
                self.assertEqual(seq.num_blocks, expected_blocks)
                self.assertEqual(len(seq.block_table), expected_blocks)
                self.assertEqual(
                    seq.last_block_num_tokens,
                    prompt_len - (expected_blocks - 1) * BLOCK_SIZE,
                )
                # Locked prefill contract: positions are [0, prompt_len).
                self.assertEqual(
                    list(
                        prefill_positions(
                            seq.num_cached_tokens, seq.num_scheduled_tokens
                        )
                    ),
                    list(range(prompt_len)),
                )
                self.assertEqual(prefill_cache_len(0, prompt_len), prompt_len)

    def test_255_stays_inside_one_block(self):
        _, seq = self.prefill(255)
        self.assertEqual(len(seq.block_table), 1)
        self.assertEqual(seq.last_block_num_tokens, 255)

    def test_256_ends_exactly_on_a_block_boundary(self):
        _, seq = self.prefill(256)
        self.assertEqual(len(seq.block_table), 1)
        self.assertEqual(seq.last_block_num_tokens, 256)

    def test_257_spans_two_blocks(self):
        _, seq = self.prefill(257)
        self.assertEqual(len(seq.block_table), 2)
        self.assertEqual(seq.last_block_num_tokens, 1)

    def test_slot_mapping_lands_each_position_in_its_own_block(self):
        for prompt_len in PROMPT_LENGTHS:
            with self.subTest(prompt_len=prompt_len):
                _, seq = self.prefill(prompt_len)
                slots = prefill_slot_mapping(
                    seq.block_table, 0, prompt_len, BLOCK_SIZE
                )
                self.assertEqual(len(slots), prompt_len)
                for position, slot in enumerate(slots):
                    block, offset = divmod(position, BLOCK_SIZE)
                    self.assertEqual(
                        slot,
                        seq.block_table[block] * BLOCK_SIZE + offset,
                        f"position {position} must land in block {block} "
                        f"offset {offset}",
                    )
                self.assertEqual(len(set(slots)), prompt_len)

    def test_slot_mapping_crosses_the_boundary_contiguously(self):
        _, seq = self.prefill(257)
        slots = prefill_slot_mapping(seq.block_table, 0, 257, BLOCK_SIZE)
        self.assertEqual(slots[255], seq.block_table[0] * BLOCK_SIZE + 255)
        self.assertEqual(slots[256], seq.block_table[1] * BLOCK_SIZE + 0)

    def test_chunked_prefill_continues_the_cached_prefix(self):
        """A chunked / prefix-cached prefill starts where the cache ends."""

        _, seq = self.prefill(300)
        suffix = prefill_slot_mapping(seq.block_table, 256, 300, BLOCK_SIZE)
        self.assertEqual(len(suffix), 44)
        self.assertEqual(
            list(prefill_positions(256, 44)), list(range(256, 300))
        )
        self.assertEqual(prefill_cache_len(256, 44), 300)
        self.assertEqual(suffix[0], seq.block_table[1] * BLOCK_SIZE + 0)
        self.assertEqual(suffix[-1], seq.block_table[1] * BLOCK_SIZE + 43)


class FirstDecodeBoundaryTests(EngineTestCase):
    def test_first_decode_position_is_the_prompt_length(self):
        for prompt_len in PROMPT_LENGTHS:
            with self.subTest(prompt_len=prompt_len):
                scheduler, seq = self.first_decode(prompt_len)
                self.assertEqual(seq.num_tokens, prompt_len + 1)
                self.assertEqual(seq.num_cached_tokens, prompt_len)
                self.assertEqual(seq.num_prompt_tokens, prompt_len)

                scheduled, is_prefill = scheduler.schedule()
                self.assertFalse(is_prefill)
                self.assertEqual(scheduled, [seq])

                num_tokens = len(seq)
                position = decode_position(num_tokens)
                self.assertEqual(position, prompt_len)
                self.assertEqual(position, seq.num_prompt_tokens)
                # cache_len == position + 1 is the attention loop bound; the
                # kernel is handed the pre-decode length (== position).
                self.assertEqual(decode_cache_len(num_tokens), position + 1)
                self.assertEqual(decode_cache_seqlens(num_tokens), position)
                self.assertEqual(
                    decode_slot_mapping(seq.block_table, num_tokens, BLOCK_SIZE),
                    seq.block_table[position // BLOCK_SIZE] * BLOCK_SIZE
                    + position % BLOCK_SIZE,
                )

    def test_decode_at_the_boundary_appends_the_next_block(self):
        scheduler, seq = self.first_decode(256)
        self.assertEqual(len(seq.block_table), 1, "256-token prefill needs one block")

        scheduler.schedule()

        self.assertEqual(len(seq.block_table), 2, "position 256 needs a new block")
        self.assertEqual(decode_position(len(seq)), 256)
        self.assertEqual(
            decode_slot_mapping(seq.block_table, len(seq), BLOCK_SIZE),
            seq.block_table[1] * BLOCK_SIZE,
        )

    def test_decode_below_the_boundary_reuses_the_last_block(self):
        scheduler, seq = self.first_decode(255)
        before = list(seq.block_table)

        scheduler.schedule()

        self.assertEqual(seq.block_table, before)
        self.assertEqual(decode_position(len(seq)), 255)
        self.assertEqual(
            decode_slot_mapping(seq.block_table, len(seq), BLOCK_SIZE),
            seq.block_table[0] * BLOCK_SIZE + 255,
        )

    def test_prompt_past_the_boundary_keeps_both_blocks_while_decoding(self):
        scheduler, seq = self.first_decode(257)
        before = list(seq.block_table)
        self.assertEqual(len(before), 2)

        scheduler.schedule()

        self.assertEqual(seq.block_table, before)
        self.assertEqual(decode_position(len(seq)), 257)
        self.assertEqual(
            decode_slot_mapping(seq.block_table, len(seq), BLOCK_SIZE),
            seq.block_table[1] * BLOCK_SIZE + 1,
        )

    def test_second_decode_crosses_into_the_next_block(self):
        """Sequence of 256 then two decodes: 256 -> block 1, then 257 -> block 1."""

        scheduler, seq = self.first_decode(256)
        scheduler.schedule()  # first decode, position 256
        scheduler.postprocess([seq], [43], False)
        self.assertEqual(seq.num_tokens, 258)

        scheduler.schedule()  # second decode, position 257

        self.assertEqual(decode_position(len(seq)), 257)
        self.assertEqual(len(seq.block_table), 2)
        self.assertEqual(
            decode_slot_mapping(seq.block_table, len(seq), BLOCK_SIZE),
            seq.block_table[1] * BLOCK_SIZE + 1,
        )


class SlotMappingGuardTests(unittest.TestCase):
    def test_prefill_slot_mapping_rejects_a_short_block_table(self):
        with self.assertRaisesRegex(RuntimeError, "need 2 blocks"):
            prefill_slot_mapping([3], 0, 257, BLOCK_SIZE)

    def test_prefill_slot_mapping_rejects_an_unallocated_table(self):
        with self.assertRaisesRegex(RuntimeError, "allocated block table"):
            prefill_slot_mapping([], 0, 4, BLOCK_SIZE)

    def test_decode_slot_mapping_rejects_a_missing_block(self):
        with self.assertRaisesRegex(RuntimeError, "may_append"):
            decode_slot_mapping([3], 257, BLOCK_SIZE)

    def test_decode_slot_mapping_rejects_a_stale_extra_block(self):
        with self.assertRaisesRegex(RuntimeError, "block table has 3"):
            decode_slot_mapping([3, 4, 5], 257, BLOCK_SIZE)

    def test_empty_prefill_range_maps_to_no_slots(self):
        self.assertEqual(prefill_slot_mapping([3], 4, 4, BLOCK_SIZE), [])

    def test_invalid_arguments_are_rejected(self):
        with self.assertRaises(ValueError):
            prefill_slot_mapping([3], 5, 4, BLOCK_SIZE)
        with self.assertRaises(ValueError):
            prefill_slot_mapping([3], 0, 1, 0)
        with self.assertRaises(ValueError):
            prefill_positions(-1, 4)
        with self.assertRaises(ValueError):
            prefill_positions(0, -1)
        with self.assertRaises(ValueError):
            decode_position(0)
        with self.assertRaises(ValueError):
            decode_slot_mapping([1], 0, BLOCK_SIZE)
        with self.assertRaises(ValueError):
            decode_slot_mapping([1], 1, 0)

    def test_decode_slot_mapping_matches_the_inlined_legacy_expression(self):
        """Regression guard for the extraction: same result as the old code.

        ``ModelRunner.prepare_decode`` used
        ``seq.block_table[-1] * block_size + seq.last_block_num_tokens - 1``.
        """

        for num_tokens in (1, 255, 256, 257, 512, 513):
            with self.subTest(num_tokens=num_tokens):
                num_blocks = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
                block_table = list(range(100, 100 + num_blocks))
                seq = make_sequence([7] * num_tokens)
                seq.block_table = list(block_table)
                legacy = (
                    block_table[-1] * BLOCK_SIZE + seq.last_block_num_tokens - 1
                )
                self.assertEqual(
                    decode_slot_mapping(block_table, num_tokens, BLOCK_SIZE),
                    legacy,
                )


class BlockTablePaddingTests(unittest.TestCase):
    def test_short_tables_are_right_padded_with_physical_block_zero(self):
        self.assertEqual(pad_block_table([7, 9], 4), [7, 9, 0, 0])
        self.assertEqual(pad_block_table([], 2), [0, 0])
        self.assertEqual(pad_block_table([1, 2], 2), [1, 2])

    def test_oversized_tables_are_rejected(self):
        with self.assertRaises(ValueError):
            pad_block_table([1, 2, 3], 2)


class KernelSideOfTheContractTests(unittest.TestCase):
    """The CUDA side of the identity, locked without a GPU.

    ``tg_attn_decode_op`` is the authority on what ``cache_seqlens`` means; if
    the kernel changes that meaning, the torch / task-graph split breaks while
    every Python test above still passes.
    """

    def test_kernel_derives_cache_len_from_the_pre_decode_length(self):
        kernel = (
            ROOT
            / "src/nanovllm/backends/megakernel/csrc/taskgraph_kernel.cu"
        ).read_text(encoding="utf-8")
        self.assertIn("int cache_len = g.cache_seqlens[batch] + 1;", kernel)
        self.assertIn("int position = cache_len - 1;", kernel)
        self.assertIn("pos_end = cache_len;", kernel)

    def test_model_runner_delegates_to_kv_layout(self):
        """Anti-drift: prepare_prefill/prepare_decode must call the helpers."""

        runner = (ROOT / "src/nanovllm/engine/model_runner.py").read_text(
            encoding="utf-8"
        )
        for call in (
            "prefill_slot_mapping(",
            "decode_slot_mapping(",
            "pad_block_table(",
            "prefill_positions(",
            "decode_position(",
        ):
            with self.subTest(call=call):
                self.assertIn(call, runner)


if __name__ == "__main__":
    unittest.main()


class ContractValidationTests(unittest.TestCase):
    """Every helper in kv_layout rejects the same illegal inputs.

    These three numbers feed ``cu_seqlens_k``, RoPE and the K/V write; a negative
    or zero value here is a silent wrong answer rather than a crash, which is why
    the module validates instead of trusting its callers.
    """

    def test_prefill_helpers_reject_negative_inputs(self) -> None:
        for call in (
            lambda: prefill_positions(-1, 4),
            lambda: prefill_positions(0, -4),
            lambda: prefill_cache_len(-1, 4),
            lambda: prefill_cache_len(0, -4),
        ):
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()

    def test_decode_helpers_reject_a_zero_length_sequence(self) -> None:
        for call in (
            lambda: decode_position(0),
            lambda: decode_cache_seqlens(0),
            lambda: decode_cache_len(0),
        ):
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()

    def test_slot_mapping_rejects_a_negative_block_size(self) -> None:
        with self.assertRaises(ValueError):
            decode_slot_mapping([0], 1, 0)
        with self.assertRaises(ValueError):
            prefill_slot_mapping([0], 0, 1, 0)
