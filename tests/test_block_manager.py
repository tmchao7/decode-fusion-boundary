"""Behavioral tests for the paged KV block manager.

The pre-server suite previously only asserted that ``block_manager.py`` *contains*
certain identifiers (a source contract). These tests drive the real
``BlockManager`` with real ``Sequence`` objects and lock the behavior the
scheduler and the prefix cache depend on:

* one block per ``block_size`` tokens, and only *completed* blocks are cacheable;
* the prefix hash chain (block N's hash depends on block N-1's tokens too, so a
  shared first block with a different second block is not a false hit);
* reference counting across sharing sequences, and reuse of a cached block that
  is currently unowned (the actual prefix-cache win);
* deallocation, free-list reuse and the peak-usage counter;
* ``can_append`` / ``may_append`` only allocate when a decode crosses a block
  boundary -- the behavior behind the 255/256/257 tests in
  ``test_kv_block_boundary.py``.

No torch, no CUDA: ``xxhash`` and ``numpy`` are the only dependencies.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine_fixtures import BLOCK_SIZE, make_sequence  # noqa: E402
from nanovllm.engine.block_manager import BlockManager  # noqa: E402
from nanovllm.engine.sequence import Sequence  # noqa: E402

NUM_BLOCKS = 8


def hash_prompt_blocks(block_manager: BlockManager, seq: Sequence) -> None:
    """Register a sequence's completed blocks, as ``Scheduler.postprocess`` does.

    ``BlockManager.hash_blocks`` hashes blocks in
    ``[num_cached_tokens, num_cached_tokens + num_scheduled_tokens)`` and
    ``postprocess`` then folds the scheduled tokens into ``num_cached_tokens``.
    This helper performs exactly that bookkeeping for a full prompt prefill.
    """

    seq.num_scheduled_tokens = seq.num_tokens - seq.num_cached_tokens
    block_manager.hash_blocks(seq)
    seq.num_cached_tokens += seq.num_scheduled_tokens
    seq.num_scheduled_tokens = 0


class BlockManagerTestCase(unittest.TestCase):
    def setUp(self):
        Sequence.block_size = BLOCK_SIZE

    def new_manager(self, num_blocks: int = NUM_BLOCKS) -> BlockManager:
        return BlockManager(num_blocks, BLOCK_SIZE)

    def prefill(self, block_manager: BlockManager, token_ids) -> Sequence:
        """Allocate + cache-hash one complete prompt, returning the sequence."""

        seq = make_sequence(token_ids)
        cached = block_manager.can_allocate(seq)
        self.assertNotEqual(cached, -1, "fresh prompts must fit a fresh manager")
        block_manager.allocate(seq, cached)
        hash_prompt_blocks(block_manager, seq)
        return seq

    def cache_counters(self, block_manager: BlockManager) -> tuple[int, int]:
        """``(hits, misses)`` for assertions on a measured phase, not totals."""

        stats = block_manager.stats()
        return stats["prefix_cache_hits"], stats["prefix_cache_misses"]


class AllocationTests(BlockManagerTestCase):
    def test_allocates_one_block_per_block_size_tokens(self):
        for num_tokens, expected_blocks in (
            (1, 1),
            (255, 1),
            (256, 1),
            (257, 2),
            (512, 2),
            (513, 3),
        ):
            with self.subTest(num_tokens=num_tokens):
                block_manager = self.new_manager()
                seq = make_sequence([1] * num_tokens)
                self.assertEqual(block_manager.can_allocate(seq), 0)
                block_manager.allocate(seq, 0)
                self.assertEqual(seq.num_blocks, expected_blocks)
                self.assertEqual(len(seq.block_table), expected_blocks)
                self.assertEqual(seq.num_cached_tokens, 0)
                self.assertEqual(
                    block_manager.stats()["allocated_blocks"], expected_blocks
                )
                self.assertEqual(
                    block_manager.stats()["free_blocks"],
                    NUM_BLOCKS - expected_blocks,
                )

    def test_reports_pressure_instead_of_over_allocating(self):
        block_manager = self.new_manager(num_blocks=2)
        self.prefill(block_manager, [1] * 256)
        big = make_sequence([2] * 512)

        self.assertEqual(block_manager.can_allocate(big), -1)
        self.assertEqual(block_manager.stats()["free_blocks"], 1)

    def test_prefix_hit_can_remove_the_pressure(self):
        """A prefix hit lowers the block demand, so a big prompt can still fit."""

        block_manager = self.new_manager(num_blocks=2)
        self.prefill(block_manager, [1] * 256)
        sharing = make_sequence([1] * 512)

        # 2 blocks wanted, 1 free -- but block 0 is a prefix-cache hit and the
        # sequence only needs one new block.
        self.assertEqual(block_manager.can_allocate(sharing), 1)

    def test_can_allocate_recovers_after_a_deallocation(self):
        block_manager = self.new_manager(num_blocks=2)
        owner = self.prefill(block_manager, [1] * 256)
        big = make_sequence([2] * 512)
        self.assertEqual(block_manager.can_allocate(big), -1)

        block_manager.deallocate(owner)

        self.assertEqual(block_manager.can_allocate(big), 0)
        self.assertEqual(block_manager.stats()["free_blocks"], 2)


class PrefixCacheTests(BlockManagerTestCase):
    def test_identical_prompt_reuses_every_completed_block(self):
        block_manager = self.new_manager()
        first = self.prefill(block_manager, [1] * 768)
        hits_before, misses_before = self.cache_counters(block_manager)

        second = make_sequence([1] * 768)
        cached = block_manager.can_allocate(second)

        # 3 blocks, but the last one is never reused: it may be incomplete.
        self.assertEqual(cached, 2, "only completed blocks are prefix-cacheable")
        block_manager.allocate(second, cached)

        self.assertEqual(second.num_cached_tokens, 2 * BLOCK_SIZE)
        hits_after, misses_after = self.cache_counters(block_manager)
        self.assertEqual(hits_after - hits_before, 2, "two blocks reused")
        self.assertEqual(misses_after - misses_before, 1, "one block allocated")
        self.assertEqual(block_manager.blocks[second.block_table[0]].ref_count, 2)
        self.assertEqual(block_manager.blocks[second.block_table[1]].ref_count, 2)
        self.assertNotIn(second.block_table[2], first.block_table)

    def test_prefix_chain_rejects_a_different_second_block(self):
        """The hash chain must include the previous block's identity."""

        block_manager = self.new_manager()
        self.prefill(block_manager, [1] * 256 + [2] * 256 + [4] * 256)

        same_prefix = make_sequence([1] * 256 + [2] * 256 + [9] * 256)
        self.assertEqual(block_manager.can_allocate(same_prefix), 2)

        diverged = make_sequence([1] * 256 + [7] * 256 + [4] * 256)
        self.assertEqual(
            block_manager.can_allocate(diverged),
            1,
            "a different second block must not inherit the first block's hit",
        )

    def test_prefix_cache_misses_for_unrelated_content(self):
        block_manager = self.new_manager()
        self.prefill(block_manager, [1] * 512)
        hits_before, misses_before = self.cache_counters(block_manager)

        other = make_sequence([5] * 512)
        self.assertEqual(block_manager.can_allocate(other), 0)
        block_manager.allocate(other, 0)

        hits_after, misses_after = self.cache_counters(block_manager)
        self.assertEqual(hits_after - hits_before, 0)
        self.assertEqual(misses_after - misses_before, 2)

    def test_partial_last_block_is_not_cached(self):
        block_manager = self.new_manager()
        seq = self.prefill(block_manager, [1] * 300)

        # Block 0 (256 tokens) is complete; block 1 has only 44 tokens.
        self.assertNotEqual(block_manager.blocks[seq.block_table[0]].hash, -1)
        self.assertEqual(block_manager.blocks[seq.block_table[1]].hash, -1)
        self.assertEqual(len(block_manager.hash_to_block_id), 1)

    def test_a_block_shorter_than_the_block_size_is_never_hashed(self):
        block_manager = self.new_manager()
        seq = make_sequence([1] * 100)
        block_manager.allocate(seq, 0)

        hash_prompt_blocks(block_manager, seq)

        self.assertEqual(block_manager.hash_to_block_id, {})

    def test_freed_block_stays_in_the_prefix_cache_and_is_reused(self):
        """The prefix-cache win: reuse a cached block that nobody owns."""

        block_manager = self.new_manager(num_blocks=4)
        owner = self.prefill(block_manager, [1] * 257)
        cached_block_id = owner.block_table[0]
        cached_hash = block_manager.blocks[cached_block_id].hash

        block_manager.deallocate(owner)

        self.assertNotIn(cached_block_id, block_manager.used_block_ids)
        self.assertEqual(
            block_manager.hash_to_block_id.get(cached_hash),
            cached_block_id,
            "deallocation must keep the hash entry; the KV data is still valid",
        )

        second = make_sequence([1] * 257)
        cached = block_manager.can_allocate(second)
        self.assertEqual(cached, 1)
        block_manager.allocate(second, cached)

        self.assertEqual(second.block_table[0], cached_block_id)
        self.assertEqual(block_manager.blocks[cached_block_id].ref_count, 1)
        self.assertEqual(
            block_manager.stats()["allocated_blocks"], 2, "block 1 is new"
        )


class RefCountAndDeallocationTests(BlockManagerTestCase):
    def test_sharing_sequences_increment_then_release_the_ref_count(self):
        block_manager = self.new_manager()
        seqs = [self.prefill(block_manager, [1] * 257) for _ in range(3)]

        shared = seqs[0].block_table[0]
        self.assertEqual(
            [seq.block_table[0] for seq in seqs], [shared, shared, shared]
        )
        self.assertEqual(block_manager.blocks[shared].ref_count, 3)
        self.assertEqual(
            block_manager.stats()["peak_used_blocks"], 4, "1 shared + 3 private"
        )

        block_manager.deallocate(seqs[0])
        self.assertEqual(block_manager.blocks[shared].ref_count, 2)
        self.assertIn(shared, block_manager.used_block_ids)

        block_manager.deallocate(seqs[1])
        block_manager.deallocate(seqs[2])

        self.assertEqual(block_manager.blocks[shared].ref_count, 0)
        self.assertEqual(block_manager.used_block_ids, set())
        self.assertEqual(len(block_manager.free_block_ids), NUM_BLOCKS)
        # 1 + 1 + 2 blocks were actually released; the shared one only once.
        self.assertEqual(block_manager.stats()["freed_blocks"], 4)

    def test_deallocate_clears_the_table_and_cached_tokens(self):
        block_manager = self.new_manager()
        seq = make_sequence([1] * 300)
        block_manager.allocate(seq, 0)
        self.assertEqual(seq.num_cached_tokens, 0, "fresh prompt has no prefix hit")
        hash_prompt_blocks(block_manager, seq)
        self.assertEqual(seq.num_cached_tokens, 300, "the whole prompt was computed")

        block_manager.deallocate(seq)

        self.assertEqual(seq.block_table, [])
        self.assertEqual(seq.num_cached_tokens, 0)
        self.assertEqual(block_manager.used_block_ids, set())

    def test_blocks_are_reused_after_being_released(self):
        block_manager = self.new_manager(num_blocks=2)
        owner = self.prefill(block_manager, [1] * 256)
        released = owner.block_table[0]

        block_manager.deallocate(owner)
        self.assertIn(released, block_manager.free_block_ids)

        first = self.prefill(block_manager, [2] * 256)
        second = self.prefill(block_manager, [3] * 256)

        self.assertIn(
            released,
            {first.block_table[0], second.block_table[0]},
            "the released block must return to the free list and be reused",
        )
        self.assertEqual(len(block_manager.free_block_ids), 0)
        self.assertEqual(block_manager.stats()["allocated_blocks"], 2)


class AppendTests(BlockManagerTestCase):
    def test_no_new_block_is_needed_at_a_block_boundary(self):
        block_manager = self.new_manager(num_blocks=1)
        seq = self.prefill(block_manager, [1] * 256)
        self.assertEqual(len(seq.block_table), 1)
        self.assertEqual(len(block_manager.free_block_ids), 0)

        # len(seq) == 256: 256 % 256 == 0, so the next decode still fits.
        self.assertTrue(block_manager.can_append(seq))
        block_manager.may_append(seq)

        self.assertEqual(len(seq.block_table), 1)
        self.assertEqual(block_manager.stats()["allocated_blocks"], 1)

    def test_below_a_boundary_never_needs_a_block_even_under_pressure(self):
        block_manager = self.new_manager(num_blocks=1)
        seq = self.prefill(block_manager, [1] * 255)
        self.assertEqual(len(seq.block_table), 1)

        # len(seq) == 255, so only 255 % 256 == 255 -> no new block required.
        self.assertTrue(block_manager.can_append(seq))
        block_manager.may_append(seq)
        self.assertEqual(len(seq.block_table), 1)

    def test_crossing_a_boundary_requires_and_takes_a_block(self):
        block_manager = self.new_manager(num_blocks=2)
        seq = self.prefill(block_manager, [1] * 256)

        seq.append_token(2)
        self.assertEqual(len(seq), 257)
        self.assertTrue(block_manager.can_append(seq))
        block_manager.may_append(seq)

        self.assertEqual(len(seq.block_table), 2)
        self.assertEqual(block_manager.stats()["allocated_blocks"], 2)

    def test_pressure_is_reported_when_no_block_is_free(self):
        block_manager = self.new_manager(num_blocks=1)
        seq = self.prefill(block_manager, [1] * 256)

        seq.append_token(2)

        self.assertFalse(block_manager.can_append(seq))
        self.assertEqual(block_manager.stats()["free_blocks"], 0)


class StatsTests(BlockManagerTestCase):
    def test_reports_layout_and_counters(self):
        block_manager = self.new_manager()

        stats = block_manager.stats()

        self.assertEqual(stats["block_size"], BLOCK_SIZE)
        self.assertEqual(stats["allocated_blocks"], 0)
        self.assertEqual(stats["free_blocks"], NUM_BLOCKS)
        self.assertEqual(stats["peak_used_blocks"], 0)
        self.assertEqual(stats["prefix_cache_hits"], 0)
        self.assertEqual(stats["prefix_cache_misses"], 0)

    def test_peak_usage_is_monotone(self):
        block_manager = self.new_manager(num_blocks=8)
        first = self.prefill(block_manager, [1] * 512)
        self.assertEqual(block_manager.stats()["peak_used_blocks"], 2)

        second = self.prefill(block_manager, [9] * 512)
        self.assertEqual(block_manager.stats()["peak_used_blocks"], 4)

        block_manager.deallocate(first)
        block_manager.deallocate(second)

        self.assertEqual(block_manager.stats()["allocated_blocks"], 0)
        self.assertEqual(
            block_manager.stats()["peak_used_blocks"], 4, "peak is historical"
        )


if __name__ == "__main__":
    unittest.main()
