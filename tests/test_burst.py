import unittest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from nanovllm.sampling_params import SamplingParams


def make_scheduler(num_blocks=64, block_size=256, max_num_seqs=8, eos=0):
    from types import SimpleNamespace

    config = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=4096,
        eos=eos,
        kvcache_block_size=block_size,
        num_kvcache_blocks=num_blocks,
    )
    return Scheduler(config)


def make_seq(num_tokens, max_tokens=128, temperature=0.0):
    sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    seq = Sequence(list(range(1, num_tokens + 1)), sp)
    seq.block_size = 256
    return seq


def fully_prefill(sched, seq):
    seq.status = SequenceStatus.RUNNING
    sched.block_manager.allocate(seq, num_cached_blocks=0)
    seq.num_cached_tokens = len(seq)


class CanAllocateBurstTests(unittest.TestCase):
    def setUp(self):
        self.bm = BlockManager(num_blocks=10, block_size=256)

    def test_returns_true_when_enough_free_blocks(self):
        seq = make_seq(10)
        self.bm.allocate(seq, num_cached_blocks=0)
        self.assertTrue(self.bm.can_allocate_burst(seq, 64))

    def test_returns_false_when_not_enough_free_blocks(self):
        seq = make_seq(10)
        self.bm.allocate(seq, num_cached_blocks=0)
        for _ in range(9):
            other = make_seq(10)
            self.bm.allocate(other, num_cached_blocks=0)
        self.assertFalse(self.bm.can_allocate_burst(seq, 512))

    def test_returns_true_when_no_new_blocks_needed(self):
        seq = make_seq(600)
        self.bm.allocate(seq, num_cached_blocks=0)
        self.assertTrue(self.bm.can_allocate_burst(seq, 64))


class TrimBurstTests(unittest.TestCase):
    def test_frees_unused_burst_blocks(self):
        bm = BlockManager(num_blocks=20, block_size=256)
        seq = make_seq(10)
        bm.allocate(seq, num_cached_blocks=0)
        before = len(seq.block_table)
        bm.allocate_burst(seq, 512)
        self.assertGreater(len(seq.block_table), before)
        freed = bm.trim_burst(seq)
        self.assertGreater(freed, 0)
        self.assertEqual(len(seq.block_table), before)

    def test_frees_nothing_when_table_already_matches(self):
        bm = BlockManager(num_blocks=20, block_size=256)
        seq = make_seq(10)
        bm.allocate(seq, num_cached_blocks=0)
        before = len(seq.block_table)
        freed = bm.trim_burst(seq)
        self.assertEqual(freed, 0)
        self.assertEqual(len(seq.block_table), before)

    def test_freed_blocks_return_to_free_pool(self):
        bm = BlockManager(num_blocks=20, block_size=256)
        seq = make_seq(10)
        bm.allocate(seq, num_cached_blocks=0)
        free_before = len(bm.free_block_ids)
        bm.allocate_burst(seq, 512)
        free_after_alloc = len(bm.free_block_ids)
        self.assertLess(free_after_alloc, free_before)
        bm.trim_burst(seq)
        free_after_trim = len(bm.free_block_ids)
        self.assertEqual(free_after_trim, free_before)

    def test_trims_after_partial_burst(self):
        bm = BlockManager(num_blocks=20, block_size=256)
        seq = make_seq(10)
        bm.allocate(seq, num_cached_blocks=0)
        bm.allocate_burst(seq, 512)
        for _ in range(100):
            seq.append_token(42)
        target = (len(seq) + bm.block_size - 1) // bm.block_size
        freed = bm.trim_burst(seq)
        self.assertEqual(len(seq.block_table), target)


class ScheduleBurstTests(unittest.TestCase):
    def test_selects_running_seqs_and_returns_max_steps(self):
        sched = make_scheduler(num_blocks=64, block_size=256, max_num_seqs=8)
        seq_a = make_seq(10, max_tokens=32)
        seq_b = make_seq(10, max_tokens=16)
        for s in (seq_a, seq_b):
            fully_prefill(sched, s)
            sched.running.append(s)
        seqs, max_steps = sched.schedule_burst(burst_cap=64, max_batch=8)
        self.assertEqual(len(seqs), 2)
        # max_steps is the SMALLEST budget, not the largest: the device burst
        # runs one length for the whole batch and never caps a sequence itself,
        # so running the largest budget would overrun the shorter sequence's
        # reserved blocks.
        self.assertEqual(max_steps, 16)

    def test_every_selected_seq_is_reserved_for_the_whole_burst(self):
        """No selected sequence may be able to write past its block table.

        This is the regression lock for the burst overrun: the kernel writes up
        to ``position + max_steps`` for *every* row, and ``ldg_kv_ptr`` has no
        bound check, so a sequence reserved for fewer than ``max_steps`` tokens
        would spill into another sequence's physical blocks.
        """

        sched = make_scheduler(num_blocks=256, block_size=256, max_num_seqs=8)
        long_seq = make_seq(10, max_tokens=64)
        short_seq = make_seq(10, max_tokens=3)
        for s in (long_seq, short_seq):
            fully_prefill(sched, s)
            sched.running.append(s)
        seqs, max_steps = sched.schedule_burst(burst_cap=64, max_batch=8)
        self.assertEqual(len(seqs), 2)
        self.assertEqual(max_steps, 3)
        for seq in seqs:
            # Blocks reserved must cover position + max_steps for this row.
            needed_blocks = (len(seq) + max_steps + 255) // 256
            self.assertGreaterEqual(
                len(seq.block_table),
                needed_blocks,
                f"seq {seq.seq_id} reserved {len(seq.block_table)} blocks but "
                f"the burst will write through position {len(seq) + max_steps}",
            )

    def test_short_budget_clamps_the_burst_not_the_reservation(self):
        """A tiny budget must shorten the burst, never leave a row unserved."""

        sched = make_scheduler(num_blocks=64, block_size=256, max_num_seqs=8)
        seqs_in = []
        for max_tokens in (64, 64, 1):
            s = make_seq(10, max_tokens=max_tokens)
            fully_prefill(sched, s)
            sched.running.append(s)
            seqs_in.append(s)
        seqs, max_steps = sched.schedule_burst(burst_cap=64, max_batch=8)
        self.assertEqual(max_steps, 1)
        self.assertEqual(len(seqs), 3)

    def test_respects_max_batch_limit(self):
        sched = make_scheduler(num_blocks=128, block_size=256, max_num_seqs=8)
        for _ in range(5):
            s = make_seq(10, max_tokens=32)
            fully_prefill(sched, s)
            sched.running.append(s)
        seqs, max_steps = sched.schedule_burst(burst_cap=64, max_batch=3)
        self.assertEqual(len(seqs), 3)

    def test_skips_seq_with_zero_budget(self):
        sched = make_scheduler(num_blocks=64, block_size=256, max_num_seqs=8)
        seq_done = make_seq(10, max_tokens=5)
        for _ in range(5):
            seq_done.append_token(42)
        fully_prefill(sched, seq_done)
        sched.running.append(seq_done)
        seqs, max_steps = sched.schedule_burst(burst_cap=64, max_batch=8)
        self.assertEqual(seqs, [])
        self.assertEqual(max_steps, 0)

    def test_does_not_rotate_running_deque(self):
        sched = make_scheduler(num_blocks=128, block_size=256, max_num_seqs=8)
        ids = []
        for _ in range(4):
            s = make_seq(10, max_tokens=32)
            fully_prefill(sched, s)
            sched.running.append(s)
            ids.append(s.seq_id)
        sched.schedule_burst(burst_cap=64, max_batch=2)
        remaining_ids = [s.seq_id for s in sched.running]
        self.assertEqual(remaining_ids, ids)


class PostprocessBurstTests(unittest.TestCase):
    def test_appends_tokens_without_finishing_when_below_max(self):
        sched = make_scheduler(num_blocks=64, block_size=256, eos=999)
        seq = make_seq(10, max_tokens=64)
        fully_prefill(sched, seq)
        sched.running.append(seq)
        token_log = [100, 101, 102, 103, 104]
        sched.postprocess_burst([seq], [token_log])
        self.assertEqual(seq.completion_token_ids, [100, 101, 102, 103, 104])
        self.assertEqual(seq.status, SequenceStatus.RUNNING)
        self.assertIn(seq, sched.running)

    def test_trims_unused_burst_blocks_for_non_finished_seq(self):
        sched = make_scheduler(num_blocks=64, block_size=256, eos=999)
        seq = make_seq(10, max_tokens=64)
        fully_prefill(sched, seq)
        sched.running.append(seq)
        table_len_before = len(seq.block_table)
        # Use a burst large enough to cross a block boundary (block_size=256).
        sched.block_manager.allocate_burst(seq, 300)
        self.assertGreater(len(seq.block_table), table_len_before)
        token_log = [100, 101, 102, 103, 104]
        sched.postprocess_burst([seq], [token_log])
        self.assertEqual(seq.status, SequenceStatus.RUNNING)
        self.assertIn(seq, sched.running)
        self.assertEqual(len(seq.block_table), table_len_before)

    def test_truncates_at_eos_inclusive(self):
        sched = make_scheduler(num_blocks=64, block_size=256, eos=999)
        seq = make_seq(10, max_tokens=64)
        fully_prefill(sched, seq)
        sched.running.append(seq)
        token_log = [100, 999, 102, 103, -1, -1]
        sched.postprocess_burst([seq], [token_log])
        self.assertEqual(seq.completion_token_ids, [100, 999])
        self.assertEqual(seq.status, SequenceStatus.FINISHED)
        self.assertNotIn(seq, sched.running)

    def test_truncates_at_sentinel_minus_one_with_ignore_eos(self):
        sched = make_scheduler(num_blocks=64, block_size=256, eos=999)
        seq = make_seq(10, max_tokens=64, temperature=0.0)
        seq.ignore_eos = True
        fully_prefill(sched, seq)
        sched.running.append(seq)
        token_log = [100, 101, -1, -1, -1]
        sched.postprocess_burst([seq], [token_log])
        self.assertEqual(seq.completion_token_ids, [100, 101])
        self.assertEqual(seq.status, SequenceStatus.RUNNING)
        self.assertIn(seq, sched.running)

    def test_truncates_at_max_tokens(self):
        sched = make_scheduler(num_blocks=64, block_size=256, eos=999)
        seq = make_seq(10, max_tokens=3)
        fully_prefill(sched, seq)
        sched.running.append(seq)
        token_log = [100, 101, 102, 103, 104]
        sched.postprocess_burst([seq], [token_log])
        self.assertEqual(seq.completion_token_ids, [100, 101, 102])
        self.assertEqual(seq.status, SequenceStatus.FINISHED)
        self.assertNotIn(seq, sched.running)

    def test_handles_multiple_seqs_with_different_eos_positions(self):
        sched = make_scheduler(num_blocks=64, block_size=256, eos=999)
        seq_a = make_seq(10, max_tokens=4)
        seq_b = make_seq(10, max_tokens=64)
        for s in (seq_a, seq_b):
            fully_prefill(sched, s)
            sched.running.append(s)
        log_a = [100, 101, 102, 103]
        log_b = [200, 999, -1, -1]
        sched.postprocess_burst([seq_a, seq_b], [log_a, log_b])
        self.assertEqual(seq_a.completion_token_ids, [100, 101, 102, 103])
        self.assertEqual(seq_a.status, SequenceStatus.FINISHED)
        self.assertNotIn(seq_a, sched.running)
        self.assertEqual(seq_b.completion_token_ids, [200, 999])
        self.assertEqual(seq_b.status, SequenceStatus.FINISHED)
        self.assertNotIn(seq_b, sched.running)


if __name__ == "__main__":
    unittest.main()
