"""Behavioral tests for the prefill/decode scheduler.

Every test here drives the **production** scheduler: ``engine_fixtures`` builds a
real ``Config`` (only ``transformers`` is replaced) and therefore a real
``Scheduler``, ``BlockManager`` and ``Sequence``. The previous phase-0 version of
these tests replaced the scheduler's imports with ``SimpleNamespace`` stubs, so
"behavior" was asserted against a hand-built object graph.

The two behaviors worth calling out:

* the decode/prefill ratio, which protects decode ITL from a long waiting
  prompt; and
* what the scheduler does under KV pressure. ``_schedule_decode`` is allowed to
  return an empty batch when preemption frees nothing (it used to ``assert
  scheduled_seqs`` and crash); ``Schedule()` then retries prefill. That fallback
  can only make progress if the pool can still fit a sequence, so the
  genuinely-undersized case is asserted to fail with an explicit error.
"""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine_fixtures import BLOCK_SIZE, make_scheduler, make_sequence  # noqa: E402
from nanovllm.engine.sequence import Sequence, SequenceStatus  # noqa: E402


class SchedulerTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        Sequence.block_size = BLOCK_SIZE

    def scheduler(self, *, num_blocks: int = 64, **overrides):
        return make_scheduler(self._tmp.name, num_blocks=num_blocks, **overrides)

    def add(self, scheduler, num_tokens: int, *, token: int = 1):
        seq = make_sequence([token] * num_tokens)
        scheduler.add(seq)
        return seq

    def prefill(self, scheduler, num_tokens: int, *, token: int = 1):
        """Add a prompt and run the prefill step that schedules exactly it."""

        seq = self.add(scheduler, num_tokens, token=token)
        seqs, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(seqs, [seq])
        return seq

    def postprocess(self, scheduler, seqs, is_prefill: bool, *, token: int = 99):
        scheduler.postprocess(seqs, [token] * len(seqs), is_prefill)


class BasicSchedulingTests(SchedulerTestCase):
    def test_a_fresh_prompt_is_scheduled_as_prefill(self):
        scheduler = self.scheduler()
        seq = self.prefill(scheduler, 256)

        self.assertEqual(seq.status, SequenceStatus.RUNNING)
        self.assertEqual(seq.num_scheduled_tokens, 256)
        self.assertEqual(len(seq.block_table), 1)
        self.assertEqual(scheduler.stats()["decode_steps_since_prefill"], 0)

    def test_is_finished_only_when_both_queues_are_empty(self):
        scheduler = self.scheduler()
        self.assertTrue(scheduler.is_finished())
        self.add(scheduler, 8)
        self.assertFalse(scheduler.is_finished())

        steps = 0
        while not scheduler.is_finished():
            seqs, is_prefill = scheduler.schedule()
            self.postprocess(scheduler, seqs, is_prefill)
            steps += 1
            self.assertLess(steps, 32, "the scheduler must terminate a request")

        self.assertTrue(scheduler.is_finished())
        self.assertEqual(scheduler.stats()["allocated_blocks"], 0)

    def test_a_prefill_budget_chunks_a_long_prompt(self):
        scheduler = self.scheduler(max_prefill_tokens_per_step=8)
        seq = self.add(scheduler, 20)

        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [seq])
        self.assertEqual(seq.num_scheduled_tokens, 8)
        self.assertEqual(seq.status, SequenceStatus.WAITING, "chunked, not done")
        self.postprocess(scheduler, scheduled, is_prefill)
        self.assertEqual(seq.num_cached_tokens, 8, "no token sampled mid-prefill")

        scheduled, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.postprocess(scheduler, scheduled, is_prefill)
        self.assertEqual(seq.num_cached_tokens, 16)
        self.assertEqual(seq.status, SequenceStatus.WAITING)


class FairnessTests(SchedulerTestCase):
    def test_decode_is_served_before_waiting_prefill(self):
        scheduler = self.scheduler(prefill_decode_ratio=1)
        running = self.prefill(scheduler, 256)
        self.postprocess(scheduler, [running], True)
        waiting = self.add(scheduler, 8, token=2)

        scheduled, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [running])
        self.assertIn(waiting, scheduler.waiting)
        self.assertEqual(scheduler.stats()["decode_steps_since_prefill"], 1)

    def test_prefill_gets_a_turn_after_the_ratio_is_spent(self):
        scheduler = self.scheduler(prefill_decode_ratio=1)
        running = self.prefill(scheduler, 256)
        self.postprocess(scheduler, [running], True)
        waiting = self.add(scheduler, 8, token=2)

        scheduled, is_prefill = scheduler.schedule()
        self.postprocess(scheduler, scheduled, is_prefill)

        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [waiting])
        self.assertEqual(waiting.num_scheduled_tokens, 8)
        self.assertEqual(scheduler.stats()["decode_steps_since_prefill"], 0)

    def test_ratio_two_spends_two_decode_steps_before_prefill(self):
        scheduler = self.scheduler(prefill_decode_ratio=2)
        running = self.prefill(scheduler, 257)
        self.postprocess(scheduler, [running], True)
        waiting = self.add(scheduler, 8, token=2)

        for step in (1, 2):
            scheduled, is_prefill = scheduler.schedule()
            self.assertFalse(is_prefill, f"decode step {step} must be served")
            self.assertEqual(scheduled, [running])
            self.postprocess(scheduler, scheduled, is_prefill)
            self.assertEqual(
                scheduler.stats()["decode_steps_since_prefill"], step
            )

        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [waiting])

    def test_without_waiting_work_decode_keeps_running(self):
        scheduler = self.scheduler(prefill_decode_ratio=1)
        running = self.prefill(scheduler, 256)
        self.postprocess(scheduler, [running], True)

        for _ in range(3):
            scheduled, is_prefill = scheduler.schedule()
            self.assertFalse(is_prefill)
            self.assertEqual(scheduled, [running])
            self.postprocess(scheduler, scheduled, is_prefill)


class KvPressureTests(SchedulerTestCase):
    def test_decode_batch_can_be_empty_instead_of_crashing(self):
        """The regression: ``_schedule_decode`` used to ``assert scheduled_seqs``."""

        scheduler = self.scheduler(num_blocks=1)
        running = self.prefill(scheduler, 256)
        self.postprocess(scheduler, [running], True)  # 257 tokens -> needs a block
        self.assertEqual(scheduler.stats()["free_blocks"], 0)

        scheduled = scheduler._schedule_decode()

        self.assertEqual(scheduled, [], "preempting everything yields no decode")
        self.assertEqual(running.status, SequenceStatus.WAITING)
        self.assertEqual(running.block_table, [])
        self.assertEqual(list(scheduler.waiting), [running])
        self.assertEqual(scheduler.stats()["free_blocks"], 1, "block returned")

    def test_preemption_keeps_the_scheduler_progressing(self):
        scheduler = self.scheduler(num_blocks=4, prefill_decode_ratio=1)
        first = self.prefill(scheduler, 511, token=1)
        self.postprocess(scheduler, [first], True)  # len 512, needs no block yet
        second = self.add(scheduler, 512, token=2)

        # One decode of `first` (no allocation at len 512), then `second` is
        # prefilled, leaving zero free blocks.
        scheduled, is_prefill = scheduler.schedule()
        self.assertFalse(is_prefill)
        self.postprocess(scheduler, scheduled, is_prefill)
        scheduled, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [second])
        self.postprocess(scheduler, scheduled, is_prefill)
        self.assertEqual(scheduler.stats()["free_blocks"], 0)
        self.assertEqual(len(first.block_table), 2)
        self.assertEqual(len(second.block_table), 2)

        waiting = self.add(scheduler, 8, token=3)

        scheduled, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [first], "the head sequence is served")
        self.assertEqual(second.status, SequenceStatus.WAITING)
        self.assertEqual(second.block_table, [], "preempted, blocks released")
        self.assertEqual(
            list(scheduler.waiting), [second, waiting], "preempted goes first"
        )

    def test_undersized_pool_fails_loudly(self):
        """No fallback can help when one sequence does not fit the whole pool.

        This is a capacity error, not a scheduler defect: the error message must
        name the failure instead of an ``assert`` that ``python -O`` strips.
        """

        scheduler = self.scheduler(num_blocks=1)
        running = self.prefill(scheduler, 256)
        self.postprocess(scheduler, [running], True)  # 257 tokens -> 2 blocks

        with self.assertRaisesRegex(RuntimeError, "scheduler made no progress"):
            scheduler.schedule()


class SchedulerStatsTests(SchedulerTestCase):
    def test_reports_queue_pressure_and_cache_counters(self):
        scheduler = self.scheduler()
        seqs = [self.add(scheduler, 8, token=token) for token in (1, 2, 3)]
        self.assertEqual(scheduler.stats()["peak_waiting_sequences"], 3)

        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, seqs, "all three fit in one prefill step")
        stats = scheduler.stats()
        self.assertEqual(stats["waiting_sequences"], 0)
        self.assertEqual(stats["running_sequences"], 3)
        self.assertEqual(stats["peak_running_sequences"], 3)
        self.assertEqual(stats["prefill_decode_ratio"], 1)
        self.assertEqual(stats["block_size"], BLOCK_SIZE)
        self.assertEqual(stats["allocated_blocks"], 3)
        self.assertEqual(stats["free_blocks"], 64 - 3)
        self.assertEqual(stats["peak_used_blocks"], 3)

    def test_peak_waiting_is_historical(self):
        scheduler = self.scheduler()
        for token in (1, 2, 3):
            self.add(scheduler, 8, token=token)
        scheduler.schedule()

        self.assertEqual(scheduler.stats()["waiting_sequences"], 0)
        self.assertEqual(scheduler.stats()["peak_waiting_sequences"], 3)


if __name__ == "__main__":
    unittest.main()
