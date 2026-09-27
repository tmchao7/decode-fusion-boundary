"""CPU-only contract tests for the pre-server inference-system phase."""

import unittest
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


class Phase0ConfigContracts(unittest.TestCase):
    def test_config_exposes_prefill_fairness_controls(self):
        text = source("src/nanovllm/config.py")
        self.assertIn("max_prefill_tokens_per_step", text)
        self.assertIn("prefill_decode_ratio", text)

    def test_config_validates_positive_fairness_controls(self):
        text = source("src/nanovllm/config.py")
        self.assertIn("max_prefill_tokens_per_step < 0", text)
        self.assertIn("prefill_decode_ratio < 1", text)


class SchedulerContracts(unittest.TestCase):
    def test_scheduler_has_decode_fairness_state(self):
        text = source("src/nanovllm/engine/scheduler.py")
        self.assertIn("prefill_decode_ratio", text)
        self.assertIn("_decode_steps_since_prefill", text)
        self.assertIn("_should_schedule_decode", text)

    def test_scheduler_exposes_runtime_cache_stats(self):
        text = source("src/nanovllm/engine/scheduler.py")
        self.assertIn("def stats", text)
        self.assertIn("block_manager.stats()", text)

    def test_block_manager_tracks_prefix_cache_stats(self):
        text = source("src/nanovllm/engine/block_manager.py")
        for name in (
            "prefix_cache_hits",
            "prefix_cache_misses",
            "allocated_blocks",
            "freed_blocks",
            "peak_used_blocks",
        ):
            self.assertIn(name, text)


class MetricsContracts(unittest.TestCase):
    def test_engine_reports_scheduler_stats(self):
        text = source("src/nanovllm/engine/llm_engine.py")
        self.assertIn("scheduler_stats", text)
        self.assertIn("self.scheduler.stats()", text)

    def test_metrics_report_accepts_scheduler_stats(self):
        text = source("src/nanovllm/metrics.py")
        self.assertIn("scheduler_stats", text)
        self.assertIn('summary["scheduler"]', text)


class MegakernelMetadataContracts(unittest.TestCase):
    def test_single_row_path_does_not_prepare_temporary_block_table(self):
        text = source("src/nanovllm/engine/model_runner.py")
        single_path = text[text.index("if len(seqs) == 1:"):text.index("else:", text.index("if len(seqs) == 1:"))]
        self.assertNotIn("prepare_block_tables(seqs)", single_path)
        self.assertIn("seqs[0].block_table", single_path)

    def test_taskgraph_row_adapter_owns_persistent_host_table(self):
        text = source("src/nanovllm/backends/megakernel/taskgraph.py")
        self.assertIn("_row_table_host", text)
        self.assertIn("_row_table_host[:cols]", text)
        self.assertIn("non_blocking=True", text)


class SchedulerBehaviorTests(unittest.TestCase):
    @staticmethod
    def make_scheduler(*, ratio=1, prefill_budget=2):
        from collections import deque
        import importlib.util
        import sys
        import types

        # Load the real scheduler while replacing only its heavy dependency
        # modules; scheduler behavior itself remains production code.
        stubs = {
            "nanovllm.config": types.SimpleNamespace(Config=object),
            "nanovllm.engine.sequence": types.SimpleNamespace(
                Sequence=object,
                SequenceStatus=types.SimpleNamespace(
                    WAITING="WAITING", RUNNING="RUNNING", FINISHED="FINISHED"
                ),
            ),
            "nanovllm.engine.block_manager": types.SimpleNamespace(
                BlockManager=object
            ),
        }
        saved = {name: sys.modules.get(name) for name in stubs}
        sys.modules.update(stubs)
        try:
            spec = importlib.util.spec_from_file_location(
                "_phase0_scheduler", ROOT / "src/nanovllm/engine/scheduler.py"
            )
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            Scheduler = module.Scheduler
        finally:
            for name, value in saved.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_num_seqs = 8
        scheduler.max_num_batched_tokens = 8
        scheduler.max_prefill_tokens_per_step = prefill_budget
        scheduler.prefill_decode_ratio = ratio
        scheduler._decode_steps_since_prefill = 0
        scheduler._peak_waiting_sequences = 0
        scheduler._peak_running_sequences = 0
        scheduler.eos = -1
        scheduler.block_size = 256
        scheduler.waiting = deque()
        scheduler.running = deque()

        class FakeBlockManager:
            def can_append(self, seq):
                return True

            def may_append(self, seq):
                return None

            def stats(self):
                return {
                    "allocated_blocks": 0,
                    "free_blocks": 8,
                    "freed_blocks": 0,
                    "peak_used_blocks": 0,
                    "prefix_cache_hits": 0,
                    "prefix_cache_misses": 0,
                }

        scheduler.block_manager = FakeBlockManager()
        return scheduler

    @staticmethod
    def make_seq(*, tokens, cached=0, block_table=None):
        return SimpleNamespace(
            num_tokens=tokens,
            num_cached_tokens=cached,
            num_scheduled_tokens=0,
            block_table=list(block_table or [0]),
            status=None,
            is_prefill=True,
        )

    def test_decode_is_served_before_waiting_prefill_then_prefill_progresses(self):
        scheduler = self.make_scheduler(ratio=1, prefill_budget=2)
        running = self.make_seq(tokens=8, cached=8)
        running.is_prefill = False
        waiting = self.make_seq(tokens=6, cached=0)
        scheduler.running.append(running)
        scheduler.waiting.append(waiting)

        first, first_is_prefill = scheduler.schedule()
        self.assertFalse(first_is_prefill)
        self.assertEqual(first, [running])

        second, second_is_prefill = scheduler.schedule()
        self.assertTrue(second_is_prefill)
        self.assertEqual(second, [waiting])
        self.assertEqual(waiting.num_scheduled_tokens, 2)
        self.assertEqual(waiting.num_cached_tokens, 0)

    def test_scheduler_stats_expose_peak_queue_pressure(self):
        scheduler = self.make_scheduler()
        scheduler.add(self.make_seq(tokens=2))
        scheduler.add(self.make_seq(tokens=2))
        scheduler.running.append(self.make_seq(tokens=2, cached=2))
        scheduler.schedule()
        stats = scheduler.stats()
        self.assertEqual(stats["peak_waiting_sequences"], 2)
        self.assertGreaterEqual(stats["peak_running_sequences"], 1)

    def test_kv_pressure_preempts_and_retries_prefill_instead_of_asserting(self):
        scheduler = self.make_scheduler(ratio=1, prefill_budget=2)
        running = self.make_seq(tokens=257, cached=257, block_table=[0, 1])
        running.is_prefill = False
        waiting = self.make_seq(tokens=2, cached=0, block_table=[])
        scheduler.running.append(running)
        scheduler.waiting.append(waiting)

        class PressureBlockManager:
            def can_append(self, seq):
                return False

            def may_append(self, seq):
                return None

            def can_allocate(self, seq):
                return 1

            def allocate(self, seq, cached_blocks):
                seq.block_table[:] = [0]

            def deallocate(self, seq):
                seq.block_table.clear()
                seq.num_cached_tokens = 0

            def stats(self):
                return {
                    "allocated_blocks": 0,
                    "free_blocks": 8,
                    "freed_blocks": 1,
                    "peak_used_blocks": 2,
                    "prefix_cache_hits": 0,
                    "prefix_cache_misses": 1,
                }

        scheduler.block_manager = PressureBlockManager()
        seqs, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        # The preempted running row is reinserted at the head of waiting so it
        # can rebuild its KV blocks before newer work; the key contract is that
        # scheduling makes progress instead of asserting on an empty decode set.
        self.assertEqual(seqs, [running])
        self.assertIn(waiting, scheduler.waiting)


class ConfigValidationBehaviorTests(unittest.TestCase):
    def test_rejects_invalid_runtime_config_without_assertions(self):
        import tempfile
        from nanovllm.config import Config

        with tempfile.TemporaryDirectory() as model:
            for kwargs, field in (
                ({"max_num_batched_tokens": 0}, "max_num_batched_tokens"),
                ({"max_num_seqs": 0}, "max_num_seqs"),
                ({"kvcache_block_size": 128}, "kvcache_block_size"),
                ({"tensor_parallel_size": 0}, "tensor_parallel_size"),
            ):
                with self.subTest(field=field):
                    with self.assertRaisesRegex(ValueError, field):
                        Config(model=model, **kwargs)


if __name__ == "__main__":
    unittest.main()
