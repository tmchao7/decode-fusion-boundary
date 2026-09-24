"""CPU-only contract tests for the pre-server inference-system phase."""

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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


if __name__ == "__main__":
    unittest.main()
