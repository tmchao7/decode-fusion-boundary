"""Source-level contracts that a CPU-only checkout cannot exercise.

Behavior is covered by ``test_scheduler_behavior``, ``test_block_manager``,
``test_config_validation`` and ``test_kv_block_boundary``. What remains here
needs the torch stack (``llm_engine``, ``model_runner``, ``taskgraph``) just to
import, or a GPU to run, so the contract is asserted against the source text
instead. Keep this file small and specific: an assertion earns its place only if
behavioral coverage is impossible.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


class EngineWiringContracts(unittest.TestCase):
    def test_engine_reports_scheduler_stats(self):
        """``_generate`` must hand the scheduler counters to the metrics report.

        ``llm_engine.py`` imports torch and multiprocessing at module level, so
        the wiring is asserted textually. The receiving side (the report embeds
        ``summary["scheduler"]``) is behaviorally tested in ``test_metrics.py``.
        """

        text = source("src/nanovllm/engine/llm_engine.py")
        self.assertIn("scheduler_stats", text)
        self.assertIn("self.scheduler.stats()", text)


class MegakernelMetadataContracts(unittest.TestCase):
    def test_single_row_path_does_not_prepare_temporary_block_table(self):
        """Batch=1 decode passes the scheduler's host table straight through."""

        text = source("src/nanovllm/engine/model_runner.py")
        start = text.index("if len(seqs) == 1:")
        single_path = text[start:text.index("else:", start)]
        self.assertNotIn("prepare_block_tables(seqs)", single_path)
        self.assertIn("seqs[0].block_table", single_path)

    def test_taskgraph_row_adapter_owns_persistent_host_table(self):
        """The task-graph decoder must reuse one pinned row table per step."""

        text = source("src/nanovllm/backends/megakernel/taskgraph.py")
        self.assertIn("_row_table_host", text)
        self.assertIn("_row_table_host[:cols]", text)
        self.assertIn("non_blocking=True", text)


class FusionGranularityWiringContracts(unittest.TestCase):
    def test_decoder_launches_one_kernel_per_segment(self):
        """``fusion_groups`` must not become a dead knob.

        ``decode_rows`` needs torch + CUDA to exercise, so the wiring is
        asserted textually: the per-segment schedule cache (built from
        ``schedule.split_layers``) feeds a loop that issues one
        ``tg_decode_step`` per segment. The schedule-side semantics are
        behaviourally tested in ``test_taskgraph_abi``.
        """

        text = source("src/nanovllm/backends/megakernel/taskgraph.py")
        self.assertIn("split_layers(", text)
        self.assertIn("for geo in geometries:", text)
        self.assertIn("self._fusion_groups", text)
        # One launch per segment: exactly one call site, inside the loop.
        self.assertEqual(text.count("self._ops.tg_decode_step("), 1)


if __name__ == "__main__":
    unittest.main()
