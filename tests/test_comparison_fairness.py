"""Fairness audit for the task-graph Megakernel versus CUDA Graph."""

import ast
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nanovllm.engine.cudagraph_buckets import (  # noqa: E402
    graph_buckets,
    select_graph_bucket,
)


class MainlineBackendTests(unittest.TestCase):
    def test_compare_exposes_only_the_two_torch_controls_and_one_megakernel(self):
        source = (REPO_ROOT / "benchmarks" / "compare.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        configs = None
        family = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "BACKEND_CONFIGS":
                    configs = ast.literal_eval(node.value)
                if isinstance(target, ast.Name) and target.id == "MEGAKERNEL_FAMILY":
                    family = ast.literal_eval(node.value)
        self.assertEqual(
            set(configs), {"torch-eager", "torch-cudagraph", "megakernel"}
        )
        self.assertEqual(family, ("megakernel",))

    def test_compare_uses_the_clean_default_backend_list(self):
        source = (REPO_ROOT / "benchmarks" / "compare.py").read_text(encoding="utf-8")
        self.assertIn('default="torch-eager,torch-cudagraph,megakernel"', source)
        self.assertNotIn("megakernel-tg", source)
        self.assertNotIn("megakernel-wmma", source)
        self.assertNotIn("megakernel-burst", source)


class CudagraphPaddingDocumentationTests(unittest.TestCase):
    """CUDA-graph bucket padding remains part of the A/B audit.

    These were source-text assertions until the bucket formula moved into
    ``engine/cudagraph_buckets.py``. Behaviour is asserted now, plus one check
    that the runner actually delegates to that single source of truth.
    """

    def test_bucket_schedule_is_reproducible_from_the_documented_rule(self) -> None:
        for max_num_seqs in (1, 2, 8, 9, 10, 16, 17, 32, 33, 64):
            with self.subTest(max_num_seqs=max_num_seqs):
                documented = [1, 2, 4, 8] + list(
                    range(16, max_num_seqs + 1, 16)
                )
                if max_num_seqs > documented[-1]:
                    documented.append(max_num_seqs)
                self.assertEqual(graph_buckets(max_num_seqs), documented)

    def test_max_bs_is_always_a_bucket_so_every_measured_batch_has_a_graph(self) -> None:
        for max_num_seqs in range(1, 65):
            with self.subTest(max_num_seqs=max_num_seqs):
                self.assertIsNotNone(
                    select_graph_bucket(max_num_seqs, graph_buckets(max_num_seqs))
                )

    def test_the_runner_delegates_to_the_shared_module(self) -> None:
        source = (
            REPO_ROOT / "src" / "nanovllm" / "engine" / "model_runner.py"
        ).read_text(encoding="utf-8")

        self.assertIn("graph_buckets(config.max_num_seqs)", source)
        self.assertIn("select_graph_bucket(bs, self.graph_bs)", source)
        # The inline list is what this module replaced; it must not come back.
        self.assertNotIn("self.graph_bs = [1, 2, 4, 8]", source)

    def test_padding_and_asymmetric_acceleration_are_documented(self) -> None:
        fairness = (
            REPO_ROOT / "docs" / "05_对比基线与公平性.md"
        ).read_text(encoding="utf-8")
        self.assertIn("bucket（桶）", fairness)
        self.assertIn("graph_bs", fairness)
        self.assertIn("FlashAttention", fairness)
        self.assertIn("scalar", fairness)
        self.assertIn("padding_factor", fairness)


if __name__ == "__main__":
    unittest.main()
