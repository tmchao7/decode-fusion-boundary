"""Fairness audit for the task-graph Megakernel versus CUDA Graph."""

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


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
    """CUDA-graph bucket padding remains part of the A/B audit."""

    def test_graph_buckets_match_the_documented_schedule(self) -> None:
        source = (
            REPO_ROOT / "src" / "nanovllm" / "engine" / "model_runner.py"
        ).read_text(encoding="utf-8")
        self.assertIn("[1, 2, 4, 8] + list(range(16, max_bs + 1, 16))", source)
        self.assertIn("next((x for x in self.graph_bs if x >= bs), None)", source)
        self.assertIn("if bucket is None:", source)

    def test_max_bs_is_always_a_bucket_so_no_gap_exists(self) -> None:
        source = (
            REPO_ROOT / "src" / "nanovllm" / "engine" / "model_runner.py"
        ).read_text(encoding="utf-8")
        self.assertIn("if max_bs > self.graph_bs[-1]:", source)
        self.assertIn("self.graph_bs.append(max_bs)", source)
        for max_bs in range(1, 65):
            graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
            if max_bs > graph_bs[-1]:
                graph_bs.append(max_bs)
            self.assertTrue(any(x >= max_bs for x in graph_bs))

    def test_padding_and_asymmetric_acceleration_are_documented(self) -> None:
        fairness = (
            REPO_ROOT / "docs" / "05_对比基线与公平性.md"
        ).read_text(encoding="utf-8")
        self.assertIn("bucket（桶）", fairness)
        self.assertIn("graph_bs", fairness)
        self.assertIn("FlashAttention", fairness)
        self.assertIn("scalar", fairness)


if __name__ == "__main__":
    unittest.main()
