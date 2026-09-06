import json
import tempfile
import unittest
from pathlib import Path

from tools.workloads import load_workloads, select_correctness_workloads


class WorkloadTests(unittest.TestCase):
    def test_loads_and_selects_workloads(self):
        payload = {
            "performance": {"prompt_lengths": [32, 128]},
            "correctness": [
                {"id": "qa", "category": "qa", "prompt": "Why?", "max_tokens": 8}
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workloads.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_workloads(path)
        selected = select_correctness_workloads(loaded, ["qa"])
        self.assertEqual(selected[0]["category"], "qa")

    def test_rejects_invalid_performance_lengths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workloads.json"
            path.write_text(json.dumps({"performance": {"prompt_lengths": [0]}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prompt_lengths"):
                load_workloads(path)


if __name__ == "__main__":
    unittest.main()

class CorrectnessSmokeHelperTests(unittest.TestCase):
    def test_builds_deterministic_synthetic_batch_cases(self):
        from tools.correctness_smoke import build_cases, compare_outputs

        cases = build_cases("benchmarks/workloads.json", [8], None, [1, 2], 0)
        self.assertEqual([case["id"] for case in cases], ["synthetic_p8_b1", "synthetic_p8_b2"])
        self.assertEqual(len(cases[1]["prompts"]), 2)
        self.assertEqual(compare_outputs([[1, 2]], [[1, 3]])[0], [0])
