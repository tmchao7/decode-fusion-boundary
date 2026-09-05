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
