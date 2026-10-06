"""Unit tests for the decode accounting report and its roofline columns.

The report exists to answer "how much of the step is device time, and how far is
that device time from the weight-streaming wall?" -- so the tests pin both halves:
the host/device split, and the fact that the denominator is a *measured* rate that
cannot be silently replaced by a spec value.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from nanovllm.roofline import (  # noqa: E402
    DEFAULT_BANDWIDTH_JSON,
    ModelDims,
    RooflineModel,
    measured_hardware_spec,
)
from tools.account_decode import (  # noqa: E402
    accounting_from_experiment,
    load_experiments,
    render_markdown,
    roofline_columns,
)

# RTX 5090, measured read bandwidth (docs/20 §8; LeanMoE-Quant EXP-p0-02).
MEASURED_5090_GBPS = 1559.2
SPEC_5090_GBPS = 1792.0


def measured_probe(mean: float = MEASURED_5090_GBPS) -> dict:
    return {
        "device": "NVIDIA GeForce RTX 5090",
        "sm_count": 170,
        "spec_gbps": SPEC_5090_GBPS,
        "achieved_gbps": {
            "count": 20,
            "mean": mean,
            "stddev": 10.6,
            "cov": 0.0068,
            "min": 1540.0,
            "max": 1575.0,
        },
    }


def experiment_with(gpu_ms: float, wall_ms: float, batch_size: int = 1) -> dict:
    """One synthetic benchmarks.compare experiment with a chosen device time."""

    decode_steps = 10
    request_count = batch_size
    return {
        "backend": "torch-cudagraph",
        "workload": {"id": f"p32_b{batch_size}", "batch_size": batch_size},
        "summary": {
            "decode_steps": decode_steps,
            "decode_time_ms_total": wall_ms * request_count * decode_steps,
            "cuda_decode_time_ms_total": gpu_ms * request_count * decode_steps,
        },
        "requests": [
            {
                "e2e_ms": 1.0,
                "tpot_ms": {"p50": wall_ms},
                "output_tokens": decode_steps,
                "backend_actual": "torch-cudagraph",
            }
        ]
        * request_count,
        "fallback": {"count": 0},
    }


class RooflineColumnTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = RooflineModel(
            ModelDims(),
            measured_hardware_spec(MEASURED_5090_GBPS, name="RTX 5090 (measured)"),
            tied_embedding=True,
        )

    def test_implied_bandwidth_is_weight_bytes_over_device_time(self) -> None:
        row = roofline_columns(self.model, gpu_ms=1.5, batch_size=1)
        expected = self.model.core_weight_bytes() / (1.5 / 1000.0) / 1e9
        self.assertAlmostEqual(row["implied_bandwidth_gbps"], expected, places=3)

    def test_gap_is_measured_against_the_core_weight_floor(self) -> None:
        row = roofline_columns(self.model, gpu_ms=1.5, batch_size=1)
        self.assertAlmostEqual(row["gap_to_core_floor_x"], 1.5 / row["core_weight_floor_ms"])

    def test_the_core_floor_is_weights_only_and_not_a_spec_floor(self) -> None:
        """1.192 GB / 1559.2 GB/s = 0.7645 ms; the spec rate would claim 0.665 ms."""

        row = roofline_columns(self.model, gpu_ms=1.0, batch_size=1)
        self.assertAlmostEqual(row["core_weight_floor_ms"], 0.7645, places=3)
        spec_floor = self.model.core_weight_bytes() / SPEC_5090_GBPS / 1e6
        self.assertNotAlmostEqual(row["core_weight_floor_ms"], spec_floor, places=3)

    def test_no_headroom_when_the_step_is_at_the_modelled_floor(self) -> None:
        """The floor is max(memory, compute) and the memory term carries KV traffic.

        Note the two thresholds: ``core_weight_floor_ms`` is weights only, while
        ``calibration_needed`` is judged against ``lower_bound_ms`` (which adds KV
        at this ``--seq-len``). A step between the two is flagged for calibration
        rather than credited with headroom -- the conservative direction.
        """

        floor = self.model.lower_bound_ms(1, seq_len=1)
        self.assertGreater(floor, self.model.core_weight_bytes() / MEASURED_5090_GBPS / 1e6)
        row = roofline_columns(self.model, gpu_ms=floor, batch_size=1)
        self.assertFalse(row["calibration_needed"])
        self.assertAlmostEqual(row["headroom_fraction"], 0.0, places=6)

    def test_a_step_below_the_floor_is_flagged_as_calibration_not_headroom(self) -> None:
        row = roofline_columns(self.model, gpu_ms=0.5, batch_size=1)
        self.assertTrue(row["calibration_needed"])
        self.assertEqual(row["headroom_ms"], 0.0)

    def test_zero_device_time_reports_unknown_rather_than_zero_bandwidth(self) -> None:
        row = roofline_columns(self.model, gpu_ms=0.0, batch_size=1)
        self.assertIsNone(row["implied_bandwidth_gbps"])
        self.assertIsNone(row["gap_to_core_floor_x"])


class AccountingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = RooflineModel(
            ModelDims(),
            measured_hardware_spec(MEASURED_5090_GBPS, name="RTX 5090 (measured)"),
            tied_embedding=True,
        )

    def test_host_residual_is_wall_minus_device(self) -> None:
        row = accounting_from_experiment(experiment_with(0.8, 3.0), self.model)
        self.assertAlmostEqual(row["ms_per_step_gpu"], 0.8, places=6)
        self.assertAlmostEqual(row["ms_per_step_wall"], 3.0, places=6)
        self.assertAlmostEqual(row["host_residual_ms"], 2.2, places=6)

    def test_rows_carry_the_roofline_distance(self) -> None:
        row = accounting_from_experiment(experiment_with(0.8, 3.0), self.model)
        for key in (
            "implied_bandwidth_gbps",
            "gap_to_core_floor_x",
            "core_weight_floor_ms",
            "calibration_needed",
        ):
            self.assertIn(key, row)

    def test_without_a_model_the_row_has_no_roofline_columns(self) -> None:
        row = accounting_from_experiment(experiment_with(0.8, 3.0))
        self.assertNotIn("core_weight_floor_ms", row)

    def test_markdown_names_the_denominator(self) -> None:
        markdown = render_markdown(
            [accounting_from_experiment(experiment_with(0.8, 3.0), self.model)],
            self.model,
        )
        self.assertIn("Roofline denominator", markdown)
        self.assertIn("1559.2", markdown)
        self.assertIn("gap to weights roofline", markdown)

    def test_markdown_warns_when_calibration_is_needed(self) -> None:
        markdown = render_markdown(
            [accounting_from_experiment(experiment_with(0.1, 3.0), self.model)],
            self.model,
        )
        self.assertIn("calibration finding", markdown)

    def test_decodable_experiments_require_decode_steps(self) -> None:
        experiment = experiment_with(0.8, 3.0)
        experiment["summary"]["decode_steps"] = 0
        self.assertIsNone(accounting_from_experiment(experiment, self.model))


class CliTests(unittest.TestCase):
    def _run(self, argv: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "tools.account_decode", *argv],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
        )

    def _write_inputs(self, tmpdir: str) -> tuple[Path, Path]:
        bandwidth = Path(tmpdir) / "bandwidth.json"
        bandwidth.write_text(json.dumps(measured_probe()), encoding="utf-8")
        compare = Path(tmpdir) / "bench.json"
        compare.write_text(
            json.dumps({"experiments": [experiment_with(0.8, 3.0)]}), encoding="utf-8"
        )
        return bandwidth, compare

    def test_cli_reports_roofline_columns_from_the_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bandwidth, compare = self._write_inputs(tmpdir)
            out = Path(tmpdir) / "account.json"
            proc = self._run(
                [str(compare), "--bandwidth-json", str(bandwidth), "--json", str(out)]
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("gap to weights roofline", proc.stdout)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertAlmostEqual(payload["roofline"]["bandwidth_gbps"], 1559.2, places=1)
            self.assertEqual(payload["roofline"]["source"], str(bandwidth))
            self.assertIn("implied_bandwidth_gbps", payload["rows"][0])

    def test_cli_fails_loudly_without_a_measured_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, compare = self._write_inputs(tmpdir)
            proc = self._run(
                [
                    str(compare),
                    "--bandwidth-json",
                    str(Path(tmpdir) / "absent.json"),
                ]
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("RUN_MODE=bandwidth", proc.stderr)

    def test_the_default_denominator_is_the_checked_in_probe_path(self) -> None:
        self.assertEqual(DEFAULT_BANDWIDTH_JSON, Path("results") / "bandwidth.json")
        # Run from a directory that has no results/bandwidth.json, so this asserts
        # the *default resolution* rather than the developer's working tree: on a
        # machine that has already run RUN_MODE=bandwidth the checked-in path
        # exists, and an earlier version of this test silently became a no-op there.
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = subprocess.run(
                [sys.executable, "-m", "tools.account_decode", "nonexistent.json"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                env={
                    "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}",
                    "PATH": "/usr/bin:/bin",
                },
            )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("RUN_MODE=bandwidth", proc.stderr)

    def test_cli_rejects_both_denominator_inputs_at_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bandwidth, compare = self._write_inputs(tmpdir)
            proc = self._run(
                [
                    str(compare),
                    "--bandwidth-json",
                    str(bandwidth),
                    "--bandwidth-gbps",
                    "1699",
                ]
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("mutually exclusive", proc.stderr)

    def test_cli_accepts_an_explicit_measured_rate_and_records_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, compare = self._write_inputs(tmpdir)
            out = Path(tmpdir) / "account.json"
            proc = self._run(
                [str(compare), "--bandwidth-gbps", "1699", "--json", str(out)]
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertAlmostEqual(payload["roofline"]["bandwidth_gbps"], 1699.0)
            self.assertEqual(payload["roofline"]["source"], "explicit --bandwidth-gbps")

    def test_load_experiments_accepts_a_bare_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "list.json"
            path.write_text(json.dumps([experiment_with(0.8, 3.0)]), encoding="utf-8")
            self.assertEqual(len(load_experiments([str(path)])), 1)


if __name__ == "__main__":
    unittest.main()
