"""Tests for tools/account_decode.py (host/device split of decode steps)."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from nanovllm.roofline import HardwareSpec, ModelDims, RooflineModel  # noqa: E402
from tools.account_decode import (  # noqa: E402
    accounting_from_crossover,
    accounting_from_experiment,
    find_calibration_warnings,
    find_inconsistencies,
    load_experiments,
    render_comparison,
    render_markdown,
)


def _crossover_payload() -> dict:
    """Mirror the schema written by tools/bench_burst_batch.py.

    Values are the documented A100 crossover scan minima (docs/07 section 9.6).
    """

    return {
        "gpu": "5",
        "model": "/models/Qwen3-0.6B",
        "prompt_length": 32,
        "max_tokens": 128,
        "warmup": 2,
        "runs": 3,
        "safe_burst_batch": 8,
        "max_batch_size": 32,
        "records": [
            {
                "batch": 1,
                "cudagraph": {"ms_per_tok_min": 2.944, "expected_tokens": 128},
                "burst": {"ms_per_tok_min": 2.241, "expected_tokens": 128},
                "speedup": 1.31,
                "verdict": "BEATS",
                "note": "",
            },
            {
                "batch": 8,
                "cudagraph": {"ms_per_tok_min": 0.429, "expected_tokens": 1024},
                "burst": {"ms_per_tok_min": 6.495, "expected_tokens": 1024},
                "speedup": 0.07,
                "verdict": "loses",
                "note": "",
            },
            {
                "batch": 32,
                "cudagraph": {"ms_per_tok_min": 0.111, "expected_tokens": 4096},
                "burst": {"ms_per_tok_min": 6.496, "expected_tokens": 4096},
                "speedup": 0.02,
                "verdict": "loses",
                "note": "serial x4",
            },
        ],
    }


def _experiment(
    *,
    backend: str,
    batch_size: int,
    decode_steps: int,
    ms_per_step_wall: float,
    ms_per_step_gpu: float,
    output_tokens: int | None = None,
    backend_actual: str | None = None,
) -> dict:
    """Build a compare.py-shaped experiment with self-consistent metrics."""

    requests = output_tokens or decode_steps
    wall_total = ms_per_step_wall * decode_steps * batch_size
    gpu_total = ms_per_step_gpu * decode_steps * batch_size
    tpot_ms = ms_per_step_wall / batch_size
    actual = backend_actual if backend_actual is not None else backend
    return {
        "backend": backend,
        "workload": {
            "id": f"perf_p128_b{batch_size}",
            "category": "controlled_batch",
            "batch_size": batch_size,
        },
        "summary": {
            "decode_steps": decode_steps,
            "decode_time_ms_total": wall_total,
            "cuda_decode_time_ms_total": gpu_total,
            "decode_tok_s": requests * batch_size / (wall_total / 1000.0),
        },
        "requests": [
            {
                "request_id": index,
                "e2e_ms": 10.0 + index,
                "tpot_ms": tpot_ms,
                "output_tokens": requests,
                "backend_actual": actual,
            }
            for index in range(batch_size)
        ],
        "fallback": {"count": 0, "reasons": []},
    }


class AccountingTests(unittest.TestCase):
    def test_documented_b8_split(self) -> None:
        """b8: 3.43 ms/step wall, 0.77 ms GPU -> ~2.66 ms host."""

        row = accounting_from_experiment(
            _experiment(
                backend="torch-cudagraph",
                backend_actual="torch",
                batch_size=8,
                decode_steps=4,
                ms_per_step_wall=3.432,
                ms_per_step_gpu=0.77,
            )
        )
        assert row is not None
        self.assertAlmostEqual(row["ms_per_step_wall"], 3.432, places=3)
        self.assertAlmostEqual(row["ms_per_step_gpu"], 0.77, places=3)
        self.assertAlmostEqual(row["host_residual_ms"], 2.662, places=3)
        self.assertAlmostEqual(row["ms_per_token_wall"], 0.429, places=3)
        self.assertAlmostEqual(row["gpu_fraction"], 0.77 / 3.432, places=6)

    def test_tpot_cross_check_is_one_for_consistent_data(self) -> None:
        row = accounting_from_experiment(
            _experiment(
                backend="torch-cudagraph",
                batch_size=4,
                decode_steps=8,
                ms_per_step_wall=3.0,
                ms_per_step_gpu=0.8,
            )
        )
        assert row is not None
        self.assertAlmostEqual(row["step_consistency_ratio"], 1.0, places=6)
        self.assertEqual(find_inconsistencies([row]), [])

    def test_inconsistent_cadence_is_flagged(self) -> None:
        experiment = _experiment(
            backend="megakernel-burst",
            batch_size=4,
            decode_steps=8,
            ms_per_step_wall=3.0,
            ms_per_step_gpu=0.8,
        )
        # Simulate a workload whose per-token cadence does not match the step
        # accounting (e.g. serialised bursts inside one step).
        for request in experiment["requests"]:
            request["tpot_ms"] = 0.1
        row = accounting_from_experiment(experiment)
        assert row is not None
        warnings = find_inconsistencies([row])
        self.assertEqual(len(warnings), 1)
        self.assertIn("megakernel-burst", warnings[0])

    def test_missing_decode_steps_is_skipped(self) -> None:
        experiment = _experiment(
            backend="x",
            batch_size=1,
            decode_steps=4,
            ms_per_step_wall=2.0,
            ms_per_step_gpu=1.0,
        )
        experiment["summary"]["decode_steps"] = 0
        self.assertIsNone(accounting_from_experiment(experiment))

    def test_no_completed_requests_is_skipped(self) -> None:
        experiment = _experiment(
            backend="x",
            batch_size=2,
            decode_steps=4,
            ms_per_step_wall=2.0,
            ms_per_step_gpu=1.0,
        )
        experiment["requests"] = []
        self.assertIsNone(accounting_from_experiment(experiment))

    def test_comparison_table_reports_host_headroom(self) -> None:
        rows = [
            accounting_from_experiment(
                _experiment(
                    backend=backend,
                    backend_actual=backend,
                    batch_size=8,
                    decode_steps=4,
                    ms_per_step_wall=wall,
                    ms_per_step_gpu=gpu,
                )
            )
            for backend, wall, gpu in (
                ("torch-cudagraph", 3.43, 0.77),
                ("megakernel-burst", 6.50, 6.30),
            )
        ]
        model = RooflineModel(ModelDims(), HardwareSpec())
        table = render_comparison([row for row in rows if row], model)
        self.assertIn("torch-cudagraph", table)
        # The fastest backend at b8 is cudagraph, whose host residual dominates.
        self.assertIn("2.660", table)
        self.assertIn("ceiling tok/s", table)


class CrossoverScanTests(unittest.TestCase):
    """The existing server artifact must be readable without a re-run."""

    def _rows(self) -> list[dict]:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "crossover_scan.json"
            path.write_text(json.dumps(_crossover_payload()), encoding="utf-8")
            experiments = load_experiments([str(path)])
        model = RooflineModel(ModelDims(), HardwareSpec())
        floor = (
            model.weight_bytes()
            / (model.hardware.effective_bandwidth_gbps() * 1e9)
            * 1000.0
        )
        return [accounting_from_crossover(exp, floor) for exp in experiments]

    def test_schema_is_auto_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "crossover_scan.json"
            path.write_text(json.dumps(_crossover_payload()), encoding="utf-8")
            experiments = load_experiments([str(path)])
        # two backends x three batches
        self.assertEqual(len(experiments), 6)
        self.assertEqual(
            {exp["backend"] for exp in experiments},
            {"torch-cudagraph", "megakernel-burst"},
        )
        self.assertTrue(
            all(
                exp["summary"]["crossover_source"].endswith("crossover_scan.json")
                for exp in experiments
            )
        )

    def test_documented_batch8_rows(self) -> None:
        rows = [row for row in self._rows() if row and row["batch_size"] == 8]
        by_backend = {row["backend"]: row for row in rows}
        cg = by_backend["torch-cudagraph"]
        burst = by_backend["megakernel-burst"]
        # tpot_ms * batch == ms/token * 8 for both backends
        self.assertAlmostEqual(cg["ms_per_step_wall"], 0.429 * 8, places=3)
        self.assertAlmostEqual(burst["ms_per_step_wall"], 6.495 * 8, places=3)
        # Device time comes from the floor, so the host residual is a bound.
        self.assertEqual(cg["device_source"], "floor")
        self.assertGreater(cg["host_residual_ms"], 2.0)
        self.assertLess(cg["host_residual_ms"], 3.0)

    def test_serial_note_is_captured(self) -> None:
        rows = [row for row in self._rows() if row and row["batch_size"] == 32]
        burst = next(row for row in rows if row["backend"] == "megakernel-burst")
        self.assertEqual(burst["serial_chunks"], 4)
        cg = next(row for row in rows if row["backend"] == "torch-cudagraph")
        self.assertEqual(cg["serial_chunks"], 1)

    def test_markdown_labels_floor_rows(self) -> None:
        rows = [row for row in self._rows() if row]
        model = RooflineModel(ModelDims(), HardwareSpec())
        text = render_markdown(rows, model)
        self.assertIn("(floor)", text)
        self.assertIn("device-less", text.replace("wall-time-only", "device-less"))

    def test_comparison_table_prefers_cudagraph_at_batch8(self) -> None:
        rows = [row for row in self._rows() if row]
        model = RooflineModel(ModelDims(), HardwareSpec())
        table = render_comparison(rows, model)
        # b1 row must name the megakernel burst as fastest, b8 the cudagraph run.
        lines = [line for line in table.splitlines() if line.startswith("| 1 |")]
        self.assertEqual(len(lines), 1)
        self.assertIn("megakernel-burst", lines[0])


class CalibrationWarningTests(unittest.TestCase):
    """0.6 ms/step for 1.503 GB implies 2505 GB/s, above the 2039 GB/s spec.

    The 0.77 ms that used to live here only looked impossible against the old
    1555 GB/s constant (the 40 GB A100 part); against the real 80 GB spec it is
    a legitimate near-peak measurement and must NOT be flagged.
    """

    def _row(self, gpu_ms: float) -> dict:
        return accounting_from_experiment(
            _experiment(
                backend="torch-cudagraph",
                backend_actual="torch",
                batch_size=8,
                decode_steps=4,
                ms_per_step_wall=3.432,
                ms_per_step_gpu=gpu_ms,
            )
        )

    def test_below_floor_is_flagged(self) -> None:
        row = self._row(0.6)
        assert row is not None
        warnings = find_calibration_warnings(
            [row], RooflineModel(ModelDims(), HardwareSpec())
        )
        self.assertEqual(len(warnings), 1)
        self.assertIn("2505 GB/s", warnings[0])
        self.assertIn("RUN_MODE=roofline", warnings[0])

    def test_near_peak_measurement_is_not_flagged(self) -> None:
        """0.77 ms/step is 97% of the A100-80GB HBM peak, not a contradiction."""

        row = self._row(0.77)
        assert row is not None
        self.assertEqual(
            find_calibration_warnings(
                [row], RooflineModel(ModelDims(), HardwareSpec())
            ),
            [],
        )

    def test_above_floor_is_not_flagged(self) -> None:
        row = self._row(1.5)
        assert row is not None
        self.assertEqual(
            find_calibration_warnings(
                [row], RooflineModel(ModelDims(), HardwareSpec())
            ),
            [],
        )

    def test_floor_rows_are_never_flagged(self) -> None:
        rows = [row for row in CrossoverScanTests()._rows() if row]
        self.assertEqual(
            find_calibration_warnings(rows, RooflineModel(ModelDims(), HardwareSpec())),
            [],
        )

    def test_cli_exit_code_ignores_calibration_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scan.json"
            path.write_text(json.dumps(_crossover_payload()), encoding="utf-8")
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.account_decode",
                    str(path),
                    "--warnings-as-errors",
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                env={
                    "PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT}",
                    "PATH": "/usr/bin:/bin",
                },
            )
        # Crossover rows carry no cadence cross-check, so nothing is a hard error.
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
