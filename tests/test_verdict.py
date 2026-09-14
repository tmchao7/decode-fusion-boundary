"""Tests for tools/verdict.py: the mechanical high-batch decision.

The thresholds are the contract in docs/07 section 10.4, so they are tested
against synthetic evidence covering every branch of the verdict tree --
including the cases that must *refuse* to promise a speedup.
"""

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
from tools.verdict import (  # noqa: E402
    VERDICT_DEVICE_AT_FLOOR,
    VERDICT_DEVICE_RECOVERABLE,
    VERDICT_DEVICE_BORDERLINE,
    VERDICT_DEVICE_BOUND_CUDA_GRAPH,
    VERDICT_DEVICE_UNREACHABLE,
    VERDICT_HOST_BOUND_OPPORTUNITY,
    VERDICT_NEEDS_CALIBRATION,
    VERDICT_NEEDS_DATA,
    compute_verdict,
    render_markdown,
)

FLOOR_MS = 0.737  # 1.503 GB (32 seqs, untied) / 2039 GB/s (A100-80GB)


def _lower_bound(
    ms_per_step: float,
    batches=(1, 8, 32),
    host_free: float | None = None,
    host_inclusive: float | None = None,
) -> dict:
    report = {
        "lower_bound_probe": {
            "rows": [
                {
                    "batch_size": batch,
                    "ms_per_step_mean": ms_per_step,
                    "verdict": "x",
                }
                for batch in batches
            ]
        }
    }
    if host_free is not None:
        report["host_free_graph"] = [
            {
                "batch_size": batch,
                "ms_per_step_host_inclusive": (
                    host_inclusive if host_inclusive is not None else host_free
                ),
                "ms_per_step_host_free": host_free,
            }
            for batch in batches
        ]
    return report


def _roofline(bandwidth_gbps: float | None, barrier_ms: float = 0.548) -> dict:
    report = {
        "device": {"spec_bandwidth_gbps": 2039.0},
        "barrier": {"ms_per_step_estimate": barrier_ms},
    }
    if bandwidth_gbps is not None:
        report["bandwidth"] = {"best_gbps": bandwidth_gbps}
    return report


def _account(gpu_ms: float, host_ms: float, backend: str = "megakernel-burst",
             batch: int = 8) -> dict:
    return {
        "rows": [
            {
                "workload_id": f"crossover_b{batch}",
                "backend_actual": backend,
                "batch_size": batch,
                "ms_per_step_wall": gpu_ms + host_ms,
                "ms_per_step_gpu": gpu_ms,
                "host_residual_ms": host_ms,
                "device_source": "measured",
            }
        ]
    }


class VerdictBranchTests(unittest.TestCase):
    def test_no_probe_is_needs_data(self) -> None:
        verdict = compute_verdict(roofline=_roofline(1600.0))
        self.assertEqual(verdict.verdict, VERDICT_NEEDS_DATA)
        self.assertIn("has not run", verdict.headline)

    def test_probe_without_bandwidth_is_needs_calibration(self) -> None:
        """A recoverable-looking floor is still untrustworthy uncalibrated.

        With tied_embedding=True (Qwen3-0.6B) the LM head correction uses the
        spec bandwidth when no measured bandwidth is available, so the full
        floor = probe + lm_head_at_spec = 1.2 + 0.153 = 1.353 ms.
        """

        verdict = compute_verdict(
            roofline=_roofline(None), lower_bound=_lower_bound(1.2)
        )
        self.assertEqual(verdict.verdict, VERDICT_NEEDS_CALIBRATION)
        self.assertIn("1.353", verdict.headline)

    def test_probe_above_stop_threshold_is_unreachable(self) -> None:
        verdict = compute_verdict(
            roofline=_roofline(1600.0), lower_bound=_lower_bound(4.0)
        )
        self.assertEqual(verdict.verdict, VERDICT_DEVICE_UNREACHABLE)
        self.assertTrue(
            any("stop the batch>1 kernel rewrite" in action for action in verdict.actions)
        )

    def test_probe_between_thresholds_is_borderline(self) -> None:
        verdict = compute_verdict(
            roofline=_roofline(1600.0), lower_bound=_lower_bound(2.2)
        )
        self.assertEqual(verdict.verdict, VERDICT_DEVICE_BORDERLINE)
        self.assertTrue(
            any("narrower justification" in action for action in verdict.actions)
        )

    def test_recoverable_probe_with_real_gap_is_recoverable(self) -> None:
        verdict = compute_verdict(
            roofline=_roofline(1600.0),
            lower_bound=_lower_bound(1.1),
            account=_account(gpu_ms=98.5, host_ms=1.0),
        )
        self.assertEqual(verdict.verdict, VERDICT_DEVICE_RECOVERABLE)
        self.assertAlmostEqual(verdict.device.probe_ms_per_step, 1.1, places=3)
        self.assertAlmostEqual(verdict.device.measured_ms_per_step, 98.5, places=3)

    def test_lm_head_correction_promotes_recoverable_to_borderline(self) -> None:
        """The probe streams per-layer weights only; the verdict adds the LM
        head read analytically.  A probe of 1.4 ms looks recoverable, but the
        LM head correction (~0.19 ms at 1600 GB/s) pushes the full floor to
        ~1.59 ms -> borderline.
        """

        verdict = compute_verdict(
            roofline=_roofline(1600.0),
            lower_bound=_lower_bound(1.4),
        )
        self.assertEqual(verdict.verdict, VERDICT_DEVICE_BORDERLINE)
        self.assertAlmostEqual(verdict.device.probe_ms_per_step, 1.4, places=3)
        self.assertIsNotNone(verdict.device.lm_head_stream_ms)
        self.assertIsNotNone(verdict.device.full_device_floor_ms)
        self.assertGreater(
            verdict.device.full_device_floor_ms, verdict.device.probe_ms_per_step
        )
        self.assertGreater(verdict.device.full_device_floor_ms, 1.5)

    def test_lm_head_correction_fields_are_in_the_json(self) -> None:
        verdict = compute_verdict(
            roofline=_roofline(1600.0),
            lower_bound=_lower_bound(1.1),
        )
        d = verdict.to_dict()["device"]
        self.assertIn("lm_head_stream_ms", d)
        self.assertIn("full_device_floor_ms", d)

    def test_recoverable_probe_but_device_at_floor_reports_host_only(self) -> None:
        """Honesty guard: no device-side promise when the gap is already closed."""

        verdict = compute_verdict(
            roofline=_roofline(1600.0),
            lower_bound=_lower_bound(1.1),
            account=_account(gpu_ms=1.2, host_ms=2.5),
        )
        self.assertEqual(verdict.verdict, VERDICT_DEVICE_AT_FLOOR)
        self.assertIn("host overhead", verdict.headline)
        self.assertTrue(
            any("host overhead" in action for action in verdict.actions)
        )


class HostOverlayTests(unittest.TestCase):
    """Experiment C: the measured opponent a persistent kernel must beat."""

    def test_host_overlay_is_measured_from_the_two_replays(self) -> None:
        verdict = compute_verdict(
            roofline=_roofline(1600.0),
            lower_bound=_lower_bound(1.1, host_free=0.9, host_inclusive=3.4),
            account=_account(gpu_ms=98.5, host_ms=1.0),
        )
        overlay = verdict.host_overlay
        self.assertAlmostEqual(overlay.host_free_ms_per_step, 0.9, places=3)
        self.assertAlmostEqual(overlay.measured_host_ms_per_step, 2.5, places=3)
        self.assertTrue(overlay.host_overlay_material)
        # probe 1.1 vs host-free 0.9: no margin to exploit.
        self.assertFalse(overlay.megakernel_could_win)

    def test_device_bound_cuda_graph_when_host_free_hits_the_floor(self) -> None:
        # probe 1.0 and host-free 0.95: no margin, and cudagraph pays a
        # material host overlay, so there is nothing left to win.
        verdict = compute_verdict(
            roofline=_roofline(1600.0),
            lower_bound=_lower_bound(1.0, host_free=0.95, host_inclusive=3.4),
            account=_account(gpu_ms=98.5, host_ms=1.0),
        )
        self.assertEqual(verdict.verdict, VERDICT_DEVICE_BOUND_CUDA_GRAPH)
        self.assertIn("No device-side win", verdict.headline)

    def test_host_bound_opportunity_when_megakernel_has_margin(self) -> None:
        verdict = compute_verdict(
            roofline=_roofline(1600.0),
            lower_bound=_lower_bound(1.0, host_free=3.0, host_inclusive=5.5),
            account=_account(gpu_ms=98.5, host_ms=1.0),
        )
        self.assertEqual(verdict.verdict, VERDICT_HOST_BOUND_OPPORTUNITY)
        self.assertTrue(verdict.host_overlay.megakernel_could_win)
        self.assertIn("Host-bound opportunity", verdict.headline)

    def test_no_host_free_data_leaves_overlay_empty(self) -> None:
        verdict = compute_verdict(
            roofline=_roofline(1600.0),
            lower_bound=_lower_bound(1.1),
            account=_account(gpu_ms=98.5, host_ms=1.0),
        )
        self.assertIsNone(verdict.host_overlay.host_free_ms_per_step)
        self.assertIsNone(verdict.host_overlay.megakernel_could_win)
        self.assertEqual(verdict.verdict, VERDICT_DEVICE_RECOVERABLE)

    def test_host_free_uses_the_largest_batch_row(self) -> None:
        """The overlay must be computed within a single row (same batch), not
        by mixing values from different batches.  The largest batch is
        selected because the verdict is about the batch>1 question."""
        lower_bound = _lower_bound(1.0, host_free=2.0, host_inclusive=3.0)
        # Batch 1 has a smaller free, but we should select batch 32.
        lower_bound["host_free_graph"][0]["ms_per_step_host_free"] = 1.4
        lower_bound["host_free_graph"][0]["ms_per_step_host_inclusive"] = 1.5
        verdict = compute_verdict(roofline=_roofline(1600.0), lower_bound=lower_bound)
        self.assertAlmostEqual(
            verdict.host_overlay.host_free_ms_per_step, 2.0, places=3
        )
        self.assertAlmostEqual(
            verdict.host_overlay.host_inclusive_ms_per_step, 3.0, places=3
        )

    def test_overlay_does_not_mix_free_and_inclusive_from_different_batches(self) -> None:
        """Regression for P0-3: _host_free_evidence used to take min(free) and
        min(inclusive) from different rows, which could pair b1's tiny free
        with b8's large inclusive and mask the true overlay, flipping
        DEVICE_BOUND_CUDA_GRAPH into DEVICE_RECOVERABLE.

        With the fix, both values come from batch=32 (the largest):
        free=0.03, inclusive=3.03 -> overlay=3.0 (material) -> DEVICE_BOUND_CUDA_GRAPH.
        The old buggy code would mix free=0.01 (b8) with inclusive=0.12 (b1)
        -> overlay=0.11 (not material) -> DEVICE_RECOVERABLE.
        """
        lower_bound = _lower_bound(1.0)
        lower_bound["host_free_graph"] = [
            {"batch_size": 1, "ms_per_step_host_free": 0.02,
             "ms_per_step_host_inclusive": 0.12},
            {"batch_size": 8, "ms_per_step_host_free": 0.01,
             "ms_per_step_host_inclusive": 2.51},
            {"batch_size": 32, "ms_per_step_host_free": 0.03,
             "ms_per_step_host_inclusive": 3.03},
        ]
        verdict = compute_verdict(roofline=_roofline(1600.0), lower_bound=lower_bound)
        self.assertAlmostEqual(
            verdict.host_overlay.host_free_ms_per_step, 0.03, places=3
        )
        self.assertAlmostEqual(
            verdict.host_overlay.host_inclusive_ms_per_step, 3.03, places=3
        )
        self.assertAlmostEqual(
            verdict.host_overlay.measured_host_ms_per_step, 3.0, places=2
        )
        self.assertTrue(verdict.host_overlay.host_overlay_material)
        self.assertEqual(verdict.verdict, VERDICT_DEVICE_BOUND_CUDA_GRAPH)

    def test_markdown_reports_the_host_overlay(self) -> None:
        verdict = compute_verdict(
            roofline=_roofline(1600.0),
            lower_bound=_lower_bound(1.0, host_free=3.0, host_inclusive=5.5),
        )
        text = render_markdown(verdict)
        self.assertIn("## Host overlay", text)
        self.assertIn("measured host overlay", text)


class CalibrationTests(unittest.TestCase):
    def test_contradiction_detected_when_the_timing_is_impossible(self) -> None:
        """0.5 ms/step implies 2384 GB/s, which is above the 2039 GB/s spec.

        The contradiction means the recording cannot have covered a whole step.
        """

        verdict = compute_verdict(
            roofline=_roofline(1600.0),
            lower_bound=_lower_bound(1.1),
            account=_account(gpu_ms=0.5, host_ms=2.2, backend="torch", batch=2),
        )
        self.assertTrue(verdict.calibration.contradiction_present)
        self.assertAlmostEqual(
            verdict.calibration.implied_bandwidth_gbps, 2384.0, delta=5.0
        )

    def test_documented_074ms_is_not_a_contradiction_on_an_80gb_a100(self) -> None:
        """The batched cudagraph step sits AT the HBM wall, it does not beat it.

        Regression for the old 1555 GB/s constant (the 40 GB part): against that
        number 0.74 ms looked impossible and sent the project hunting for a
        missing nsys capture. Against the real 2039 GB/s spec it is 99.7% of
        peak with the untied 1.503 GB accounting, and a plausible 79% with
        Qwen3's true tied traffic.
        """

        verdict = compute_verdict(
            roofline=_roofline(2039.0),
            lower_bound=_lower_bound(1.1),
            account=_account(gpu_ms=0.74, host_ms=2.2, backend="torch", batch=2),
        )
        self.assertFalse(verdict.calibration.contradiction_present)

    def test_contradiction_absent_when_device_is_above_floor(self) -> None:
        verdict = compute_verdict(
            roofline=_roofline(1600.0),
            lower_bound=_lower_bound(1.1),
            account=_account(gpu_ms=2.0, host_ms=1.5),
        )
        self.assertFalse(verdict.calibration.contradiction_present)
        self.assertIsNone(verdict.calibration.implied_bandwidth_gbps)

    def test_measured_bandwidth_resolves_calibration(self) -> None:
        # The implied rate (2384 GB/s at 0.5 ms with Qwen3's tied 1.192 GB
        # traffic) exceeds both the 2039 GB/s spec and the measured 1500 GB/s,
        # so the contradiction is resolved as "NOT reachable": the recording
        # cannot have covered a whole step.
        verdict = compute_verdict(
            roofline=_roofline(1500.0),
            lower_bound=_lower_bound(1.1),
            account=_account(gpu_ms=0.5, host_ms=2.2, backend="torch", batch=2),
        )
        self.assertTrue(verdict.calibration.resolved)
        self.assertEqual(verdict.calibration.measured_bandwidth_gbps, 1500.0)
        self.assertTrue(any("NOT reachable" in note for note in verdict.calibration.notes))

    def test_needs_calibration_mentions_the_implied_rate(self) -> None:
        verdict = compute_verdict(
            roofline=_roofline(None),
            lower_bound=_lower_bound(1.1),
            account=_account(gpu_ms=0.5, host_ms=2.2, backend="torch", batch=2),
        )
        self.assertEqual(verdict.verdict, VERDICT_NEEDS_CALIBRATION)
        self.assertTrue(any("2384" in action for action in verdict.actions))


class DeviceRowSelectionTests(unittest.TestCase):
    def test_megakernel_rows_preferred_over_cudagraph(self) -> None:
        account = {
            "rows": [
                {**_account(0.77, 2.66, backend="torch", batch=8)["rows"][0]},
                {**_account(98.5, 1.0, backend="megakernel-burst", batch=8)["rows"][0]},
            ]
        }
        verdict = compute_verdict(
            roofline=_roofline(1600.0),
            lower_bound=_lower_bound(1.1),
            account=account,
        )
        self.assertEqual(verdict.device.measured_backend, "megakernel-burst")
        self.assertAlmostEqual(verdict.device.measured_ms_per_step, 98.5, places=3)

    def test_batch1_is_ignored_for_the_batch_gt_1_question(self) -> None:
        verdict = compute_verdict(
            roofline=_roofline(1600.0),
            lower_bound=_lower_bound(1.1),
            account=_account(gpu_ms=1.89, host_ms=0.4, backend="megakernel", batch=1),
        )
        self.assertIsNone(verdict.device.measured_ms_per_step)

    def test_probe_floor_uses_the_fastest_row(self) -> None:
        lower_bound = {
            "lower_bound_probe": {
                "rows": [
                    {"batch_size": 1, "ms_per_step_mean": 1.30},
                    {"batch_size": 8, "ms_per_step_mean": 1.05},
                    {"batch_size": 32, "ms_per_step_mean": 1.12},
                ]
            }
        }
        verdict = compute_verdict(
            roofline=_roofline(1600.0), lower_bound=lower_bound
        )
        self.assertAlmostEqual(verdict.device.probe_ms_per_step, 1.05, places=3)
        self.assertEqual(verdict.device.probe_batch_sizes, [1, 8, 32])


class ReportTests(unittest.TestCase):
    def test_markdown_carries_verdict_evidence_and_actions(self) -> None:
        verdict = compute_verdict(
            roofline=_roofline(1600.0), lower_bound=_lower_bound(4.0)
        )
        text = render_markdown(verdict)
        self.assertIn(VERDICT_DEVICE_UNREACHABLE, text)
        self.assertIn("## Evidence", text)
        self.assertIn("## Actions", text)
        self.assertIn("device lower bound probe", text)

    def test_markdown_reports_lm_head_correction(self) -> None:
        """The LM head correction must appear in the markdown evidence."""
        verdict = compute_verdict(
            roofline=_roofline(1600.0), lower_bound=_lower_bound(1.4)
        )
        text = render_markdown(verdict)
        self.assertIn("LM head", text)
        self.assertIn("full device floor", text)

    def test_json_round_trips_all_fields(self) -> None:
        verdict = compute_verdict(
            roofline=_roofline(1600.0),
            lower_bound=_lower_bound(1.1),
            account=_account(gpu_ms=98.5, host_ms=1.0),
        )
        payload = json.loads(json.dumps(verdict.to_dict()))
        self.assertEqual(payload["verdict"], VERDICT_DEVICE_RECOVERABLE)
        self.assertAlmostEqual(payload["device"]["probe_ms_per_step"], 1.1, places=3)
        self.assertIn("calibration", payload)


class CliTests(unittest.TestCase):
    def _write(self, directory: Path, name: str, payload: dict) -> None:
        (directory / name).write_text(json.dumps(payload), encoding="utf-8")

    def test_cli_defaults_to_results_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            results = Path(tmpdir)
            self._write(results, "a100_roofline_probe.json", _roofline(1600.0))
            self._write(results, "a100_lower_bound.json", _lower_bound(4.0))
            self._write(
                results, "a100_account_breakdown.json", _account(98.5, 1.0)
            )
            out = results / "verdict.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.verdict",
                    "--results-dir",
                    str(results),
                    "--json",
                    str(out),
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                env={
                    "PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT}",
                    "PATH": "/usr/bin:/bin",
                },
            )
            self.assertIn(proc.returncode, (0, 1), proc.stderr)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["verdict"], VERDICT_DEVICE_UNREACHABLE)

    def test_cli_without_inputs_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = subprocess.run(
                [sys.executable, "-m", "tools.verdict", "--results-dir", tmpdir],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                env={
                    "PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT}",
                    "PATH": "/usr/bin:/bin",
                },
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("no probe inputs found", proc.stderr)


class ThresholdAlignmentTests(unittest.TestCase):
    def test_thresholds_match_the_documented_floor(self) -> None:
        model = RooflineModel(ModelDims(), HardwareSpec())
        floor = model.weight_bytes() / (
            model.hardware.effective_bandwidth_gbps() * 1e9
        ) * 1000.0
        self.assertAlmostEqual(floor, FLOOR_MS, places=3)
        # The recoverable threshold must sit above the physical floor, and the
        # stop threshold above that, or the decision tree is nonsensical.
        from tools.verdict import RECOVERABLE_MS, STOP_MS

        self.assertGreater(RECOVERABLE_MS, floor)
        self.assertGreater(STOP_MS, RECOVERABLE_MS)


if __name__ == "__main__":
    unittest.main()
