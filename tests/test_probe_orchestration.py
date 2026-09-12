"""Exercise the GPU tools' orchestration with fakes (no CUDA required).

The CUDA loops in ``bench_roofline_a100`` and ``bench_burst_lower_bound`` are
the last untested layer before the A100 session.  They cannot run here, but the
Python around the kernel -- launch arguments, warmup/run counting, failure
propagation, statistics, report assembly, and the handoff to ``tools/verdict``
-- can, by injecting fake modules and a fake torch.

This is also the end-to-end check on the decision pipeline: synthetic probe
outputs are written to a temporary directory and fed through ``tools.verdict``
as the server would, so a schema/key drift between writer and reader fails here
instead of on the GPU box.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from nanovllm.roofline import ModelDims  # noqa: E402
from tools.bench_burst_lower_bound import (  # noqa: E402
    BARRIERS_PER_LAYER,
    bench_lower_bound,
    cooperative_preflight,
)
from tools.bench_roofline_a100 import bench_barrier  # noqa: E402
from tools.verdict import (  # noqa: E402
    VERDICT_DEVICE_RECOVERABLE,
    VERDICT_NEEDS_DATA,
    compute_verdict,
)


class _FakeEvent:
    """CUDA event stub whose elapsed time is scripted per sample."""

    def __init__(self, clock: "_FakeClock") -> None:
        self.clock = clock
        self.mark: float | None = None

    def record(self) -> None:
        self.mark = self.clock.now

    def elapsed_time(self, other: "_FakeEvent") -> float:
        assert self.mark is not None and other.mark is not None
        return other.mark - self.mark


class _FakeClock:
    def __init__(self, durations: list[float]) -> None:
        self.now = 0.0
        self.durations = list(durations)
        self.index = 0
        self.syncs = 0

    def advance(self, duration: float) -> None:
        self.now += duration

    def next_duration(self) -> float:
        value = self.durations[self.index % len(self.durations)]
        self.index += 1
        return value


class _FakeTorch:
    def __init__(self, durations: list[float]) -> None:
        self.clock = _FakeClock(durations)
        self.cuda = SimpleNamespace(
            Event=lambda enable_timing=True: self._event(),
            synchronize=self._synchronize,
        )

    def _event(self) -> _FakeEvent:
        # Recording the start event advances nothing; the module call advances
        # the clock by the next scripted duration, so elapsed == that duration.
        return _FakeEvent(self.clock)

    def _synchronize(self) -> None:
        self.clock.syncs += 1


class _FakeBarrierModule:
    def __init__(self, clock: _FakeClock, kernel_ms: float = 1.68) -> None:
        self.clock = clock
        self.kernel_ms = kernel_ms
        self.calls: list[tuple[int, int, int]] = []

    def run_barrier_probe(self, grid: int, block: int, iterations: int) -> float:
        self.calls.append((grid, block, iterations))
        # The host-side event delta is what the tool reports; the module's own
        # return value is the in-kernel check.
        self.clock.advance(self.clock.next_duration())
        return self.kernel_ms


class _FakeLowerBoundModule:
    def __init__(self, clock: _FakeClock, scripted: list[float]) -> None:
        self.clock = clock
        self.scripted = list(scripted)
        self.calls: list[dict] = []

    def run_lower_bound_probe(self, ptr, bytes_per_layer, buffer_bytes, nlayers,
                              barriers, grid, block, iterations) -> float:
        self.calls.append(
            {
                "ptr": ptr,
                "bytes_per_layer": bytes_per_layer,
                "buffer_bytes": buffer_bytes,
                "nlayers": nlayers,
                "barriers": barriers,
                "grid": grid,
                "block": block,
                "iterations": iterations,
            }
        )
        return self.scripted.pop(0)


class _FakeTensor:
    def __init__(self, numel: int, element_size: int) -> None:
        self._numel = numel
        self._element_size = element_size

    def numel(self) -> int:
        return self._numel

    def element_size(self) -> int:
        return self._element_size

    def data_ptr(self) -> int:
        return 4096


class BarrierProbeTests(unittest.TestCase):
    def _run(self, durations, *, warmup=2, runs=3, kernel_ms=1.68):
        torch = _FakeTorch(durations)
        # The event clock must advance while the module call happens; patch
        # advance onto the module through the shared clock.
        module = _FakeBarrierModule(torch.clock, kernel_ms=kernel_ms)
        report = bench_barrier(
            80, 512, 336, warmup, runs, module=module, torch_module=torch
        )
        return report, module, torch

    def test_warmup_runs_are_not_recorded(self) -> None:
        report, module, torch = self._run([1.60, 1.62, 1.64, 1.66, 1.68])
        self.assertEqual(len(module.calls), 2 + 3)
        self.assertEqual(len(report["samples_ms"]), 3)
        self.assertEqual(report["warmup_runs"], 2)

    def test_reports_microseconds_per_barrier(self) -> None:
        report, module, _ = self._run([1.68])
        # advance() happens once per call, consuming the same scripted duration
        # (1.68 ms) every time, so mean is 1.68 ms and 5 us/barrier for 336.
        self.assertAlmostEqual(report["mean_ms"], 1.68, places=6)
        self.assertAlmostEqual(report["us_per_barrier"], 1.68 * 1000.0 / 336, places=6)

    def test_launch_arguments_are_the_production_shape(self) -> None:
        _, module, _ = self._run([1.6])
        grid, block, iterations = module.calls[0]
        self.assertEqual((grid, block, iterations), (80, 512, 336))
        self.assertEqual(iterations, BARRIERS_PER_LAYER * 28)

    def test_negative_status_propagates(self) -> None:
        torch = _FakeTorch([1.6])
        module = _FakeBarrierModule(torch.clock, kernel_ms=-4.0)
        with self.assertRaises(RuntimeError) as ctx:
            bench_barrier(80, 512, 336, 0, 1, module=module, torch_module=torch)
        self.assertIn("cuda status", str(ctx.exception))

    def test_zero_runs_still_samples_once(self) -> None:
        report, _, _ = self._run([1.6], warmup=0, runs=0)
        self.assertEqual(len(report["samples_ms"]), 1)


class LowerBoundProbeTests(unittest.TestCase):
    def _weights(self, layers: int = 28, per_layer_bytes: int = 2 * 1024 * 1024):
        return _FakeTensor(per_layer_bytes * layers // 2, 2), per_layer_bytes * layers

    def test_launch_arguments_use_the_production_barrier_count(self) -> None:
        clock = _FakeClock([1.0])
        module = _FakeLowerBoundModule(clock, [1.4] * 6)
        weights = self._weights()
        report = bench_lower_bound(
            "unused", ModelDims(), [1, 8], 80, 512, 4, 1, 2,
            module=module, weights=weights,
        )
        first = module.calls[0]
        self.assertEqual(first["barriers"], BARRIERS_PER_LAYER)
        self.assertEqual(first["nlayers"], 28)
        self.assertEqual((first["grid"], first["block"]), (80, 512))
        self.assertEqual(report["rows"][0]["batch_size"], 1)
        # warmup(1) + runs(2) calls per batch, two batches.
        self.assertEqual(len(module.calls), 2 * 3)

    def test_refuses_when_the_buffer_cannot_hold_a_distinct_layer(self) -> None:
        # A loader that returns far less memory than the model needs: the
        # buffer (2048 B) is below one layer's stride (57344/28 = 2048 B, so
        # equal), meaning every layer would read the same addresses.
        weights = (_FakeTensor(1024, 2), 28 * 2048)
        with self.assertRaises(RuntimeError) as ctx:
            bench_lower_bound(
                "unused", ModelDims(), [1], 80, 512, 1, 0, 1,
                module=_FakeLowerBoundModule(_FakeClock([1.0]), [1.0]),
                weights=weights,
            )
        self.assertIn("distinct weights per layer", str(ctx.exception))

    def test_negative_status_propagates_with_grid_context(self) -> None:
        clock = _FakeClock([1.0])
        module = _FakeLowerBoundModule(clock, [-5.0])
        with self.assertRaises(RuntimeError) as ctx:
            bench_lower_bound(
                "unused", ModelDims(), [8], 80, 512, 1, 0, 1,
                module=module, weights=self._weights(),
            )
        self.assertIn("grid=80", str(ctx.exception))

    def test_verdict_thresholds_are_applied_to_the_probe_result(self) -> None:
        clock = _FakeClock([1.0])
        module = _FakeLowerBoundModule(clock, [4.0])
        report = bench_lower_bound(
            "unused", ModelDims(), [8], 80, 512, 1, 0, 1,
            module=module, weights=self._weights(),
        )
        row = report["rows"][0]
        self.assertEqual(row["ms_per_step_mean"], 4.0)
        self.assertIn("NOT achievable", row["verdict"])

    def test_gbps_is_computed_from_real_weight_bytes(self) -> None:
        clock = _FakeClock([1.0])
        module = _FakeLowerBoundModule(clock, [2.0])
        _, total_bytes = self._weights()
        report = bench_lower_bound(
            "unused", ModelDims(), [1], 80, 512, 1, 0, 1,
            module=module, weights=self._weights(),
        )
        expected = total_bytes / (2.0 / 1000.0) / 1e9
        self.assertAlmostEqual(
            report["rows"][0]["weight_stream_gbps"], expected, places=3
        )


class _Props:
    def __init__(self, sm_count: int = 108, cooperative=True) -> None:
        self.multi_processor_count = sm_count
        if cooperative is not None:
            self.cooperativeLaunch = cooperative


class _FakeCudaTorch:
    def __init__(self, props: _Props) -> None:
        self.cuda = SimpleNamespace(get_device_properties=lambda index: props)


class CooperativePreflightTests(unittest.TestCase):
    """Fail before compiling if the cooperative launch shape is impossible."""

    def test_a100_shape_is_accepted(self) -> None:
        result = cooperative_preflight(
            512, 80, torch_module=_FakeCudaTorch(_Props(108))
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["max_grid"], 108)
        self.assertEqual(result["sm_count"], 108)
        self.assertIsNone(result["error"])

    def test_grid_above_sm_count_is_rejected_with_the_fix(self) -> None:
        result = cooperative_preflight(
            512, 200, torch_module=_FakeCudaTorch(_Props(108))
        )
        self.assertFalse(result["ok"])
        self.assertIn("exceeds the cooperative limit 108", result["error"])
        self.assertIn("--grid-blocks 108", result["error"])

    def test_device_without_cooperative_launch_is_rejected(self) -> None:
        result = cooperative_preflight(
            512, 80, torch_module=_FakeCudaTorch(_Props(80, cooperative=False))
        )
        self.assertFalse(result["ok"])
        self.assertIn("cooperativeLaunch=False", result["error"])

    def test_missing_flag_is_tolerated(self) -> None:
        """Older torch builds do not expose cooperativeLaunch."""

        result = cooperative_preflight(
            512, 80, torch_module=_FakeCudaTorch(_Props(108, cooperative=None))
        )
        self.assertTrue(result["ok"])
        self.assertIsNone(result["cooperative_launch_flag"])

    def test_report_records_the_preflight(self) -> None:
        clock = _FakeClock([1.0])
        module = _FakeBarrierModule(clock)
        report = bench_barrier(
            80, 512, 336, 0, 1,
            module=module, torch_module=_FakeTorch([1.6]),
        )
        # Injected module means no preflight was run, but the key must exist so
        # downstream consumers can rely on its presence.
        self.assertIn("preflight", report)
        self.assertIsNone(report["preflight"])


class PipelineTests(unittest.TestCase):
    """Whatever the probes write, the verdict tool must be able to read."""

    def _write_probe_outputs(self, directory: Path, *, probe_ms: float) -> None:
        (directory / "a100_roofline_probe.json").write_text(
            json.dumps(
                {
                    "device": {"spec_bandwidth_gbps": 1555.0},
                    "bandwidth": {"best_gbps": 1650.0},
                    "barrier": {"ms_per_step_estimate": 0.548},
                }
            ),
            encoding="utf-8",
        )
        (directory / "a100_lower_bound.json").write_text(
            json.dumps(
                {
                    "lower_bound_probe": {
                        "rows": [
                            {
                                "batch_size": batch,
                                "ms_per_step_mean": probe_ms,
                                "weight_stream_gbps": 900.0,
                                "verdict": "x",
                            }
                            for batch in (1, 8, 32)
                        ]
                    },
                    "host_free_graph": [
                        {
                            "batch_size": batch,
                            "ms_per_step_host_inclusive": 3.4,
                            "ms_per_step_host_free": 0.9,
                        }
                        for batch in (1, 8, 32)
                    ],
                }
            ),
            encoding="utf-8",
        )

    def _run_verdict(self, directory: Path) -> dict:
        out = directory / "verdict.json"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.verdict",
                "--results-dir",
                str(directory),
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
        return json.loads(out.read_text(encoding="utf-8"))

    def test_probe_outputs_flow_into_a_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            self._write_probe_outputs(directory, probe_ms=1.0)
            payload = self._run_verdict(directory)
        # probe 1.0 vs host-free 0.9: no margin, so no device-side win.
        self.assertEqual(payload["verdict"], "DEVICE_BOUND_CUDA_GRAPH")
        self.assertTrue(payload["calibration"]["resolved"])

    def test_missing_probe_is_needs_data_not_a_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            (directory / "a100_roofline_probe.json").write_text(
                json.dumps({"bandwidth": {"best_gbps": 1650.0}}), encoding="utf-8"
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.verdict",
                    "--results-dir",
                    str(directory),
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                env={
                    "PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT}",
                    "PATH": "/usr/bin:/bin",
                },
            )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("NEEDS_DATA", proc.stdout)

    def test_verdict_functions_and_cli_agree(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            self._write_probe_outputs(directory, probe_ms=1.0)
            payload = self._run_verdict(directory)
            direct = compute_verdict(
                roofline=json.loads(
                    (directory / "a100_roofline_probe.json").read_text()
                ),
                lower_bound=json.loads(
                    (directory / "a100_lower_bound.json").read_text()
                ),
            )
        self.assertEqual(payload["verdict"], direct.verdict)

    def test_recoverable_pipeline_needs_probe_below_host_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            self._write_probe_outputs(directory, probe_ms=0.5)
            payload = self._run_verdict(directory)
        self.assertEqual(payload["verdict"], "HOST_BOUND_OPPORTUNITY")


class NoDataTests(unittest.TestCase):
    def test_compute_verdict_with_nothing_is_needs_data(self) -> None:
        self.assertEqual(compute_verdict().verdict, VERDICT_NEEDS_DATA)

    def test_recoverable_constant_is_not_reachable_without_a_probe(self) -> None:
        self.assertNotEqual(compute_verdict().verdict, VERDICT_DEVICE_RECOVERABLE)


if __name__ == "__main__":
    unittest.main()
