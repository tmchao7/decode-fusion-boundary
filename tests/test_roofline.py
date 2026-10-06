"""Unit tests for the analytical decode-step roofline model."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nanovllm.roofline import (  # noqa: E402
    DEFAULT_BANDWIDTH_JSON,
    A100_SXM4_80GB,
    DEFAULT_KERNEL_SOURCE,
    RTX_5090,
    HardwareSpec,
    ModelDims,
    RooflineModel,
    account_step,
    assert_kernel_constants,
    gap_table,
    hardware_from_bandwidth_json,
    load_measured_bandwidth,
    measured_hardware_spec,
    parse_kernel_constants,
    read_bandwidth_probe,
    render_markdown,
    step_accounting_from_metrics,
)

# Documented A100 crossover-scan wall times (docs/07 §9.6), ms/token/sequence.
DOCUMENTED_WALL_MS_PER_TOKEN = {1: 2.944, 2: 1.634, 8: 0.429, 32: 0.111}


class ModelDimsTests(unittest.TestCase):
    def test_streaming_params_are_layers_plus_one_lm_head(self) -> None:
        """The LM head is streamed once; the embedding copy is not a stream.

        ``total_params`` counts the parameters that cross HBM in one decode
        step, so it is layer weights + one LM head matrix for a tied model and
        layer weights + embedding table + LM head for an untied one.
        """

        dims = ModelDims()
        # Verified against the instantiated model: its layer-0 tensors sum to exactly
        # 15,730,944 (11 tensors). The earlier hand count here used 2 norms and no
        # q_norm/k_norm, which is 256 short; a separately amortised final norm brought
        # the total back to within 0.002%, which is exactly why it went unnoticed.
        #   q 2048x1024 + k 1024x1024 + v 1024x1024 + o 1024x2048 = 6,291,456
        #   input + post_attention norm = 2,048
        #   q_norm + k_norm = 2 x 128 = 256
        #   gate + up + down = 3 x 1024 x 3072 = 9,437,184
        self.assertEqual(dims.params_per_layer(), 15_730_944)
        self.assertEqual(dims.final_norm_params(), 1024)
        layers = dims.num_layers * dims.params_per_layer()
        lm_head = dims.params_lm_head()
        self.assertEqual(lm_head, 151936 * 1024)
        # Qwen3-0.6B has tie_word_embeddings=True.
        self.assertEqual(dims.total_params(tied_embedding=True), layers + 1024 + lm_head)
        self.assertEqual(
            dims.total_params(tied_embedding=False),
            layers + 1024 + dims.params_embedding() + lm_head,
        )
        # The instantiated model reports exactly this total.
        self.assertEqual(dims.total_params(tied_embedding=True), 596_049_920)

    def test_weight_bytes_counts_the_lm_head_exactly_once(self) -> None:
        model = RooflineModel()
        # Untied: layer weights + embedding table + LM head.
        self.assertAlmostEqual(model.weight_bytes() / 1e9, 1.503, places=3)
        # The batch-independent core is layers + one LM head, which is also the
        # correct per-step traffic for Qwen3-0.6B (tied embeddings).
        self.assertAlmostEqual(
            model.weight_bytes_excl_embedding() / 1e9, 1.192, places=3
        )
        self.assertAlmostEqual(model.core_weight_bytes() / 1e9, 1.192, places=3)
        self.assertAlmostEqual(model.embedding_bytes() / 1e9, 0.311, places=3)
        tied = RooflineModel(tied_embedding=True)
        self.assertEqual(tied.weight_bytes(), tied.core_weight_bytes())

    def test_tied_embedding_does_not_double_count_the_matrix(self) -> None:
        """Tying removes the *second* copy of the 311 MB matrix, not the LM head.

        The regression this locks: committing 2d21fd8-era code counted the
        embedding table and the (tied) LM head as separate 311 MB reads, which
        inflated the Qwen3 floor by 26% and made a real 0.74 ms/step look faster
        than HBM.
        """

        tied = RooflineModel(tied_embedding=True)
        untied = RooflineModel()
        delta = untied.weight_bytes() - tied.weight_bytes()
        self.assertEqual(delta, 151936 * 1024 * 2)
        # Sanity: with the correct numbers 0.74 ms is a plausible 79% of peak.
        implied_gbps = tied.weight_bytes() / 0.74e-3 / 1e9
        self.assertGreater(implied_gbps, 1400.0)
        self.assertLess(implied_gbps, tied.hardware.bandwidth_gbps)

    def test_weight_breakdown_sums_to_total(self) -> None:
        model = RooflineModel()
        parts = model.weight_bytes_breakdown()
        # The final norm is its own row because it is one tensor for the whole model:
        # folding it into `layers_bytes` is what let it be counted an extra time.
        self.assertEqual(
            parts["layers_bytes"]
            + parts["final_norm_bytes"]
            + parts["lm_head_bytes"]
            + parts["embedding_bytes"],
            parts["weight_bytes_total"],
        )
        self.assertEqual(parts["final_norm_bytes"], 2048)
        self.assertEqual(parts["weight_bytes_total"], model.weight_bytes())
        self.assertEqual(
            parts["layers_bytes"] + parts["final_norm_bytes"] + parts["lm_head_bytes"],
            model.core_weight_bytes(),
        )


class HardwareTests(unittest.TestCase):
    def test_default_hardware_is_a100(self) -> None:
        hw = HardwareSpec()
        # A100-SXM4-80GB HBM2e peak; 1555 GB/s is the 40 GB part.
        self.assertEqual(hw.bandwidth_gbps, 2039.0)
        self.assertEqual(hw.peak_tflops, 312.0)

    def test_compute_efficiency_scales_tflops(self) -> None:
        hw = HardwareSpec(peak_tflops=100.0, compute_efficiency=0.25)
        self.assertEqual(hw.effective_tflops(), 25.0)


class RooflineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = RooflineModel()

    def test_memory_bound_across_practical_batches(self) -> None:
        for batch in (1, 2, 8, 32, 64, 128):
            self.assertTrue(
                self.model.is_memory_bound(batch, seq_len=224),
                f"batch={batch} should still be weight-bandwidth bound",
            )

    def test_lower_bound_is_max_of_memory_and_compute(self) -> None:
        for batch in (1, 8, 32):
            self.assertAlmostEqual(
                self.model.lower_bound_ms(batch, 224),
                max(
                    self.model.memory_bound_ms(batch, 224),
                    self.model.compute_bound_ms(batch, 224),
                ),
                places=9,
            )

    def test_weight_streaming_floor_is_batch_independent(self) -> None:
        """The weight term is the same for every batch size."""

        floor = self.model.weight_bytes() / (2039.0 * 1e9) * 1000.0
        for batch in (1, 8, 32):
            self.assertGreaterEqual(self.model.memory_bound_ms(batch, 1), floor)

    def test_compute_bound_crossover_is_in_the_hundreds(self) -> None:
        # 2 * batch * 596e6 params vs 1.192 GB / 2039 GB/s at 312 TFLOP/s
        crossover = self.model.compute_bound_batch(seq_len=224)
        self.assertGreater(crossover, 100.0)
        self.assertLess(crossover, 1000.0)
        half = max(1, int(crossover * 0.5))
        double = int(crossover * 4)
        self.assertTrue(self.model.is_memory_bound(half, 224, ignore_kv=True))
        self.assertFalse(self.model.is_memory_bound(double, 224, ignore_kv=True))

    def test_attainable_tok_s_scales_with_batch(self) -> None:
        previous = 0.0
        for batch in (1, 2, 4, 8):
            current = self.model.attainable_tok_s(batch, 224)
            self.assertGreater(current, previous)
            previous = current

    def test_kv_traffic_grows_with_seq_len_and_amplification(self) -> None:
        amp1 = RooflineModel(kv_read_amp=1)
        amp8 = RooflineModel(kv_read_amp=8)
        self.assertEqual(amp8.kv_bytes(1, 224), 8 * amp1.kv_bytes(1, 224))
        self.assertGreater(amp1.kv_bytes(1, 448), amp1.kv_bytes(1, 224))

    def test_arithmetic_intensity_low_enough_to_share_hbm_wall(self) -> None:
        """At batch=1 intensity is far below the machine balance point, so both
        cuBLAS and a fused kernel are bounded by the same HBM bandwidth."""

        intensity = self.model.arithmetic_intensity(1, 224)
        machine_balance = (
            self.model.hardware.effective_tflops()
            * 1e12
            / (self.model.hardware.effective_bandwidth_gbps() * 1e9)
        )
        self.assertLess(intensity, machine_balance)
        self.assertLess(intensity, 1.0)


class KernelConstantContractTests(unittest.TestCase):
    def test_parses_checked_in_kernel(self) -> None:
        constants = parse_kernel_constants(DEFAULT_KERNEL_SOURCE)
        self.assertEqual(constants["hidden_size"], 1024)
        self.assertEqual(constants["intermediate_size"], 3072)
        self.assertEqual(constants["num_q_heads"], 16)
        self.assertEqual(constants["num_kv_heads"], 8)
        self.assertEqual(constants["head_dim"], 128)
        self.assertEqual(constants["vocab_size"], 151936)

    def test_default_dims_match_kernel(self) -> None:
        assert_kernel_constants(ModelDims(), DEFAULT_KERNEL_SOURCE)

    def test_mismatch_is_reported(self) -> None:
        with self.assertRaises(ValueError):
            assert_kernel_constants(
                ModelDims(hidden_size=2048), DEFAULT_KERNEL_SOURCE
            )

    def test_missing_constant_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "kernel.cu"
            path.write_text("constexpr int HIDDEN_SIZE = 1024;\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_kernel_constants(path)


class StepAccountingTests(unittest.TestCase):
    def test_wall_time_is_per_token_times_batch(self) -> None:
        acc = account_step(
            backend="torch-cudagraph",
            batch_size=8,
            ms_per_token_wall=0.429,
            kernel_ms_per_step=0.77,
        )
        self.assertAlmostEqual(acc.ms_per_step_wall, 3.432, places=3)
        self.assertAlmostEqual(acc.host_residual_ms, 2.662, places=3)
        self.assertAlmostEqual(acc.gpu_fraction, 0.77 / 3.432, places=6)

    def test_documented_host_residual_range(self) -> None:
        """The whole thesis: ~2.2-2.8 ms/step is host, not device."""

        for batch, kernel_ms in ((1, 0.74), (8, 0.77), (32, 0.77)):
            acc = account_step(
                backend="torch-cudagraph",
                batch_size=batch,
                ms_per_token_wall=DOCUMENTED_WALL_MS_PER_TOKEN[batch],
                kernel_ms_per_step=kernel_ms,
            )
            self.assertGreater(acc.host_residual_ms, 2.0)
            self.assertLess(acc.host_residual_ms, 3.0)
            self.assertLess(acc.gpu_fraction, 0.30)

    def test_metrics_summary_accounting(self) -> None:
        summary = {
            "decode_steps": 4,
            "decode_time_ms_total": 13.728,
            "cuda_decode_time_ms_total": 3.08,
        }
        acc = step_accounting_from_metrics(
            backend="torch-cudagraph", batch_size=8, metrics={"summary": summary}
        )
        self.assertIsNotNone(acc)
        assert acc is not None
        self.assertAlmostEqual(acc.ms_per_step_wall, 3.432, places=3)
        self.assertAlmostEqual(acc.ms_per_step_gpu, 0.77, places=3)

    def test_metrics_without_decode_steps_returns_none(self) -> None:
        self.assertIsNone(
            step_accounting_from_metrics(
                backend="x", batch_size=1, metrics={"summary": {"decode_steps": 0}}
            )
        )


class GapTableTests(unittest.TestCase):
    def test_flags_calibration_when_measured_beats_model(self) -> None:
        """0.75 ms/step for 1.503 GB implies >2 TB/s, above the A100 spec."""

        model = RooflineModel()
        rows = gap_table(
            model,
            [1, 8],
            seq_len=224,
            kernel_ms_per_step=0.75,
            ms_per_token_wall=DOCUMENTED_WALL_MS_PER_TOKEN,
        )
        for row in rows:
            self.assertTrue(row["calibration_needed"])
            self.assertEqual(row["effective_lower_bound_ms"], 0.75)
            self.assertGreater(row["kernel_implied_bandwidth_gbps"], 2000.0)

    def test_no_calibration_when_measured_above_model(self) -> None:
        model = RooflineModel()
        rows = gap_table(
            model,
            [1],
            seq_len=224,
            kernel_ms_per_step=1.5,
            ms_per_token_wall={1: 2.944},
        )
        row = rows[0]
        self.assertFalse(row["calibration_needed"])
        self.assertAlmostEqual(row["headroom_ms"], 1.5 - row["lower_bound_ms"], places=9)
        # The ceiling is bounded by the measured 1.5 ms, which is above the
        # modeled 0.838 ms floor, so it sits below the 1193 tok/s bound and
        # above what the wall time achieves (1/2.944 ms = 340 tok/s).
        ceiling = row["ceiling_tok_s_if_host_removed"]
        self.assertGreater(ceiling, 900.0)
        self.assertLess(ceiling, 1200.0)

    def test_markdown_contains_warning_for_calibration(self) -> None:
        model = RooflineModel()
        rows = gap_table(
            model,
            [1],
            seq_len=224,
            kernel_ms_per_step=0.75,
            ms_per_token_wall={1: 2.944},
        )
        text = render_markdown(rows, model)
        self.assertIn("WARNING", text)
        self.assertIn("1.503 GB", text)  # untied model passed to gap_table here


class CliTests(unittest.TestCase):
    def test_cli_emits_json_and_checks_kernel(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "roofline.json"
            # An explicit measured rate: the CLI default is now the probe
            # artifact (results/bandwidth.json), which a checkout does not carry.
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "nanovllm.roofline",
                    "--batch-sizes",
                    "1,8",
                    "--seq-len",
                    "224",
                    "--bandwidth-gbps",
                    "1699",
                    "--check-kernel",
                    "--json",
                    str(out),
                ],
                cwd=str(Path(__file__).resolve().parents[1]),
                capture_output=True,
                text=True,
                env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            payload = json.loads(out.read_text(encoding="utf-8"))
            # The CLI's default target is Qwen3-0.6B, whose embeddings are tied,
            # so it reports the tied footprint unless asked otherwise.
            self.assertEqual(
                payload["weight_bytes"], RooflineModel(tied_embedding=True).weight_bytes()
            )
            self.assertAlmostEqual(payload["weight_bytes"] / 1e9, 1.192, places=3)
            self.assertEqual([row["batch_size"] for row in payload["rows"]], [1, 8])
            self.assertEqual(payload["bandwidth_source"]["kind"], "explicit_cli")
            self.assertEqual(payload["hardware"]["name"], "explicit --bandwidth-gbps")

    def test_cli_rejects_malformed_wall_pairs(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "nanovllm.roofline",
                "--ms-per-token-wall",
                "nonsense",
            ],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
        )
        self.assertNotEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()


class HardwareSpecPresetTests(unittest.TestCase):
    """Named device presets are spec values; the roofline must use them as given."""

    def test_the_a100_preset_is_the_dataclass_default(self) -> None:
        self.assertEqual(A100_SXM4_80GB.name, "A100-SXM4-80GB")
        self.assertEqual(A100_SXM4_80GB.bandwidth_gbps, 2039.0)

    def test_the_5090_preset_carries_its_own_spec(self) -> None:
        self.assertEqual(RTX_5090.name, "RTX 5090")
        self.assertEqual(RTX_5090.bandwidth_gbps, 1792.0)
        self.assertLess(RTX_5090.bandwidth_gbps, A100_SXM4_80GB.bandwidth_gbps)

    def test_the_5090_weight_stream_floor_is_tighter_than_the_spec_gap_suggests(self) -> None:
        """1.192 GB/step: 0.665 ms at 1792 GB/s spec vs 0.585 ms at 2039 GB/s.

        The two cards land within ~5% of each other once the A100's measured 83%
        of spec is accounted for, which is the point of calibrating per machine.
        """

        step_bytes = 1.192e9
        self.assertAlmostEqual(step_bytes / RTX_5090.bandwidth_gbps / 1e6, 0.665, places=3)
        self.assertAlmostEqual(
            step_bytes / A100_SXM4_80GB.bandwidth_gbps / 1e6, 0.585, places=3
        )


class MeasuredBandwidthTests(unittest.TestCase):
    """The roofline denominator is a measurement, never a spec number.

    ``docs/20`` §8 pins the 5090 number at 1559.2 GB/s (measured) against 1792
    GB/s (spec). These tests lock the loader to the probe artifact's real schema
    (``achieved_gbps.mean``) and lock it against silently guessing.
    """

    def _write(self, payload) -> Path:
        tmpdir = Path(tempfile.mkdtemp())
        path = tmpdir / "bandwidth.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _probe_payload(self, mean: float = 1559.2) -> dict:
        return {
            "requested_bytes": 1_200_000_000,
            "measured_bytes": 1_200_000_000,
            "iterations": 20,
            "warmup": 5,
            "device": "NVIDIA GeForce RTX 5090",
            "sm_count": 170,
            "spec_gbps": 1792.0,
            "achieved_gbps": {
                "count": 20,
                "mean": mean,
                "stddev": 10.6,
                "cov": 0.0068,
                "min": 1540.0,
                "max": 1575.0,
            },
            "achieved_fraction_of_spec": mean / 1792.0,
        }

    def test_reads_the_probe_artifacts_mean(self) -> None:
        path = self._write(self._probe_payload())
        self.assertAlmostEqual(load_measured_bandwidth(path), 1559.2, places=1)

    def test_builds_a_spec_whose_bandwidth_is_the_measurement(self) -> None:
        hw = hardware_from_bandwidth_json(self._write(self._probe_payload()))
        self.assertAlmostEqual(hw.bandwidth_gbps, 1559.2, places=1)
        self.assertIn("RTX 5090", hw.name)
        self.assertIn("measured", hw.name)

    def test_the_measured_denominator_is_not_the_spec_value(self) -> None:
        """1792 GB/s is the spec; using it understates the floor by ~13%."""

        hw = hardware_from_bandwidth_json(self._write(self._probe_payload()))
        model = RooflineModel(ModelDims(), hw, tied_embedding=True)
        measured_floor_ms = model.core_weight_bytes() / hw.bandwidth_gbps / 1e6
        spec_floor_ms = model.core_weight_bytes() / RTX_5090.bandwidth_gbps / 1e6
        self.assertGreater(measured_floor_ms, spec_floor_ms)
        self.assertAlmostEqual(measured_floor_ms / spec_floor_ms, 1792.0 / 1559.2, places=4)

    def test_probe_spec_gbps_never_reaches_the_hardware_spec(self) -> None:
        hw = hardware_from_bandwidth_json(self._write(self._probe_payload()))
        self.assertNotEqual(hw.bandwidth_gbps, 1792.0)
        self.assertNotEqual(hw.effective_bandwidth_gbps(), 1792.0)

    def test_missing_artifact_names_the_command_that_creates_it(self) -> None:
        missing = Path(tempfile.mkdtemp()) / "nope.json"
        with self.assertRaises(FileNotFoundError) as caught:
            load_measured_bandwidth(missing)
        self.assertIn("RUN_MODE=bandwidth", str(caught.exception))

    def test_a_spec_only_artifact_is_rejected(self) -> None:
        """A device spec without a measurement must not become a denominator."""

        path = self._write({"device": "RTX 5090", "spec_gbps": 1792.0})
        with self.assertRaises(ValueError) as caught:
            load_measured_bandwidth(path)
        self.assertIn("achieved_gbps", str(caught.exception))

    def test_null_mean_is_rejected_instead_of_read_as_zero(self) -> None:
        payload = self._probe_payload()
        payload["achieved_gbps"]["mean"] = None
        with self.assertRaises(ValueError):
            load_measured_bandwidth(self._write(payload))

    def test_non_positive_mean_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_measured_bandwidth(self._write(self._probe_payload(mean=0.0)))

    def test_measured_hardware_spec_refuses_a_nonsense_rate(self) -> None:
        for bad in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(bandwidth=bad):
                with self.assertRaises(ValueError):
                    measured_hardware_spec(bad)

    def test_cli_defaults_to_the_probe_artifact(self) -> None:
        """No --bandwidth-gbps and no artifact must fail, not fall back to spec."""

        with tempfile.TemporaryDirectory() as tmpdir:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "nanovllm.roofline",
                    "--batch-sizes",
                    "1",
                    "--bandwidth-json",
                    str(Path(tmpdir) / "absent.json"),
                ],
                cwd=str(Path(__file__).resolve().parents[1]),
                capture_output=True,
                text=True,
                env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("RUN_MODE=bandwidth", proc.stderr)

    def test_cli_records_where_the_denominator_came_from(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bandwidth.json"
            path.write_text(json.dumps(self._probe_payload()), encoding="utf-8")
            out = Path(tmpdir) / "roofline.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "nanovllm.roofline",
                    "--batch-sizes",
                    "1",
                    "--bandwidth-json",
                    str(path),
                    "--json",
                    str(out),
                ],
                cwd=str(Path(__file__).resolve().parents[1]),
                capture_output=True,
                text=True,
                env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["bandwidth_source"]["kind"], "measured_probe")
            self.assertEqual(payload["bandwidth_source"]["path"], str(path))
            self.assertAlmostEqual(payload["hardware"]["bandwidth_gbps"], 1559.2, places=1)

    def test_cli_rejects_both_denominator_inputs_at_once(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "nanovllm.roofline",
                "--bandwidth-gbps",
                "1559.2",
                "--bandwidth-json",
                "results/bandwidth.json",
            ],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("mutually exclusive", proc.stderr)
