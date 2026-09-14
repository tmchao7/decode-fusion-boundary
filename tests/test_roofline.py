"""Unit tests for the analytical decode-step roofline model."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nanovllm.roofline import (  # noqa: E402
    DEFAULT_KERNEL_SOURCE,
    HardwareSpec,
    ModelDims,
    RooflineModel,
    account_step,
    assert_kernel_constants,
    gap_table,
    parse_kernel_constants,
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
        # Hand count: qkv 1024*(2048+2048)=4.19M, o 2048*1024=2.10M,
        # mlp 3*1024*3072=9.44M, 2 norms 2048 -> 15,730,688 per layer
        self.assertEqual(dims.params_per_layer(), 15_730_688)
        layers = dims.num_layers * dims.params_per_layer()
        lm_head = dims.params_lm_head()
        self.assertEqual(lm_head, 151936 * 1024)
        # Qwen3-0.6B has tie_word_embeddings=True.
        self.assertEqual(dims.total_params(tied_embedding=True), layers + lm_head)
        self.assertEqual(
            dims.total_params(tied_embedding=False),
            layers + dims.params_embedding() + lm_head,
        )

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
        self.assertEqual(
            parts["layers_bytes"] + parts["lm_head_bytes"] + parts["embedding_bytes"],
            parts["weight_bytes_total"],
        )
        self.assertEqual(parts["weight_bytes_total"], model.weight_bytes())


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
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "nanovllm.roofline",
                    "--batch-sizes",
                    "1,8",
                    "--seq-len",
                    "224",
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
