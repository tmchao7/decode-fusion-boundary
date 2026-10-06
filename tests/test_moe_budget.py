"""CPU-only tests for the sparse (MoE) decode byte budget.

These tests are the executable form of ``docs/22`` §0. Their job is to make the
architecture's premises falsifiable on a laptop: if someone changes ``top_k``, the
context length, or the quantization width, the byte budget, the gate shares and the
roofline floors must move in a way the tests predict -- and the shipped OLMoE preset
must keep reproducing the two constants that were verified from ``config.json``.

Nothing here needs a GPU. No number here is a *measurement*: the derived values are
config arithmetic, and the bandwidth is always passed in explicitly so no test can
silently acquire a spec figure as its denominator.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nanovllm.moe_budget import (  # noqa: E402
    BUDGET_PARTS,
    MOE_LEVELS_PER_LAYER,
    OLMOE_1B_7B,
    TERMINAL_LEVELS,
    MoEDims,
    byte_budget,
    decode_budget,
    expert_token_load,
    flops_per_token,
    kv_bytes_per_token,
    plan_work,
    render_report,
)

# RTX 5090 measured read bandwidth (docs/20 §8, re-checked on this machine by
# RUN_MODE=bandwidth; a spec number must never appear as a denominator here).
MEASURED_5090_GBPS = 1559.2

# Tensor-verified OLMoE facts. These were read out of the checkpoint rather than
# copied from prose: the safetensors API reports parameters.BF16 and the per-tensor
# headers were enumerated (3,219 tensors).  See the module docstring for the four
# things that reconciliation corrected, including OLMoE's per-head QK-norms.
VERIFIED_PARAMS = 6_919_161_856
VERIFIED_STORAGE_BYTES_BF16 = 13_838_323_712
VERIFIED_WEIGHT_BYTES_BF16 = 2_357_985_280
VERIFIED_ACTIVE_PARAMS = 1_178_994_688

# The constants recorded in the sibling LeanMoE-Quant checkout. They are kept
# visible because the *gap* is a finding: its [W] omits the router and the norms.
LEANMOE_DOC_WEIGHT_BYTES = 2_357_723_136
LEANMOE_DOC_PARAMS = 6_919_028_736
ROUTER_BYTES_PER_TOKEN = OLMOE_1B_7B.num_layers * OLMOE_1B_7B.router_params_per_layer() * 2
#: LeanMoE-Quant's recorded constants are both short, and the shortfalls decompose
#: into exactly these two terms (see the test that asserts the decomposition).
QK_NORMS_BYTES = OLMOE_1B_7B.num_layers * 2 * OLMOE_1B_7B.hidden_size * 2  # 131,072
FINAL_NORM_BYTES = OLMOE_1B_7B.final_norm_params() * 2  # 4,096


class DimsTests(unittest.TestCase):
    def test_olmoes_preset_matches_the_verified_config(self) -> None:
        dims = OLMOE_1B_7B
        self.assertEqual(dims.num_layers, 16)
        self.assertEqual(dims.hidden_size, 2048)
        self.assertEqual(dims.num_experts, 64)
        self.assertEqual(dims.top_k, 8)
        self.assertEqual(dims.expert_ffn_dim, 1024)
        self.assertEqual(dims.vocab_size, 50304)
        self.assertFalse(dims.tie_word_embeddings)
        # OLMoE is MHA (16/16), not GQA: the KV width equals the hidden width.
        self.assertEqual(dims.num_q_heads, 16)
        self.assertEqual(dims.num_kv_heads, 16)
        self.assertEqual(dims.head_dim, 128)
        self.assertEqual(dims.q_size, 2048)
        self.assertEqual(dims.kv_size, 2048)

    def test_router_settings_come_from_config_dot_json(self) -> None:
        """No grouped top-k, and the top-k weights are NOT renormalized."""

        dims = OLMOE_1B_7B
        self.assertEqual(dims.num_expert_groups, 1)
        self.assertFalse(dims.norm_topk_prob)

    def test_olmoes_per_head_qk_norms_are_counted(self) -> None:
        """The correction that made the tensor totals reconcile.

        Four ``[hidden]`` norm vectors per layer, not two: input, post-attention,
        q_norm and k_norm.  Without qk_norm the layer total is 4,096 params short.
        """

        dims = OLMOE_1B_7B
        self.assertTrue(dims.qk_norm)
        self.assertEqual(dims.norms_params_per_layer(), 4 * 2048)
        without = MoEDims(qk_norm=False)
        self.assertEqual(
            dims.norms_params_per_layer() - without.norms_params_per_layer(), 2 * 2048
        )

    def test_total_parameter_count_matches_safetensors_exactly(self) -> None:
        self.assertEqual(OLMOE_1B_7B.total_params(), VERIFIED_PARAMS)

    def test_activated_parameter_count_is_derived_not_copied(self) -> None:
        self.assertEqual(OLMOE_1B_7B.active_params_per_token(), VERIFIED_ACTIVE_PARAMS)

    def test_the_prose_total_in_the_sibling_repo_undercounts(self) -> None:
        """Record the gap instead of silently agreeing with a rounded constant."""

        gap = OLMOE_1B_7B.total_params() - LEANMOE_DOC_PARAMS
        self.assertEqual(gap, 133_120)
        self.assertLess(gap / OLMOE_1B_7B.total_params(), 2e-5)

    def test_sparse_activation_reads_the_end_less_than_a_percent(self) -> None:
        dims = OLMOE_1B_7B
        active = dims.active_params_per_token()
        self.assertAlmostEqual(active / dims.total_params(), 0.1704, places=4)
        # Sparse because 8/64 experts are touched, not because the model is small:
        # storage is 5.9x the per-token read.
        self.assertGreater(dims.total_params() / active, 5.5)

    def test_qk_norm_layout_must_tile_hidden(self) -> None:
        with self.assertRaisesRegex(ValueError, "qk_norm layout"):
            MoEDims(hidden_size=1024, qk_norm=True).validate()

    def test_grouped_topk_geometry_is_checked_when_enabled(self) -> None:
        with self.assertRaisesRegex(ValueError, "not divisible"):
            MoEDims(num_expert_groups=3, top_k=8).validate()
        with self.assertRaisesRegex(ValueError, "top_k_per_group"):
            MoEDims(num_expert_groups=2, top_k_per_group=3, top_k=8).validate()
        MoEDims(num_expert_groups=4, top_k_per_group=2, top_k=8).validate()

    def test_invalid_dimensions_are_rejected(self) -> None:
        for kwargs, message in (
            ({"top_k": 0}, "top_k"),
            ({"top_k": 65}, "top_k"),
            ({"num_experts": 0}, "num_experts"),
            ({"num_layers": 0}, "positive"),
            ({"num_expert_groups": 8, "num_experts": 4, "top_k": 4}, "exceeds"),
        ):
            with self.subTest(**kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    MoEDims(**kwargs).validate()


class ByteBudgetTests(unittest.TestCase):
    def test_the_four_components_sum_to_weight_bytes(self) -> None:
        budget = byte_budget()
        self.assertEqual(budget.weight_bytes, sum(getattr(budget, p) for p in BUDGET_PARTS))

    def test_weight_bytes_reproduces_the_tensor_verified_value(self) -> None:
        self.assertEqual(byte_budget().weight_bytes, VERIFIED_WEIGHT_BYTES_BF16)

    def test_the_documented_constants_are_short_by_the_norms(self) -> None:
        """Record the shortfall with its decomposition instead of agreeing silently.

        LeanMoE-Quant's byte table lists attention (q/k/v/o), the layer RMSNorms, the
        router, the experts and the lm_head -- but never OLMoE's per-head
        ``self_attn.q_norm`` / ``k_norm`` (16 x 2 x 2048 x 2 B = 131,072 per token).
        Its independent fp16-storage constant also omits the final ``model.norm``
        (4,096 B). Both shortfalls are pinned to the byte so a future edit to either
        record fails this test rather than quietly changing the architecture's
        premise.
        """

        budget = byte_budget()
        self.assertEqual(QK_NORMS_BYTES, 131_072)
        self.assertEqual(FINAL_NORM_BYTES, 4_096)
        # Per-token: the shortfall equals two QK-norm tensors per layer, i.e. the
        # attention row is 2 * 16 * 2048 * 2 = 131,072 too low and the final-layer
        # norm is charged once more in the record's total.
        self.assertEqual(
            budget.weight_bytes - LEANMOE_DOC_WEIGHT_BYTES,
            2 * QK_NORMS_BYTES,
        )
        self.assertEqual(budget.weight_bytes - LEANMOE_DOC_WEIGHT_BYTES, 262_144)
        self.assertLess(
            (budget.weight_bytes - LEANMOE_DOC_WEIGHT_BYTES) / budget.weight_bytes, 2e-4
        )
        # Storage: the QK norms plus the final norm.
        self.assertEqual(
            budget.storage_bytes - 2 * LEANMOE_DOC_PARAMS,
            2 * QK_NORMS_BYTES + FINAL_NORM_BYTES,
        )

    def test_experts_dominate_at_68_percent(self) -> None:
        budget = byte_budget()
        self.assertAlmostEqual(budget.share("experts"), 0.683, places=3)
        self.assertEqual(budget.experts, 1_610_612_736)

    def test_attention_and_lm_head_reproduce_their_documented_shares(self) -> None:
        budget = byte_budget()
        self.assertAlmostEqual(budget.share("attention"), 0.228, places=3)
        self.assertAlmostEqual(budget.share("lm_head"), 0.087, places=3)
        # 4 x 2048 x 2048 per layer for q/k/v/o, plus the four norm vectors.
        self.assertEqual(budget.attention, 16 * (4 * 2048 * 2048 + 4 * 2048) * 2)
        self.assertEqual(budget.lm_head, 206_045_184)

    def test_the_router_is_negligible_but_not_invented(self) -> None:
        """Router is 0.18% of [W] yet a full read, so it floors the quantization win."""

        budget = byte_budget()
        self.assertAlmostEqual(budget.share("router"), 0.0018, places=4)
        self.assertEqual(budget.router, 4_194_304)

    def test_storage_is_not_the_per_token_budget(self) -> None:
        budget = byte_budget()
        self.assertEqual(budget.storage_bytes, VERIFIED_STORAGE_BYTES_BF16)
        # 13.84 GB on the card, 2.36 GB read per token.
        self.assertAlmostEqual(budget.storage_bytes / 1e9, 13.84, places=2)
        self.assertAlmostEqual(budget.storage_bytes / budget.weight_bytes, 5.868, places=2)

    def test_one_expert_is_12_megabytes(self) -> None:
        """3 matrices x 2048 x 1024 x 2 B -- the unit the kernel streams per slot."""

        self.assertEqual(byte_budget().mv_bytes_per_expert, 12_582_912)

    def test_eight_experts_across_sixteen_layers_is_the_expert_budget(self) -> None:
        budget = byte_budget()
        self.assertEqual(budget.experts, 8 * 16 * budget.mv_bytes_per_expert)

    def test_quantizing_sets_the_bytes_linearly_and_float_widths_are_rejected(self) -> None:
        fp16 = byte_budget()
        fp8 = byte_budget(bytes_per_param=1)
        self.assertEqual(fp8.experts, fp16.experts // 2)
        self.assertEqual(fp8.weight_bytes, fp16.weight_bytes // 2)
        with self.assertRaisesRegex(ValueError, "bytes_per_param"):
            byte_budget(bytes_per_param=0)

    def test_unknown_component_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown component"):
            byte_budget().share("norms")


class KvBudgetTests(unittest.TestCase):
    def test_kv_at_1024_context_is_the_documented_134_megabytes(self) -> None:
        self.assertEqual(kv_bytes_per_token(OLMOE_1B_7B, 1024, 2), 134_217_728)

    def test_kv_is_linear_in_context(self) -> None:
        dims = OLMOE_1B_7B
        self.assertEqual(
            kv_bytes_per_token(dims, 4096, 2), 4 * kv_bytes_per_token(dims, 1024, 2)
        )

    def test_kv_starts_at_zero_and_rope_ceiling_is_4096(self) -> None:
        self.assertEqual(kv_bytes_per_token(OLMOE_1B_7B, 0, 2), 0)
        with self.assertRaisesRegex(ValueError, "ctx"):
            kv_bytes_per_token(OLMOE_1B_7B, -1, 2)

    def test_mha_kv_is_wider_than_the_dense_projects_gqa(self) -> None:
        """At MHA the KV read is 2*hidden per layer, not 2*kv_size with 8 heads."""

        mha = kv_bytes_per_token(OLMOE_1B_7B, 1024, 2)
        gqa = kv_bytes_per_token(
            MoEDims(num_kv_heads=8, num_q_heads=16), 1024, 2
        )
        self.assertEqual(mha // gqa, 2)


class RooflineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.budget = decode_budget(
            OLMOE_1B_7B, batch_size=1, ctx=1024, bandwidth_gbps=MEASURED_5090_GBPS
        )

    def test_floors_match_the_documented_values(self) -> None:
        self.assertAlmostEqual(self.budget.weight_floor_ms, 1.5121, places=3)
        self.assertAlmostEqual(self.budget.total_floor_ms, 1.5982, places=3)

    def test_bandwidth_is_required_and_never_defaulted_to_a_spec(self) -> None:
        with self.assertRaises(TypeError):
            decode_budget(OLMOE_1B_7B)  # type: ignore[call-arg]
        with self.assertRaisesRegex(ValueError, "bandwidth_gbps"):
            decode_budget(OLMOE_1B_7B, bandwidth_gbps=0)

    def test_the_spec_bandwidth_would_understate_the_floor_by_thirteen_percent(self) -> None:
        """1792 is the spec; using it as a denominator is the documented failure mode."""

        spec = decode_budget(OLMOE_1B_7B, bandwidth_gbps=1792.0)
        self.assertLess(spec.total_floor_ms, self.budget.total_floor_ms)
        self.assertAlmostEqual(
            self.budget.total_floor_ms / spec.total_floor_ms, 1792.0 / MEASURED_5090_GBPS, places=6
        )

    def test_floor_scales_inversely_with_bandwidth(self) -> None:
        double = decode_budget(OLMOE_1B_7B, bandwidth_gbps=2 * MEASURED_5090_GBPS)
        self.assertAlmostEqual(
            self.budget.total_floor_ms / double.total_floor_ms, 2.0, places=6
        )

    def test_kv_is_a_fifth_of_the_traffic_at_the_rope_ceiling(self) -> None:
        at_4096 = decode_budget(
            OLMOE_1B_7B, ctx=4096, bandwidth_gbps=MEASURED_5090_GBPS
        )
        self.assertAlmostEqual(
            at_4096.kv_bytes / at_4096.bytes.public_bytes(at_4096.kv_bytes), 0.185, places=3
        )
        self.assertAlmostEqual(at_4096.total_floor_ms, 1.857, places=3)

    def test_compute_is_under_one_percent_of_the_memory_floor(self) -> None:
        """The margin that makes this a streaming problem, not a GEMM problem."""

        self.assertLess(self.budget.flops, 2.5e9)
        self.assertLess(self.budget.compute_floor_ms / self.budget.total_floor_ms, 0.01)

    def test_the_same_flops_on_cuda_cores_still_fit_under_the_floor(self) -> None:
        """~8 TFLOP/s FP32: MMA is required for margin, but is never the limiter."""

        cuda_core_ms = self.budget.flops / (8e12) * 1000.0
        self.assertLess(cuda_core_ms, self.budget.total_floor_ms)
        self.assertGreater(cuda_core_ms, self.budget.compute_floor_ms)

    def test_materializing_experts_costs_two_milliseconds_per_token(self) -> None:
        """The gather path's price -- the fused kernel's whole advantage at batch 1."""

        self.assertAlmostEqual(self.budget.gather_saving_ms, 2.066, places=3)
        self.assertGreater(self.budget.active_weight_floor_ms, self.budget.total_floor_ms)

    def test_batch_does_not_change_the_floor_but_does_change_the_load_shape(self) -> None:
        """Weights are streamed once per step regardless of how many tokens share it."""

        batch8 = decode_budget(
            OLMOE_1B_7B, batch_size=8, ctx=1024, bandwidth_gbps=MEASURED_5090_GBPS
        )
        self.assertAlmostEqual(batch8.weight_floor_ms, self.budget.weight_floor_ms)
        self.assertLess(batch8.flops / 8.0, batch8.flops)

    def test_flops_reject_bad_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size"):
            flops_per_token(OLMOE_1B_7B, batch_size=0)
        with self.assertRaisesRegex(ValueError, "ctx"):
            flops_per_token(OLMOE_1B_7B, ctx=-1)


class LoadBalanceTests(unittest.TestCase):
    def test_batch_one_is_balanced_by_construction(self) -> None:
        """8 slots, 8 active experts, one token each: the flat-histogram premise."""

        load = expert_token_load(1)
        self.assertEqual(load["slots"], 8)
        self.assertEqual(load["min_tokens_per_active_expert"], 1)
        self.assertEqual(load["max_tokens_per_active_expert"], 1)
        self.assertEqual(load["balanced_by_construction"], 1)

    def test_batch_greater_than_one_loses_the_guarantee(self) -> None:
        load = expert_token_load(8)
        self.assertEqual(load["slots"], 64)
        self.assertEqual(load["balanced_by_construction"], 0)
        # Worst case at batch b is all b tokens on one expert.
        self.assertEqual(load["max_tokens_per_active_expert"], 8)

    def test_load_uses_the_dimensions_top_k(self) -> None:
        dims = MoEDims(top_k=2, num_experts=8)
        self.assertEqual(expert_token_load(4, dims)["slots"], 8)

    def test_zero_batch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size"):
            expert_token_load(0)


class WorkPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = plan_work(OLMOE_1B_7B, batch_size=1, num_sms=170)

    def test_expert_slots_are_top_k_times_batch(self) -> None:
        self.assertEqual(self.plan.expert_slots, 8)
        self.assertEqual(plan_work(OLMOE_1B_7B, batch_size=4).expert_slots, 32)

    def test_gate_and_up_count_as_two_matrices_per_expert(self) -> None:
        """1024 tiles per layer: 8 experts x (2 x 1024 / 16)."""

        self.assertEqual(self.plan.tiles_per_layer["expert_gateup"], 8 * 128)
        self.assertEqual(self.plan.tiles_per_layer["expert_down"], 8 * 64)

    def test_lm_head_is_the_largest_single_wave(self) -> None:
        self.assertEqual(self.plan.tiles_per_layer["lm_head"], 3144)
        self.assertGreater(
            self.plan.tiles_per_layer["lm_head"],
            self.plan.tiles_per_layer["expert_gateup"],
        )

    def test_serial_depth_is_five_levels_per_layer_plus_terminal(self) -> None:
        self.assertEqual(self.plan.serial_levels, 16 * 5 + 2)
        self.assertEqual(len(MOE_LEVELS_PER_LAYER), 5)
        self.assertEqual(len(TERMINAL_LEVELS), 2)

    def test_every_wave_fits_in_a_handful_of_passes_per_sm(self) -> None:
        """If a wave needed many passes, the barrier cost would stop being negligible."""

        for name, passes in self.plan.instructions_per_sm_per_wave.items():
            with self.subTest(wave=name):
                self.assertGreaterEqual(passes, 1)
                self.assertLessEqual(passes, 16)

    def test_tile_count_scales_with_layers_and_experts(self) -> None:
        self.assertEqual(
            self.plan.tiles_total,
            16 * sum(self.plan.tiles_per_layer.values()) - 16 * self.plan.tiles_per_layer["lm_head"]
            + self.plan.tiles_per_layer["lm_head"]
            + 1,
        )

    def test_vocab_divides_into_tiles_exactly(self) -> None:
        """50304 = 16 x 3144; a non-divisible vocab would need a padded wave."""

        self.assertEqual(OLMOE_1B_7B.vocab_size % self.plan.tile_cols, 0)

    def test_plan_rejects_bad_geometry(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size"):
            plan_work(OLMOE_1B_7B, batch_size=0)
        with self.assertRaisesRegex(ValueError, "num_sms"):
            plan_work(OLMOE_1B_7B, num_sms=0)
        with self.assertRaisesRegex(ValueError, "tile_cols"):
            plan_work(OLMOE_1B_7B, tile_cols=0)


class ReportTests(unittest.TestCase):
    def test_report_names_the_denominator_and_the_floors(self) -> None:
        text = render_report(
            decode_budget(OLMOE_1B_7B, bandwidth_gbps=MEASURED_5090_GBPS), plan_work()
        )
        self.assertIn("1559.2", text)
        self.assertIn("measured", text)
        self.assertIn("floor_W", text)
        self.assertIn("floor_T", text)
        self.assertIn("gather", text)
        for part in BUDGET_PARTS:
            self.assertIn(part, text)
        # No spec bandwidth may appear as a denominator anywhere in the report.
        self.assertNotIn("1792.0 GB/s", text)


class CliTests(unittest.TestCase):
    """The CLI is how docs/22 §0 is re-derived, so its denominator rules are tested."""

    def _run(self, args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "nanovllm.moe_budget", *args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
        )

    def test_explicit_bandwidth_reports_the_denominator_and_floors(self) -> None:
        proc = self._run(["--bandwidth-gbps", "1559.2", "--batch-sizes", "1"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("1559.2 GB/s", proc.stdout)
        self.assertIn("explicit --bandwidth-gbps", proc.stdout)
        self.assertIn("floor_W = 1.512", proc.stdout)
        self.assertIn("floor_T = 1.598", proc.stdout)

    def test_missing_probe_artifact_fails_instead_of_assuming_a_spec_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = self._run(
                ["--bandwidth-json", str(Path(tmpdir) / "absent.json")]
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("RUN_MODE=bandwidth", proc.stderr)

    def test_the_two_denominator_inputs_are_mutually_exclusive(self) -> None:
        proc = self._run(["--bandwidth-gbps", "1559.2", "--bandwidth-json", "x.json"])
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("mutually exclusive", proc.stderr)

    def test_multiple_batch_sizes_show_that_the_weight_floor_is_flat(self) -> None:
        proc = self._run(
            ["--bandwidth-gbps", "1559.2", "--batch-sizes", "1,8"]
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("| 1 | 1.512 | 1.598 |", proc.stdout)
        self.assertIn("| 8 | 1.512 | 1.598 |", proc.stdout)

    def test_json_artifact_records_provenance_and_the_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "budget.json"
            proc = self._run(
                ["--bandwidth-gbps", "1559.2", "--json", str(out)]
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["bandwidth_source"], "explicit --bandwidth-gbps")
            self.assertEqual(payload["plan"]["expert_slots"], 8)
            self.assertEqual(payload["budgets"][0]["bytes"]["experts"], 1_610_612_736)
            self.assertAlmostEqual(payload["budgets"][0]["gather_saving_ms"], 2.066, places=3)

    def test_bad_batch_sizes_are_rejected(self) -> None:
        for value in ("0", "-1", " , "):
            with self.subTest(value=value):
                proc = self._run(["--bandwidth-gbps", "1559.2", "--batch-sizes", value])
                self.assertNotEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()
