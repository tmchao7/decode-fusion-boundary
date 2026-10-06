"""CPU-only tests for the fusion-granularity sweep tool.

The tool drives GPU benchmarks, but its plan/validation logic must be testable
without torch — that is what keeps the sweep honest: an invalid granularity or a
row that cannot fit ``max_model_len`` fails here instead of producing a
misleading row on the server. The correctness gate and the reference-arm caching
are tested by injecting fake ``run_case``/``mark_correctness`` callables, because
"every row passed a real gate" is exactly the property that used to be silently
skipped (``mark_correctness`` ignores a group with no ``torch-eager`` member).
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tools.bench_fusion_granularity import (  # noqa: E402
    BACKEND,
    BASE_DEVICE_MS,
    DEFAULT_BATCH_SIZES,
    DEFAULT_GROUPS,
    DEFAULT_PROMPT_LENGTHS,
    DEFAULT_REFERENCE_BACKENDS,
    build_parser,
    cross_check_against_baseline,
    failed_rows,
    fused_prediction,
    parse_reference_backends,
    plan_reference_work,
    reference_key,
    roofline_columns,
    run_sweep,
    sweep_plan,
    validate_groups,
    validate_lengths,
    validate_reference_backends,
)
from nanovllm.roofline import (  # noqa: E402
    ModelDims,
    RooflineModel,
    measured_hardware_spec,
)
from tools.cli import parse_int_list

TOOL = ROOT / "tools" / "bench_fusion_granularity.py"
SERVER_COMMANDS = ROOT / "SERVER_COMMANDS.txt"

MEASURED_5090_GBPS = 1559.2


def measured_model() -> RooflineModel:
    return RooflineModel(
        ModelDims(),
        measured_hardware_spec(MEASURED_5090_GBPS, name="RTX 5090 (measured)"),
        tied_embedding=True,
    )


def fake_experiment(device_ms: float, wall_ms: float, batch_size: int = 1) -> dict:
    decode_steps = 10
    return {
        "backend": "megakernel",
        "workload": {
            "id": f"fusion_g1_p32_b{batch_size}",
            "category": "controlled_batch",
            "batch_size": batch_size,
            "prompt_lengths": [32],
        },
        "summary": {
            "decode_steps": decode_steps,
            "decode_time_ms_total": wall_ms * decode_steps,
            "cuda_decode_time_ms_total": device_ms * decode_steps,
        },
        "requests": [],
    }


class FakeHarness:
    """Records run_case calls and applies a stand-in correctness gate."""

    def __init__(self, *, passes: bool = True) -> None:
        self.calls: list[tuple[str, int, int]] = []
        self.fusion_groups: list[int | None] = []
        self.correctness_groups: list[list[dict]] = []
        self.passes = passes

    def __call__(self, *_args, **kwargs) -> dict:
        backend = _args[1]
        batch_size = _args[6]
        prompt_lengths = _args[5]
        self.calls.append((backend, batch_size, prompt_lengths[0]))
        self.fusion_groups.append(kwargs.get("fusion_groups"))
        experiment = fake_experiment(0.9, 3.0, batch_size)
        experiment["backend"] = backend
        experiment["workload"]["id"] = _args[2]
        return experiment

    def mark(self, group: list[dict]) -> None:
        self.correctness_groups.append(list(group))
        for experiment in group:
            if experiment["backend"] == BACKEND:
                experiment["correctness"] = {"passed": self.passes}

    def synthetic_factory(self, lengths, seed):  # noqa: ANN001 - test double
        return lambda run_index, slot: lengths  # noqa: ARG005 - unused


class ParseIntListTests(unittest.TestCase):
    def test_parses_a_comma_separated_list(self):
        self.assertEqual(parse_int_list("1,2,4", name="x"), [1, 2, 4])
        self.assertEqual(parse_int_list(" 28 ", name="x"), [28])

    def test_rejects_empty_duplicate_and_non_positive_values(self):
        for value, message in (
            ("", "non-empty"),
            (" , ", "non-empty"),
            ("1,1", "repeat"),
            ("0", ">= 1"),
            ("-1", ">= 1"),
            ("1,x", "integers"),
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, message):
                    parse_int_list(value, name="fusion-groups")


class GroupValidationTests(unittest.TestCase):
    def test_accepts_the_documented_sweep(self):
        validate_groups(parse_int_list(DEFAULT_GROUPS, name="g"))

    def test_rejects_granularities_the_schedule_cannot_represent(self):
        with self.assertRaisesRegex(ValueError, ">= 1"):
            validate_groups([0])
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            validate_groups([29])

    def test_shares_the_valid_range_with_the_schedule_builder(self):
        """A duplicated range here could drift from the schedule builder."""

        from nanovllm.backends.megakernel import schedule as tg

        for groups in (1, 2, 28, 29, 0):
            with self.subTest(groups=groups):
                try:
                    tg.split_layers(28, groups)
                    builder_ok = True
                except ValueError:
                    builder_ok = False
                try:
                    validate_groups([groups])
                    tool_ok = True
                except ValueError:
                    tool_ok = False
                self.assertEqual(tool_ok, builder_ok)


class LengthValidationTests(unittest.TestCase):
    def test_accepts_the_last_row_that_fits(self):
        validate_lengths([1920], 128)

    def test_rejects_rows_past_max_model_len(self):
        with self.assertRaisesRegex(ValueError, "exceeds max_model_len"):
            validate_lengths([1920], 129)
        with self.assertRaisesRegex(ValueError, "exceeds max_model_len"):
            validate_lengths([32, 2048], 128)


class SweepPlanTests(unittest.TestCase):
    def test_plan_is_group_major_and_complete(self):
        plan = sweep_plan([1, 2], [1, 8], [32, 512])

        self.assertEqual(len(plan), 8)
        self.assertEqual(
            plan[0],
            {"fusion_groups": 1, "batch_size": 1, "prompt_length": 32},
        )
        self.assertEqual(
            [row["fusion_groups"] for row in plan], [1] * 4 + [2] * 4
        )
        self.assertEqual(
            {(row["batch_size"], row["prompt_length"]) for row in plan},
            {(1, 32), (1, 512), (8, 32), (8, 512)},
        )
        # Deterministic: same inputs, same plan.
        self.assertEqual(plan, sweep_plan([1, 2], [1, 8], [32, 512]))

    def test_documented_defaults_produce_the_documented_grid(self):
        groups = parse_int_list(DEFAULT_GROUPS, name="g")
        batches = parse_int_list(DEFAULT_BATCH_SIZES, name="b")
        lengths = parse_int_list(DEFAULT_PROMPT_LENGTHS, name="p")
        plan = sweep_plan(groups, batches, lengths)

        self.assertEqual(len(plan), 6 * 4 * 3)
        validate_groups(groups)
        validate_lengths(lengths, 128)


class CommandLineTests(unittest.TestCase):
    def test_defaults_match_the_documented_sweep(self):
        args = build_parser().parse_args(["--model", "m"])
        self.assertEqual(args.fusion_groups, DEFAULT_GROUPS)
        self.assertEqual(args.batch_sizes, DEFAULT_BATCH_SIZES)
        self.assertEqual(args.prompt_lengths, DEFAULT_PROMPT_LENGTHS)
        self.assertEqual(args.max_tokens, 128)
        self.assertEqual(args.warmup, 2)
        self.assertEqual(args.runs, 5)

    def test_overrides_are_parsed(self):
        args = build_parser().parse_args(
            ["--model", "m", "--fusion-groups", "1,4", "--runs", "5"]
        )
        self.assertEqual(parse_int_list(args.fusion_groups, name="g"), [1, 4])
        self.assertEqual(args.runs, 5)


class WiringTests(unittest.TestCase):
    def test_sweep_drives_compare_run_case_on_the_megakernel_arm(self):
        text = TOOL.read_text(encoding="utf-8")
        self.assertIn("from benchmarks.compare import mark_correctness, run_case", text)
        self.assertIn("fusion_groups=groups", text)
        self.assertEqual(BACKEND, "megakernel")

    def test_the_gate_is_the_shared_compare_helper_not_a_copy(self):
        """A duplicated pass/fail rule would drift from the main table's."""

        text = TOOL.read_text(encoding="utf-8")
        self.assertIn("mark_correctness_fn(group)", text)
        self.assertNotIn("first_token_matches", text)

    def test_tool_imports_torch_lazily(self):
        """The plan/validation logic must import on a CPU-only checkout."""

        text = TOOL.read_text(encoding="utf-8")
        self.assertNotIn("\nimport torch", text)
        self.assertIn("    from benchmarks.compare import", text)

    def test_server_commands_expose_the_sweep(self):
        text = SERVER_COMMANDS.read_text(encoding="utf-8")
        self.assertIn("taskgraph-fusion)", text)
        self.assertIn("run_taskgraph_fusion()", text)
        self.assertIn("tools.bench_fusion_granularity", text)


class ReferenceBackendTests(unittest.TestCase):
    def test_documented_default_is_an_independent_implementation(self):
        self.assertEqual(parse_reference_backends(DEFAULT_REFERENCE_BACKENDS), ["torch-eager"])

    def test_parsing_rejects_an_empty_list(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            parse_reference_backends(" , ")

    def test_parsing_accepts_the_step_1_pair(self):
        self.assertEqual(
            parse_reference_backends("torch-eager,torch-cudagraph"),
            ["torch-eager", "torch-cudagraph"],
        )

    def test_the_swept_arm_cannot_be_its_own_reference(self):
        with self.assertRaisesRegex(ValueError, "cannot also be a reference"):
            validate_reference_backends([BACKEND], known=[BACKEND, "torch-eager"])

    def test_unknown_backends_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown reference backend"):
            validate_reference_backends(["torch-magic"], known=["torch-eager"])


class ReferenceWorkTests(unittest.TestCase):
    def test_one_reference_run_per_grid_point_per_backend(self):
        work = plan_reference_work([1, 8], [32, 512], ["torch-eager", "torch-cudagraph"])
        self.assertEqual(len(work), 2 * 2 * 2)

    def test_the_key_is_the_grid_point_not_the_granularity(self):
        """Reusing a reference across ``g`` is the whole point of the cache key."""

        self.assertEqual(
            reference_key("torch-eager", 1, 32, 128, 0),
            reference_key("torch-eager", 1, 32, 128, 0),
        )
        self.assertNotEqual(
            reference_key("torch-eager", 1, 32, 128, 0),
            reference_key("torch-eager", 1, 512, 128, 0),
        )

    def test_reference_arms_run_at_fusion_groups_one(self):
        """Passing g to a non-megakernel backend makes compare.py raise."""

        harness = FakeHarness()
        run_sweep(
            plan=sweep_plan([1, 4], [1], [32]),
            model_path="m",
            max_tokens=128,
            warmup=1,
            runs=1,
            seed=0,
            reference_backends=["torch-eager"],
            run_case_fn=harness,
            synthetic_factory_fn=harness.synthetic_factory,
            mark_correctness_fn=harness.mark,
        )
        # The reference arm is the only call without fusion_groups (the megakernel
        # call is recorded first; the gate runs after both exist).
        self.assertEqual(
            [call[0] for call in harness.calls],
            ["megakernel", "torch-eager", "megakernel"],
        )
        self.assertIsNone(harness.fusion_groups[1])
        self.assertEqual(harness.fusion_groups[0], 1)
        self.assertEqual(harness.fusion_groups[2], 4)

    def test_references_are_cached_across_granularities(self):
        harness = FakeHarness()
        _, cache = run_sweep(
            plan=sweep_plan([1, 2, 4, 28], [1, 8], [32, 512]),
            model_path="m",
            max_tokens=128,
            warmup=1,
            runs=1,
            seed=0,
            reference_backends=["torch-eager"],
            run_case_fn=harness,
            synthetic_factory_fn=harness.synthetic_factory,
            mark_correctness_fn=harness.mark,
        )
        grid_points = 2 * 2
        self.assertEqual(len(cache), grid_points)
        reference_calls = [c for c in harness.calls if c[0] == "torch-eager"]
        self.assertEqual(len(reference_calls), grid_points)
        self.assertEqual(
            len([c for c in harness.calls if c[0] == BACKEND]), 4 * 2 * 2
        )

    def test_every_row_is_grouped_with_a_reference_before_the_gate(self):
        harness = FakeHarness()
        rows, _ = run_sweep(
            plan=sweep_plan([1, 4], [1], [32]),
            model_path="m",
            max_tokens=128,
            warmup=1,
            runs=1,
            seed=0,
            reference_backends=["torch-eager"],
            run_case_fn=harness,
            synthetic_factory_fn=harness.synthetic_factory,
            mark_correctness_fn=harness.mark,
        )
        self.assertEqual(len(harness.correctness_groups), len(rows))
        for group in harness.correctness_groups:
            self.assertEqual(
                [experiment["backend"] for experiment in group],
                ["torch-eager", BACKEND],
            )


class CorrectnessGateTests(unittest.TestCase):
    def _rows(self, *, passes: bool) -> list[dict]:
        harness = FakeHarness(passes=passes)
        rows, _ = run_sweep(
            plan=sweep_plan([1, 4], [1], [32]),
            model_path="m",
            max_tokens=128,
            warmup=1,
            runs=1,
            seed=0,
            reference_backends=["torch-eager"],
            run_case_fn=harness,
            synthetic_factory_fn=harness.synthetic_factory,
            mark_correctness_fn=harness.mark,
        )
        return rows

    def test_passing_rows_produce_no_failures(self):
        self.assertEqual(failed_rows(self._rows(passes=True)), [])

    def test_a_row_without_a_gate_counts_as_a_failure(self):
        rows = self._rows(passes=True)
        del rows[0]["experiment"]["correctness"]
        self.assertEqual(len(failed_rows(rows)), 1)

    def test_a_failed_gate_is_reported(self):
        rows = self._rows(passes=False)
        self.assertEqual(len(failed_rows(rows)), len(rows))

    def test_a_row_with_no_reference_at_all_is_a_failure(self):
        """This is the regression: a megakernel-only table used to pass silently."""

        experiment = fake_experiment(0.9, 3.0)
        self.assertEqual(len(failed_rows([{"fusion_groups": 1, "experiment": experiment}])), 1)


class RooflineColumnTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = measured_model()

    def test_device_time_comes_from_the_cuda_event_total(self):
        row = roofline_columns(self.model, fake_experiment(0.8, 3.0), 1)
        self.assertAlmostEqual(row["device_ms_per_step"], 0.8, places=6)
        self.assertAlmostEqual(row["wall_ms_per_step"], 3.0, places=6)
        self.assertAlmostEqual(row["host_residual_ms"], 2.2, places=6)

    def test_gap_uses_the_measured_denominator(self):
        row = roofline_columns(self.model, fake_experiment(1.5, 3.0), 1)
        self.assertAlmostEqual(row["core_weight_floor_ms"], 0.7645, places=3)
        self.assertAlmostEqual(
            row["gap_to_core_floor_x"], 1.5 / row["core_weight_floor_ms"]
        )
        self.assertNotAlmostEqual(row["core_weight_floor_ms"], 0.665, places=3)

    def test_below_the_floor_is_a_calibration_finding(self):
        row = roofline_columns(self.model, fake_experiment(0.2, 3.0), 1)
        self.assertTrue(row["calibration_needed"])

    def test_above_the_floor_is_not_a_calibration_finding(self):
        row = roofline_columns(self.model, fake_experiment(2.0, 3.0), 1)
        self.assertFalse(row["calibration_needed"])

    def test_a_row_without_decode_steps_reports_unknown_not_zero(self):
        experiment = fake_experiment(0.8, 3.0)
        experiment["summary"]["decode_steps"] = 0
        row = roofline_columns(self.model, experiment, 1)
        self.assertIsNone(row["device_ms_per_step"])
        self.assertIsNone(row["implied_bandwidth_gbps"])

    def test_zero_device_time_reports_unknown_bandwidth(self):
        row = roofline_columns(self.model, fake_experiment(0.0, 3.0), 1)
        self.assertIsNone(row["implied_bandwidth_gbps"])
        self.assertIsNone(row["gap_to_core_floor_x"])


class PreRegistrationTests(unittest.TestCase):
    def test_the_base_is_the_archived_a100_value_and_says_so(self):
        prediction = fused_prediction([1, 28], [1])
        self.assertEqual(prediction["base_device_ms"], BASE_DEVICE_MS)
        self.assertIn("archived", prediction["base_device_ms_source"])
        self.assertIn("NOT a 5090", prediction["base_device_ms_source"])

    def test_a_different_base_rescales_the_curve_and_its_relative_bound(self):
        """The g penalty is a fixed *absolute* cost, so the relative bound moves.

        This is why the base must be recorded: reporting the 1.89 ms-scaled
        ``+15.7%`` as a 5090 prediction would be a statement about a device whose
        step is ~11 ms, not ~2 ms.
        """

        default = fused_prediction([1, 28], [1])
        rescaled = fused_prediction(
            [1, 28], [1], base_device_ms=10.8, base_device_ms_source="x"
        )
        self.assertAlmostEqual(
            default["refuted_if_measured_delta_exceeds"], 0.157, places=3
        )
        self.assertAlmostEqual(
            rescaled["refuted_if_measured_delta_exceeds"], 0.0275, places=4
        )
        # The absolute g=28 penalty is unchanged; only the denominator is.
        default_penalty = (
            default["device_ms_upper_bound"][28] - default["base_device_ms"]
        )
        rescaled_penalty = (
            rescaled["device_ms_upper_bound"][28] - rescaled["base_device_ms"]
        )
        self.assertAlmostEqual(default_penalty, rescaled_penalty, places=6)
        self.assertEqual(rescaled["base_device_ms_source"], "x")

    def test_the_documented_falsification_bound_is_preserved(self):
        prediction = fused_prediction([1, 2, 4, 7, 14, 28], [1, 2, 4, 8])
        self.assertAlmostEqual(
            prediction["refuted_if_measured_delta_exceeds"], 0.157, places=3
        )


class CrossCheckTests(unittest.TestCase):
    def test_g_one_matches_a_baseline_row_within_tolerance(self):
        rows = [{"fusion_groups": 1, "experiment": fake_experiment(0.8, 3.0)}]
        findings = cross_check_against_baseline(rows, [fake_experiment(0.79, 3.0)])
        self.assertEqual(len(findings), 1)
        self.assertTrue(findings[0]["within_tolerance"])

    def test_drift_beyond_tolerance_is_reported(self):
        rows = [{"fusion_groups": 1, "experiment": fake_experiment(1.0, 3.0)}]
        findings = cross_check_against_baseline(rows, [fake_experiment(0.8, 3.0)])
        self.assertFalse(findings[0]["within_tolerance"])
        self.assertAlmostEqual(findings[0]["relative_delta"], 0.25, places=6)

    def test_a_missing_baseline_row_is_not_silently_ok(self):
        rows = [{"fusion_groups": 1, "experiment": fake_experiment(0.8, 3.0)}]
        findings = cross_check_against_baseline(rows, [])
        self.assertIsNone(findings[0]["baseline"])
        self.assertFalse(findings[0]["within_tolerance"])

    def test_rows_above_g_one_are_not_cross_checked(self):
        rows = [{"fusion_groups": 4, "experiment": fake_experiment(0.8, 3.0)}]
        self.assertEqual(cross_check_against_baseline(rows, []), [])

    def test_a_non_megakernel_baseline_is_ignored(self):
        baseline = fake_experiment(0.8, 3.0)
        baseline["backend"] = "torch-cudagraph"
        findings = cross_check_against_baseline(
            [{"fusion_groups": 1, "experiment": fake_experiment(0.8, 3.0)}], [baseline]
        )
        self.assertIsNone(findings[0]["baseline"])

    def test_the_join_ignores_workload_ids(self):
        """The two tools name the same grid point differently, by construction.

        ``benchmarks.compare`` writes ``perf_p32_b1`` for a grid row; this tool
        writes ``fusion_g1_p32_b1`` because several granularities share one point.
        A join on the id would therefore never match, and the acceptance-2 check
        would be present but silently inert.
        """

        sweep_row = {"fusion_groups": 1, "experiment": fake_experiment(0.8, 3.0)}
        sweep_row["experiment"]["workload"]["id"] = "fusion_g1_p32_b1"
        baseline_row = fake_experiment(0.81, 3.0)
        baseline_row["workload"]["id"] = "perf_p32_b1"

        findings = cross_check_against_baseline([sweep_row], [baseline_row])
        self.assertEqual(len(findings), 1)
        self.assertIsNotNone(findings[0]["baseline"])
        self.assertTrue(findings[0]["within_tolerance"])
        self.assertEqual(findings[0]["workload_id"], "fusion_g1_p32_b1")

    def test_join_distinguishes_different_grid_points(self):
        sweep_row = {"fusion_groups": 1, "experiment": fake_experiment(0.8, 3.0)}
        sweep_row["experiment"]["workload"]["prompt_lengths"] = [512]
        baseline_row = fake_experiment(0.8, 3.0)
        baseline_row["workload"]["prompt_lengths"] = [32]
        findings = cross_check_against_baseline([sweep_row], [baseline_row])
        self.assertIsNone(findings[0]["baseline"])

    def test_a_row_without_prompt_lengths_is_skipped_not_mismatched(self):
        sweep_row = {"fusion_groups": 1, "experiment": fake_experiment(0.8, 3.0)}
        sweep_row["experiment"]["workload"].pop("prompt_lengths")
        self.assertEqual(cross_check_against_baseline([sweep_row], []), [])


class RooflineFlagsTests(unittest.TestCase):
    def test_help_and_plan_paths_do_not_need_a_bandwidth_artifact(self):
        args = build_parser().parse_args(["--model", "m"])
        self.assertIsNone(args.bandwidth_json)
        self.assertIsNone(args.bandwidth_gbps)

    def test_new_flags_have_safe_defaults(self):
        args = build_parser().parse_args(["--model", "m"])
        self.assertEqual(args.reference_backends, DEFAULT_REFERENCE_BACKENDS)
        self.assertIsNone(args.baseline_json)
        self.assertAlmostEqual(args.cross_check_tolerance, 0.05)
        self.assertFalse(args.no_fail_on_correctness)
        self.assertEqual(args.runs, 5)

    def test_server_commands_pass_the_measured_denominator(self):
        text = SERVER_COMMANDS.read_text(encoding="utf-8")
        self.assertIn("--bandwidth-json", text)
        self.assertIn("--reference-backends", text)


if __name__ == "__main__":
    unittest.main()
