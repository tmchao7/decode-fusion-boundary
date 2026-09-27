"""CPU-only tests for the fusion-granularity sweep tool.

The tool drives GPU benchmarks, but its plan/validation logic must be testable
without torch — that is what keeps the sweep honest: an invalid granularity or a
row that cannot fit ``max_model_len`` fails here instead of producing a
misleading row on the server.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tools.bench_fusion_granularity import (  # noqa: E402
    BACKEND,
    DEFAULT_BATCH_SIZES,
    DEFAULT_GROUPS,
    DEFAULT_PROMPT_LENGTHS,
    build_parser,
    parse_int_list,
    sweep_plan,
    validate_groups,
    validate_lengths,
)

TOOL = ROOT / "tools" / "bench_fusion_granularity.py"
SERVER_COMMANDS = ROOT / "SERVER_COMMANDS.txt"


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
        self.assertIn("from benchmarks.compare import run_case, synthetic_factory", text)
        self.assertIn("fusion_groups=row[\"fusion_groups\"]", text)
        self.assertEqual(BACKEND, "megakernel")

    def test_tool_imports_torch_lazily(self):
        """The plan/validation logic must import on a CPU-only checkout."""

        text = TOOL.read_text(encoding="utf-8")
        self.assertNotIn("\nimport torch", text)
        self.assertIn("    from benchmarks.compare import run_case", text)

    def test_server_commands_expose_the_sweep(self):
        text = SERVER_COMMANDS.read_text(encoding="utf-8")
        self.assertIn("taskgraph-fusion)", text)
        self.assertIn("run_taskgraph_fusion()", text)
        self.assertIn("tools.bench_fusion_granularity", text)


if __name__ == "__main__":
    unittest.main()
