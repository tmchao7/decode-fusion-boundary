import unittest

from tools.correctness_smoke import (
    backend_config,
    classify_comparison,
    compare_outputs,
    serialize_backend_result,
)


class CorrectnessSmokeProcessTests(unittest.TestCase):
    def test_reports_first_token_mismatch_details(self):
        mismatches, first = compare_outputs([[11, 12]], [[11, 99]])
        self.assertEqual(mismatches, [0])
        self.assertEqual(
            first,
            {"request_index": 0, "token_index": 1, "expected": 12, "actual": 99},
        )

    def test_serializes_only_json_safe_backend_result(self):
        result = {
            "outputs": [{"token_ids": [1, 2, 3]}],
            "metrics": {"summary": {"backend_actuals": ["megakernel"]}},
        }
        self.assertEqual(
            serialize_backend_result(result),
            {
                "outputs": [[1, 2, 3]],
                "metrics": {"summary": {"backend_actuals": ["megakernel"]}},
            },
        )


class ClassifyComparisonTests(unittest.TestCase):
    """Tests for the PASS / CASCADE / FAIL classification.

    Exercises the mainline Megakernel correctness classification: token 0 is the
    prefill output and must always match; divergences at decode tokens
    (>= 1) are the documented ~1-ulp cascade and are tolerated.
    """

    def test_identical_outputs_are_pass(self):
        status, details = classify_comparison([[1, 2, 3]], [[1, 2, 3]])
        self.assertEqual(status, "PASS")
        self.assertEqual(details, [])

    def test_decode_token_divergence_is_cascade(self):
        status, details = classify_comparison([[1, 2, 3]], [[1, 99, 3]])
        self.assertEqual(status, "CASCADE")
        self.assertEqual(len(details), 1)
        self.assertEqual(details[0]["kind"], "token")
        self.assertEqual(details[0]["first"], 1)

    def test_prefill_token_zero_divergence_is_fail(self):
        status, details = classify_comparison([[1, 2, 3]], [[99, 2, 3]])
        self.assertEqual(status, "FAIL")
        self.assertEqual(details[0]["first"], 0)

    def test_length_mismatch_is_fail(self):
        status, details = classify_comparison([[1, 2, 3]], [[1, 2]])
        self.assertEqual(status, "FAIL")
        self.assertEqual(details[0]["kind"], "length")

    def test_mixed_prefill_and_cascade_is_fail(self):
        """If any request diverges at token 0, the whole case is FAIL."""
        status, _ = classify_comparison(
            [[1, 2, 3], [10, 20, 30]],
            [[99, 2, 3], [10, 99, 30]],
        )
        self.assertEqual(status, "FAIL")

    def test_all_cascade_is_cascade(self):
        status, _ = classify_comparison(
            [[1, 2, 3], [10, 20, 30]],
            [[1, 99, 3], [10, 20, 99]],
        )
        self.assertEqual(status, "CASCADE")

    def test_request_count_mismatch_is_fail(self):
        status, details = classify_comparison([[1, 2], [3, 4]], [[1, 2]])
        self.assertEqual(status, "FAIL")
        self.assertTrue(any(d["kind"] == "count" for d in details))

    def test_empty_outputs_are_pass(self):
        status, details = classify_comparison([], [])
        self.assertEqual(status, "PASS")
        self.assertEqual(details, [])

    def test_single_token_prefill_only_mismatch_is_fail(self):
        """With only one token (prefill), any mismatch is FAIL (token 0)."""
        status, _ = classify_comparison([[1]], [[99]])
        self.assertEqual(status, "FAIL")


class BackendConfigTests(unittest.TestCase):
    """``--attn-splits`` must reach the engine, and only where it means something."""

    def test_default_is_the_plain_backend_config(self) -> None:
        self.assertEqual(
            backend_config("megakernel"),
            {"decode_backend": "megakernel", "enforce_eager": False},
        )

    def test_attn_splits_reaches_the_megakernel(self) -> None:
        config = backend_config("megakernel", 2)

        self.assertEqual(config["megakernel_attn_splits"], 2)
        self.assertEqual(config["decode_backend"], "megakernel")

    def test_attn_splits_on_a_torch_backend_is_a_caller_error(self) -> None:
        with self.assertRaises(ValueError):
            backend_config("torch-cudagraph", 2)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            backend_config("nope")
        with self.assertRaises(ValueError):
            backend_config("megakernel", 0)

    def test_attn_splits_survives_the_isolated_process_handoff(self) -> None:
        """``--attn-splits 2`` must reach the child process that runs the backend.

        Regression: ``run_backend_isolated`` accepted ``attn_splits`` and never
        put it in ``Process(args=...)``, and ``_backend_worker`` accepted it and
        called ``run_backend`` without it. The gate therefore ran at split=1
        while ``results/taskgraph_attn_correctness.json`` recorded
        ``"attn_splits": 2`` — a false record that looks like passing coverage.

        Asserted structurally (AST) rather than by executing: the child would
        need a GPU, and this bug is purely "is the argument forwarded".
        """

        import ast
        import inspect
        import textwrap

        from tools import correctness_smoke

        def calls_in(func) -> list[ast.Call]:
            tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
            return [n for n in ast.walk(tree) if isinstance(n, ast.Call)]

        # 1. The Process(...) handoff must pass attn_splits positionally.
        isolated = correctness_smoke.run_backend_isolated
        process_calls = [
            call
            for call in calls_in(isolated)
            if getattr(call.func, "attr", None) == "Process"
        ]
        self.assertEqual(len(process_calls), 1, "expected one Process(...) call")
        args_kw = next(k for k in process_calls[0].keywords if k.arg == "args")
        tunnel = [getattr(a, "id", None) for a in args_kw.value.elts]
        self.assertIn(
            "attn_splits",
            tunnel,
            f"attn_splits is not forwarded to the child process: args={tunnel}",
        )

        # 2. The worker must forward it to run_backend.
        worker = [c for c in calls_in(correctness_smoke._backend_worker)
                  if getattr(c.func, "id", None) == "run_backend"]
        self.assertEqual(len(worker), 1, "expected one run_backend(...) call")
        forwarded = [getattr(a, "id", None) for a in worker[0].args]
        forwarded += [k.value.id for k in worker[0].keywords
                      if getattr(k.value, "id", None)]
        self.assertIn(
            "attn_splits",
            forwarded,
            f"the worker drops attn_splits: run_backend args={forwarded}",
        )
