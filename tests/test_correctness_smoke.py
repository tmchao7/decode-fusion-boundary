import unittest

from tools.correctness_smoke import (
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

    Mirrors the logic in ``correctness_wmma_vs_scalar.py``: token 0 is the
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
