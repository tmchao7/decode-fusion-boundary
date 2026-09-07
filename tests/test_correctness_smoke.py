import unittest

from tools.correctness_smoke import compare_outputs, serialize_backend_result


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
