import unittest

from tools.correctness_smoke import serialize_backend_result


class CorrectnessSmokeProcessTests(unittest.TestCase):
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
