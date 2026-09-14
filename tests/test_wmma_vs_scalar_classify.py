"""CPU-safe tests for the WMMA-vs-scalar correctness gate's classifier.

``classify`` is what turns a token comparison into PASS / CASCADE / FAIL, so a
hole in it makes the correctness gate itself unreliable.  It iterated
``zip(scalar, wmma)`` and only recorded divergences, so a backend that returned
*fewer requests* than the other had its extra requests silently dropped --
reported as PASS.  These tests pin every branch, including the count mismatch.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from tools.correctness_wmma_vs_scalar import classify  # noqa: E402


class ClassifyTests(unittest.TestCase):
    def test_identical_is_pass(self) -> None:
        status, details = classify([[1, 2, 3], [4, 5]], [[1, 2, 3], [4, 5]])
        self.assertEqual(status, "PASS")
        self.assertEqual(details, [])

    def test_empty_both_sides_is_pass(self) -> None:
        self.assertEqual(classify([], [])[0], "PASS")

    def test_decode_token_divergence_is_cascade(self) -> None:
        # token 0 (the prefill output) matches; the flip is at a decode token.
        status, details = classify([[1, 2, 3]], [[1, 9, 3]])
        self.assertEqual(status, "CASCADE")
        self.assertEqual(details[0]["first"], 1)

    def test_prefill_divergence_is_fail(self) -> None:
        self.assertEqual(classify([[1, 2, 3]], [[7, 2, 3]])[0], "FAIL")

    def test_per_request_length_mismatch_is_fail(self) -> None:
        self.assertEqual(classify([[1, 2, 3]], [[1, 2]])[0], "FAIL")

    def test_shorter_wmma_is_fail(self) -> None:
        """The regression: scalar returned 2 requests, wmma only 1.

        zip() dropped the second scalar request, so this used to be reported as
        PASS even though the WMMA backend never produced it.
        """
        scalar = [[1, 2, 3], [4, 5, 6]]
        wmma = [[1, 2, 3]]
        status, details = classify(scalar, wmma)
        self.assertEqual(status, "FAIL")
        self.assertTrue(any(d["kind"] == "count" for d in details))

    def test_shorter_scalar_is_fail(self) -> None:
        status, details = classify([[1, 2, 3]], [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(status, "FAIL")
        self.assertTrue(any(d["kind"] == "count" for d in details))

    def test_count_mismatch_dominates_an_earlier_cascade(self) -> None:
        scalar = [[1, 2, 3], [4, 5, 6]]
        wmma = [[1, 9, 3]]
        status, details = classify(scalar, wmma)
        self.assertEqual(status, "FAIL")
        self.assertTrue(any(d["kind"] == "count" for d in details))


if __name__ == "__main__":
    unittest.main()
