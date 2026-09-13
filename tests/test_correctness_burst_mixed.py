"""CPU-safe tests for the mixed max_tokens burst guard's pure logic.

``tools/correctness_burst_mixed.py`` needs a GPU, but the case construction and
the scheduler tripwire decision it makes are pure Python.  The contract is: the
device burst runs one ``max_steps`` for the whole batch, so ``max_steps`` must
equal the minimum per-sequence budget and every selected sequence must be
reserved for that many tokens.  ``burst_guard_violations`` is the tool's own
pass/fail function, tested directly so the decision cannot silently weaken.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from tools.correctness_burst_mixed import (  # noqa: E402
    burst_guard_violations,
    caps_for_batch,
)


class CapsForBatchTests(unittest.TestCase):
    def test_alternates_short_and_long(self) -> None:
        self.assertEqual(caps_for_batch(2, 1, 200), [1, 200])
        self.assertEqual(caps_for_batch(4, 1, 200), [1, 200, 1, 200])
        self.assertEqual(caps_for_batch(5, 1, 200), [1, 200, 1, 200, 1])

    def test_every_batch_has_both_a_short_and_a_long_budget(self) -> None:
        # The regression only exists when the batch mixes budgets: the short
        # row is the one that would overrun under max_steps = max(budget).
        for batch_size in range(2, 9):
            caps = caps_for_batch(batch_size, 1, 200)
            self.assertIn(1, caps)
            self.assertIn(200, caps)

    def test_min_budget_is_the_short_cap(self) -> None:
        for batch_size in (2, 3, 4, 5):
            caps = caps_for_batch(batch_size, 7, 200)
            self.assertEqual(min(caps), 7)

    def test_caps_length_matches_the_batch(self) -> None:
        for batch_size in (1, 2, 5, 8):
            self.assertEqual(len(caps_for_batch(batch_size, 1, 200)), batch_size)


class TripwireDecisionTests(unittest.TestCase):
    """The tool must fail on any scheduler-contract violation."""

    def test_clean_guard_has_no_violations(self) -> None:
        guard = {
            "bursts_observed": 3,
            "max_steps_violations": 0,
            "reservation_violations": 0,
        }
        self.assertEqual(burst_guard_violations(guard), 0)

    def test_missing_guard_is_not_a_violation_by_itself(self) -> None:
        # A backend that never burst has no guard; the tool rejects that
        # separately via bursts_observed <= 0.
        self.assertEqual(burst_guard_violations(None), 0)

    def test_max_steps_regression_is_a_violation(self) -> None:
        # This is the §5.6 defect: max_steps = max(budget) instead of min.
        guard = {
            "bursts_observed": 1,
            "max_steps_violations": 1,
            "reservation_violations": 0,
            "last_max_steps": 200,
            "last_budgets": [1, 200],
        }
        self.assertGreater(burst_guard_violations(guard), 0)

    def test_short_reservation_is_a_violation(self) -> None:
        guard = {
            "bursts_observed": 1,
            "max_steps_violations": 0,
            "reservation_violations": 1,
            "last_max_steps": 200,
            "last_reserved_tokens": [200, 1],
        }
        self.assertGreater(burst_guard_violations(guard), 0)


if __name__ == "__main__":
    unittest.main()
