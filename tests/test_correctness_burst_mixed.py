"""CPU-safe tests for the mixed max_tokens burst guard's pure logic.

``tools/correctness_burst_mixed.py`` needs a GPU, but the case construction and
the pass/fail decision it makes are pure Python.  Two contracts are tested:

* the device burst runs one ``max_steps`` for the whole batch, so ``max_steps``
  must equal the minimum per-sequence budget and every selected sequence must be
  reserved for that many tokens;
* the run must contain at least one burst with **mixed** per-sequence budgets,
  otherwise ``min == max`` and the contract is vacuous -- which is exactly what
  a ``--short-tokens 1`` run produces, because prefill consumes that sequence's
  only token and it never reaches ``schedule_burst``.

``burst_guard_violations`` / ``burst_guard_rejection`` are the tool's own
decision functions, tested directly so they cannot silently weaken.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from tools.correctness_burst_mixed import (  # noqa: E402
    burst_guard_rejection,
    burst_guard_violations,
    caps_for_batch,
)


class CapsForBatchTests(unittest.TestCase):
    def test_alternates_short_and_long(self) -> None:
        self.assertEqual(caps_for_batch(2, 2, 200), [2, 200])
        self.assertEqual(caps_for_batch(4, 2, 200), [2, 200, 2, 200])
        self.assertEqual(caps_for_batch(5, 2, 200), [2, 200, 2, 200, 2])

    def test_every_batch_has_both_a_short_and_a_long_budget(self) -> None:
        # The regression only exists when the batch mixes budgets: the short
        # row is the one that would overrun under max_steps = max(budget).
        for batch_size in range(2, 9):
            caps = caps_for_batch(batch_size, 2, 200)
            self.assertIn(2, caps)
            self.assertIn(200, caps)

    def test_min_budget_is_the_short_cap(self) -> None:
        for batch_size in (2, 3, 4, 5):
            caps = caps_for_batch(batch_size, 7, 200)
            self.assertEqual(min(caps), 7)

    def test_caps_length_matches_the_batch(self) -> None:
        for batch_size in (1, 2, 5, 8):
            self.assertEqual(len(caps_for_batch(batch_size, 2, 200)), batch_size)


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
            "mixed_budget_bursts": 1,
            "max_steps_violations": 1,
            "reservation_violations": 0,
            "last_max_steps": 200,
            "last_budgets": [1, 200],
        }
        self.assertGreater(burst_guard_violations(guard), 0)

    def test_short_reservation_is_a_violation(self) -> None:
        guard = {
            "bursts_observed": 1,
            "mixed_budget_bursts": 1,
            "max_steps_violations": 0,
            "reservation_violations": 1,
            "last_max_steps": 200,
            "last_reserved_tokens": [200, 1],
        }
        self.assertGreater(burst_guard_violations(guard), 0)


class BurstGuardRejectionTests(unittest.TestCase):
    """A run that cannot exercise the mixed path must not be reported as a pass."""

    def _valid(self, **overrides):
        guard = {
            "bursts_observed": 4,
            "mixed_budget_bursts": 1,
            "max_steps_violations": 0,
            "reservation_violations": 0,
            "last_max_steps": 1,
            "last_budgets": [1, 64],
            "last_reserved_tokens": [4, 240],
        }
        guard.update(overrides)
        return guard

    def test_valid_mixed_run_is_accepted(self) -> None:
        self.assertIsNone(burst_guard_rejection(self._valid(), short_tokens=2))

    def test_no_burst_is_rejected(self) -> None:
        reason = burst_guard_rejection(
            self._valid(bursts_observed=0), short_tokens=2
        )
        self.assertIsNotNone(reason)
        self.assertIn("no burst", reason)

    def test_violation_is_rejected(self) -> None:
        reason = burst_guard_rejection(
            self._valid(max_steps_violations=1), short_tokens=2
        )
        self.assertIsNotNone(reason)
        self.assertIn("contract was violated", reason)

    def test_uniform_budgets_are_rejected_as_vacuous(self) -> None:
        """The round-14 server run: 4 uniform bursts and 0 violations.

        That passed the counter check but never formed a mixed batch, because
        ``--short-tokens 1`` finishes the short row during prefill.  It must be
        a failure, otherwise the guard is a green light for nothing.
        """

        guard = self._valid(
            bursts_observed=4,
            mixed_budget_bursts=0,
            last_max_steps=64,
            last_budgets=[64, 64],
        )
        reason = burst_guard_rejection(guard, short_tokens=1)
        self.assertIsNotNone(reason)
        self.assertIn("no burst had mixed per-sequence budgets", reason)
        self.assertIn("--short-tokens 2", reason)

    def test_missing_counter_is_treated_as_vacuous(self) -> None:
        # A scheduler build without the mixed counter must not silently pass.
        guard = self._valid()
        del guard["mixed_budget_bursts"]
        self.assertIsNotNone(burst_guard_rejection(guard, short_tokens=2))


if __name__ == "__main__":
    unittest.main()
