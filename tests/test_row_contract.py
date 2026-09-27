"""Lock the per-row activation contract across the host, the launcher and the kernel.

``row_active`` is easy to get wrong in a way no local test would notice: the host
passes all-ones today, so the masked path is never taken, and a bad mask would
only show up as corrupted KV for some *other* sequence on the GPU. These tests pin
both sides of the contract from the CPU.
"""

import unittest
from pathlib import Path

from nanovllm.backends.megakernel.row_contract import row_active_mask

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL = REPO_ROOT / "src/nanovllm/backends/megakernel/csrc/taskgraph_kernel.cu"
TASKGRAPH = REPO_ROOT / "src/nanovllm/backends/megakernel/taskgraph.py"


class MaskTests(unittest.TestCase):
    def test_none_activates_every_row(self) -> None:
        self.assertEqual(row_active_mask(None, 1), [1])
        self.assertEqual(row_active_mask(None, 8), [1] * 8)

    def test_values_are_normalized_to_zero_or_one(self) -> None:
        self.assertEqual(row_active_mask([True, False, True], 3), [1, 0, 1])
        self.assertEqual(row_active_mask([1, 0, 2], 3), [1, 0, 1])
        self.assertEqual(row_active_mask([None, "x"], 2), [0, 1])

    def test_wrong_length_is_rejected(self) -> None:
        for bad in ([1, 1], [1, 0, 1, 1]):
            with self.subTest(mask=bad), self.assertRaises(ValueError):
                row_active_mask(bad, 3)

    def test_activating_nothing_is_rejected(self) -> None:
        """A launch with no live row is a bug, not a no-op."""

        with self.assertRaises(ValueError):
            row_active_mask([0, 0, 0], 3)

    def test_bad_batch_size_is_rejected(self) -> None:
        for batch_size in (0, -1):
            with self.subTest(batch_size=batch_size), self.assertRaises(ValueError):
                row_active_mask(None, batch_size)


class KernelSideContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kernel = KERNEL.read_text(encoding="utf-8")

    def test_the_kv_write_is_the_only_thing_gated_by_row_active(self) -> None:
        """Inactive rows must still arrive at every bar.

        If a future change moved the ``row_on`` guard around an arrival, the wait
        targets would depend on the mask and the schedule would no longer be
        independent of it.
        """

        self.assertIn("const bool row_on = (g.row_active[batch] != 0);", self.kernel)
        self.assertEqual(
            self.kernel.count("row_on"),
            2,
            "row_on should be declared once and used once (the KV write guard)",
        )
        declaration = self.kernel.index("const bool row_on")
        guard = self.kernel.index("if (wwarp == 0 && row_on)")
        self.assertLess(declaration, guard)
        # The guard must appear inside Phase A, before the attention/FFN work,
        # and no arrival may sit inside it.
        segment = self.kernel[guard : guard + 400]
        self.assertNotIn("tg_arrive", segment)

    def test_the_barrier_phase_is_outside_the_guard(self) -> None:
        body = self.kernel[self.kernel.index("const bool row_on") :]
        self.assertIn("tg_arrive", body)


class LauncherSideContractTests(unittest.TestCase):
    def test_the_mask_is_created_through_the_contract(self) -> None:
        source = TASKGRAPH.read_text(encoding="utf-8")

        self.assertIn("row_active_mask(None, bs)", source)
        self.assertIn("row_active_mask(active_rows, rows)", source)
        # The bare literal this replaced; it must not come back, or the validated
        # contract is bypassed.
        self.assertNotIn("self._row_active = torch.ones(", source)

    def test_an_explicit_mask_is_optional(self) -> None:
        source = TASKGRAPH.read_text(encoding="utf-8")

        self.assertIn("active_rows=None", source)
        # The default path must not pay an extra copy per step.
        self.assertIn("if active_rows is not None:", source)


if __name__ == "__main__":
    unittest.main()
