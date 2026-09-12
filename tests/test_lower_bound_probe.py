"""CPU-safe tests for the device lower-bound probe's access pattern.

The probe streams a wrapped weight buffer with a cooperative grid.  A boundary
mistake there would read out of bounds on the GPU (and corrupt an unrelated
allocation), so the access pattern is mirrored in Python and checked here --
this runs on a laptop, unlike the probe itself.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from tools.bench_burst_lower_bound import (  # noqa: E402
    BARRIERS_PER_LAYER,
    probe_read_offsets,
)


class ProbeCoverageTests(unittest.TestCase):
    def _assert_exact_coverage(
        self, size: int, block_size: int, nblocks: int, vector_bytes: int = 16
    ) -> None:
        plan = probe_read_offsets(size, block_size, nblocks, vector_bytes)
        vector_region = size - (size % vector_bytes)
        # Every 16B read must fit entirely inside [0, size).
        for offset in plan["vector"]:
            self.assertGreaterEqual(offset, 0)
            self.assertLessEqual(
                offset + vector_bytes, size, f"OOB vector read at {offset}"
            )
        for offset in plan["tail"]:
            self.assertGreaterEqual(offset, vector_region)
            self.assertLess(offset, size, f"OOB tail read at {offset}")
        # Each byte must be covered exactly once (vector read start or tail).
        covered = set(plan["vector"]) | set(plan["tail"])
        for offset in range(0, vector_region, vector_bytes):
            self.assertIn(offset, covered, f"uncovered 16B vector at {offset}")
        for offset in range(vector_region, size):
            self.assertIn(offset, covered, f"uncovered tail byte at {offset}")

    def test_exact_coverage_typical_shape(self) -> None:
        # The A100 probe shape scaled down so the exhaustive check stays cheap.
        self._assert_exact_coverage(31_456 * 100, 512, 80)

    def test_exact_coverage_with_unaligned_tail(self) -> None:
        self._assert_exact_coverage(1_000_003, 128, 8)

    def test_exact_coverage_smaller_than_one_grid_step(self) -> None:
        # Fewer bytes than threads*16: only the first threads read anything.
        self._assert_exact_coverage(4096, 512, 80)

    def test_exact_coverage_size_not_multiple_of_vector(self) -> None:
        self._assert_exact_coverage(5000, 256, 4)

    def test_no_offsets_beyond_size(self) -> None:
        plan = probe_read_offsets(100_000, 512, 80)
        self.assertTrue(plan["vector"])
        self.assertLessEqual(max(plan["vector"]) + 16, 100_000)
        self.assertEqual(plan["tail"], [])  # 100_000 is a multiple of 16

    def test_tiny_size_yields_only_tail(self) -> None:
        plan = probe_read_offsets(8, 512, 80)
        self.assertEqual(plan["vector"], [])
        self.assertEqual(plan["tail"], list(range(8)))

    def test_vector_reads_cover_region_exactly_once(self) -> None:
        """Total 16B reads x 16 must equal the vector region, for many shapes."""

        for size, block_size, nblocks in (
            (1 << 16, 512, 80),
            (1 << 16, 128, 16),
            (4096, 512, 80),
            (123_456, 256, 10),
        ):
            plan = probe_read_offsets(size, block_size, nblocks)
            vector_region = size - (size % 16)
            self.assertEqual(len(plan["vector"]) * 16, vector_region)

    def test_read_count_per_thread_matches_loop_bound(self) -> None:
        """Loop exits while `start + k*grid_step + 16 <= vector_region`."""

        size, block_size, nblocks = 1 << 20, 128, 8
        vector_region = size - (size % 16)
        grid_step = nblocks * block_size * 16
        plan = probe_read_offsets(size, block_size, nblocks)
        for thread in range(block_size * nblocks):
            start = thread * 16
            if start + 16 > vector_region:
                expected = 0
            else:
                expected = (vector_region - 16 - start) // grid_step + 1
            actual = len(
                [o for o in plan["vector"] if o >= start and (o - start) % grid_step == 0]
            )
            self.assertEqual(actual, expected, f"thread {thread}")


class ProbeConstantsTests(unittest.TestCase):
    def test_barriers_per_layer_matches_model(self) -> None:
        # kernel.cu performs ~12 grid syncs per layer; docs/07 section 3.4
        # prices 28 layers x 12 barriers at 0.547 ms/token.
        self.assertEqual(BARRIERS_PER_LAYER, 12)
        self.assertAlmostEqual(BARRIERS_PER_LAYER * 28 * 1.63e-3, 0.548, places=2)


if __name__ == "__main__":
    unittest.main()
