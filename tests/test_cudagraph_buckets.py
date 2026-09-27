"""Lock the CUDA-graph bucket scheme and the padding it implies.

The bucket list is a fairness parameter, not an implementation detail: a batch
that is smaller than the bucket it replays does the GEMM of the whole bucket, and
that time lands on the cudagraph arm's per-token number. Two protocols are pinned
here so neither can drift silently:

* the current benchmark (``max_num_seqs == batch_size``), where only batch
  3/5/6/7 pad, because ``max_bs`` is itself appended as a bucket once it is > 8;
* the archived crossover scan (``max_num_seqs = max(batch_size, 32)``), whose
  cudagraph rows carry up to 1.88x padding.
"""

import unittest

from nanovllm.engine.cudagraph_buckets import (
    GRAPH_BATCH_CAP,
    batch_padding_factor,
    graph_buckets,
    padding_report,
    select_graph_bucket,
)

# The bucket formula the runner used inline before this module existed, kept here
# as an independent re-derivation rather than a copy of the implementation.
def reference_buckets(max_num_seqs: int) -> list[int]:
    max_bs = min(max_num_seqs, 512)
    buckets = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    if max_bs > buckets[-1]:
        buckets.append(max_bs)
    return buckets


class GraphBucketTests(unittest.TestCase):
    def test_small_and_exact_shapes(self):
        self.assertEqual(graph_buckets(1), [1, 2, 4, 8])
        self.assertEqual(graph_buckets(8), [1, 2, 4, 8])
        self.assertEqual(graph_buckets(16), [1, 2, 4, 8, 16])
        self.assertEqual(graph_buckets(32), [1, 2, 4, 8, 16, 32])

    def test_max_bs_is_appended_so_a_non_bucket_batch_has_an_exact_graph(self):
        self.assertEqual(graph_buckets(10), [1, 2, 4, 8, 10])
        self.assertEqual(graph_buckets(17), [1, 2, 4, 8, 16, 17])
        self.assertEqual(graph_buckets(33), [1, 2, 4, 8, 16, 32, 33])

    def test_cap_and_validation(self):
        self.assertEqual(graph_buckets(1024)[-1], GRAPH_BATCH_CAP)
        self.assertEqual(graph_buckets(GRAPH_BATCH_CAP + 1), graph_buckets(GRAPH_BATCH_CAP))
        for value in (0, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                graph_buckets(value)

    def test_matches_an_independent_re_derivation(self):
        for max_num_seqs in (1, 2, 3, 8, 10, 15, 16, 17, 32, 33, 64, 100, 512, 513):
            with self.subTest(max_num_seqs=max_num_seqs):
                self.assertEqual(graph_buckets(max_num_seqs), reference_buckets(max_num_seqs))


class GraphBucketSelectionTests(unittest.TestCase):
    def test_exact_bucket_is_reused(self):
        self.assertEqual(select_graph_bucket(8, graph_buckets(32)), 8)
        self.assertEqual(select_graph_bucket(32, graph_buckets(32)), 32)

    def test_between_buckets_rounds_up(self):
        self.assertEqual(select_graph_bucket(9, graph_buckets(32)), 16)
        self.assertEqual(select_graph_bucket(31, graph_buckets(32)), 32)

    def test_above_the_cap_has_no_bucket(self):
        """None is the runner's eager fallback, not an exception."""

        self.assertIsNone(select_graph_bucket(513, graph_buckets(512)))
        self.assertIsNone(batch_padding_factor(513, graph_buckets(512)))

    def test_validation(self):
        with self.assertRaises(ValueError):
            select_graph_bucket(0, graph_buckets(8))


class CurrentProtocolPaddingTests(unittest.TestCase):
    """``max_num_seqs == batch_size``: what benchmarks/compare.py builds."""

    def _factor(self, batch_size: int) -> float | None:
        return batch_padding_factor(batch_size, graph_buckets(batch_size))

    def test_the_measured_grid_is_exact(self):
        for batch_size in (1, 2, 4, 8, 16, 32):
            with self.subTest(batch_size=batch_size):
                self.assertEqual(self._factor(batch_size), 1.0)

    def test_only_three_five_six_seven_pad(self):
        """Above 8, max_bs is appended as its own bucket, so those are exact."""

        self.assertAlmostEqual(self._factor(3), 4 / 3, places=6)
        self.assertAlmostEqual(self._factor(5), 8 / 5, places=6)
        self.assertAlmostEqual(self._factor(6), 8 / 6, places=6)
        self.assertAlmostEqual(self._factor(7), 8 / 7, places=6)
        for batch_size in (9, 10, 15, 17, 31, 33):
            with self.subTest(batch_size=batch_size):
                self.assertEqual(self._factor(batch_size), 1.0)


class ArchivedCrossoverProtocolPaddingTests(unittest.TestCase):
    """``max_num_seqs = max(batch_size, 32)``: the docs/06-07 crossover scan."""

    def _factor(self, batch_size: int) -> float | None:
        max_num_seqs = max(batch_size, 32)
        return batch_padding_factor(batch_size, graph_buckets(max_num_seqs))

    def test_documented_rows_carry_padding(self):
        expected = {3: 4 / 3, 5: 8 / 5, 6: 8 / 6, 7: 8 / 7, 10: 16 / 10, 17: 32 / 17}
        for batch_size, factor in expected.items():
            with self.subTest(batch_size=batch_size):
                self.assertAlmostEqual(self._factor(batch_size), factor, places=6)

    def test_exact_rows_are_unaffected(self):
        for batch_size in (1, 2, 4, 8, 16, 32):
            with self.subTest(batch_size=batch_size):
                self.assertEqual(self._factor(batch_size), 1.0)

    def test_worst_case_is_below_two(self):
        self.assertLess(max(self._factor(bs) for bs in range(1, 33)), 2.0)


class PaddingReportTests(unittest.TestCase):
    def test_report_shape_for_the_current_protocol(self):
        report = padding_report(5, 5, uses_cuda_graph=True)

        self.assertTrue(report["uses_cuda_graph"])
        self.assertEqual(report["max_num_seqs"], 5)
        self.assertEqual(report["buckets"], [1, 2, 4, 8])
        self.assertEqual(report["bucket"], 8)
        self.assertAlmostEqual(report["padding_factor"], 1.6, places=6)

    def test_non_graph_backends_are_marked(self):
        report = padding_report(8, 8, uses_cuda_graph=False)

        self.assertFalse(report["uses_cuda_graph"])
        self.assertEqual(report["padding_factor"], 1.0)


if __name__ == "__main__":
    unittest.main()
