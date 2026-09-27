"""Unit tests for the bandwidth calibration tool's pure parts.

The CUDA path needs a GPU and is exercised on the server; everything here runs
anywhere. The A100 numbers are pinned because they are the calibration this
project actually uses: 1593 MHz x2 x (5120/8) = 2039 GB/s spec, and the measured
1699 GB/s is what turned "0.74 ms/step implies 1.61 TB/s" from a suspected
measurement bug into a verified assertion.
"""

import unittest

from tools.bench_memory_bandwidth import (
    achieved_fraction,
    bandwidth_gbps,
    build_parser,
    device_spec_bandwidth_gbps,
    parse_byte_count,
    summarize_bandwidth,
)


class ByteCountParsingTests(unittest.TestCase):
    def test_scientific_and_suffixed_forms_agree(self):
        self.assertEqual(parse_byte_count("1.2e9"), 1_200_000_000)
        self.assertEqual(parse_byte_count("1200MB"), 1_200_000_000)
        self.assertEqual(parse_byte_count("1200mb"), 1_200_000_000)
        self.assertEqual(parse_byte_count("2GiB"), 2 * 2**30)
        self.assertEqual(parse_byte_count("512"), 512)

    def test_rejects_empty_nonsense_and_zero(self):
        for value in ("", "  ", "abc", "0", "-5", "GB"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_byte_count(value)


class BandwidthMathTests(unittest.TestCase):
    def test_gbps_uses_decimal_gigabytes(self):
        self.assertEqual(bandwidth_gbps(1_000_000_000, 0.5), 2.0)

    def test_zero_bytes_or_time_is_unknown_not_zero(self):
        self.assertIsNone(bandwidth_gbps(0, 0.5))
        self.assertIsNone(bandwidth_gbps(1_000, 0.0))
        self.assertIsNone(bandwidth_gbps(-1, 1.0))

    def test_a100_80gb_spec_is_reproduced_from_clock_and_bus(self):
        """1593 MHz x 2 x (5120 bit / 8) = 2039 GB/s, the published figure."""

        self.assertAlmostEqual(device_spec_bandwidth_gbps(1593, 5120), 2039.0, places=1)

    def test_the_40gb_part_is_distinguishable(self):
        """1215 MHz gives 1555 GB/s -- the HBM2 part that used to understate the 80 GB card."""

        self.assertAlmostEqual(device_spec_bandwidth_gbps(1215, 5120), 1555.2, places=1)

    def test_missing_clock_or_bus_width_is_none(self):
        self.assertIsNone(device_spec_bandwidth_gbps(None, 5120))
        self.assertIsNone(device_spec_bandwidth_gbps(1593, None))
        self.assertIsNone(device_spec_bandwidth_gbps(0, 5120))
        self.assertIsNone(device_spec_bandwidth_gbps(1593, 0))

    def test_achieved_fraction_reproduces_the_measured_a100_calibration(self):
        self.assertAlmostEqual(achieved_fraction(1699.0, 2039.0), 0.8333, places=3)

    def test_an_unknown_limit_is_not_guessed(self):
        self.assertIsNone(achieved_fraction(1699.0, None))
        self.assertIsNone(achieved_fraction(None, 2039.0))
        self.assertIsNone(achieved_fraction(1699.0, 0.0))

    def test_summary_reuses_the_shared_statistics(self):
        summary = summarize_bandwidth([100.0, 110.0, 120.0])

        for key in ("count", "mean", "stddev", "cov", "min", "max", "p50", "p95"):
            with self.subTest(key=key):
                self.assertIn(key, summary)
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["min"], 100.0)
        self.assertEqual(summary["max"], 120.0)


class ArgumentValidationTests(unittest.TestCase):
    """The argument checks must happen before torch is imported."""

    def test_defaults(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.iterations, 20)
        self.assertEqual(args.warmup, 5)
        self.assertEqual(parse_byte_count(args.bytes), 1_200_000_000)
        self.assertIsNone(args.json)

    def test_measure_rejects_a_bad_iteration_count(self):
        from tools.bench_memory_bandwidth import measure_read_bandwidth

        with self.assertRaises(ValueError):
            measure_read_bandwidth(1024, iterations=0)
        with self.assertRaises(ValueError):
            measure_read_bandwidth(1024, warmup=-1)


if __name__ == "__main__":
    unittest.main()
