import json
import statistics
import unittest
from types import SimpleNamespace

from nanovllm.metrics import (
    CudaEventTimer,
    MetricsCollector,
    RequestMetrics,
    aggregate_requests,
    compare_metric_summaries,
    percentile,
    summarize,
    summarize_runs,
)


class MetricsMathTests(unittest.TestCase):
    def test_percentiles_are_deterministic_and_empty_is_none(self):
        self.assertIsNone(percentile([], 0.95))
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0, 4.0], 0.95), 3.85)

    def test_request_metrics_derives_tpot_and_itl(self):
        request = RequestMetrics(
            request_id=7,
            prompt_tokens=4,
            arrival_time=10.0,
            token_times=[11.0, 11.5, 12.25],
            output_tokens=3,
            prefill_time=0.25,
            decode_time=1.25,
            backend_requested="megakernel",
            backend_actual="torch+megakernel",
        )
        sample = request.to_dict()
        self.assertEqual(sample["ttft_ms"], 1000.0)
        self.assertEqual(sample["itl_ms"], [500.0, 750.0])
        self.assertEqual(sample["e2e_ms"], 2250.0)
        self.assertEqual(sample["tpot_ms"], 625.0)

    def test_single_token_tpot_is_none_not_nan(self):
        request = RequestMetrics(
            request_id=1,
            prompt_tokens=1,
            arrival_time=0.0,
            token_times=[0.25],
            output_tokens=1,
            prefill_time=0.1,
            decode_time=0.0,
            backend_requested="torch",
            backend_actual="torch",
        )
        self.assertIsNone(request.to_dict()["tpot_ms"])

    def test_aggregate_latencies_stay_in_milliseconds(self):
        """Regression: the aggregate used to divide ms by 1000 and report seconds.

        ``RequestMetrics.to_dict()`` emits milliseconds, so a 1000x understated
        ``ttft_ms``/``tpot_ms``/``itl_ms``/``e2e_ms`` reached every generated
        report and plot (which label those columns "ms").
        """

        request = RequestMetrics(
            request_id=3,
            prompt_tokens=4,
            arrival_time=0.0,
            token_times=[0.1, 0.2],
            output_tokens=2,
            prefill_time=0.05,
            decode_time=0.1,
            backend_requested="torch",
            backend_actual="torch",
        )
        summary = aggregate_requests([request], wall_time=1.0)
        self.assertAlmostEqual(summary["ttft_ms"]["p50"], 100.0, places=6)
        self.assertAlmostEqual(summary["tpot_ms"]["p50"], 100.0, places=6)
        self.assertAlmostEqual(summary["itl_ms"]["p50"], 100.0, places=6)
        self.assertAlmostEqual(summary["e2e_ms"]["p50"], 200.0, places=6)

    def test_aggregate_and_comparison(self):
        requests = [
            RequestMetrics(1, 4, 0.0, [0.1, 0.2], 2, 0.05, 0.1, "torch", "torch"),
            RequestMetrics(2, 4, 0.0, [0.2, 0.4], 2, 0.05, 0.2, "torch", "torch"),
        ]
        summary = aggregate_requests(requests)
        self.assertEqual(summary["request_count"], 2)
        self.assertEqual(summary["output_tokens"], 4)
        self.assertIsNotNone(summary["ttft_ms"]["p95"])

        comparison = compare_metric_summaries(
            {
                "prefill_tok_s": 200.0,
                "output_tok_s": 100.0,
                "peak_memory_allocated_bytes": 200.0,
                "e2e_ms": {"p50": 20.0},
            },
            {
                "prefill_tok_s": 250.0,
                "output_tok_s": 125.0,
                "peak_memory_allocated_bytes": 150.0,
                "e2e_ms": {"p50": 15.0},
            },
        )
        self.assertEqual(comparison["output_tok_s"]["speedup"], 1.25)
        self.assertEqual(comparison["prefill_tok_s"]["improvement_pct"], 25.0)
        self.assertEqual(comparison["peak_memory_allocated_bytes"]["improvement_pct"], 25.0)
        self.assertEqual(comparison["e2e_ms_p50"]["improvement_pct"], 25.0)

    def test_metrics_are_json_serializable(self):
        request = RequestMetrics(1, 2, 0.0, [1.0], 1, 0.1, 0.2, "torch", "torch")
        json.dumps(request.to_dict())

    def test_cuda_event_timer_is_safe_without_cuda(self):
        timer = CudaEventTimer()
        self.assertIsNone(timer.elapsed_seconds())

    def test_collector_tracks_cuda_phase_time(self):
        class FakeTimer:
            def elapsed_seconds(self):
                return 0.05

        collector = MetricsCollector("torch")
        collector.add_request(4, 8, 0.0)
        seq = SimpleNamespace(seq_id=4)
        collector.record_step([seq], True, 0.1, "torch", None, cuda_timer=FakeTimer())
        report = collector.report(wall_time=0.1)
        self.assertEqual(report["summary"]["cuda_prefill_time_ms_total"], 50.0)
        self.assertEqual(report["requests"][0]["cuda_prefill_time_ms"], 50.0)

    def test_collector_tracks_phase_tokens_and_backend_fallback(self):
        collector = MetricsCollector("megakernel")
        collector.add_request(3, 8, 0.0)
        seq = SimpleNamespace(seq_id=3)
        collector.record_step([seq], True, 0.2, "torch", "decode-only")
        collector.record_token(3, 0.3)
        collector.record_step([seq], False, 0.1, "megakernel", None)
        collector.record_token(3, 0.4)
        report = collector.report(wall_time=0.4)
        self.assertEqual(report["summary"]["backend_actuals"], ["megakernel", "torch"])
        self.assertEqual(report["summary"]["fallback_count"], 1)
        self.assertEqual(report["requests"][0]["output_tokens"], 2)


class MetricsMergeTests(unittest.TestCase):
    def test_merge_reports_preserves_requests_and_adds_wall_time(self):
        from nanovllm.metrics import merge_metric_reports

        first = MetricsCollector("torch")
        first.add_request(1, 4, 0.0)
        first.record_token(1, 0.2)
        second = MetricsCollector("torch")
        second.add_request(2, 8, 0.0)
        second.record_token(2, 0.3)
        merged = merge_metric_reports(
            [first.report(wall_time=0.2), second.report(wall_time=0.3)]
        )
        self.assertEqual(merged["summary"]["request_count"], 2)
        self.assertEqual(merged["summary"]["output_tokens"], 2)
        self.assertEqual(merged["summary"]["wall_time_s"], 0.5)
        self.assertEqual(len(merged["requests"]), 2)

class BatchThroughputTests(unittest.TestCase):
    def test_summary_throughput_uses_step_time_once_for_a_batch(self):
        collector = MetricsCollector("torch")
        collector.add_request(1, 10, 0.0)
        collector.add_request(2, 20, 0.0)
        seqs = [SimpleNamespace(seq_id=1), SimpleNamespace(seq_id=2)]
        collector.record_step(seqs, True, 0.5, "torch", None)
        collector.record_token(1, 0.6)
        collector.record_token(2, 0.6)
        summary = collector.report(wall_time=0.6)["summary"]
        self.assertEqual(summary["prefill_tok_s"], 60.0)


class SchedulerStatsPassthroughTests(unittest.TestCase):
    def test_report_embeds_the_scheduler_counters(self):
        """``LLMEngine._generate`` passes ``scheduler.stats()`` to ``report``."""

        collector = MetricsCollector("megakernel")
        stats = {
            "waiting_sequences": 1,
            "running_sequences": 2,
            "peak_waiting_sequences": 3,
            "peak_running_sequences": 4,
            "decode_steps_since_prefill": 0,
            "prefill_decode_ratio": 1,
            "block_size": 256,
            "allocated_blocks": 5,
            "free_blocks": 6,
            "peak_used_blocks": 7,
            "prefix_cache_hits": 8,
            "prefix_cache_misses": 9,
        }

        report = collector.report(wall_time=1.0, scheduler_stats=stats)

        self.assertEqual(report["summary"]["scheduler"], stats)
        # Reported only when the engine supplies it.
        self.assertNotIn(
            "scheduler", MetricsCollector("torch").report(wall_time=1.0)["summary"]
        )


class SummaryStatisticsTests(unittest.TestCase):
    def test_reports_dispersion_not_just_the_mean(self):
        """A mean without dispersion cannot answer "is this difference real?"."""

        summary = summarize([1.0, 2.0, 3.0, 4.0])

        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["max"], 4.0)
        self.assertAlmostEqual(summary["stddev"], statistics.stdev([1.0, 2.0, 3.0, 4.0]))
        self.assertAlmostEqual(summary["cov"], summary["stddev"] / 2.5)
        self.assertEqual(summary["p5"], percentile([1.0, 2.0, 3.0, 4.0], 0.05))

    def test_single_sample_has_no_dispersion(self):
        summary = summarize([7.0])

        self.assertEqual(summary["count"], 1)
        self.assertEqual(summary["mean"], 7.0)
        self.assertIsNone(summary["stddev"])
        self.assertIsNone(summary["cov"])

    def test_empty_sample_is_all_none(self):
        summary = summarize([])

        for key in ("count", "mean", "stddev", "cov", "min", "max", "p5", "p50"):
            with self.subTest(key=key):
                self.assertIsNone(summary[key])

    def test_zero_mean_has_no_coefficient_of_variation(self):
        summary = summarize([-1.0, 1.0])

        self.assertEqual(summary["mean"], 0.0)
        self.assertIsNone(summary["cov"])

    def test_scale_is_applied_before_the_statistics(self):
        summary = summarize([1.0, 2.0], scale=1000.0)

        self.assertEqual(summary["mean"], 1500.0)
        self.assertEqual(summary["min"], 1000.0)


class RunSpreadTests(unittest.TestCase):
    def test_bootstrap_interval_brackets_the_mean(self):
        spread = summarize_runs([10.0, 11.0, 12.0, 13.0, 14.0])

        self.assertEqual(spread["count"], 5)
        self.assertLessEqual(spread["ci95_low"], spread["mean"])
        self.assertGreaterEqual(spread["ci95_high"], spread["mean"])
        self.assertAlmostEqual(
            spread["ci95_half_width"],
            (spread["ci95_high"] - spread["ci95_low"]) / 2.0,
        )
        self.assertGreater(spread["resamples"], 0)

    def test_interval_is_reproducible_for_a_fixed_seed(self):
        values = [10.0, 11.0, 12.0, 13.0, 14.0]

        self.assertEqual(summarize_runs(values), summarize_runs(values))

    def test_constant_runs_have_zero_spread(self):
        spread = summarize_runs([5.0] * 5)

        self.assertEqual(spread["stddev"], 0.0)
        self.assertEqual(spread["cov"], 0.0)
        self.assertEqual(spread["ci95_low"], 5.0)
        self.assertEqual(spread["ci95_high"], 5.0)

    def test_single_run_has_no_interval(self):
        spread = summarize_runs([5.0])

        self.assertEqual(spread["count"], 1)
        self.assertIsNone(spread["ci95_low"])
        self.assertIsNone(spread["ci95_high"])
        self.assertIsNone(spread["ci95_half_width"])

    def test_empty_is_all_none(self):
        spread = summarize_runs([])

        self.assertIsNone(spread["mean"])
        self.assertIsNone(spread["ci95_low"])
        self.assertIsNone(spread["stddev"])

    def test_more_runs_narrow_the_interval(self):
        """Quoting a CI is only useful if it shrinks as runs are added.

        Compares the same value distribution sampled 4x more often, so the
        spread is unchanged and only ``n`` differs (a 2-sample baseline cannot
        estimate spread, so it is not a valid comparison).
        """

        base = [9.94, 10.03, 9.98, 10.06]

        loose = summarize_runs(base)
        tight = summarize_runs(base * 4)

        self.assertLess(tight["ci95_half_width"], loose["ci95_half_width"])


if __name__ == "__main__":
    unittest.main()
