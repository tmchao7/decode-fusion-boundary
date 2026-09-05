import json
import unittest
from types import SimpleNamespace

from nanovllm.metrics import (
    MetricsCollector,
    RequestMetrics,
    aggregate_requests,
    compare_metric_summaries,
    percentile,
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


if __name__ == "__main__":
    unittest.main()
