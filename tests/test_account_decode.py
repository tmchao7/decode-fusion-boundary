"""Tests for tools/account_decode.py (host/device split of decode steps)."""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from nanovllm.roofline import HardwareSpec, ModelDims, RooflineModel  # noqa: E402
from tools.account_decode import (  # noqa: E402
    accounting_from_experiment,
    find_inconsistencies,
    render_comparison,
)


def _experiment(
    *,
    backend: str,
    batch_size: int,
    decode_steps: int,
    ms_per_step_wall: float,
    ms_per_step_gpu: float,
    output_tokens: int | None = None,
    backend_actual: str | None = None,
) -> dict:
    """Build a compare.py-shaped experiment with self-consistent metrics."""

    requests = output_tokens or decode_steps
    wall_total = ms_per_step_wall * decode_steps * batch_size
    gpu_total = ms_per_step_gpu * decode_steps * batch_size
    tpot_ms = ms_per_step_wall / batch_size
    actual = backend_actual if backend_actual is not None else backend
    return {
        "backend": backend,
        "workload": {
            "id": f"perf_p128_b{batch_size}",
            "category": "controlled_batch",
            "batch_size": batch_size,
        },
        "summary": {
            "decode_steps": decode_steps,
            "decode_time_ms_total": wall_total,
            "cuda_decode_time_ms_total": gpu_total,
            "decode_tok_s": requests * batch_size / (wall_total / 1000.0),
        },
        "requests": [
            {
                "request_id": index,
                "e2e_ms": 10.0 + index,
                "tpot_ms": tpot_ms,
                "output_tokens": requests,
                "backend_actual": actual,
            }
            for index in range(batch_size)
        ],
        "fallback": {"count": 0, "reasons": []},
    }


class AccountingTests(unittest.TestCase):
    def test_documented_b8_split(self) -> None:
        """b8: 3.43 ms/step wall, 0.77 ms GPU -> ~2.66 ms host."""

        row = accounting_from_experiment(
            _experiment(
                backend="torch-cudagraph",
                backend_actual="torch",
                batch_size=8,
                decode_steps=4,
                ms_per_step_wall=3.432,
                ms_per_step_gpu=0.77,
            )
        )
        assert row is not None
        self.assertAlmostEqual(row["ms_per_step_wall"], 3.432, places=3)
        self.assertAlmostEqual(row["ms_per_step_gpu"], 0.77, places=3)
        self.assertAlmostEqual(row["host_residual_ms"], 2.662, places=3)
        self.assertAlmostEqual(row["ms_per_token_wall"], 0.429, places=3)
        self.assertAlmostEqual(row["gpu_fraction"], 0.77 / 3.432, places=6)

    def test_tpot_cross_check_is_one_for_consistent_data(self) -> None:
        row = accounting_from_experiment(
            _experiment(
                backend="torch-cudagraph",
                batch_size=4,
                decode_steps=8,
                ms_per_step_wall=3.0,
                ms_per_step_gpu=0.8,
            )
        )
        assert row is not None
        self.assertAlmostEqual(row["step_consistency_ratio"], 1.0, places=6)
        self.assertEqual(find_inconsistencies([row]), [])

    def test_inconsistent_cadence_is_flagged(self) -> None:
        experiment = _experiment(
            backend="megakernel-burst",
            batch_size=4,
            decode_steps=8,
            ms_per_step_wall=3.0,
            ms_per_step_gpu=0.8,
        )
        # Simulate a workload whose per-token cadence does not match the step
        # accounting (e.g. serialised bursts inside one step).
        for request in experiment["requests"]:
            request["tpot_ms"] = 0.1
        row = accounting_from_experiment(experiment)
        assert row is not None
        warnings = find_inconsistencies([row])
        self.assertEqual(len(warnings), 1)
        self.assertIn("megakernel-burst", warnings[0])

    def test_missing_decode_steps_is_skipped(self) -> None:
        experiment = _experiment(
            backend="x",
            batch_size=1,
            decode_steps=4,
            ms_per_step_wall=2.0,
            ms_per_step_gpu=1.0,
        )
        experiment["summary"]["decode_steps"] = 0
        self.assertIsNone(accounting_from_experiment(experiment))

    def test_no_completed_requests_is_skipped(self) -> None:
        experiment = _experiment(
            backend="x",
            batch_size=2,
            decode_steps=4,
            ms_per_step_wall=2.0,
            ms_per_step_gpu=1.0,
        )
        experiment["requests"] = []
        self.assertIsNone(accounting_from_experiment(experiment))

    def test_comparison_table_reports_host_headroom(self) -> None:
        rows = [
            accounting_from_experiment(
                _experiment(
                    backend=backend,
                    backend_actual=backend,
                    batch_size=8,
                    decode_steps=4,
                    ms_per_step_wall=wall,
                    ms_per_step_gpu=gpu,
                )
            )
            for backend, wall, gpu in (
                ("torch-cudagraph", 3.43, 0.77),
                ("megakernel-burst", 6.50, 6.30),
            )
        ]
        model = RooflineModel(ModelDims(), HardwareSpec())
        table = render_comparison([row for row in rows if row], model)
        self.assertIn("torch-cudagraph", table)
        # The fastest backend at b8 is cudagraph, whose host residual dominates.
        self.assertIn("2.660", table)
        self.assertIn("ceiling tok/s", table)


if __name__ == "__main__":
    unittest.main()
