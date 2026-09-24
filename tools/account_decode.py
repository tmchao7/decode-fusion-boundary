"""Split active nano-vLLM decode steps into GPU time and host residual."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nanovllm.roofline import HardwareSpec, ModelDims, RooflineModel  # noqa: E402


def _p50(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("p50", "mean"):
            if value.get(key) is not None:
                return float(value[key])
        return None
    return float(value)


def load_experiments(paths: list[str]) -> list[dict]:
    experiments: list[dict] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "experiments" in payload:
            experiments.extend(payload["experiments"])
        elif isinstance(payload, list):
            experiments.extend(payload)
        else:
            raise ValueError(f"{path}: expected benchmarks.compare JSON")
    return experiments


def accounting_from_experiment(experiment: dict) -> dict | None:
    summary = experiment.get("summary") or {}
    workload = experiment.get("workload") or {}
    requests = [r for r in experiment.get("requests", []) if r.get("e2e_ms") is not None]
    decode_steps = float(summary.get("decode_steps") or 0)
    if decode_steps <= 0 or not requests:
        return None

    wall_total = float(summary.get("decode_time_ms_total") or 0.0)
    gpu_total = float(summary.get("cuda_decode_time_ms_total") or 0.0)
    if wall_total <= 0:
        return None
    request_count = len(requests)
    wall_ms = wall_total / request_count / decode_steps
    gpu_ms = gpu_total / request_count / decode_steps
    tpot = next((_p50(r.get("tpot_ms")) for r in requests if r.get("tpot_ms") is not None), None)
    return {
        "workload_id": workload.get("id"),
        "category": workload.get("category"),
        "backend": experiment.get("backend"),
        "backend_actual": ",".join(sorted({str(r.get("backend_actual")) for r in requests if r.get("backend_actual")})),
        "batch_size": int(workload.get("batch_size") or request_count),
        "decode_steps": decode_steps,
        "output_tokens_per_request": statistics.fmean(float(r.get("output_tokens") or 0) for r in requests),
        "ms_per_step_wall": wall_ms,
        "ms_per_step_gpu": gpu_ms,
        "host_residual_ms": wall_ms - gpu_ms,
        "gpu_fraction": gpu_ms / wall_ms if wall_ms else 0.0,
        "ms_per_token_wall": wall_ms / request_count,
        "tpot_step_ms": None if tpot is None else tpot * request_count,
        "fallback_count": int((experiment.get("fallback") or {}).get("count", 0)),
    }


def render_markdown(rows: list[dict]) -> str:
    lines = [
        "# Decode accounting",
        "",
        "| workload | backend | batch | wall ms/step | GPU ms/step | host residual ms | GPU share |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['workload_id']} | {row['backend']} | {row['batch_size']} | "
            f"{row['ms_per_step_wall']:.3f} | {row['ms_per_step_gpu']:.3f} | "
            f"{row['host_residual_ms']:.3f} | {row['gpu_fraction']:.1%} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_paths", nargs="+")
    parser.add_argument("--json", dest="json_out")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--bandwidth-gbps", type=float, default=None)
    args = parser.parse_args(argv)

    # Construct the model here so the CLI still validates the active taskgraph
    # dimensions and keeps the bandwidth option discoverable for reports.
    HardwareSpec(bandwidth_gbps=args.bandwidth_gbps or HardwareSpec().bandwidth_gbps)
    RooflineModel(ModelDims(), HardwareSpec())
    rows = [row for item in load_experiments(args.json_paths) if (row := accounting_from_experiment(item))]
    if not rows:
        print("no decodable experiments found", file=sys.stderr)
        return 2
    markdown = render_markdown(rows)
    print(markdown, end="")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
