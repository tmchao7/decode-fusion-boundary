"""Split nano-vLLM decode steps into device time and host residual.

The central question this answers: when cudagraph beats an unfused megakernel
at batch>1, *how much of the step is actually GPU work?*  The answer from the
A100 data in docs/07 is that the GPU is only ~21-25% of a decode step; the
bulk is per-step host/launch overhead that was already paid before the GPU
started.  If that is true, then the recoverable cost is host-side and it is
recoverable for *both* backends -- which changes what "megakernel beats
cudagraph" can even mean at large batch.

Usage::

    python -m tools.account_decode results/a100-performance.json
    python -m tools.account_decode results/*.json --bandwidth-gbps 1555

The input is the JSON written by ``python -m benchmarks.compare --json``
(schema_version 2).  Each decode step emits one token per running sequence, so
the wall time of one step is ``ms_per_token * batch_size``; the per-step GPU
time is ``cuda_decode_time_ms_total / decode_steps``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nanovllm.roofline import (  # noqa: E402
    HardwareSpec,
    ModelDims,
    RooflineModel,
)

# A cross-check discrepancy larger than this means the metric semantics differ
# from our model of "one step == one token per sequence".
CONSISTENCY_TOLERANCE = 0.05


def _p50(value: Any) -> float | None:
    """Accept a scalar or a ``{"p50": ...}`` summary block."""

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
            raise ValueError(f"{path}: unrecognised payload shape")
    return experiments


def accounting_from_experiment(experiment: dict) -> dict | None:
    """Return the per-step host/device split for one experiment row."""

    summary = experiment.get("summary") or {}
    workload = experiment.get("workload") or {}
    requests = experiment.get("requests") or []
    batch_size = int(workload.get("batch_size") or len(requests) or 1)
    decode_steps = float(summary.get("decode_steps") or 0)
    completed = [r for r in requests if r.get("e2e_ms") is not None]
    if decode_steps <= 0 or not completed:
        return None

    request_count = len(completed)
    # decode_time_ms_total is the sum over requests of their decode wall time;
    # every request sees the same (batched) steps, so dividing by
    # requests*steps yields the true wall duration of one step.
    wall_total_ms = float(summary.get("decode_time_ms_total") or 0.0)
    gpu_total_ms = float(summary.get("cuda_decode_time_ms_total") or 0.0)
    if wall_total_ms <= 0:
        return None
    per_request_wall = wall_total_ms / request_count
    per_request_gpu = gpu_total_ms / request_count
    ms_per_step_wall = per_request_wall / decode_steps
    ms_per_step_gpu = per_request_gpu / decode_steps

    # Independent cross-check: TPOT * batch is the same quantity measured from
    # per-token timestamps rather than from step boundaries.
    tpot_ms = None
    for request in completed:
        value = _p50(request.get("tpot_ms"))
        if value is not None:
            tpot_ms = value
            break
    tpot_step_ms = None if tpot_ms is None else tpot_ms * request_count
    consistency = None
    if tpot_step_ms and ms_per_step_wall > 0:
        consistency = tpot_step_ms / ms_per_step_wall

    backend_actuals = sorted(
        {
            str(request.get("backend_actual") or "unknown")
            for request in completed
            if request.get("backend_actual")
        }
    )
    fallback = experiment.get("fallback") or {}
    return {
        "workload_id": workload.get("id"),
        "category": workload.get("category"),
        "backend": experiment.get("backend"),
        "backend_actual": ",".join(backend_actuals) or experiment.get("backend"),
        "batch_size": request_count,
        "decode_steps": decode_steps,
        "output_tokens_per_request": statistics.fmean(
            float(request.get("output_tokens") or 0) for request in completed
        ),
        "ms_per_step_wall": ms_per_step_wall,
        "ms_per_step_gpu": ms_per_step_gpu,
        "host_residual_ms": ms_per_step_wall - ms_per_step_gpu,
        "gpu_fraction": ms_per_step_gpu / ms_per_step_wall,
        "ms_per_token_wall": ms_per_step_wall / request_count,
        "tpot_step_ms": tpot_step_ms,
        "step_consistency_ratio": consistency,
        "decode_tok_s": _p50(summary.get("decode_tok_s")),
        "fallback_count": fallback.get("count", 0),
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_markdown(rows: list[dict], model: RooflineModel) -> str:
    hw = model.hardware
    lines = [
        f"# Decode-step host/device accounting ({hw.name}, "
        f"{hw.effective_bandwidth_gbps():.0f} GB/s)",
        "",
        f"Roofline weight floor: {model.weight_bytes() / 1e9:.3f} GB/step, "
        f"{model.weight_bytes() / (hw.effective_bandwidth_gbps() * 1e9) * 1000.0:.3f} ms",
        "",
        "| workload | backend (actual) | batch | wall ms/step | GPU ms/step | "
        "host residual ms | GPU share | ms/token | tpot check | decode tok/s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        check = row["step_consistency_ratio"]
        check_text = "n/a" if check is None else f"{check:.2f}x"
        lines.append(
            "| {wid} | {backend} | {batch} | {wall} | {gpu} | {host} | {share} | "
            "{mtok} | {check} | {tok} |".format(
                wid=row["workload_id"],
                backend=row["backend_actual"],
                batch=row["batch_size"],
                wall=_fmt(row["ms_per_step_wall"]),
                gpu=_fmt(row["ms_per_step_gpu"]),
                host=_fmt(row["host_residual_ms"]),
                share=f"{row['gpu_fraction']:.1%}",
                mtok=_fmt(row["ms_per_token_wall"]),
                check=check_text,
                tok=_fmt(row["decode_tok_s"], 1),
            )
        )
    return "\n".join(lines) + "\n"


def render_comparison(rows: list[dict], model: RooflineModel) -> str:
    """For each batch, compare backends and quantify the recoverable host cost."""

    by_batch: dict[int, list[dict]] = {}
    for row in rows:
        by_batch.setdefault(int(row["batch_size"]), []).append(row)
    lines = [
        "# Host-side recoverable headroom",
        "",
        "For each batch this reports the best measured step time, and what the "
        "step time would be if only the device part remained. 'ceiling' is the "
        "decode tok/s at that bound.",
        "",
        "| batch | fastest backend | wall ms/step | GPU ms/step | host ms/step | "
        "wall/gpu | step-time if host removed | ceiling tok/s | measured tok/s |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for batch in sorted(by_batch):
        best = min(by_batch[batch], key=lambda row: row["ms_per_step_wall"])
        lower = model.lower_bound_ms(batch, seq_len=224)
        device_bound = max(lower, best["ms_per_step_gpu"])
        lines.append(
            "| {batch} | {backend} | {wall} | {gpu} | {host} | {ratio:.2f}x | "
            "{bound} | {ceiling:.0f} | {measured} |".format(
                batch=batch,
                backend=best["backend_actual"],
                wall=_fmt(best["ms_per_step_wall"]),
                gpu=_fmt(best["ms_per_step_gpu"]),
                host=_fmt(best["host_residual_ms"]),
                ratio=best["ms_per_step_wall"] / best["ms_per_step_gpu"]
                if best["ms_per_step_gpu"] > 0
                else float("inf"),
                bound=_fmt(device_bound),
                ceiling=1000.0 * batch / device_bound if device_bound > 0 else 0.0,
                measured=_fmt(best["decode_tok_s"], 1),
            )
        )
    return "\n".join(lines) + "\n"


def find_inconsistencies(rows: list[dict]) -> list[str]:
    warnings: list[str] = []
    for row in rows:
        ratio = row["step_consistency_ratio"]
        if ratio is None:
            continue
        if abs(ratio - 1.0) > CONSISTENCY_TOLERANCE:
            warnings.append(
                f"{row['workload_id']} {row['backend_actual']} batch={row['batch_size']}: "
                f"tpot*batch / (decode_time/decode_steps) = {ratio:.2f}x "
                f"(outside +/-{CONSISTENCY_TOLERANCE:.0%}); step accounting may not "
                "match this workload's cadence"
            )
    return warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Host/device split of nano-vLLM decode steps."
    )
    parser.add_argument("json_paths", nargs="+", help="benchmarks.compare JSON outputs")
    parser.add_argument("--bandwidth-gbps", type=float, default=None)
    parser.add_argument("--seq-len", type=int, default=224)
    parser.add_argument("--json", dest="json_out")
    parser.add_argument(
        "--warnings-as-errors",
        action="store_true",
        help="exit non-zero when the tpot cross-check disagrees",
    )
    args = parser.parse_args(argv)

    experiments = load_experiments(args.json_paths)
    rows = [
        row
        for row in (accounting_from_experiment(exp) for exp in experiments)
        if row is not None
    ]
    if not rows:
        print("no decodable experiments found", file=sys.stderr)
        return 2

    hw = HardwareSpec(
        bandwidth_gbps=args.bandwidth_gbps or HardwareSpec().bandwidth_gbps,
        peak_tflops=HardwareSpec().peak_tflops,
    )
    model = RooflineModel(ModelDims(), hw)

    # Only "controlled_batch" rows are relevant: they hold batch constant while
    # scaling batch size, which is exactly the comparison being made.
    batch_rows = [row for row in rows if row["category"] == "controlled_batch"]
    if not batch_rows:
        batch_rows = rows

    print(render_markdown(batch_rows, model))
    print(render_comparison(batch_rows, model))

    warnings = find_inconsistencies(batch_rows)
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "hardware": asdict(hw),
                    "rows": batch_rows,
                    "comparison": render_comparison(batch_rows, model),
                    "warnings": warnings,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.json_out}")

    if warnings and args.warnings_as_errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
