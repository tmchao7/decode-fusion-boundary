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
import re
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
    """Load compare.py experiment rows from one or more payloads.

    Also accepts ``tools/bench_burst_batch.py`` crossover scans, whose schema is
    different (``records`` with per-backend ``ms_per_tok_min``); those rows are
    converted so the same accounting/reporting path applies.  The crossover
    scan carries no per-step GPU time, so its rows are reported against the
    roofline floor instead (``device_source="floor"``).
    """

    experiments: list[dict] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "experiments" in payload:
            experiments.extend(payload["experiments"])
        elif isinstance(payload, dict) and "records" in payload:
            experiments.extend(_experiments_from_crossover(payload, path))
        elif isinstance(payload, list):
            experiments.extend(payload)
        else:
            raise ValueError(f"{path}: unrecognised payload shape")
    return experiments


def _serial_factor(note: str | None) -> int:
    """Parse the ``serial xN`` marker the crossover scan puts in ``note``."""

    if not note:
        return 1
    match = re.search(r"serial\s*[x×]\s*(\d+)", note)
    return max(1, int(match.group(1))) if match else 1


def _experiments_from_crossover(payload: dict, path: str) -> list[dict]:
    prompt_length = int(payload.get("prompt_length") or 0)
    max_tokens = int(payload.get("max_tokens") or 0)
    experiments = []
    for record in payload.get("records") or []:
        batch = int(record.get("batch") or 0)
        note = record.get("note")
        serial = _serial_factor(note)
        for backend, label in (("cudagraph", "torch-cudagraph"), ("burst", "megakernel-burst")):
            block = record.get(backend) or {}
            ms_per_tok = block.get("ms_per_tok_min")
            if ms_per_tok is None:
                continue
            expected = int(block.get("expected_tokens") or max_tokens * batch)
            experiments.append(
                {
                    "backend": label,
                    "workload": {
                        "id": f"crossover_b{batch}",
                        "category": "crossover_scan",
                        "batch_size": batch,
                        "prompt_lengths": [prompt_length],
                        "max_tokens": max_tokens,
                    },
                    "summary": {
                        "decode_steps": float(max_tokens),
                        "output_tokens": expected,
                        "crossover_source": path,
                        # The serial marker describes the burst backend only:
                        # hybrid dispatch splits batch>safe_burst_batch into
                        # several serial bursts, while cudagraph does not.
                        "serial_chunks": serial if backend == "burst" else 1,
                        "record_note": note,
                    },
                    "requests": [
                        {
                            "request_id": index,
                            # ms_per_tok is elapsed/total_tokens, i.e. the true
                            # per-sequence per-token wall time.
                            "tpot_ms": float(ms_per_tok),
                            "output_tokens": expected,
                            "e2e_ms": 1.0,
                            "backend_actual": label,
                        }
                        for index in range(batch)
                    ],
                    "fallback": {"count": 0, "reasons": []},
                }
            )
    return experiments


def accounting_from_crossover(experiment: dict, floor_ms: float) -> dict | None:
    """Host/device split for a crossover-scan row.

    The scan measures only wall time, so the device time is taken from the
    roofline floor (the best any kernel could do).  The reported "host
    residual" is therefore a *lower bound* on host overhead, and "ceiling
    tok/s" an upper bound on what removing it could buy.
    """

    workload = experiment.get("workload") or {}
    summary = experiment.get("summary") or {}
    requests = experiment.get("requests") or []
    batch = int(workload.get("batch_size") or len(requests) or 1)
    ms_per_tok = _p50((requests[0] if requests else {}).get("tpot_ms"))
    if ms_per_tok is None or ms_per_tok <= 0:
        return None
    ms_per_step_wall = ms_per_tok * batch
    return {
        "workload_id": workload.get("id"),
        "category": workload.get("category"),
        "backend": experiment.get("backend"),
        "backend_actual": experiment.get("backend"),
        "batch_size": batch,
        "decode_steps": float(summary.get("decode_steps") or 0),
        "output_tokens_per_request": float(ms_per_tok),
        "ms_per_step_wall": ms_per_step_wall,
        "ms_per_step_gpu": floor_ms,
        "host_residual_ms": ms_per_step_wall - floor_ms,
        "gpu_fraction": floor_ms / ms_per_step_wall if ms_per_step_wall > 0 else 0.0,
        "ms_per_token_wall": ms_per_tok,
        # No independent cadence cross-check is possible from a wall-time-only
        # scan, so leave it unset rather than reporting a tautology.
        "tpot_step_ms": None,
        "step_consistency_ratio": None,
        "decode_tok_s": _p50(
            (experiment.get("summary") or {}).get("decode_tok_s")
        ),
        "fallback_count": 0,
        "device_source": "floor",
        "serial_chunks": int(summary.get("serial_chunks") or 1),
        "note": summary.get("record_note"),
    }


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
        "device_source": "measured",
        "serial_chunks": 1,
        "note": None,
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
        gpu_text = _fmt(row["ms_per_step_gpu"])
        if row.get("device_source") == "floor":
            gpu_text += " (floor)"
        lines.append(
            "| {wid} | {backend} | {batch} | {wall} | {gpu} | {host} | {share} | "
            "{mtok} | {check} | {tok} |".format(
                wid=row["workload_id"],
                backend=row["backend_actual"],
                batch=row["batch_size"],
                wall=_fmt(row["ms_per_step_wall"]),
                gpu=gpu_text,
                host=_fmt(row["host_residual_ms"]),
                share=f"{row['gpu_fraction']:.1%}",
                mtok=_fmt(row["ms_per_token_wall"]),
                check=check_text,
                tok=_fmt(row["decode_tok_s"], 1),
            )
        )
    if any(row.get("device_source") == "floor" for row in rows):
        lines += [
            "",
            "> Rows marked `(floor)` come from a wall-time-only scan (crossover "
            "scan); their device column is the roofline floor, so the host "
            "residual is a lower bound and GPU share an upper bound.",
        ]
    return "\n".join(lines) + "\n"


def render_comparison(rows: list[dict], model: RooflineModel) -> str:
    """For each batch, compare backends and quantify the recoverable host cost."""

    by_batch: dict[int, list[dict]] = {}
    for row in rows:
        by_batch.setdefault(int(row["batch_size"]), []).append(row)
    # Device anchor, best available per batch:
    #  * a measured per-step GPU time (compare.py / nsys) is the strongest;
    #  * otherwise use the batch-independent weight floor, which is what the
    #    wall-time-only crossover scan can be compared against.
    # The KV read-amplification term is deliberately excluded: it only becomes
    # significant at much longer contexts and would otherwise inflate the floor
    # and hide headroom.
    weight_floor_ms = (
        model.weight_bytes() / (model.hardware.effective_bandwidth_gbps() * 1e9) * 1000.0
    )
    lines = [
        "# Host-side recoverable headroom",
        "",
        "For each batch this reports the best measured step time, and what the "
        "step time would be if only the device part remained. 'ceiling' is the "
        "decode tok/s at that bound. The device anchor is the measured per-step "
        f"GPU time when available, otherwise the {weight_floor_ms:.3f} ms/step "
        "batch-independent weight floor.",
        "",
        "| batch | fastest backend | wall ms/step | device anchor ms | host ms/step | "
        "wall/anchor | step-time if host removed | ceiling tok/s | measured tok/s |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for batch in sorted(by_batch):
        candidates = by_batch[batch]
        best = min(candidates, key=lambda row: row["ms_per_step_wall"])
        measured_gpu = [
            row["ms_per_step_gpu"]
            for row in candidates
            if row.get("device_source") == "measured" and row["ms_per_step_gpu"] > 0
        ]
        anchor = max(weight_floor_ms, max(measured_gpu)) if measured_gpu else weight_floor_ms
        device_bound = max(anchor, best["ms_per_step_gpu"])
        lines.append(
            "| {batch} | {backend} | {wall} | {anchor} | {host} | {ratio:.2f}x | "
            "{bound} | {ceiling:.0f} | {measured} |".format(
                batch=batch,
                backend=best["backend_actual"],
                wall=_fmt(best["ms_per_step_wall"]),
                anchor=_fmt(anchor),
                host=_fmt(best["host_residual_ms"]),
                ratio=best["ms_per_step_wall"] / anchor if anchor > 0 else float("inf"),
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


def find_calibration_warnings(rows: list[dict], model: RooflineModel) -> list[str]:
    """Flag measured device time that is physically impossible.

    A decode step must stream ``weight_bytes`` from HBM, so a measured per-step
    GPU time below ``weight_bytes / bandwidth`` implies either (a) the timing
    only covered part of the step, or (b) the device bandwidth exceeds the
    configured value.  Either way the roofline parameters need calibrating
    before any "how much headroom is left" claim is trusted -- this is exactly
    the docs/07 section 10.4 contradiction.
    """

    floor_ms = (
        model.weight_bytes()
        / (model.hardware.effective_bandwidth_gbps() * 1e9)
        * 1000.0
    )
    warnings: list[str] = []
    for row in rows:
        if row.get("device_source") != "measured":
            continue
        gpu = row["ms_per_step_gpu"]
        if gpu <= 0 or gpu >= floor_ms * 0.95:
            continue
        implied = model.weight_bytes() / (gpu / 1000.0) / 1e9
        warnings.append(
            f"{row['workload_id']} {row['backend_actual']} batch={row['batch_size']}: "
            f"measured GPU {gpu:.3f} ms/step is below the {floor_ms:.3f} ms weight "
            f"floor (implies {implied:.0f} GB/s > "
            f"{model.hardware.effective_bandwidth_gbps():.0f} GB/s spec). Calibrate "
            "with RUN_MODE=roofline (bandwidth section) before trusting the "
            "headroom numbers for this row"
        )
    return warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Host/device split of nano-vLLM decode steps."
    )
    parser.add_argument(
        "json_paths",
        nargs="+",
        help="benchmarks.compare JSON outputs (or bench_burst_batch crossover scans)",
    )
    parser.add_argument("--bandwidth-gbps", type=float, default=None)
    parser.add_argument("--seq-len", type=int, default=224)
    parser.add_argument(
        "--tied-embedding",
        action="store_true",
        default=True,
        help="whether the model ties LM head to the embedding (Qwen3-0.6B: True)",
    )
    parser.add_argument(
        "--no-tied-embedding",
        dest="tied_embedding",
        action="store_false",
        help="override: model does NOT tie LM head to embedding",
    )
    parser.add_argument(
        "--floor-ms-per-step",
        type=float,
        default=None,
        help="override the roofline floor used for device-less scans",
    )
    parser.add_argument("--json", dest="json_out")
    parser.add_argument(
        "--warnings-as-errors",
        action="store_true",
        help="exit non-zero when the tpot cross-check disagrees",
    )
    args = parser.parse_args(argv)

    hw = HardwareSpec(
        bandwidth_gbps=args.bandwidth_gbps or HardwareSpec().bandwidth_gbps,
        peak_tflops=HardwareSpec().peak_tflops,
    )
    model = RooflineModel(ModelDims(), hw, tied_embedding=args.tied_embedding)

    experiments = load_experiments(args.json_paths)
    # For device-less scans use the batch-independent *weight* floor.  The KV
    # read-amplification term only matters at much longer contexts and would
    # otherwise inflate the floor (and hide headroom) at large batch.
    weight_floor_ms = (
        model.weight_bytes() / (model.hardware.effective_bandwidth_gbps() * 1e9) * 1000.0
    )
    rows = []
    for experiment in experiments:
        if "crossover_source" in (experiment.get("summary") or {}):
            floor = args.floor_ms_per_step or weight_floor_ms
            row = accounting_from_crossover(experiment, floor)
        else:
            row = accounting_from_experiment(experiment)
        if row is not None:
            rows.append(row)
    if not rows:
        print("no decodable experiments found", file=sys.stderr)
        return 2

    # Controlled batch rows hold batch constant while scaling batch size, which
    # is exactly the comparison being made; crossover scans are already keyed
    # by batch.
    batch_rows = [
        row
        for row in rows
        if row["category"] in ("controlled_batch", "crossover_scan")
    ]
    if not batch_rows:
        batch_rows = rows

    print(render_markdown(batch_rows, model))
    print(render_comparison(batch_rows, model))

    cadence_warnings = find_inconsistencies(batch_rows)
    calibration_warnings = find_calibration_warnings(batch_rows, model)
    warnings = cadence_warnings + calibration_warnings
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "hardware": asdict(hw),
                    "seq_len": args.seq_len,
                    "rows": batch_rows,
                    "comparison": render_comparison(batch_rows, model),
                    "warnings": warnings,
                    "cadence_warnings": cadence_warnings,
                    "calibration_warnings": calibration_warnings,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.json_out}")

    # Only a cadence mismatch is a hard failure: a calibration warning is a
    # finding to act on (run the bandwidth probe), not a tool error.
    if cadence_warnings and args.warnings_as_errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
