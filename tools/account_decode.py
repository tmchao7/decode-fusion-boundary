"""Split active nano-vLLM decode steps into GPU time and host residual.

Every row also reports its **distance to the weight-streaming roofline**, because
a latency number without a roofline denominator cannot say whether anything was
left on the table (the project's honesty rule). That distance is only meaningful
against a bandwidth measured *on this machine*: the denominator comes from
``RUN_MODE=bandwidth`` (``results/bandwidth.json``), never from a spec value and
never inherited from another device. Pass ``--bandwidth-gbps`` only when you have
a measured rate from elsewhere and want to make that explicit in the report.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nanovllm.roofline import (  # noqa: E402
    DEFAULT_BANDWIDTH_JSON,
    ModelDims,
    RooflineModel,
    hardware_from_bandwidth_json,
    measured_hardware_spec,
)


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


def roofline_columns(
    model: RooflineModel, gpu_ms: float, batch_size: int, seq_len: int = 1
) -> dict[str, Any]:
    """Distance from one measured step to the weight-streaming floor.

    ``implied_bandwidth_gbps`` is the rate the step would need to hit to move the
    weight bytes in exactly ``gpu_ms``; ``gap_to_core_floor_x`` > 1 means the step
    is slower than streaming the weights once. When the measurement is *below* the
    modelled floor the model is wrong on this machine -- that is
    ``calibration_needed``, a calibration finding, never a win for the kernel.
    """

    peak_bw = model.hardware.effective_bandwidth_gbps() * 1e9
    core_bytes = model.core_weight_bytes()
    core_floor_ms = core_bytes / peak_bw * 1000.0
    # The floor is the max of memory and compute so a batch where tensor-core
    # compute dominates cannot be reported as "above the bandwidth wall".
    memory_ms = model.memory_bound_ms(batch_size, seq_len=seq_len)
    lower_ms = model.lower_bound_ms(batch_size, seq_len=seq_len)
    row: dict[str, Any] = {
        "weight_bytes": model.weight_bytes(),
        "core_weight_bytes": core_bytes,
        "core_weight_floor_ms": core_floor_ms,
        "memory_bound_ms": memory_ms,
        "lower_bound_ms": lower_ms,
        "calibration_needed": False,
        "headroom_ms": 0.0,
        "headroom_fraction": 0.0,
        "implied_bandwidth_gbps": None,
        "gap_to_core_floor_x": None,
    }
    if gpu_ms <= 0:
        return row
    row["implied_bandwidth_gbps"] = core_bytes / (gpu_ms / 1000.0) / 1e9
    if gpu_ms < lower_ms:
        row["calibration_needed"] = True
    else:
        row["headroom_ms"] = gpu_ms - lower_ms
        row["headroom_fraction"] = (gpu_ms - lower_ms) / gpu_ms
    if core_floor_ms > 0:
        row["gap_to_core_floor_x"] = gpu_ms / core_floor_ms
    return row


def accounting_from_experiment(
    experiment: dict, model: RooflineModel | None = None, seq_len: int = 1
) -> dict | None:
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
    batch_size = int(workload.get("batch_size") or request_count)
    tpot = next((_p50(r.get("tpot_ms")) for r in requests if r.get("tpot_ms") is not None), None)
    row = {
        "workload_id": workload.get("id"),
        "category": workload.get("category"),
        "backend": experiment.get("backend"),
        "backend_actual": ",".join(sorted({str(r.get("backend_actual")) for r in requests if r.get("backend_actual")})),
        "batch_size": batch_size,
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
    if model is not None:
        row.update(roofline_columns(model, gpu_ms, batch_size, seq_len))
    return row


def render_markdown(rows: list[dict], model: RooflineModel | None = None) -> str:
    header = [
        "# Decode accounting",
        "",
    ]
    if model is not None:
        header += [
            f"Roofline denominator: **{model.hardware.effective_bandwidth_gbps():.1f} GB/s** "
            f"({model.hardware.name}); weight bytes/step "
            f"= {model.weight_bytes() / 1e9:.3f} GB, core floor "
            f"= {model.core_weight_bytes() / model.hardware.effective_bandwidth_gbps() / 1e6:.3f} ms/step.",
            "",
        ]
    lines = header + [
        "| workload | backend | batch | wall ms/step | GPU ms/step | host residual ms | GPU share "
        "| implied GB/s | gap to weights roofline |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        implied = row.get("implied_bandwidth_gbps")
        gap = row.get("gap_to_core_floor_x")
        gap_text = "--" if gap is None else f"{gap:.2f}x"
        implied_text = "--" if implied is None else f"{implied:.0f}"
        if row.get("calibration_needed"):
            gap_text += " (calibration_needed)"
        lines.append(
            f"| {row['workload_id']} | {row['backend']} | {row['batch_size']} | "
            f"{row['ms_per_step_wall']:.3f} | {row['ms_per_step_gpu']:.3f} | "
            f"{row['host_residual_ms']:.3f} | {row['gpu_fraction']:.1%} | "
            f"{implied_text} | {gap_text} |"
        )
    if any(row.get("calibration_needed") for row in rows):
        lines += [
            "",
            "> WARNING: at least one measured GPU step is below the modelled roofline "
            "floor. The bandwidth denominator or the KV read-amplification assumption "
            "is wrong for this machine -- this is a calibration finding, not kernel "
            "headroom.",
        ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_paths", nargs="+")
    parser.add_argument("--json", dest="json_out")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1,
        help=(
            "context length used by the KV term of the memory bound. Decode step 1 "
            "of a short prompt is ~1; the archived reports used 2048, which charges "
            "a long KV read the benchmark never performed."
        ),
    )
    parser.add_argument(
        "--bandwidth-json",
        default=None,
        help=(
            "measured-bandwidth artifact (default results/bandwidth.json). This is "
            "the roofline denominator and must come from this machine."
        ),
    )
    parser.add_argument(
        "--bandwidth-gbps",
        type=float,
        default=None,
        help="explicit measured GB/s override; mutually exclusive with --bandwidth-json",
    )
    parser.add_argument("--no-roofline", action="store_true", help=argparse.SUPPRESS)
    return parser


def resolve_model(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> tuple[RooflineModel | None, str | None]:
    """Build the roofline model plus its provenance, or ``(None, None)``.

    The provenance string is what the JSON artifact records, so a report can
    always state which measurement produced its denominator.
    """

    if args.no_roofline:
        return None, None
    dims = ModelDims()
    if args.bandwidth_gbps is not None:
        if args.bandwidth_json is not None:
            parser.error("--bandwidth-gbps and --bandwidth-json are mutually exclusive")
        hardware = measured_hardware_spec(
            float(args.bandwidth_gbps), name="explicit --bandwidth-gbps"
        )
        bandwidth_source = "explicit --bandwidth-gbps"
    else:
        bandwidth_path = Path(args.bandwidth_json or DEFAULT_BANDWIDTH_JSON)
        hardware = hardware_from_bandwidth_json(bandwidth_path)
        bandwidth_source = str(bandwidth_path)
    return RooflineModel(dims, hardware, tied_embedding=True), bandwidth_source


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    model, bandwidth_source = resolve_model(args, parser)
    rows = [
        row
        for item in load_experiments(args.json_paths)
        if (row := accounting_from_experiment(item, model, args.seq_len))
    ]
    if not rows:
        print("no decodable experiments found", file=sys.stderr)
        return 2
    markdown = render_markdown(rows, model)
    print(markdown, end="")
    if args.json_out:
        payload: dict[str, Any] = {"rows": rows}
        if model is not None:
            payload["roofline"] = {
                "bandwidth_gbps": model.hardware.effective_bandwidth_gbps(),
                "device": model.hardware.name,
                "source": bandwidth_source,
                "weight_bytes": model.weight_bytes(),
                "core_weight_floor_ms": model.core_weight_bytes()
                / model.hardware.effective_bandwidth_gbps()
                / 1e6,
            }
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
