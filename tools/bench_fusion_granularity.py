"""Fusion-granularity sweep for the task-graph decoder.

The step's instruction stream can be built as ``g`` contiguous layer segments
and launched ``g`` times per decode step (``fusion_groups`` in ``Config``).
``g == 1`` is the single-megakernel baseline; ``g == 28`` (one launch per layer)
is the other extreme. This tool measures the same (batch, context) grid as
``benchmarks.compare`` for several ``g`` values, so the granularity axis can be
read off one table.

Every row goes through ``benchmarks.compare.run_case``, and every row carries a
**real correctness gate**: the tool runs reference arms (``--reference-backends``,
default ``torch-eager``) through the same ``run_case``/``synthetic_factory``
harness and hands the whole group to ``benchmarks.compare.mark_correctness``.
That matters because ``mark_correctness`` silently skips a group with no
``torch-eager`` member -- a megakernel-only table has no gate at all, which would
violate ``docs/14`` §6 acceptance 1. Reference arms always run at
``fusion_groups=1`` and are cached per (backend, batch, prompt), so they cost one
extra pass per grid point rather than one per row.

Each row also reports its **distance to the weight-streaming roofline**, because a
device time without a bandwidth denominator cannot say whether anything was left
on the table. That denominator is a *measurement* on this machine
(``--bandwidth-json``, written by ``RUN_MODE=bandwidth``); a spec number never
enters the arithmetic.

The segmentation itself is verified on the CPU before any GPU time is spent:
``tests/test_taskgraph_abi.py::FusionGranularityScheduleTests`` proves the
segments partition the layers, ship every instruction exactly once, and stay
deadlock-free for every split and grid size.

Usage::

    python -m tools.bench_fusion_granularity --model "$MODEL" \
        --fusion-groups 1,2,4,7,14,28 \
        --batch-sizes 1,2,4,8 \
        --prompt-lengths 32,512,1920 \
        --max-tokens 128 --warmup 2 --runs 5 \
        --reference-backends torch-eager,torch-cudagraph \
        --bandwidth-json results/bandwidth.json \
        --json results/fusion_granularity.json

Only stdlib and torch-free nanovllm modules are imported at module level, so the
plan/validation logic below is unit-testable on a CPU-only checkout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from nanovllm.backends.megakernel.schedule import split_layers  # noqa: E402
from nanovllm.backends.megakernel.taskgraph import NUM_LAYERS  # noqa: E402
from nanovllm.roofline import (  # noqa: E402
    DEFAULT_BANDWIDTH_JSON,
    ModelDims,
    RooflineModel,
    hardware_from_bandwidth_json,
    measured_hardware_spec,
)
from tools.cli import parse_int_list  # noqa: E402

DEFAULT_GROUPS = "1,2,4,7,14,28"
DEFAULT_BATCH_SIZES = "1,2,4,8"
DEFAULT_PROMPT_LENGTHS = "32,512,1920"
DEFAULT_REFERENCE_BACKENDS = "torch-eager"
MAX_MODEL_LEN = 2048
BACKEND = "megakernel"

#: Archived, documented device time per decode step at batch=1.
#: This is the **archive `kernel.cu` line on A100** (``docs/14`` §1), not a
#: measurement of the current task-graph implementation and not a 5090 number.
#: It is only the scale factor of the pre-registered granularity curve, whose
#: output is a *relative* delta; the CLI records which base was used.
BASE_DEVICE_MS = 1.89
BASE_DEVICE_MS_SOURCE = (
    "archived kernel.cu line, A100 batch=1 (docs/14 §1); NOT this machine's "
    "taskgraph baseline and NOT a 5090 number"
)


def validate_groups(group_counts: list[int], *, num_layers: int = NUM_LAYERS) -> None:
    """Reject granularities the schedule builder cannot represent.

    Delegates to ``split_layers`` so the sweep tool and the decoder agree on the
    valid range by construction rather than by a duplicated constant.
    """

    for groups in group_counts:
        split_layers(num_layers, groups)


def validate_lengths(
    prompt_lengths: list[int], max_tokens: int, *, max_model_len: int = MAX_MODEL_LEN
) -> None:
    """A row must fit the RoPE table and the KV pool (same rule as compare.py)."""

    for prompt_length in prompt_lengths:
        total = prompt_length + max_tokens
        if total > max_model_len:
            raise ValueError(
                f"prompt_length={prompt_length} + max_tokens={max_tokens} = "
                f"{total} exceeds max_model_len {max_model_len}"
            )


def parse_reference_backends(value: str) -> list[str]:
    backends = [part.strip() for part in value.split(",") if part.strip()]
    if not backends:
        raise ValueError("--reference-backends must name at least one backend")
    return backends


def validate_reference_backends(
    backends: list[str], *, known: list[str] | None = None
) -> None:
    """Reference arms must be independent: never the swept megakernel arm."""

    if known is None:
        from benchmarks.compare import BACKEND_CONFIGS

        known = sorted(BACKEND_CONFIGS)
    for backend in backends:
        if backend == BACKEND:
            raise ValueError(
                f"{BACKEND!r} is the swept arm and cannot also be a reference; "
                "the gate needs an independent implementation"
            )
        if backend not in known:
            raise ValueError(
                f"unknown reference backend {backend!r}; expected one of {known}"
            )


def fused_prediction(
    group_counts: list[int],
    batch_sizes: list[int],
    *,
    base_device_ms: float = BASE_DEVICE_MS,
    base_device_ms_source: str = BASE_DEVICE_MS_SOURCE,
) -> dict[str, object]:
    """The pre-registered granularity prediction, computed without a GPU.

    Kept torch-free so ``--help``/validation paths and the CPU test suite can call
    it; see ``docs/18`` for the model and what would refute it.
    """

    from nanovllm.cost_model import preregistered_granularity_prediction

    return {
        **preregistered_granularity_prediction(
            base_device_ms=base_device_ms, groups=group_counts
        ),
        "batch_sizes": list(batch_sizes),
        "base_device_ms_source": base_device_ms_source,
    }


def sweep_plan(
    group_counts: list[int],
    batch_sizes: list[int],
    prompt_lengths: list[int],
) -> list[dict[str, int]]:
    """Deterministic sweep plan, granularity outermost so one g reads together."""

    return [
        {
            "fusion_groups": groups,
            "batch_size": batch_size,
            "prompt_length": prompt_length,
        }
        for groups in group_counts
        for batch_size in batch_sizes
        for prompt_length in prompt_lengths
    ]


def reference_key(
    backend: str, batch_size: int, prompt_length: int, max_tokens: int, seed: int
) -> tuple[str, int, int, int, int]:
    """Cache key for a reference arm: one run per grid point, reused by every g."""

    return (backend, int(batch_size), int(prompt_length), int(max_tokens), int(seed))


def plan_reference_work(
    batch_sizes: list[int], prompt_lengths: list[int], backends: list[str]
) -> list[tuple[str, int, int]]:
    """The grid points that will need a reference arm, in a stable order."""

    return [
        (backend, batch_size, prompt_length)
        for backend in backends
        for batch_size in batch_sizes
        for prompt_length in prompt_lengths
    ]


def roofline_device_ms(experiment: dict) -> float | None:
    """Pure-GPU decode step from a compare.py experiment, or ``None``."""

    summary = experiment.get("summary") or {}
    decode_steps = float(summary.get("decode_steps") or 0)
    if decode_steps <= 0:
        return None
    total = float(summary.get("cuda_decode_time_ms_total") or 0.0)
    return total / decode_steps if total > 0 else None


def roofline_columns(
    model: RooflineModel, experiment: dict, batch_size: int
) -> dict[str, Any]:
    """Device time, implied bandwidth and gap to the weight-streaming floor."""

    wall_ms = None
    summary = experiment.get("summary") or {}
    decode_steps = float(summary.get("decode_steps") or 0)
    device_ms = roofline_device_ms(experiment)
    if decode_steps > 0:
        wall_ms = float(summary.get("decode_time_ms_total") or 0.0) / decode_steps
    peak_bw = model.hardware.effective_bandwidth_gbps() * 1e9
    core_bytes = model.core_weight_bytes()
    core_floor_ms = core_bytes / peak_bw * 1000.0
    lower_ms = model.lower_bound_ms(batch_size, seq_len=1)
    row: dict[str, Any] = {
        "device_ms_per_step": device_ms,
        "wall_ms_per_step": wall_ms,
        "core_weight_floor_ms": core_floor_ms,
        "implied_bandwidth_gbps": None,
        "gap_to_core_floor_x": None,
        "calibration_needed": False,
    }
    if wall_ms is not None and device_ms is not None:
        row["host_residual_ms"] = wall_ms - device_ms
        row["gpu_fraction"] = device_ms / wall_ms if wall_ms else 0.0
    if device_ms:
        row["implied_bandwidth_gbps"] = core_bytes / (device_ms / 1000.0) / 1e9
        row["gap_to_core_floor_x"] = device_ms / core_floor_ms if core_floor_ms else None
        # Below the modelled floor is a calibration finding, never kernel
        # headroom: the denominator or a traffic assumption is wrong here.
        row["calibration_needed"] = bool(device_ms < lower_ms)
    return row


def resolve_roofline(
    bandwidth_json: str | None, bandwidth_gbps: float | None
) -> RooflineModel:
    """Build the roofline model from a measurement, never from a spec preset."""

    dims = ModelDims()
    if bandwidth_gbps is not None:
        if bandwidth_json is not None:
            raise ValueError(
                "--bandwidth-gbps and --bandwidth-json are mutually exclusive"
            )
        hardware = measured_hardware_spec(
            float(bandwidth_gbps), name="explicit --bandwidth-gbps"
        )
    else:
        hardware = hardware_from_bandwidth_json(bandwidth_json or DEFAULT_BANDWIDTH_JSON)
    return RooflineModel(dims, hardware, tied_embedding=True)


def failed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows whose correctness gate did not pass, including a missing gate."""

    return [
        row
        for row in rows
        if not (row.get("experiment", {}).get("correctness") or {}).get("passed", False)
    ]


def cross_check_against_baseline(
    rows: list[dict[str, Any]],
    baseline_experiments: list[dict],
    *,
    tolerance: float = 0.05,
) -> list[dict[str, Any]]:
    """Compare every ``g=1`` row with the main table's megakernel row.

    ``docs/14`` §6 acceptance 2 requires the sweep's ``g=1`` to agree with
    ``RUN_MODE=taskgraph-baseline``; without this check the two artifacts can drift
    apart with nothing noticing.
    """

    def row_key(experiment: dict) -> tuple[str, int, int]:
        workload = experiment.get("workload") or {}
        prompt_lengths = workload.get("prompt_lengths") or [0]
        return (
            str(workload.get("id")),
            int(workload.get("batch_size") or 0),
            int(prompt_lengths[0]),
        )

    # compare.py builds the workload id from the (prompt, batch) pair, so the id
    # alone is enough; the tuple keys keep a mismatched baseline visible instead
    # of silently matching the wrong row.
    baseline: dict[tuple[str, int, int], float] = {}
    for experiment in baseline_experiments:
        if experiment.get("backend") != BACKEND:
            continue
        device = roofline_device_ms(experiment)
        if device is None:
            continue
        baseline[row_key(experiment)] = device

    findings: list[dict[str, Any]] = []
    for row in rows:
        if row.get("fusion_groups") != 1:
            continue
        experiment = row.get("experiment") or {}
        measured = roofline_device_ms(experiment)
        if measured is None:
            continue
        key = row_key(experiment)
        reference = baseline.get(key)
        finding: dict[str, Any] = {
            "workload_id": key[0],
            "batch_size": key[1],
            "prompt_length": key[2],
            "measured": measured,
            "baseline": reference,
            "relative_delta": None,
            "within_tolerance": False,
            "reason": None,
        }
        if reference is None or reference <= 0:
            finding["reason"] = "no matching megakernel row in the baseline artifact"
            findings.append(finding)
            continue
        delta = (measured - reference) / reference
        finding["relative_delta"] = delta
        finding["within_tolerance"] = abs(delta) <= tolerance
        findings.append(finding)
    return findings


def run_sweep(
    *,
    plan: list[dict[str, int]],
    model_path: str,
    max_tokens: int,
    warmup: int,
    runs: int,
    seed: int,
    reference_backends: list[str],
    run_case_fn: Callable[..., dict],
    synthetic_factory_fn: Callable[..., Any],
    mark_correctness_fn: Callable[[list[dict]], None],
    progress: Callable[[str], None] = lambda _message: None,
) -> tuple[list[dict[str, Any]], dict[tuple, dict]]:
    """Execute the sweep, gating every row with the reference arms.

    The three ``*_fn`` arguments default to the real ``benchmarks.compare`` entry
    points in ``main`` but are injectable here so the correctness gate and the
    reference caching are testable without a GPU.
    """

    reference_cache: dict[tuple, dict] = {}

    def reference_arms(batch_size: int, prompt_length: int) -> list[dict]:
        arms: list[dict] = []
        for backend in reference_backends:
            key = reference_key(backend, batch_size, prompt_length, max_tokens, seed)
            cached = reference_cache.get(key)
            if cached is None:
                progress(
                    f"reference {backend} batch={batch_size} prompt={prompt_length}"
                )
                cached = run_case_fn(
                    model_path,
                    backend,
                    f"fusion_ref_{backend}_p{prompt_length}_b{batch_size}",
                    "controlled_batch",
                    synthetic_factory_fn([prompt_length], seed),
                    [prompt_length],
                    batch_size,
                    max_tokens,
                    warmup,
                    runs,
                    seed,
                    ignore_eos=True,
                )
                reference_cache[key] = cached
            arms.append(cached)
        return arms

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(plan, start=1):
        progress(
            f"[{index}/{len(plan)}] fusion_groups={row['fusion_groups']} "
            f"batch={row['batch_size']} prompt={row['prompt_length']}"
        )
        groups = row["fusion_groups"]
        experiment = run_case_fn(
            model_path,
            BACKEND,
            f"fusion_g{groups}_p{row['prompt_length']}_b{row['batch_size']}",
            "controlled_batch",
            synthetic_factory_fn([row["prompt_length"]], seed),
            [row["prompt_length"]],
            row["batch_size"],
            max_tokens,
            warmup,
            runs,
            seed,
            ignore_eos=True,
            fusion_groups=groups,
        )
        # mark_correctness mutates each experiment's `correctness` key and needs
        # the reference in the very same call, so group the pair first.
        group = reference_arms(row["batch_size"], row["prompt_length"]) + [experiment]
        mark_correctness_fn(group)
        rows.append({**row, "experiment": experiment})
    return rows, reference_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fusion-groups", default=DEFAULT_GROUPS)
    parser.add_argument("--batch-sizes", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--prompt-lengths", default=DEFAULT_PROMPT_LENGTHS)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default="results/fusion_granularity.json")
    parser.add_argument(
        "--reference-backends",
        default=DEFAULT_REFERENCE_BACKENDS,
        help=(
            "arms used as the correctness reference, run through the same harness "
            "at fusion_groups=1 (default torch-eager). The taskgraph-fusion "
            "SERVER_COMMANDS mode adds torch-cudagraph so one artifact also carries "
            "the CUDA Graph device-time opponent."
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
    parser.add_argument(
        "--base-device-ms",
        type=float,
        default=BASE_DEVICE_MS,
        help=(
            "device ms/step the pre-registered granularity curve is centred on. Only "
            "the relative g=28 delta is model output, so a different base rescales the "
            "absolute column; the source is recorded in the artifact."
        ),
    )
    parser.add_argument("--base-device-ms-source", default=BASE_DEVICE_MS_SOURCE)
    parser.add_argument(
        "--baseline-json",
        default=None,
        help=(
            "a taskgraph-baseline artifact; compares this sweep's g=1 device time "
            "against the main table (docs/14 §6 acceptance 2)"
        ),
    )
    parser.add_argument(
        "--cross-check-tolerance",
        type=float,
        default=0.05,
        help="relative tolerance for the g=1 cross-check (default 5%%)",
    )
    parser.add_argument(
        "--no-fail-on-correctness",
        action="store_true",
        help=(
            "record instead of failing when a row's correctness gate does not pass. "
            "Off by default: an ungated latency number must not be written as a result."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    group_counts = parse_int_list(args.fusion_groups, name="fusion-groups")
    batch_sizes = parse_int_list(args.batch_sizes, name="batch-sizes")
    prompt_lengths = parse_int_list(args.prompt_lengths, name="prompt-lengths")
    try:
        reference_backends = parse_reference_backends(args.reference_backends)
        validate_groups(group_counts)
        validate_lengths(prompt_lengths, args.max_tokens)
        validate_reference_backends(reference_backends)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        model = resolve_roofline(args.bandwidth_json, args.bandwidth_gbps)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    plan = sweep_plan(group_counts, batch_sizes, prompt_lengths)
    prediction = fused_prediction(
        group_counts,
        batch_sizes,
        base_device_ms=args.base_device_ms,
        base_device_ms_source=args.base_device_ms_source,
    )
    reference_work = plan_reference_work(batch_sizes, prompt_lengths, reference_backends)

    # Printed before any measurement exists: the whole point of pre-registering is
    # that the expectation is on the record first (docs/18).
    print("pre-registered prediction (docs/18):")
    print(
        f"  device ms/step at g=1: {prediction['base_device_ms']} "
        f"-- base source: {prediction['base_device_ms_source']}"
    )
    for fraction in ("device_ms_lower_bound", "device_ms_upper_bound"):
        curve = prediction[fraction]
        label = "lower" if "lower" in fraction else "upper"
        print(f"  {label} bound: " + ", ".join(f"g={g}:{ms:.3f}" for g, ms in curve.items()))
    print(
        f"  refuted if the measured g={max(group_counts)} device delta exceeds "
        f"{prediction['refuted_if_measured_delta_exceeds']:.1%}"
    )
    print(
        "  structure is invariant in g (verifier), so a bigger effect means "
        "launch/drain cost or lost L2 reuse, not granularity"
    )
    print(
        f"roofline denominator: {model.hardware.effective_bandwidth_gbps():.1f} GB/s "
        f"({model.hardware.name}); core weight floor "
        f"{model.core_weight_bytes() / model.hardware.effective_bandwidth_gbps() / 1e6:.4f} ms/step"
    )
    print(
        f"correctness gate: {len(reference_work)} reference run(s) "
        f"({', '.join(reference_backends)}) against {len(plan)} swept row(s)"
    )

    # Imported lazily: benchmarks.compare pulls in torch/transformers, which the
    # plan/validation logic above must not require.
    from benchmarks.compare import mark_correctness, run_case, synthetic_factory
    from nanovllm.metrics import dump_json

    rows, reference_cache = run_sweep(
        plan=plan,
        model_path=args.model,
        max_tokens=args.max_tokens,
        warmup=args.warmup,
        runs=args.runs,
        seed=args.seed,
        reference_backends=reference_backends,
        run_case_fn=run_case,
        synthetic_factory_fn=synthetic_factory,
        mark_correctness_fn=mark_correctness,
        progress=lambda message: print(message, file=sys.stderr),
    )
    for row in rows:
        row.update(roofline_columns(model, row["experiment"], row["batch_size"]))

    baseline_experiments: list[dict] = []
    if args.baseline_json:
        from tools.account_decode import load_experiments

        baseline_experiments = load_experiments([args.baseline_json])
    findings = (
        cross_check_against_baseline(
            rows, baseline_experiments, tolerance=args.cross_check_tolerance
        )
        if baseline_experiments
        else []
    )
    failures = failed_rows(rows)

    payload = {
        "schema": "fusion_granularity_v2",
        "backend": BACKEND,
        "model": args.model,
        "num_layers": NUM_LAYERS,
        "max_tokens": args.max_tokens,
        "warmup": args.warmup,
        "runs": args.runs,
        "seed": args.seed,
        "reference_backends": reference_backends,
        "reference_runs": len(reference_cache),
        "plan": plan,
        "pre_registered_prediction": prediction,
        "roofline": {
            "bandwidth_gbps": model.hardware.effective_bandwidth_gbps(),
            "device": model.hardware.name,
            "source": (
                "explicit --bandwidth-gbps"
                if args.bandwidth_gbps is not None
                else str(args.bandwidth_json or DEFAULT_BANDWIDTH_JSON)
            ),
            "weight_bytes": model.weight_bytes(),
            "core_weight_bytes": model.core_weight_bytes(),
            "core_weight_floor_ms": model.core_weight_bytes()
            / model.hardware.effective_bandwidth_gbps()
            / 1e6,
        },
        "correctness": {
            "gate": "benchmarks.compare.mark_correctness",
            "rows_checked": len(rows),
            "rows_failed": len(failures),
            "failed_rows": [
                {
                    "fusion_groups": row["fusion_groups"],
                    "batch_size": row["batch_size"],
                    "prompt_length": row["prompt_length"],
                    "correctness": row["experiment"].get("correctness"),
                }
                for row in failures
            ],
        },
        "cross_check": {
            "baseline_json": args.baseline_json,
            "tolerance": args.cross_check_tolerance,
            "findings": findings,
        },
        "rows": rows,
    }
    dump_json(args.json, payload)
    print(f"wrote {args.json} ({len(rows)} rows)")

    if findings:
        print("\ng=1 vs baseline (docs/14 §6 acceptance 2):", file=sys.stderr)
        for finding in findings:
            if finding["baseline"] is None:
                print(f"  {finding['workload_id']}: {finding['reason']}", file=sys.stderr)
            else:
                flag = "" if finding["within_tolerance"] else "  <-- OUT OF TOLERANCE"
                print(
                    f"  {finding['workload_id']}: sweep "
                    f"{finding['measured']:.3f} ms vs baseline "
                    f"{finding['baseline']:.3f} ms "
                    f"({finding['relative_delta']:+.1%}){flag}",
                    file=sys.stderr,
                )
    if failures:
        print(
            f"correctness gate failed for {len(failures)}/{len(rows)} rows",
            file=sys.stderr,
        )
        if not args.no_fail_on_correctness:
            return 1
    if any(
        finding["baseline"] is None or not finding["within_tolerance"]
        for finding in findings
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
