"""Fusion-granularity sweep for the task-graph decoder.

The step's instruction stream can be built as ``g`` contiguous layer segments
and launched ``g`` times per decode step (``fusion_groups`` in ``Config``).
``g == 1`` is the single-megakernel baseline; ``g == 28`` (one launch per layer)
is the other extreme. This tool measures the same (batch, context) grid as
``benchmarks.compare`` for several ``g`` values on the **megakernel arm only**,
so the granularity axis can be read off one table.

Every row goes through ``benchmarks.compare.run_case``, so it inherits that
tool's workload construction, warmup/runs protocol, correctness gate and metrics
report — the only changed variable is the segmentation. That is the point of the
experiment: granularity must not be confounded with implementation language.

The segmentation itself is verified on the CPU before any GPU time is spent:
``tests/test_taskgraph_abi.py::FusionGranularityScheduleTests`` proves the
segments partition the layers, ship every instruction exactly once, and stay
deadlock-free for every split and grid size.

Usage::

    python -m tools.bench_fusion_granularity --model "$MODEL" \
        --fusion-groups 1,2,4,7,14,28 \
        --batch-sizes 1,2,4,8 \
        --prompt-lengths 32,512,1920 \
        --max-tokens 128 --warmup 2 --runs 3 \
        --json results/fusion_granularity.json

Only stdlib and torch-free nanovllm modules are imported at module level, so the
plan/validation logic below is unit-testable on a CPU-only checkout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from nanovllm.backends.megakernel.schedule import split_layers  # noqa: E402
from nanovllm.backends.megakernel.taskgraph import NUM_LAYERS  # noqa: E402

DEFAULT_GROUPS = "1,2,4,7,14,28"
DEFAULT_BATCH_SIZES = "1,2,4,8"
DEFAULT_PROMPT_LENGTHS = "32,512,1920"
MAX_MODEL_LEN = 2048
BACKEND = "megakernel"


def parse_int_list(value: str, *, name: str) -> list[int]:
    """Parse ``"1,2,4"`` into positive ints, rejecting empties and duplicates."""

    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{name} must be a non-empty comma-separated list")
    try:
        values = [int(item) for item in items]
    except ValueError as exc:
        raise ValueError(f"{name} must contain integers; got {value!r}") from exc
    for parsed in values:
        if parsed < 1:
            raise ValueError(f"{name} values must be >= 1; got {parsed}")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not repeat values; got {values}")
    return values


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


#: Archived, documented device time per decode step at batch=1 (docs/06/07).
BASE_DEVICE_MS = 1.89


def fused_prediction(group_counts, batch_sizes) -> dict[str, object]:
    """The pre-registered granularity prediction, computed without a GPU.

    Kept torch-free so ``--help``/validation paths and the CPU test suite can call
    it; see ``docs/18`` for the model and what would refute it.
    """

    from nanovllm.cost_model import preregistered_granularity_prediction

    return {
        **preregistered_granularity_prediction(
            base_device_ms=BASE_DEVICE_MS, groups=group_counts
        ),
        "batch_sizes": list(batch_sizes),
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    group_counts = parse_int_list(args.fusion_groups, name="fusion-groups")
    batch_sizes = parse_int_list(args.batch_sizes, name="batch-sizes")
    prompt_lengths = parse_int_list(args.prompt_lengths, name="prompt-lengths")
    validate_groups(group_counts)
    validate_lengths(prompt_lengths, args.max_tokens)
    plan = sweep_plan(group_counts, batch_sizes, prompt_lengths)
    prediction = fused_prediction(group_counts, batch_sizes)

    # Printed before any measurement exists: the whole point of pre-registering is
    # that the expectation is on the record first (docs/18).
    print("pre-registered prediction (docs/18):")
    print(
        f"  device ms/step at g=1: {prediction['base_device_ms']} "
        f"(documented archived value)"
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

    # Imported lazily: benchmarks.compare pulls in torch/transformers, which the
    # plan/validation logic above must not require.
    from benchmarks.compare import run_case, synthetic_factory
    from nanovllm.metrics import dump_json

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(plan, start=1):
        print(
            f"[{index}/{len(plan)}] fusion_groups={row['fusion_groups']} "
            f"batch={row['batch_size']} prompt={row['prompt_length']}",
            file=sys.stderr,
        )
        experiment = run_case(
            args.model,
            BACKEND,
            f"fusion_g{row['fusion_groups']}_p{row['prompt_length']}"
            f"_b{row['batch_size']}",
            "controlled_batch",
            synthetic_factory([row["prompt_length"]], args.seed),
            [row["prompt_length"]],
            row["batch_size"],
            args.max_tokens,
            args.warmup,
            args.runs,
            args.seed,
            ignore_eos=True,
            fusion_groups=row["fusion_groups"],
        )
        rows.append({**row, "experiment": experiment})

    payload = {
        "schema": "fusion_granularity_v1",
        "backend": BACKEND,
        "model": args.model,
        "num_layers": NUM_LAYERS,
        "max_tokens": args.max_tokens,
        "warmup": args.warmup,
        "runs": args.runs,
        "plan": plan,
        "pre_registered_prediction": prediction,
        "rows": rows,
    }
    dump_json(args.json, payload)
    print(f"wrote {args.json} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
