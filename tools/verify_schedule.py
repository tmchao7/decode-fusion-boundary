"""Verify task-graph decode schedules offline, without a GPU.

Runs the static verifier over a geometry matrix and reports, per configuration,
whether the instruction stream can deadlock and how many adversarially mutated
schedules it rejects. This is the pre-server gate for anything that changes
``schedule.py``: a schedule bug found here costs nothing, and the same bug found
on the A100 costs a session and can hang the device.

The tool is CPU-only and torch-free, so it runs on the development machine.
Adversarial mutants are sampled with a fixed stride (see
``nanovllm.backends.megakernel.verifier.mutants``): the population is tens of
thousands per configuration, and a coverage number that changed between runs
would not be evidence. The full population size is reported next to the sampled
count.

Usage::

    python -m tools.verify_schedule --json results/schedule_verification.json
    python -m tools.verify_schedule --batch-sizes 1,8 --num-sms 108,170 \
        --attn-splits 1,2 --sample 300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from nanovllm.backends.megakernel import schedule as tg  # noqa: E402
from nanovllm.backends.megakernel import verifier as V  # noqa: E402
from tools.cli import parse_int_list  # noqa: E402

DEFAULT_BATCH_SIZES = "1,2,4,8"
DEFAULT_NUM_SMS = "1,4,108,170"
DEFAULT_FUSION_GROUPS = "1,2,4,7,14,28"
DEFAULT_ATTN_SPLITS = "1,2"
DEFAULT_SAMPLE = 200


def build_segments(geo: tg.TgGeometry, num_sms: int, groups: int) -> list[Any]:
    """One schedule per layer segment, in launch order."""

    if groups == 1:
        return [tg.build_schedule(geo, num_sms)]
    ranges = tg.split_layers(geo.num_layers, groups)
    return [tg.build_schedule(geo, num_sms, layer_range=lr) for lr in ranges]


def verify_matrix(
    *,
    batch_sizes: list[int],
    num_sms_values: list[int],
    fusion_groups: list[int],
    attn_splits: list[int],
    sample: int,
) -> dict[str, Any]:
    """Verify every configuration; mutate the single-launch stream of each."""

    configs: list[dict[str, Any]] = []
    totals = {
        "configs": 0,
        "passed": 0,
        "failed": 0,
        "valid_accepted": 0,
        "valid_total": 0,
        "mutants_sampled": 0,
        "mutants_rejected": 0,
        "mutant_population": 0,
        "false_accepts": 0,
        "rejected_by_check": {},
    }

    for batch_size in batch_sizes:
        for attn_split in attn_splits:
            geo = tg.default_geometry(batch_size, attn_splits=attn_split)
            for num_sms in num_sms_values:
                # One reference stream per (batch, attn_splits, num_sms): the
                # segmentation below slices this same wave structure, so mutants
                # of it cover the segmented variants too.
                reference = tg.build_schedule(geo, num_sms)
                adversarial = V.adversarial_report(
                    [(f"BS{batch_size}/sms{num_sms}/attn{attn_split}", reference, num_sms)],
                    sample=sample,
                )

                for groups in fusion_groups:
                    segments = build_segments(geo, num_sms, groups)
                    report = V.verify_segmented(segments, num_sms)
                    configs.append(
                        {
                            "batch_size": batch_size,
                            "attn_splits": attn_split,
                            "num_sms": num_sms,
                            "fusion_groups": groups,
                            "segments": len(segments),
                            "instructions": report.instructions,
                            "ok": report.ok,
                            "failures": report.failures,
                            "stalled_sms": list(report.stalled_sms),
                            "cycle": report.cycle,
                            "arrivals": report.arrival_audit.describe(),
                        }
                    )
                    totals["configs"] += 1
                    totals["passed" if report.ok else "failed"] += 1

                totals["valid_accepted"] += adversarial.accepted_valid
                totals["valid_total"] += adversarial.valid_total
                totals["mutants_sampled"] += adversarial.mutant_total
                totals["mutants_rejected"] += adversarial.rejected_mutants
                totals["mutant_population"] += adversarial.candidate_total
                totals["false_accepts"] += len(adversarial.false_accepts)
                for check, count in adversarial.rejected_checks.items():
                    key = check.split(":")[0]
                    totals["rejected_by_check"][key] = (
                        totals["rejected_by_check"].get(key, 0) + count
                    )

    return {"configs": configs, "totals": totals}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--num-sms", default=DEFAULT_NUM_SMS)
    parser.add_argument("--fusion-groups", default=DEFAULT_FUSION_GROUPS)
    parser.add_argument("--attn-splits", default=DEFAULT_ATTN_SPLITS)
    parser.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_SAMPLE,
        help=(
            "adversarial mutants per configuration, stratified by mutation kind "
            "(0 = the whole population, ~2.5 h over the default matrix)"
        ),
    )
    parser.add_argument("--json", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = verify_matrix(
        batch_sizes=parse_int_list(args.batch_sizes, name="batch-sizes"),
        num_sms_values=parse_int_list(args.num_sms, name="num-sms"),
        fusion_groups=parse_int_list(args.fusion_groups, name="fusion-groups"),
        attn_splits=parse_int_list(args.attn_splits, name="attn-splits"),
        sample=args.sample if args.sample > 0 else None,
    )

    totals = payload["totals"]
    for config in payload["configs"]:
        mark = "PASS" if config["ok"] else "FAIL"
        detail = "" if config["ok"] else f"  {config['failures'][:1]}"
        print(
            f"{mark} BS={config['batch_size']} sms={config['num_sms']:>3} "
            f"attn={config['attn_splits']} g={config['fusion_groups']:>2} "
            f"segs={config['segments']:>2} inst={config['instructions']:>5}{detail}"
        )

    print(
        f"\nschedules: {totals['passed']}/{totals['configs']} pass; "
        f"valid accepted {totals['valid_accepted']}/{totals['valid_total']}; "
        f"mutants rejected {totals['mutants_rejected']}/{totals['mutants_sampled']} "
        f"sampled of {totals['mutant_population']}; "
        f"false accepts {totals['false_accepts']}"
    )
    print(f"rejected by check: {totals['rejected_by_check']}")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"wrote {args.json}")

    return 0 if totals["failed"] == 0 and totals["false_accepts"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
