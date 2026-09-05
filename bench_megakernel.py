"""Reproducible nano-vLLM A/B benchmark with request-level metrics.

Run on an A100/V100 server. The script intentionally keeps CUDA profiling out
of latency measurements; use the NVTX ranges with Nsight separately.
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.metrics import (
    GpuTelemetrySampler,
    compare_metric_summaries,
    dump_json,
    environment_metadata,
)


BACKEND_CONFIGS = {
    "torch-eager": {"decode_backend": "torch", "enforce_eager": True},
    "torch-cudagraph": {"decode_backend": "torch", "enforce_eager": False},
    "megakernel": {"decode_backend": "megakernel", "enforce_eager": False},
}


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def prompt_ids(length: int, seed: int, index: int) -> list[int]:
    rng = random.Random(seed + index * 104729 + length * 1009)
    return [rng.randrange(1, 10000) for _ in range(length)]


def git_metadata() -> dict[str, Any]:
    def run_git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return None

    return {
        "commit": run_git("rev-parse", "HEAD"),
        "dirty": bool(run_git("status", "--porcelain")),
    }


def _backend_actual_is_megakernel(summary: dict[str, Any]) -> bool:
    return "megakernel" in summary.get("backend_actuals", [])


def run_case(
    model_path: str,
    backend_name: str,
    prompt_length: int,
    batch_size: int,
    max_tokens: int,
    warmup: int,
    runs: int,
    seed: int,
) -> dict[str, Any]:
    if backend_name not in BACKEND_CONFIGS:
        raise ValueError(f"unknown backend: {backend_name}")
    if backend_name == "megakernel" and batch_size != 1:
        raise ValueError("Megakernel benchmark cases must use batch_size=1")

    backend_config = BACKEND_CONFIGS[backend_name]
    llm = LLM(
        model_path,
        **backend_config,
        max_num_seqs=batch_size,
        max_model_len=2048,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    warmup_prompts = [prompt_ids(prompt_length, seed + 17, index) for index in range(warmup)]
    measured_count = runs * batch_size
    measured_prompts = [
        prompt_ids(prompt_length, seed, index + warmup + 1)
        for index in range(measured_count)
    ]

    try:
        for prompt in warmup_prompts:
            llm.generate([prompt], sampling_params, use_tqdm=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        telemetry = GpuTelemetrySampler().start()
        try:
            result = llm.generate_with_metrics(
                measured_prompts,
                sampling_params,
                use_tqdm=False,
            )
        finally:
            telemetry.stop()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        summary = result["metrics"]["summary"]
        if torch.cuda.is_available():
            summary["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
            summary["peak_memory_reserved_bytes"] = torch.cuda.max_memory_reserved()
        summary["telemetry"] = telemetry.summary()
        return {
            "backend": backend_name,
            "backend_config": backend_config,
            "workload": {
                "prompt_length": prompt_length,
                "batch_size": batch_size,
                "max_tokens": max_tokens,
                "warmup": warmup,
                "runs": runs,
                "seed": seed,
            },
            "summary": summary,
            "requests": result["metrics"]["requests"],
            "outputs": [output["token_ids"] for output in result["outputs"]],
            "backend_actual": summary.get("backend_actuals", []),
            "fallback": {
                "count": summary.get("fallback_count", 0),
                "reasons": summary.get("fallback_reasons", []),
            },
            "profiling": {
                "source": "external_nsight",
                "kernel_launches_per_token": None,
                "memcpy_operations_per_token": None,
                "memcpy_bytes_per_token": None,
                "dram_throughput": None,
                "l2_throughput": None,
                "achieved_occupancy": None,
                "sm_throughput": None,
            },
        }
    finally:
        llm.exit()


def mark_correctness(experiments: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for experiment in experiments:
        workload = experiment["workload"]
        key = (workload["prompt_length"], workload["batch_size"])
        grouped.setdefault(key, []).append(experiment)

    for group in grouped.values():
        reference = next(
            (item for item in group if item["backend"] == "torch-eager"), None
        )
        if reference is None:
            continue
        reference_outputs = reference["outputs"]
        for experiment in group:
            mismatches = [
                index
                for index, (expected, actual) in enumerate(
                    zip(reference_outputs, experiment["outputs"])
                )
                if expected != actual
            ]
            mismatches.extend(
                range(min(len(reference_outputs), len(experiment["outputs"])),
                      max(len(reference_outputs), len(experiment["outputs"])))
            )
            experiment["correctness"] = {
                "reference_backend": "torch-eager",
                "passed": not mismatches,
                "mismatched_requests": mismatches,
            }


def add_comparisons(
    experiments: list[dict[str, Any]], baseline_backend: str
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for experiment in experiments:
        workload = experiment["workload"]
        grouped.setdefault(
            (workload["prompt_length"], workload["batch_size"]), []
        ).append(experiment)

    comparisons = []
    for (prompt_length, batch_size), group in grouped.items():
        baseline = next(
            (item for item in group if item["backend"] == baseline_backend), None
        )
        if baseline is None:
            continue
        for optimized in group:
            if optimized is baseline:
                continue
            comparison = compare_metric_summaries(
                baseline["summary"], optimized["summary"]
            )
            if optimized["backend"] == "megakernel" and (
                not _backend_actual_is_megakernel(optimized["summary"])
                or not optimized.get("correctness", {}).get("passed", False)
            ):
                for metric in comparison.values():
                    metric["comparable"] = False
                    metric["improvement_pct"] = None
                    metric["speedup"] = None
                comparison_status = "fallback/not_applicable"
            else:
                comparison_status = "comparable"
            comparisons.append(
                {
                    "baseline_backend": baseline_backend,
                    "optimized_backend": optimized["backend"],
                    "prompt_length": prompt_length,
                    "batch_size": batch_size,
                    "status": comparison_status,
                    "metrics": comparison,
                }
            )
    return comparisons


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# nano-vLLM Inference Benchmark",
        "",
        f"- Timestamp: `{payload['environment'].get('timestamp_utc')}`",
        f"- GPU: `{payload['environment'].get('gpu_name', 'unknown')}`",
        f"- Commit: `{payload['git'].get('commit', 'unknown')}`",
        "",
        "## Request Metrics",
        "",
        "| Backend | Prompt | Batch | TTFT p50 (ms) | TPOT p50 (ms) | ITL p50 (ms) | E2E p50 (ms) | Host decode p50 (ms) | CUDA decode p50 (ms) | Decode tok/s | Peak memory (GiB) | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for experiment in payload["experiments"]:
        summary = experiment["summary"]
        peak = summary.get("peak_memory_allocated_bytes")
        peak_gib = "n/a" if peak is None else f"{peak / 2**30:.2f}"
        def value(metric: str) -> str:
            item = summary.get(metric, {}).get("p50")
            return "n/a" if item is None else f"{item:.3f}"
        lines.append(
            f"| {experiment['backend']} | {experiment['workload']['prompt_length']} | "
            f"{experiment['workload']['batch_size']} | {value('ttft_ms')} | "
            f"{value('tpot_ms')} | {value('itl_ms')} | {value('e2e_ms')} | "
            f"{value('decode_time_ms')} | {value('cuda_decode_time_ms')} | "
            f"{summary.get('decode_tok_s', 0.0) or 0.0:.2f} | {peak_gib} | "
            f"{','.join(experiment.get('backend_actual', [])) or 'unknown'} |"
        )
    lines.extend(["", "## Baseline Comparisons", ""])
    for comparison in payload["comparisons"]:
        lines.append(
            f"### prompt={comparison['prompt_length']}, batch={comparison['batch_size']} "
            f"({comparison['optimized_backend']} vs {comparison['baseline_backend']})"
        )
        lines.append(f"Status: **{comparison['status']}**")
        lines.append("")
        lines.append("| Metric | Speedup | Improvement | Comparable |")
        lines.append("|---|---:|---:|---|")
        for metric, values in comparison["metrics"].items():
            speedup = values["speedup"]
            improvement = values["improvement_pct"]
            lines.append(
                f"| {metric} | {'n/a' if speedup is None else f'{speedup:.3f}x'} | "
                f"{'n/a' if improvement is None else f'{improvement:.2f}%'} | "
                f"{values['comparable']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Profiling",
            "",
            "Use Nsight Systems/Compute separately to collect kernel launches per token, memcpy bytes, DRAM/L2 throughput, SM throughput, and occupancy.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Local Qwen3-0.6B model directory")
    parser.add_argument(
        "--backends",
        default="torch-eager,torch-cudagraph,megakernel",
        help="Comma-separated backend names",
    )
    parser.add_argument("--prompt-lengths", type=parse_int_list, default=[32, 128, 512])
    parser.add_argument("--batch-sizes", type=parse_int_list, default=[1, 8, 32])
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", dest="json_path")
    parser.add_argument("--markdown", dest="markdown_path")
    parser.add_argument("--skip-batch", action="store_true")
    parser.add_argument("--baseline", default="torch-cudagraph")
    args = parser.parse_args()

    backends = [item.strip() for item in args.backends.split(",") if item.strip()]
    invalid = set(backends) - set(BACKEND_CONFIGS)
    if invalid:
        parser.error(f"unknown backend(s): {', '.join(sorted(invalid))}")
    if args.baseline not in backends:
        parser.error("--baseline must be included in --backends")

    experiments = []
    for prompt_length in args.prompt_lengths:
        for backend in backends:
            experiments.append(
                run_case(
                    args.model,
                    backend,
                    prompt_length,
                    1,
                    args.max_tokens,
                    args.warmup,
                    args.runs,
                    args.seed,
                )
            )
    if not args.skip_batch:
        for batch_size in args.batch_sizes:
            for backend in [item for item in backends if item != "megakernel"]:
                experiments.append(
                    run_case(
                        args.model,
                        backend,
                        128,
                        batch_size,
                        args.max_tokens,
                        args.warmup,
                        args.runs,
                        args.seed,
                    )
                )

    mark_correctness(experiments)
    payload = {
        "schema_version": 1,
        "environment": environment_metadata(),
        "git": git_metadata(),
        "model": os.path.abspath(args.model),
        "configuration": vars(args),
        "experiments": experiments,
        "comparisons": add_comparisons(experiments, args.baseline),
    }
    if args.json_path:
        output = Path(args.json_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        dump_json(str(output), payload)
    markdown = render_markdown(payload)
    if args.markdown_path:
        output = Path(args.markdown_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
