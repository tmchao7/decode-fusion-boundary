"""Reproducible nano-vLLM A/B benchmark with request-level metrics.

Run on an A100/V100 server. Controlled token-length workloads are useful for
kernel analysis; the realistic workload suite exercises natural-language
requests separately. CUDA profiling stays outside latency measurements.
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.metrics import (
    GpuTelemetrySampler,
    compare_metric_summaries,
    merge_metric_reports,
    dump_json,
    environment_metadata,
)
from tools.workloads import load_workloads


BACKEND_CONFIGS = {
    "torch-eager": {"decode_backend": "torch", "enforce_eager": True},
    "torch-cudagraph": {"decode_backend": "torch", "enforce_eager": False},
    "megakernel": {"decode_backend": "megakernel", "enforce_eager": False},
}
PromptFactory = Callable[[int, int], str | list[int]]


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of positive integers"
        )
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


def synthetic_factory(prompt_lengths: list[int], seed: int) -> PromptFactory:
    def factory(run_index: int, slot: int) -> list[int]:
        length = prompt_lengths[slot % len(prompt_lengths)]
        index = max(0, run_index) * 1000 + slot
        return prompt_ids(length, seed, index)

    return factory


def realistic_factory(prompt: str) -> PromptFactory:
    def factory(run_index: int, slot: int) -> str:
        if run_index <= 0:
            return prompt
        return f"{prompt}\n\nBenchmark variation {run_index}; keep the answer focused."

    return factory


def merge_telemetry(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge best-effort NVML summaries without inventing unavailable values."""

    available = [summary for summary in summaries if summary.get("available")]
    if not available:
        errors = [summary.get("error") for summary in summaries if summary.get("error")]
        return {
            "available": False,
            "error": errors[0] if errors else None,
            "samples": 0,
        }
    merged = {
        "available": True,
        "error": next(
            (summary.get("error") for summary in summaries if summary.get("error")),
            None,
        ),
        "samples": sum(int(summary.get("samples", 0)) for summary in available),
    }
    for key in (
        "max_gpu_utilization_pct",
        "max_memory_utilization_pct",
        "max_memory_used_bytes",
        "max_power_watts",
        "max_temperature_c",
    ):
        values = [summary[key] for summary in available if key in summary]
        if values:
            merged[key] = max(values)
    utilization = [
        (summary["mean_gpu_utilization_pct"], summary.get("samples", 0))
        for summary in available
        if "mean_gpu_utilization_pct" in summary
    ]
    total_samples = sum(samples for _, samples in utilization)
    if total_samples:
        merged["mean_gpu_utilization_pct"] = sum(
            value * samples for value, samples in utilization
        ) / total_samples
    return merged


def run_case(
    model_path: str,
    backend_name: str,
    workload_id: str,
    category: str,
    prompt_factory: PromptFactory,
    prompt_lengths: list[int] | None,
    batch_size: int,
    max_tokens: int,
    warmup: int,
    runs: int,
    seed: int,
    ignore_eos: bool,
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
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        ignore_eos=ignore_eos,
    )
    try:
        for warmup_index in range(warmup):
            warmup_prompts = [
                prompt_factory(-(warmup_index + 1), slot)
                for slot in range(batch_size)
            ]
            llm.generate(warmup_prompts, sampling_params, use_tqdm=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        run_results = []
        for run_index in range(1, runs + 1):
            measured_prompts = [
                prompt_factory(run_index, slot)
                for slot in range(batch_size)
            ]
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

            run_results.append(
                {
                    "metrics": result["metrics"],
                    "outputs": [output["token_ids"] for output in result["outputs"]],
                    "telemetry": telemetry.summary(),
                }
            )

        merged_metrics = merge_metric_reports(
            [run_result["metrics"] for run_result in run_results]
        )
        summary = merged_metrics["summary"]
        if torch.cuda.is_available():
            summary["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
            summary["peak_memory_reserved_bytes"] = torch.cuda.max_memory_reserved()
        summary["telemetry"] = merge_telemetry(
            [run_result["telemetry"] for run_result in run_results]
        )
        return {
            "backend": backend_name,
            "backend_config": backend_config,
            "workload": {
                "id": workload_id,
                "category": category,
                "prompt_lengths": prompt_lengths,
                "batch_size": batch_size,
                "max_tokens": max_tokens,
                "warmup": warmup,
                "runs": runs,
                "seed": seed,
                "ignore_eos": ignore_eos,
            },
            "summary": summary,
            "requests": merged_metrics["requests"],
            "outputs": [
                output
                for run_result in run_results
                for output in run_result["outputs"]
            ],
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
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def mark_correctness(experiments: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for experiment in experiments:
        grouped.setdefault(experiment["workload"]["id"], []).append(experiment)

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
                range(
                    min(len(reference_outputs), len(experiment["outputs"])),
                    max(len(reference_outputs), len(experiment["outputs"])),
                )
            )
            experiment["correctness"] = {
                "reference_backend": "torch-eager",
                "passed": not mismatches,
                "mismatched_requests": mismatches,
            }


def add_comparisons(
    experiments: list[dict[str, Any]], baseline_backend: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for experiment in experiments:
        grouped.setdefault(experiment["workload"]["id"], []).append(experiment)

    comparisons = []
    for workload_id, group in grouped.items():
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
                comparison_status = "fallback_or_incorrect"
            else:
                comparison_status = "comparable"
            comparisons.append(
                {
                    "workload_id": workload_id,
                    "category": optimized["workload"]["category"],
                    "baseline_backend": baseline_backend,
                    "optimized_backend": optimized["backend"],
                    "batch_size": optimized["workload"]["batch_size"],
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
        "| Workload | Backend | Prompt lengths | Batch | TTFT p50 (ms) | TPOT p50 (ms) | ITL p50 (ms) | E2E p50 (ms) | Host decode p50 (ms) | CUDA decode p50 (ms) | Decode tok/s | Peak memory (GiB) | Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for experiment in payload["experiments"]:
        summary = experiment["summary"]
        workload = experiment["workload"]
        peak = summary.get("peak_memory_allocated_bytes")
        peak_gib = "n/a" if peak is None else f"{peak / 2**30:.2f}"

        def value(metric: str) -> str:
            item = summary.get(metric, {}).get("p50")
            return "n/a" if item is None else f"{item:.3f}"

        lengths = workload.get("prompt_lengths") or ["natural"]
        lines.append(
            f"| {workload['id']} | {experiment['backend']} | {','.join(map(str, lengths))} | "
            f"{workload['batch_size']} | {value('ttft_ms')} | {value('tpot_ms')} | "
            f"{value('itl_ms')} | {value('e2e_ms')} | {value('decode_time_ms')} | "
            f"{value('cuda_decode_time_ms')} | {summary.get('decode_tok_s', 0.0) or 0.0:.2f} | "
            f"{peak_gib} | {','.join(experiment.get('backend_actual', [])) or 'unknown'} |"
        )
    lines.extend(["", "## Baseline Comparisons", ""])
    for comparison in payload["comparisons"]:
        lines.append(
            f"### {comparison['workload_id']} "
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
    parser.add_argument("--workloads", default="benchmarks/workloads.json")
    parser.add_argument("--suite", choices=("performance", "realistic", "all"), default="performance")
    parser.add_argument(
        "--backends",
        default="torch-eager,torch-cudagraph,megakernel",
        help="Comma-separated backend names",
    )
    parser.add_argument("--prompt-lengths", type=parse_int_list)
    parser.add_argument("--batch-sizes", type=parse_int_list, default=[1, 8, 32])
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", dest="json_path")
    parser.add_argument("--markdown", dest="markdown_path")
    parser.add_argument("--skip-batch", action="store_true")
    parser.add_argument("--skip-mixed-batch", action="store_true")
    parser.add_argument("--baseline", default="torch-cudagraph")
    args = parser.parse_args()

    workload_payload = load_workloads(args.workloads)
    backends = [item.strip() for item in args.backends.split(",") if item.strip()]
    invalid = set(backends) - set(BACKEND_CONFIGS)
    if invalid:
        parser.error(f"unknown backend(s): {', '.join(sorted(invalid))}")
    if args.baseline not in backends:
        print(
            f"No baseline backend {args.baseline!r} in this run; "
            "comparisons will be omitted.",
            file=sys.stderr,
        )

    experiments = []
    performance_lengths = args.prompt_lengths or workload_payload["performance"]["prompt_lengths"]
    if args.suite in ("performance", "all"):
        for prompt_length in performance_lengths:
            for backend in backends:
                experiments.append(
                    run_case(
                        args.model,
                        backend,
                        f"perf_p{prompt_length}_b1",
                        "controlled_performance",
                        synthetic_factory([prompt_length], args.seed),
                        [prompt_length],
                        1,
                        args.max_tokens,
                        args.warmup,
                        args.runs,
                        args.seed,
                        ignore_eos=True,
                    )
                )
        if not args.skip_batch:
            for batch_size in args.batch_sizes:
                for backend in backends:
                    experiments.append(
                        run_case(
                            args.model,
                            backend,
                            f"perf_p128_b{batch_size}",
                            "controlled_batch",
                            synthetic_factory([128], args.seed),
                            [128],
                            batch_size,
                            args.max_tokens,
                            args.warmup,
                            args.runs,
                            args.seed,
                            ignore_eos=True,
                        )
                    )
        if not args.skip_mixed_batch:
            for mix in workload_payload["performance"].get("batch_mixes", []):
                mix_lengths = [int(length) for length in mix["prompt_lengths"]]
                for backend in backends:
                    experiments.append(
                        run_case(
                            args.model,
                            backend,
                            mix["id"],
                            mix["category"],
                            synthetic_factory(mix_lengths, args.seed),
                            mix_lengths,
                            len(mix_lengths),
                            args.max_tokens,
                            args.warmup,
                            args.runs,
                            args.seed,
                            ignore_eos=True,
                        )
                    )

    if args.suite in ("realistic", "all"):
        for item in workload_payload["correctness"]:
            for backend in backends:
                experiments.append(
                    run_case(
                        args.model,
                        backend,
                        item["id"],
                        item["category"],
                        realistic_factory(item["prompt"]),
                        None,
                        1,
                        int(item["max_tokens"]),
                        max(1, min(args.warmup, 2)),
                        max(1, min(args.runs, 3)),
                        args.seed,
                        ignore_eos=False,
                    )
                )

    if not experiments:
        parser.error("no workloads selected")
    mark_correctness(experiments)
    payload = {
        "schema_version": 2,
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
