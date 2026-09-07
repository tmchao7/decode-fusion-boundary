"""Short greedy correctness checks for the server before long benchmarks."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
from pathlib import Path

from tools.workloads import load_workloads, select_correctness_workloads

BACKENDS = {
    "torch-eager": {"decode_backend": "torch", "enforce_eager": True},
    "torch-cudagraph": {"decode_backend": "torch", "enforce_eager": False},
    "megakernel": {"decode_backend": "megakernel", "enforce_eager": False},
}


def prompt_ids(length: int, seed: int, index: int = 0) -> list[int]:
    rng = random.Random(seed + index * 104729 + length * 1009)
    return [rng.randrange(1, 10000) for _ in range(length)]


def build_cases(
    workload_path: str,
    prompt_lengths: list[int] | None,
    workload_ids: list[str] | None,
    batch_sizes: list[int],
    seed: int,
) -> list[dict]:
    """Build deterministic, equally-shaped correctness cases."""

    if prompt_lengths:
        cases = []
        for length in prompt_lengths:
            for batch_size in batch_sizes:
                cases.append(
                    {
                        "id": f"synthetic_p{length}_b{batch_size}",
                        "category": "synthetic",
                        "prompt_lengths": [length] * batch_size,
                        "prompts": [
                            prompt_ids(length, seed, slot)
                            for slot in range(batch_size)
                        ],
                    }
                )
        return cases

    payload = load_workloads(workload_path)
    selected = select_correctness_workloads(payload, workload_ids)
    cases = []
    for item in selected:
        for batch_size in batch_sizes:
            prompts = [
                f"{item['prompt']}\n\nBatch slot {slot}; answer concisely."
                if batch_size > 1
                else item["prompt"]
                for slot in range(batch_size)
            ]
            cases.append(
                {
                    "id": f"{item['id']}_b{batch_size}",
                    "category": item["category"],
                    "prompt_lengths": None,
                    "prompts": prompts,
                }
            )
    return cases


def run_backend(model: str, name: str, prompts, max_tokens: int):
    import torch
    from nanovllm import LLM, SamplingParams

    llm = LLM(
        model,
        **BACKENDS[name],
        max_num_seqs=len(prompts),
        max_model_len=2048,
    )
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    try:
        result = llm.generate_with_metrics(prompts, params, use_tqdm=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return result
    finally:
        llm.exit()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def compare_outputs(reference, actual):
    mismatches = []
    first_mismatch = None
    for request_index, (expected, observed) in enumerate(zip(reference, actual)):
        for token_index, (expected_token, observed_token) in enumerate(
            zip(expected, observed)
        ):
            if expected_token != observed_token:
                mismatches.append(request_index)
                if first_mismatch is None:
                    first_mismatch = {
                        "request_index": request_index,
                        "token_index": token_index,
                        "expected": expected_token,
                        "actual": observed_token,
                    }
                break
        if len(expected) != len(observed):
            if request_index not in mismatches:
                mismatches.append(request_index)
            if first_mismatch is None:
                first_mismatch = {
                    "request_index": request_index,
                    "token_index": min(len(expected), len(observed)),
                    "expected": expected[min(len(expected), len(observed))]
                    if len(expected) > len(observed)
                    else None,
                    "actual": observed[min(len(expected), len(observed))]
                    if len(observed) > len(expected)
                    else None,
                }
    if len(reference) != len(actual):
        mismatches.extend(
            index
            for index in range(min(len(reference), len(actual)), max(len(reference), len(actual)))
            if index not in mismatches
        )
    return sorted(set(mismatches)), first_mismatch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--workloads", default="benchmarks/workloads.json")
    parser.add_argument("--workload-ids")
    parser.add_argument(
        "--prompt-lengths",
        type=lambda value: [int(item) for item in value.split(",") if item],
        help="Use synthetic prompts instead of realistic workloads",
    )
    parser.add_argument(
        "--batch-sizes",
        type=lambda value: [int(item) for item in value.split(",") if item],
        default=[1, 2, 8],
    )
    parser.add_argument("--backends", default="torch-eager,torch-cudagraph,megakernel")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require-megakernel", action="store_true")
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args()

    backend_names = [name.strip() for name in args.backends.split(",") if name.strip()]
    unknown = set(backend_names) - set(BACKENDS)
    if unknown:
        parser.error(f"unknown backend(s): {', '.join(sorted(unknown))}")
    if "torch-eager" not in backend_names:
        parser.error("--backends must include torch-eager as the correctness reference")
    if any(size <= 0 for size in args.batch_sizes):
        parser.error("--batch-sizes must contain positive integers")
    workload_ids = (
        [value.strip() for value in args.workload_ids.split(",") if value.strip()]
        if args.workload_ids
        else None
    )
    cases = build_cases(
        args.workloads,
        args.prompt_lengths,
        workload_ids,
        args.batch_sizes,
        args.seed,
    )

    results = []
    for case in cases:
        backend_results = {}
        for backend in backend_names:
            print(f"Running correctness: case={case['id']} backend={backend}", flush=True)
            result = run_backend(args.model, backend, case["prompts"], args.max_tokens)
            backend_results[backend] = {
                "outputs": [item["token_ids"] for item in result["outputs"]],
                "metrics": result["metrics"],
            }
        reference = backend_results["torch-eager"]["outputs"]
        checks = {}
        for backend, result in backend_results.items():
            actuals = result["metrics"]["summary"].get("backend_actuals", [])
            mismatches, first_mismatch = compare_outputs(reference, result["outputs"])
            used_megakernel = "megakernel" in actuals
            checks[backend] = {
                "passed": not mismatches,
                "mismatched_requests": mismatches,
                "first_mismatch": first_mismatch,
                "backend_actuals": actuals,
                "used_megakernel": used_megakernel,
                "fallback_reasons": result["metrics"]["summary"].get(
                    "fallback_reasons", []
                ),
            }
            if args.require_megakernel and backend == "megakernel" and not used_megakernel:
                raise SystemExit(
                    f"Megakernel was requested but did not execute for {case['id']}"
                )
            if mismatches:
                raise SystemExit(
                    f"correctness failed for {backend}/{case['id']}: requests {mismatches}"
                )
        results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "prompt_lengths": case["prompt_lengths"],
                "batch_size": len(case["prompts"]),
                "checks": checks,
            }
        )

    payload = {
        "model": os.path.abspath(args.model),
        "max_tokens": args.max_tokens,
        "batch_sizes": args.batch_sizes,
        "cases": results,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_path:
        output = Path(args.json_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
