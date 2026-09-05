"""Short greedy correctness check for the server before long benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch

from nanovllm import LLM, SamplingParams

BACKENDS = {
    "torch-eager": {"decode_backend": "torch", "enforce_eager": True},
    "torch-cudagraph": {"decode_backend": "torch", "enforce_eager": False},
    "megakernel": {"decode_backend": "megakernel", "enforce_eager": False},
}


def prompt_ids(length: int, seed: int) -> list[int]:
    rng = random.Random(seed + length * 1009)
    return [rng.randrange(1, 10000) for _ in range(length)]


def run_backend(model: str, name: str, prompts: list[list[int]], max_tokens: int):
    llm = LLM(model, **BACKENDS[name], max_num_seqs=1, max_model_len=2048)
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    try:
        result = llm.generate_with_metrics(prompts, params, use_tqdm=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return result
    finally:
        llm.exit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--backends", default="torch-eager,torch-cudagraph,megakernel")
    parser.add_argument("--prompt-lengths", default="8,32,128")
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
    lengths = [int(value) for value in args.prompt_lengths.split(",") if value]
    prompts = [prompt_ids(length, args.seed) for length in lengths]

    results = {}
    for backend in backend_names:
        result = run_backend(args.model, backend, prompts, args.max_tokens)
        results[backend] = {
            "outputs": [item["token_ids"] for item in result["outputs"]],
            "metrics": result["metrics"],
        }

    reference = results["torch-eager"]["outputs"]
    checks = {}
    for backend, result in results.items():
        actuals = result["metrics"]["summary"].get("backend_actuals", [])
        mismatches = [
            index
            for index, (expected, actual) in enumerate(zip(reference, result["outputs"]))
            if expected != actual
        ]
        mismatches.extend(range(min(len(reference), len(result["outputs"])), max(len(reference), len(result["outputs"]))))
        used_megakernel = "megakernel" in actuals
        checks[backend] = {
            "passed": not mismatches,
            "mismatched_requests": mismatches,
            "backend_actuals": actuals,
            "used_megakernel": used_megakernel,
            "fallback_reasons": result["metrics"]["summary"].get("fallback_reasons", []),
        }
        if args.require_megakernel and backend == "megakernel" and not used_megakernel:
            raise SystemExit("Megakernel was requested but did not execute")
        if mismatches:
            raise SystemExit(f"correctness failed for {backend}: requests {mismatches}")

    payload = {
        "model": os.path.abspath(args.model),
        "prompt_lengths": lengths,
        "max_tokens": args.max_tokens,
        "checks": checks,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_path:
        output = Path(args.json_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
