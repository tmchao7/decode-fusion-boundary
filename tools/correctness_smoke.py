"""Short greedy correctness checks for the server before long benchmarks."""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import random
import traceback
from pathlib import Path

from tools.workloads import load_workloads, select_correctness_workloads

BACKENDS = {
    "torch-eager": {"decode_backend": "torch", "enforce_eager": True},
    "torch-cudagraph": {"decode_backend": "torch", "enforce_eager": False},
    "megakernel": {"decode_backend": "megakernel", "enforce_eager": False},
}


def backend_config(name: str, attn_splits: int = 1) -> dict:
    """Engine kwargs for one backend, with the attention split count applied.

    ``attn_splits`` only reaches the megakernel: it selects how many SM groups
    produce unnormalized attention partials, which the combine wave then
    renormalizes (see docs/11 §11.3). The torch backends have nothing to split,
    so passing a value there is a caller error rather than a silent no-op.
    """

    if name not in BACKENDS:
        raise ValueError(f"unknown backend: {name}")
    if attn_splits < 1:
        raise ValueError(f"attn_splits must be >= 1; got {attn_splits}")
    config = dict(BACKENDS[name])
    if attn_splits != 1:
        if name != "megakernel":
            raise ValueError(
                f"attn_splits only applies to the megakernel backend, not {name}"
            )
        config["megakernel_attn_splits"] = attn_splits
    return config


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


def run_backend(model: str, name: str, prompts, max_tokens: int, attn_splits: int = 1):
    import torch
    from nanovllm import LLM, SamplingParams

    llm = LLM(
        model,
        **backend_config(name, attn_splits),
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
        # If a Megakernel device fault occurred, CUDA is poisoned for this
        # process. Avoid a second empty_cache() failure masking the original
        # transformer/LM-head error and terminate this isolated backend run.
        llm.exit()
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as exc:
                print(f"CUDA cleanup skipped after backend failure: {exc}", flush=True)


def serialize_backend_result(result: dict) -> dict:
    """Keep only JSON/IPC-safe data returned from an isolated backend process."""

    return {
        "outputs": [item["token_ids"] for item in result["outputs"]],
        "metrics": result["metrics"],
    }


def _backend_worker(connection, model: str, name: str, prompts, max_tokens: int, attn_splits: int = 1):
    """Run one CUDA backend in a fresh process and return JSON-safe data."""

    try:
        # attn_splits MUST be forwarded: dropping it here silently downgraded
        # ``--attn-splits 2`` to 1 while the JSON artifact still recorded 2.
        result = run_backend(model, name, prompts, max_tokens, attn_splits)
        connection.send({"ok": True, "result": serialize_backend_result(result)})
    except BaseException:
        connection.send({"ok": False, "error": traceback.format_exc()})
    finally:
        connection.close()


def run_backend_isolated(model: str, name: str, prompts, max_tokens: int, attn_splits: int = 1) -> dict:
    """Run a backend in a clean process so CUDA allocator state cannot leak.

    A backend may reserve most of the GPU for its paged KV pool. Reusing the
    same Python process for the next backend is unsafe even after ``empty_cache``
    because CUDA Graphs, module references, and allocator state can survive the
    Python-level cleanup path. The child process exits after one backend.
    """

    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_backend_worker,
        args=(child, model, name, prompts, max_tokens, attn_splits),
    )
    process.start()
    child.close()
    process.join()
    payload = parent.recv() if parent.poll() else None
    parent.close()
    if payload is None:
        raise RuntimeError(
            f"isolated backend process exited without a result: "
            f"backend={name}, exitcode={process.exitcode}"
        )
    if not payload["ok"]:
        raise RuntimeError(
            f"isolated backend failed: backend={name}\n{payload['error']}"
        )
    if process.exitcode != 0:
        raise RuntimeError(
            f"isolated backend exited with code {process.exitcode}: backend={name}"
        )
    return payload["result"]


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


def classify_comparison(reference, actual):
    """Classify a reference-vs-actual comparison (PASS / CASCADE / FAIL).

    This is the mainline Megakernel correctness classification:

      - PASS: token-for-token identical.
      - CASCADE (tolerated): every divergence starts at a decode token
        (token >= 1; token 0 is the prefill output and must always match)
        and no request has a length mismatch.  This is the documented
        ~1-ulp bf16 reduction-order cascade, not a correctness failure.
      - FAIL: any divergence at token 0 (prefill mismatch) or any length
        mismatch -- a real bug.
    """
    details = []
    real_fail = False
    for i, (expected, observed) in enumerate(zip(reference, actual)):
        if expected == observed:
            continue
        if len(expected) != len(observed):
            real_fail = True
            details.append({"seq": i, "kind": "length", "d": len(expected) - len(observed)})
            continue
        first = next(j for j, (a, b) in enumerate(zip(expected, observed)) if a != b)
        real_fail |= first == 0
        details.append(
            {
                "seq": i,
                "kind": "token",
                "first": first,
                "expected": expected[first],
                "actual": observed[first],
            }
        )
    if len(reference) != len(actual):
        real_fail = True
        details.append(
            {"seq": len(actual), "kind": "count", "d": len(reference) - len(actual)}
        )
    if real_fail:
        status = "FAIL"
    elif details:
        status = "CASCADE"
    else:
        status = "PASS"
    return status, details


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
    parser.add_argument(
        "--attn-splits",
        type=int,
        default=1,
        help=(
            "megakernel_attn_splits for the megakernel backend: how many SM "
            "groups produce partial attention (the combine wave renormalizes). "
            "Only valid with the megakernel backend."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require-megakernel", action="store_true")
    parser.add_argument(
        "--allow-cascade",
        action="store_true",
        help=(
            "Tolerate the documented ~1-ulp bf16 cascade: divergences at decode "
            "tokens (token >= 1) are classified as CASCADE and do not abort. "
            "Prefill-token-0 mismatches and length mismatches still fail. "
            "Use this for megakernel-family backends vs torch-eager."
        ),
    )
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
            backend_results[backend] = run_backend_isolated(
                args.model, backend, case["prompts"], args.max_tokens, args.attn_splits
            )
        reference = backend_results["torch-eager"]["outputs"]
        checks = {}
        for backend, result in backend_results.items():
            actuals = result["metrics"]["summary"].get("backend_actuals", [])
            mismatches, first_mismatch = compare_outputs(reference, result["outputs"])
            status, cascade_details = classify_comparison(reference, result["outputs"])
            used_megakernel = "megakernel" in actuals
            checks[backend] = {
                "passed": not mismatches,
                "status": status,
                "mismatched_requests": mismatches,
                "first_mismatch": first_mismatch,
                "cascade_details": cascade_details if cascade_details else None,
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
            if status == "FAIL":
                raise SystemExit(
                    f"correctness FAILED for {backend}/{case['id']}: "
                    f"prefill-token-0 or length mismatch; "
                    f"details={cascade_details}; "
                    f"reference_outputs={reference}; actual_outputs={result['outputs']}"
                )
            if status == "CASCADE" and not args.allow_cascade:
                raise SystemExit(
                    f"correctness cascade for {backend}/{case['id']}: "
                    f"decode-token divergence (use --allow-cascade to tolerate); "
                    f"details={cascade_details}"
                )
            if status == "CASCADE":
                print(
                    f"correctness CASCADE (tolerated) for {backend}/{case['id']}: "
                    f"details={cascade_details}",
                    flush=True,
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
        # Recorded because two runs of this tool can differ only in this value.
        "attn_splits": args.attn_splits,
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
