"""Mixed max_tokens burst correctness guard (docs/10 §5.6 regression).

The ``schedule_burst`` fix (``max_steps = min(budget)``; every selected
sequence reserved for the whole burst, ``scheduler.py``) prevents a short
sequence in a mixed-length batch from writing K/V past its reserved blocks
into a neighbour's slot on the device-side burst loop. The reservation
invariant is unit-tested in ``tests/test_burst.py``.

This tool has two arms, because a token-level comparison alone **cannot** see
the regression:

1. **Scheduler tripwire (the real detector).** The run reads the scheduler's
   ``burst_max_steps_violations`` / ``burst_reservation_violations`` counters
   and fails if any burst ran longer than the minimum selected budget or
   reserved fewer tokens than it ran. This is deterministic and independent of
   numerical tolerance.
2. **Token comparison (secondary smoke).** ``torch-eager`` (reference) and
   ``megakernel-burst`` on the same batch whose per-sequence ``max_tokens``
   alternate between ``--short-tokens`` and ``--long-tokens``, all with
   ``ignore_eos=True`` so the burst path is taken. The comparison is classified
   PASS / CASCADE / FAIL exactly like ``tools/correctness_smoke``.

   Why arm 2 is not the detector: an overrun corrupts a *neighbour's* KV, which
   shows up as that sequence diverging at a decode token >= 1 -- i.e. exactly
   the documented ~1-ulp bf16 cascade that ``--allow-cascade`` tolerates.
   ``length_check`` cannot see it either, because ``postprocess_burst``
   truncates the short row back to its cap. So ``--allow-cascade`` is safe here
   *only because* arm 1 carries the detection.

**``--short-tokens`` must be >= 2, and the tool requires at least one genuinely
mixed-length burst.**  Prefill already produces the first completion token, so
a request with ``max_tokens=1`` is FINISHED during prefill
(``scheduler.postprocess``) and never enters a burst -- the batch would then be
all-long, ``min(budget) == max(budget)``, and the tripwire would pass without
having exercised the mixed path at all.  ``--short-tokens 2`` is the smallest
value that leaves the short row in the burst with a 1-token budget next to the
long row's ``burst_cap`` (64), and it is also the value that actually overruns
under the bug: with prompt 250, ``len(seq) = 252`` fits in one 256-token block,
so reserving that row's own budget and then running 64 steps writes into the
next block-table entry.  The ``mixed_budget_bursts`` counter makes this
self-checking: zero mixed bursts is a hard failure, not a pass.

The reference prompt length is chosen near ``kvcache_block_size`` (256) so that
a ``max_steps = max(budget)`` regression would make the short row write across
a block boundary into the padded block-table entry (physical block 0 after the
§8 padding fix), corrupting whichever sequence owns it.
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import traceback
from pathlib import Path

from tools.correctness_smoke import (
    BACKENDS,
    classify_comparison,
    compare_outputs,
    prompt_ids,
)


def caps_for_batch(batch_size: int, short_tokens: int, long_tokens: int) -> list[int]:
    """Alternate short/long max_tokens across the batch slots."""

    return [short_tokens if slot % 2 == 0 else long_tokens for slot in range(batch_size)]


def burst_guard_violations(guard: dict | None) -> int:
    """Total scheduler burst-contract violations recorded by the tripwire.

    ``max_steps`` must equal the minimum selected budget (otherwise a shorter
    sequence runs past the blocks it reserved) and every selected sequence must
    be reserved for the full burst length.  Either violation is a hard failure;
    the tool must not fall back to the token comparison for this decision.
    """

    guard = guard or {}
    return int(guard.get("max_steps_violations") or 0) + int(
        guard.get("reservation_violations") or 0
    )


def burst_guard_rejection(guard: dict | None, *, short_tokens: int) -> str | None:
    """Reason this run does not validate the mixed-length burst guard, if any.

    Returns ``None`` only when the run actually exercised the contract: at
    least one burst was scheduled, at least one of them had *mixed* per-sequence
    budgets, and none violated the max_steps / reservation contract.  A run
    where every burst had uniform budgets proves nothing (``min == max`` makes
    the contract vacuous), which is what a ``--short-tokens 1`` run produces
    because prefill consumes that sequence's only token.
    """

    guard = guard or {}
    observed = int(guard.get("bursts_observed") or 0)
    if observed <= 0:
        return (
            "the megakernel-burst backend ran but the scheduler observed no "
            "burst; the mixed-length path was not exercised"
        )
    violations = burst_guard_violations(guard)
    if violations:
        return (
            "the schedule_burst contract was violated (max_steps must equal the "
            "minimum budget and every selected sequence must be reserved for it)"
        )
    mixed = int(guard.get("mixed_budget_bursts") or 0)
    if mixed <= 0:
        return (
            "no burst had mixed per-sequence budgets, so the overrun this guard "
            f"targets could not occur (min == max everywhere); --short-tokens="
            f"{short_tokens} leaves no short row in the burst. Prefill emits the "
            "first completion token, so max_tokens=1 finishes during prefill and "
            "never reaches schedule_burst(); use --short-tokens 2 so the short "
            "row enters the burst with a 1-token budget next to the long row's "
            "burst_cap (64)"
        )
    return None


def run_backend(model: str, name: str, prompts: list[list[int]], caps: list[int]) -> dict:
    import torch
    from nanovllm import LLM, SamplingParams

    llm = LLM(
        model,
        **BACKENDS[name],
        max_num_seqs=len(prompts),
        max_model_len=2048,
    )
    params = [
        SamplingParams(temperature=0.0, max_tokens=cap, ignore_eos=True)
        for cap in caps
    ]
    try:
        result = llm.generate_with_metrics(prompts, params, use_tqdm=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # Oracle-free half of this guard: the scheduler's own tripwire.  The
        # device burst runs one max_steps for the whole batch, so max_steps must
        # be the minimum per-sequence budget and every selected sequence must be
        # reserved for that length.  A token comparison against torch-eager
        # cannot see an overrun (it looks exactly like the documented bf16 ulp
        # cascade), so this is the signal that actually catches a regression.
        scheduler = getattr(llm, "scheduler", None)
        result["burst_guard"] = {
            "bursts_observed": int(getattr(scheduler, "bursts_observed", 0) or 0),
            "mixed_budget_bursts": int(
                getattr(scheduler, "burst_mixed_budget_bursts", 0) or 0
            ),
            "max_steps_violations": int(
                getattr(scheduler, "burst_max_steps_violations", 0) or 0
            ),
            "reservation_violations": int(
                getattr(scheduler, "burst_reservation_violations", 0) or 0
            ),
            "last_max_steps": int(getattr(scheduler, "last_burst_max_steps", 0) or 0),
            "last_budgets": list(getattr(scheduler, "last_burst_budgets", []) or []),
            "last_reserved_tokens": list(
                getattr(scheduler, "last_burst_reserved_tokens", []) or []
            ),
        }
        return result
    finally:
        llm.exit()
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as exc:
                print(f"CUDA cleanup skipped after backend failure: {exc}", flush=True)


def _worker(connection, model: str, name: str, prompts, caps):
    try:
        result = run_backend(model, name, prompts, caps)
        connection.send(
            {
                "ok": True,
                "result": {
                    "outputs": [item["token_ids"] for item in result["outputs"]],
                    "metrics": result["metrics"],
                    "burst_guard": result.get("burst_guard"),
                },
            }
        )
    except BaseException:
        connection.send({"ok": False, "error": traceback.format_exc()})
    finally:
        connection.close()


def run_backend_isolated(model, name, prompts, caps) -> dict:
    """Run one backend in a fresh process so CUDA state cannot leak."""

    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker, args=(child, model, name, prompts, caps)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mixed max_tokens burst correctness guard (§5.6)."
    )
    parser.add_argument("model")
    parser.add_argument(
        "--batch-sizes",
        type=lambda value: [int(item) for item in value.split(",") if item],
        default=[2, 4, 5],
    )
    parser.add_argument("--prompt-length", type=int, default=250)
    parser.add_argument(
        "--short-tokens",
        type=int,
        default=2,
        help=(
            "per-sequence max_tokens for the short rows.  Must be >= 2: prefill "
            "emits the first completion token, so max_tokens=1 finishes during "
            "prefill and never enters a burst, which would make the mixed-budget "
            "check vacuous (see the module docstring)."
        ),
    )
    parser.add_argument("--long-tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backends", default="torch-eager,megakernel-burst")
    parser.add_argument("--require-megakernel", action="store_true")
    parser.add_argument(
        "--allow-cascade",
        action="store_true",
        help="Tolerate the documented ~1-ulp bf16 cascade at decode tokens >= 1.",
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
    if args.short_tokens < 2 or args.long_tokens < 1:
        parser.error(
            "--short-tokens must be >= 2 (prefill consumes the first token, so "
            "max_tokens=1 never reaches the burst) and --long-tokens >= 1"
        )
    if args.short_tokens >= args.long_tokens:
        parser.error("--short-tokens must be smaller than --long-tokens")

    results = []
    for batch_size in args.batch_sizes:
        caps = caps_for_batch(batch_size, args.short_tokens, args.long_tokens)
        prompts = [
            prompt_ids(args.prompt_length, args.seed, slot)
            for slot in range(batch_size)
        ]
        case_id = (
            f"mixed_p{args.prompt_length}_b{batch_size}_"
            f"s{args.short_tokens}_l{args.long_tokens}"
        )
        backend_results = {}
        for backend in backend_names:
            print(
                f"Running mixed burst correctness: case={case_id} backend={backend}",
                flush=True,
            )
            backend_results[backend] = run_backend_isolated(
                args.model, backend, prompts, caps
            )
        reference = backend_results["torch-eager"]["outputs"]
        checks = {}
        for backend, result in backend_results.items():
            actuals = result["metrics"]["summary"].get("backend_actuals", [])
            mismatches, first_mismatch = compare_outputs(reference, result["outputs"])
            status, cascade_details = classify_comparison(reference, result["outputs"])
            used_megakernel = "megakernel" in actuals
            # With ignore_eos=True every sequence must yield exactly its cap.
            length_check = {
                "expected": caps,
                "actual": [len(out) for out in result["outputs"]],
                "ok": [len(out) == cap for out, cap in zip(result["outputs"], caps)],
            }
            checks[backend] = {
                "status": status,
                "mismatched_requests": mismatches,
                "first_mismatch": first_mismatch,
                "cascade_details": cascade_details if cascade_details else None,
                "backend_actuals": actuals,
                "used_megakernel": used_megakernel,
                "length_check": length_check,
                "fallback_reasons": result["metrics"]["summary"].get(
                    "fallback_reasons", []
                ),
            }
            if args.require_megakernel and backend == "megakernel-burst" and not used_megakernel:
                raise SystemExit(
                    f"Megakernel-burst was requested but did not execute for {case_id}"
                )
            burst_guard = result.get("burst_guard") or {}
            checks[backend]["burst_guard"] = burst_guard
            if backend == "megakernel-burst" and used_megakernel:
                rejection = burst_guard_rejection(
                    burst_guard, short_tokens=args.short_tokens
                )
                if rejection is not None:
                    raise SystemExit(
                        f"mixed burst guard did not validate {case_id}: "
                        f"{rejection}; burst_guard={burst_guard}"
                    )
            if status == "FAIL":
                raise SystemExit(
                    f"mixed burst correctness FAILED for {backend}/{case_id}: "
                    f"prefill-token-0 or length mismatch; "
                    f"details={cascade_details}; length_check={length_check}; "
                    f"reference_outputs={reference}; actual_outputs={result['outputs']}"
                )
            if status == "CASCADE" and not args.allow_cascade:
                raise SystemExit(
                    f"mixed burst correctness cascade for {backend}/{case_id}: "
                    f"use --allow-cascade to tolerate; details={cascade_details}"
                )
            if not all(length_check["ok"]):
                raise SystemExit(
                    f"mixed burst length mismatch for {backend}/{case_id}: "
                    f"a sequence did not produce exactly max_tokens under "
                    f"ignore_eos=True; {length_check}"
                )
            if status == "CASCADE":
                print(
                    f"mixed burst CASCADE (tolerated) for {backend}/{case_id}: "
                    f"details={cascade_details}",
                    flush=True,
                )
        results.append(
            {
                "id": case_id,
                "batch_size": batch_size,
                "prompt_length": args.prompt_length,
                "caps": caps,
                "checks": checks,
            }
        )

    payload = {
        "model": os.path.abspath(args.model),
        "prompt_length": args.prompt_length,
        "short_tokens": args.short_tokens,
        "long_tokens": args.long_tokens,
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
