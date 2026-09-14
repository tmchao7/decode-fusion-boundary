"""Multi-batch megakernel burst vs torch-cudagraph crossover benchmark.

Unlike benchmarks.compare (which uses ignore_eos=True and disables burst),
this uses ignore_eos=False so the burst path actually triggers. Each batch
size runs in a fresh process. Prompts are synthetic token IDs chosen to
avoid early EOS in ``max_tokens`` tokens.

The scan produces the crossover curve (decode ms/token and tok/s vs batch
size) that定位 MegaKernel 由胜转败的拐点。Architecture notes baked into the
measurement:

- ``LDG_ATTN_BLOCKS=8`` (build default) with an 80-block cooperative grid
  caps the single-launch WMMA batch kernel at batch ≈ 10. We set
  ``megakernel_burst_batch = min(batch, 8)`` so batch > 8 splits into serial
  bursts of ≤ 8 (the tested-safe max). This honestly reflects the fixed-grid
  ceiling: megakernel throughput plateaus at batch ≈ 8 while cudagraph keeps
  scaling.
- ``MAX_BATCH_SIZE = 32``; batch > 32 disables megakernel (recorded as null)
  because the compatibility gate rejects it.
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DEFAULT_MODEL = os.environ.get(
    "NANOVLLM_MODEL_PATH", os.path.expanduser("~/models/Qwen3-0.6B")
)
MAX_BATCH_SIZE = 32
SAFE_BURST_BATCH = 8


def prompt_ids(length: int, seed: int) -> list[int]:
    import random
    rng = random.Random(seed + length * 1009)
    return [rng.randrange(1, 10000) for _ in range(length)]


def _worker(connection, backend: str, batch_size: int, prompt_len: int,
            max_tokens: int, warmup: int, runs: int, model: str):
    try:
        import torch
        from nanovllm import LLM, SamplingParams

        prompts = [prompt_ids(prompt_len, i * 17 + 3) for i in range(batch_size)]
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)

        # nano-vLLM captures CUDA graphs at batch sizes [1,2,4,8] + multiples
        # of 16 up to max_num_seqs. Use max(batch, 32) so a graph >= batch is
        # always captured (batch=10 pads to graph 16, etc.).
        max_num_seqs = max(batch_size, 32)
        if backend == "burst":
            burst_batch = min(batch_size, SAFE_BURST_BATCH)
            llm = LLM(model, decode_backend="megakernel", enforce_eager=False,
                      max_num_seqs=max_num_seqs,
                      max_model_len=2048,
                      megakernel_batch_kernel="wmma", megakernel_burst=True,
                      megakernel_burst_batch=burst_batch,
                      megakernel_batch_decode=True)
        else:
            llm = LLM(model, decode_backend="torch", enforce_eager=False,
                      max_num_seqs=max_num_seqs,
                      max_model_len=2048)

        warm_sp = SamplingParams(temperature=0.0, max_tokens=10)
        for _ in range(warmup):
            llm.generate(prompts, warm_sp, use_tqdm=False)
        torch.cuda.synchronize()

        walls: list[float] = []
        total_tokens: int = 0
        for _ in range(runs):
            t0 = time.perf_counter()
            outs = llm.generate(prompts, sp, use_tqdm=False)
            torch.cuda.synchronize()
            walls.append(time.perf_counter() - t0)
            total_tokens = sum(len(o["token_ids"]) for o in outs)

        # Detect whether the megakernel backend was actually active (not
        # silently disabled/fallback). disabled_reason is None when active.
        disabled_reason = None
        if backend == "burst":
            mr = getattr(llm, "engine", llm).model_runner
            mk = getattr(mr, "megakernel", None)
            if mk is None:
                disabled_reason = "megakernel_not_initialized"
            elif getattr(mk, "disabled_reason", None):
                disabled_reason = str(mk.disabled_reason)

        llm.exit()
        del llm
        gc.collect()
        connection.send({
            "ok": True,
            "walls": walls,
            "total_tokens": total_tokens,
            "expected_tokens": batch_size * max_tokens,
            "disabled_reason": disabled_reason,
        })
    except BaseException as exc:
        import traceback
        connection.send({"ok": False, "error": traceback.format_exc()})
    finally:
        connection.close()


def measure(backend: str, batch_size: int, prompt_len: int, max_tokens: int,
            warmup: int, runs: int, model: str) -> dict:
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_worker,
        args=(child, backend, batch_size, prompt_len, max_tokens, warmup, runs, model),
    )
    proc.start()
    proc.join(timeout=900)
    payload = parent.recv() if parent.poll() else None
    if payload is None or not payload["ok"]:
        raise RuntimeError(
            f"{backend} b={batch_size} failed: {payload and payload['error']}"
        )
    return payload


def summarize(walls: list[float], total_tokens: int) -> dict:
    wall_med = statistics.median(walls)
    wall_min = min(walls)
    ms_per_tok_med = wall_med * 1000 / total_tokens if total_tokens else float("inf")
    tok_s_med = total_tokens / wall_med if wall_med > 0 else 0.0
    ms_per_tok_min = wall_min * 1000 / total_tokens if total_tokens else float("inf")
    return {
        "ms_per_tok": ms_per_tok_med,
        "ms_per_tok_min": ms_per_tok_min,
        "tok_s": tok_s_med,
        "wall_s": wall_med,
        "wall_min_s": wall_min,
        "total_tokens": total_tokens,
        "runs": walls,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-length", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--batch-sizes", default="1,2,3,4,5,6,7,8,10,16,32")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    print(f"prompt_len={args.prompt_length} max_tokens={args.max_tokens} "
          f"warmup={args.warmup} runs={args.runs} (primary metric = min-of-runs)")
    print(f"{'batch':>6s} {'cg ms/tok':>10s} {'cg tok/s':>10s} "
          f"{'burst ms/tok':>13s} {'burst tok/s':>13s} {'speedup':>8s} "
          f"{'verdict':>8s} {'note':>10s}")

    records = []
    for bs in batch_sizes:
        cg = measure("cudagraph", bs, args.prompt_length, args.max_tokens,
                     args.warmup, args.runs, args.model)
        cg_s = summarize(cg["walls"], cg["total_tokens"])
        note = ""
        if bs > MAX_BATCH_SIZE:
            note = "mega>32"
            rec = {
                "batch": bs,
                "cudagraph": {**cg_s, "expected_tokens": cg["expected_tokens"]},
                "burst": None,
                "speedup": None,
                "verdict": "skipped",
                "note": "megakernel MAX_BATCH_SIZE=32; burst skipped",
            }
            print(f"{bs:>6d} {cg_s['ms_per_tok_min']:>9.3f} "
                  f"{cg_s['tok_s']:>9.1f} {'n/a':>12s} {'n/a':>12s} "
                  f"{'n/a':>7s} {'skip':>7s} {note:>9s}")
            records.append(rec)
            continue

        bu = measure("burst", bs, args.prompt_length, args.max_tokens,
                     args.warmup, args.runs, args.model)
        bu_s = summarize(bu["walls"], bu["total_tokens"])
        # Primary metric = min-of-runs (kernel-benchmark convention: filters
        # scheduler/scheduling interference, reflects kernel capability).
        speedup = (cg_s["ms_per_tok_min"] / bu_s["ms_per_tok_min"]
                   if bu_s["ms_per_tok_min"] else 0.0)
        verdict = "BEATS" if bu_s["ms_per_tok_min"] < cg_s["ms_per_tok_min"] else "loses"
        if bu.get("disabled_reason"):
            note = "fallback"
        elif bu["total_tokens"] < bu["expected_tokens"]:
            note = "eos-trunc"
        elif bs > SAFE_BURST_BATCH:
            note = f"serial×{(bs + SAFE_BURST_BATCH - 1) // SAFE_BURST_BATCH}"
        print(f"{bs:>6d} {cg_s['ms_per_tok_min']:>9.3f} {cg_s['tok_s']:>9.1f} "
              f"{bu_s['ms_per_tok_min']:>12.3f} {bu_s['tok_s']:>12.1f} "
              f"{speedup:>7.2f}x {verdict:>7s} {note:>9s}")
        records.append({
            "batch": bs,
            "cudagraph": {**cg_s, "expected_tokens": cg["expected_tokens"]},
            "burst": {**bu_s, "expected_tokens": bu["expected_tokens"],
                      "disabled_reason": bu.get("disabled_reason")},
            "speedup": speedup,
            "verdict": verdict,
            "note": note,
        })

    out = {
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
        "model": args.model,
        "prompt_length": args.prompt_length,
        "max_tokens": args.max_tokens,
        "warmup": args.warmup,
        "runs": args.runs,
        "safe_burst_batch": SAFE_BURST_BATCH,
        "max_batch_size": MAX_BATCH_SIZE,
        "records": records,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nJSON written to {args.json}")


if __name__ == "__main__":
    main()
