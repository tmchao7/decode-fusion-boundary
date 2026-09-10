"""Multi-batch megakernel burst vs torch-cudagraph benchmark.

Unlike benchmarks.compare (which uses ignore_eos=True and disables burst),
this uses ignore_eos=False so the burst path actually triggers. Each batch
size runs in a fresh process. Prompts are synthetic token IDs chosen to
avoid early EOS in 128 tokens.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

MODEL = "/home/t50063365/models/Qwen3-0.6B"


def prompt_ids(length: int, seed: int) -> list[int]:
    import random
    rng = random.Random(seed + length * 1009)
    return [rng.randrange(1, 10000) for _ in range(length)]


def _worker(connection, backend: str, batch_size: int, prompt_len: int, max_tokens: int):
    try:
        import gc
        import torch
        from nanovllm import LLM, SamplingParams

        prompts = [prompt_ids(prompt_len, i * 17 + 3) for i in range(batch_size)]
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)

        if backend == "burst":
            llm = LLM(MODEL, decode_backend="megakernel", enforce_eager=False,
                      max_num_seqs=max(batch_size, 8), max_model_len=2048,
                      megakernel_batch_kernel="wmma", megakernel_burst=True)
        else:
            llm = LLM(MODEL, decode_backend="torch", enforce_eager=False,
                      max_num_seqs=max(batch_size, 8), max_model_len=2048)

        # warmup
        warm_sp = SamplingParams(temperature=0.0, max_tokens=10)
        llm.generate(prompts, warm_sp, use_tqdm=False)
        torch.cuda.synchronize()

        # measured run
        t0 = time.perf_counter()
        llm.generate(prompts, sp, use_tqdm=False)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        total_tokens = batch_size * max_tokens
        ms_per_tok = wall * 1000 / total_tokens
        llm.exit()
        del llm
        gc.collect()
        connection.send({"ok": True, "value": ms_per_tok, "wall": wall})
    except BaseException as exc:
        import traceback
        connection.send({"ok": False, "error": traceback.format_exc()})
    finally:
        connection.close()


def measure(backend: str, batch_size: int, prompt_len: int, max_tokens: int) -> float:
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_worker, args=(child, backend, batch_size, prompt_len, max_tokens))
    proc.start()
    proc.join(timeout=600)
    payload = parent.recv() if parent.poll() else None
    if payload is None or not payload["ok"]:
        raise RuntimeError(f"{backend} b={batch_size} failed: {payload and payload['error']}")
    return payload["value"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-length", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    print(f"prompt_len={args.prompt_length} max_tokens={args.max_tokens}")
    print(f"{'batch':>6s} {'cudagraph':>12s} {'burst':>12s} {'speedup':>8s} {'verdict':>8s}")
    for bs in (1, 2, 4, 5):
        cg = measure("cudagraph", bs, args.prompt_length, args.max_tokens)
        bu = measure("burst", bs, args.prompt_length, args.max_tokens)
        speedup = cg / bu
        verdict = "BEATS" if bu < cg else "loses"
        print(f"{bs:>6d} {cg:>10.2f}ms {bu:>10.2f}ms {speedup:>7.2f}x {verdict:>8s}")


if __name__ == "__main__":
    main()
