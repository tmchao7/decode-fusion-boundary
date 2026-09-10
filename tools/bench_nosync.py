"""Reproducible megakernel no-sync vs torch-cudagraph decode benchmark.

Each path runs in a fresh process so the 72 GB paged KV pool of one backend
does not starve the next. Prints end-to-end wall time per generated token.
"""

from __future__ import annotations

import argparse
import json
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


def _worker(connection, path: str, prompt, prompt_len: int, N: int):
    try:
        import gc
        import torch
        from nanovllm import LLM, SamplingParams

        bt = torch.tensor([0], dtype=torch.int32, device="cuda")
        if path == "mega-sync":
            llm = LLM(MODEL, decode_backend="megakernel", enforce_eager=False,
                      max_num_seqs=1, max_model_len=2048)
            llm.model_runner.megakernel.ensure_ready()
            dec = llm.model_runner.megakernel.decoder
            cur = int(prompt[-1]); pos = prompt_len
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(N):
                cur = dec.step(cur, position=pos, block_table=bt); pos += 1
            torch.cuda.synchronize()
            value = (time.perf_counter() - t0) * 1000 / N
            llm.exit(); del llm; del dec
        elif path == "mega-nosync":
            llm = LLM(MODEL, decode_backend="megakernel", enforce_eager=False,
                      max_num_seqs=1, max_model_len=2048)
            llm.model_runner.megakernel.ensure_ready()
            dec = llm.model_runner.megakernel.decoder
            dec.generate_paged(int(prompt[-1]), prompt_len, [0], 100)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            dec.generate_paged(int(prompt[-1]), prompt_len, [0], N)
            torch.cuda.synchronize()
            value = (time.perf_counter() - t0) * 1000 / N
            llm.exit(); del llm; del dec
        elif path == "mega-burst":
            # End-to-end through the engine with burst path enabled.
            # The engine's step() detects burst eligibility (WMMA + greedy +
            # no waiting) and dispatches to schedule_burst → generate_paged_batch
            # → postprocess_burst automatically.
            llm = LLM(MODEL, decode_backend="megakernel", enforce_eager=False,
                      max_num_seqs=1, max_model_len=2048,
                      megakernel_batch_kernel="wmma", megakernel_burst=True)
            sp = SamplingParams(temperature=0.0, max_tokens=10)
            warm = llm.add_request(prompt, sp)
            while not llm.is_finished():
                llm.step()
            torch.cuda.synchronize()
            seq = llm.add_request(prompt, SamplingParams(temperature=0.0, max_tokens=N))
            t0 = time.perf_counter()
            while not llm.is_finished():
                llm.step()
            torch.cuda.synchronize()
            value = (time.perf_counter() - t0) * 1000 / N
            llm.exit(); del llm
        else:  # torch-cudagraph
            llm = LLM(MODEL, decode_backend="torch", enforce_eager=False,
                      max_num_seqs=1, max_model_len=2048)
            warm = llm.add_request(prompt, SamplingParams(temperature=0.0, max_tokens=10))
            while not llm.is_finished():
                llm.step()
            torch.cuda.synchronize()
            seq = llm.add_request(prompt, SamplingParams(temperature=0.0, max_tokens=N))
            t0 = time.perf_counter()
            while not llm.is_finished():
                llm.step()
            torch.cuda.synchronize()
            value = (time.perf_counter() - t0) * 1000 / N
            llm.exit(); del llm
        gc.collect()
        connection.send({"ok": True, "value": value})
    except BaseException as exc:  # noqa: BLE001
        import traceback
        connection.send({"ok": False, "error": traceback.format_exc()})
    finally:
        connection.close()


def measure(path: str, prompt, prompt_len: int, N: int) -> float:
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_worker, args=(child, path, prompt, prompt_len, N))
    proc.start(); proc.join()
    payload = parent.recv() if parent.poll() else None
    if payload is None or not payload["ok"]:
        raise RuntimeError(f"{path} failed: {payload and payload['error']}")
    return payload["value"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-length", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    prompt = prompt_ids(args.prompt_length, args.seed)
    N = args.tokens
    results = {}
    for path in ("mega-sync", "mega-nosync", "mega-burst", "torch-cudagraph"):
        print(f"measuring {path} ...", flush=True)
        results[path] = measure(path, prompt, args.prompt_length, N)

    print(f"\ntokens={N} prompt_len={args.prompt_length} seed={args.seed}")
    print(f"{'megakernel sync (.item())':34s} {results['mega-sync']:8.2f} ms/token")
    print(f"{'megakernel no-sync':34s} {results['mega-nosync']:8.2f} ms/token")
    print(f"{'megakernel burst (engine E2E)':34s} {results['mega-burst']:8.2f} ms/token")
    print(f"{'torch-cudagraph':34s} {results['torch-cudagraph']:8.2f} ms/token")
    print(f"no-sync vs cudagraph: {results['torch-cudagraph'] / results['mega-nosync']:.2f}x "
          f"({'BEATS' if results['mega-nosync'] < results['torch-cudagraph'] else 'ties/loses'})")
    print(f"burst vs cudagraph:   {results['torch-cudagraph'] / results['mega-burst']:.2f}x "
          f"({'BEATS' if results['mega-burst'] < results['torch-cudagraph'] else 'ties/loses'})")
    print(f"no-sync vs sync:     {results['mega-sync'] / results['mega-nosync']:.2f}x faster")


if __name__ == "__main__":
    main()
