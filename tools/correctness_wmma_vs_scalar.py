"""Compare WMMA batch kernel vs scalar megakernel token-by-token.

The scalar megakernel itself diverges from torch-eager due to ~1-ulp
bf16 reduction-order differences (documented in AGENTS.md).  The correct
gate for the WMMA conversion is therefore: WMMA must match the scalar
megakernel, not torch-eager.

Each (batch_size, backend) combination runs in a fresh process so a
CUDA fault cannot poison the other run.
"""

from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

BACKEND_CONFIGS = {
    "scalar": {
        "decode_backend": "megakernel",
        "enforce_eager": False,
        "megakernel_batch_kernel": "scalar",
        "megakernel_burst": False,
    },
    "wmma": {
        "decode_backend": "megakernel",
        "enforce_eager": False,
        "megakernel_batch_kernel": "wmma",
        "megakernel_burst": False,
    },
    "burst": {
        "decode_backend": "megakernel",
        "enforce_eager": False,
        "megakernel_batch_kernel": "wmma",
        "megakernel_burst": True,
    },
}


def prompt_ids(length: int, seed: int, index: int = 0) -> list[int]:
    import random
    rng = random.Random(seed + index * 104729 + length * 1009)
    return [rng.randrange(1, 10000) for _ in range(length)]


def _worker(connection, backend: str, model: str, batch_size: int,
            prompt_len: int, max_tokens: int):
    try:
        import torch
        from nanovllm import LLM, SamplingParams

        prompts = [prompt_ids(prompt_len, i * 17 + 3) for i in range(batch_size)]
        llm = LLM(model, max_num_seqs=max(batch_size, 8), max_model_len=2048,
                  **BACKEND_CONFIGS[backend])
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        result = llm.generate(prompts, sp, use_tqdm=False)
        torch.cuda.synchronize()
        token_ids = [item["token_ids"] for item in result]
        llm.exit()
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        connection.send({"ok": True, "token_ids": token_ids})
    except BaseException:
        connection.send({"ok": False, "error": traceback.format_exc()})
    finally:
        connection.close()


def measure(backend: str, model: str, batch_size: int, prompt_len: int,
            max_tokens: int) -> list[list[int]]:
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_worker, args=(child, backend, model, batch_size,
                                              prompt_len, max_tokens))
    proc.start()
    proc.join(timeout=600)
    payload = parent.recv() if parent.poll() else None
    if payload is None or not payload["ok"]:
        raise RuntimeError(f"{backend} b={batch_size} failed: "
                           f"{payload and payload['error']}")
    return payload["token_ids"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--prompt-lengths", type=str, default="32,128")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,5")
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()

    prompt_lengths = [int(x) for x in args.prompt_lengths.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    all_pass = True
    for plen in prompt_lengths:
        for bs in batch_sizes:
            scalar = measure("scalar", args.model, bs, plen, args.max_tokens)
            wmma = measure("wmma", args.model, bs, plen, args.max_tokens)
            match = scalar == wmma
            status = "PASS" if match else "FAIL"
            if not match:
                all_pass = False
                # Find first mismatch
                for i, (s, w) in enumerate(zip(scalar, wmma)):
                    for j, (st, wt) in enumerate(zip(s, w)):
                        if st != wt:
                            print(f"  first mismatch: seq={i} token={j} "
                                  f"scalar={st} wmma={wt}")
                            break
                    else:
                        if len(s) != len(w):
                            print(f"  length mismatch: seq={i} "
                                  f"scalar={len(s)} wmma={len(w)}")
            print(f"p{plen}_b{bs}: {status}  "
                  f"scalar={scalar}  wmma={wmma}")

    print()
    if all_pass:
        print("All WMMA vs scalar comparisons: PASS")
    else:
        print("Some comparisons FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
