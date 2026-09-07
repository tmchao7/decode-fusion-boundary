"""Validate the GPU server environment before compiling or benchmarking."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys


REQUIRED_PACKAGES = ("transformers", "triton", "flash_attn", "safetensors")
OPTIONAL_PACKAGES = ("pynvml", "matplotlib")


def _print_nvcc() -> bool:
    executable = shutil.which("nvcc")
    print(f"nvcc: {executable or 'not found'}")
    if executable is None:
        return False
    result = subprocess.run(
        [executable, "--version"], capture_output=True, text=True, check=False
    )
    print(result.stdout.strip() or result.stderr.strip())
    return result.returncode == 0


def _check_model(path: str) -> bool:
    print(f"model path: {path}")
    if not os.path.isdir(path):
        print(f"ERROR: model path does not exist: {path}", file=sys.stderr)
        return False
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(path)
    except Exception as exc:
        print(f"ERROR: failed to load model config: {exc}", file=sys.stderr)
        return False

    expected = {
        "model_type": "qwen3",
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "vocab_size": 151936,
    }
    mismatches = {
        name: (getattr(config, name, None), value)
        for name, value in expected.items()
        if getattr(config, name, None) != value
    }
    if mismatches:
        print(f"ERROR: unsupported Qwen3-0.6B config: {mismatches}", file=sys.stderr)
        return False
    print("model config: Qwen3-0.6B compatible")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-megakernel", action="store_true")
    parser.add_argument("--skip-nvcc", action="store_true")
    parser.add_argument("--model", default=os.environ.get("NANOVLLM_MODEL_PATH"))
    args = parser.parse_args()

    print(f"python: {sys.version.split()[0]}")
    print(f"executable: {sys.executable}")
    checks_ok = True if args.skip_nvcc else _print_nvcc()

    try:
        import torch
    except Exception as exc:
        print(f"ERROR: torch import failed: {exc}", file=sys.stderr)
        return 1

    print(f"torch: {torch.__version__}")
    print(f"torch CUDA runtime: {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("ERROR: torch.cuda.is_available() is False", file=sys.stderr)
        return 1

    print(f"GPU count: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(index)
        capability = torch.cuda.get_device_capability(index)
        print(f"GPU {index}: {name}, capability={capability}")
    capability = torch.cuda.get_device_capability(0)
    actual_arch = f"sm_{capability[0]}{capability[1]}"
    requested_arch = os.environ.get("NANOVLLM_CUDA_ARCH")
    if requested_arch:
        requested_arch = requested_arch if requested_arch.startswith("sm_") else f"sm_{requested_arch}"
        print(f"requested build arch: {requested_arch}")
        if requested_arch != actual_arch:
            print(f"WARNING: requested arch differs from visible GPU arch {actual_arch}")
    bf16_supported = bool(torch.cuda.is_bf16_supported())
    print(f"BF16 supported: {bf16_supported}")
    if args.require_megakernel and (capability[0] < 8 or not bf16_supported):
        print(
            "ERROR: Megakernel requires an A100-class BF16-capable GPU",
            file=sys.stderr,
        )
        checks_ok = False

    try:
        import nanovllm

        print(f"nanovllm import: OK ({nanovllm.__file__})")
    except Exception as exc:
        print(f"ERROR: nanovllm import failed: {exc}", file=sys.stderr)
        checks_ok = False

    missing_required = [
        name for name in REQUIRED_PACKAGES if importlib.util.find_spec(name) is None
    ]
    if missing_required:
        print(
            f"ERROR: missing required Python packages: {', '.join(missing_required)}",
            file=sys.stderr,
        )
        checks_ok = False
    print("Python required dependencies: OK" if not missing_required else "Python required dependencies: FAILED")
    missing_optional = [
        name for name in OPTIONAL_PACKAGES if importlib.util.find_spec(name) is None
    ]
    if missing_optional:
        print(f"Optional packages unavailable: {', '.join(missing_optional)}")

    if args.model:
        if not _check_model(args.model):
            checks_ok = False
    else:
        print("model path: not checked (pass --model or set NANOVLLM_MODEL_PATH)")

    print("server environment: OK" if checks_ok else "server environment: FAILED")
    return 0 if checks_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
