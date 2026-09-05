"""Validate the server environment before compiling or benchmarking."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys


def main() -> int:
    print(f"python: {sys.version.split()[0]}")
    print(f"executable: {sys.executable}")
    print(f"nvcc: {shutil.which('nvcc') or 'not found'}")
    if shutil.which("nvcc"):
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, check=False
        )
        print(result.stdout.strip() or result.stderr.strip())

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
        print(
            f"GPU {index}: {torch.cuda.get_device_name(index)}, "
            f"capability={torch.cuda.get_device_capability(index)}"
        )

    required = [
        "transformers",
        "triton",
        "flash_attn",
        "safetensors",
        "pynvml",
        "matplotlib",
    ]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        print(f"ERROR: missing Python packages: {', '.join(missing)}", file=sys.stderr)
        return 1
    print("Python dependencies: OK")

    model_path = os.environ.get("NANOVLLM_MODEL_PATH")
    if model_path:
        print(f"model path: {model_path}")
        if not os.path.isdir(model_path):
            print(f"ERROR: model path does not exist: {model_path}", file=sys.stderr)
            return 1
    else:
        print("model path: not checked (set NANOVLLM_MODEL_PATH to validate one)")
    print("server environment: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
