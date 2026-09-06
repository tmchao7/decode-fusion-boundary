# A100/V100 Server Benchmark Runbook

All CUDA commands in this document are intended for the GPU server, not a development Mac.

## 1. Environment

The recommended server setup uses `uv` for Python isolation and fast dependency installation. The script reuses an existing server Python 3.10–3.12 and disables uv-managed CPython downloads. The NVIDIA driver, `nvidia-smi`, CUDA toolkit, and `nvcc` remain system/module dependencies. PyTorch's wheel supplies the CUDA runtime used by Python; it does not replace the driver or the compiler required by the Megakernel.

From the repository root:

```bash
bash setup_server.sh
source .venv/bin/activate
```

The script creates `.venv` with an existing compatible Python, installs the repository's CUDA PyTorch wheel and Python requirements, installs the project in editable mode, and runs a basic environment check. Ordinary Python packages, PyTorch, and `nvidia-*` CUDA dependencies prefer the Huawei Cloud PyPI mirror; the official PyTorch CUDA index remains only as a fallback for packages missing from the mirror. To use another PyTorch CUDA wheel index:

```bash
NANOVLLM_TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu126 \
bash setup_server.sh
```

To change only the ordinary Python package mirror:

```bash
NANOVLLM_PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
bash setup_server.sh
```

Validate a local Qwen3-0.6B model:

```bash
NANOVLLM_MODEL_PATH=/path/to/Qwen3-0.6B \
python -m tools.verify_server_env --require-megakernel
```

The setup script uses `UV_HTTP_TIMEOUT=300` and `UV_LINK_MODE=copy`; the latter is appropriate when uv's cache and the repository are on different filesystems. If uv reports `invalid peer certificate: UnknownIssuer`, do not disable TLS verification. Set `UV_SYSTEM_CERTS=true` when the enterprise CA is installed in the system trust store, or set `SSL_CERT_FILE=/path/to/company-ca-bundle.pem` to the PEM bundle supplied by the server administrator. If `nvcc` is unavailable, load the server's CUDA module before running the setup script:

```bash
module avail cuda
module load cuda/<server-version>
```

Do not commit model weights, credentials, benchmark results, or profiler reports.

## 2. Static, build, and smoke checks

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall src/nanovllm tools
```

Compile the extension separately before running a correctness test:

```bash
NANOVLLM_CUDA_ARCH=sm_80 \
python -m nanovllm.backends.megakernel.build
```

Run a short correctness check before performance testing:

```bash
CUDA_LAUNCH_BLOCKING=1 \
TORCH_SHOW_CPP_STACKTRACES=1 \
NANOVLLM_CUDA_ARCH=sm_80 \
python -m tools.correctness_smoke /path/to/Qwen3-0.6B \
  --workload-ids short_qa,math_reasoning,code_generation,code_debugging \
  --max-tokens 8 \
  --require-megakernel \
  --json results/a100_correctness.json
```

For V100, use `sm_70` and omit `--require-megakernel`; a fallback is expected until the FP16 path is validated.

## 3. Request-level benchmark

```bash
NANOVLLM_CUDA_ARCH=sm_80 \
python -m benchmarks.compare /path/to/Qwen3-0.6B \
  --backends torch-eager,torch-cudagraph,megakernel \
  --prompt-lengths 32,128,512 \
  --batch-sizes 1,8,32 \
  --max-tokens 128 \
  --warmup 2 \
  --runs 10 \
  --json results/a100.json \
  --markdown results/a100.md
```

Generate plots from the JSON:

```bash
python -m tools.plot_metrics results/a100.json --output-dir results/a100-plots
```

## 4. Nsight profiling

Profile a short run only; do not use profiling output as the latency benchmark.

```bash
NANOVLLM_CUDA_ARCH=sm_80 \
nsys profile --trace=cuda,nvtx \
  --output=results/a100_nsys \
  python -m benchmarks.compare /path/to/Qwen3-0.6B \
    --backends megakernel --prompt-lengths 32 --max-tokens 32 --warmup 1 --runs 1

NANOVLLM_CUDA_ARCH=sm_80 \
ncu --set full --target-processes all \
  --export results/a100_ncu \
  python -m benchmarks.compare /path/to/Qwen3-0.6B \
    --backends megakernel --prompt-lengths 32 --max-tokens 32 --warmup 1 --runs 1
```

Save the JSON/Markdown results and the profiler report names together with GPU model, driver, CUDA, PyTorch, git commit, and whether the tree was dirty.
