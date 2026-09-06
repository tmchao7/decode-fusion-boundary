# A100/V100 Server Benchmark Runbook

All CUDA commands in this document are intended for the GPU server, not a development Mac.

## 1. Environment

The recommended setup is Miniconda for Python isolation, the official CUDA PyTorch wheel for the runtime, and the server's NVIDIA driver/CUDA toolkit for `nvidia-smi` and `nvcc`. The PyTorch wheel does not replace the driver, and compiling the Megakernel extension still requires `nvcc`.

From the repository root:

```bash
bash setup_server.sh
conda activate nano-vllm-server
```

The script uses Python 3.11, PyTorch 2.8.0 with the CUDA 12.8 wheel, and the pinned project runtime requirements. Override the defaults when the server requires another supported wheel:

```bash
NANOVLLM_TORCH_VERSION=2.8.0 \
NANOVLLM_TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu126 \
bash setup_server.sh
```

Check a specific local model too:

```bash
NANOVLLM_MODEL_PATH=/path/to/Qwen3-0.6B \
python -m tools.verify_server_env --require-megakernel
```

Use a local Qwen3-0.6B model directory. Do not commit model weights, Hugging Face tokens, profiler reports, or generated result files.

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
