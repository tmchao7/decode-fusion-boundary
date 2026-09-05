<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

For a normal Python installation:

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

For the A100/V100 server workflow, use the repository setup script instead:

```bash
bash setup_server.sh
conda activate nano-vllm-server
python tools/verify_server_env.py
```

The script creates a Python 3.11 Miniconda environment, installs the CUDA
PyTorch wheel, installs `requirements.server.txt`, and installs this repository
in editable mode. It requires `nvidia-smi` and `nvcc` to already be available
from the server driver/toolkit or module system.

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

### Optional Qwen Megakernel Decode Backend

The experimental backend targets Qwen3-0.6B BF16 greedy decode on an A100. It
keeps prefill and unsupported requests on the standard path, and falls back
when the model, batch, sampling mode, or GPU is incompatible:

```python
llm = LLM(
    "/YOUR/QWEN3-0.6B/PATH",
    decode_backend="megakernel",
    max_model_len=2048,
    max_num_seqs=1,
)
sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
```

Build the optional CUDA extension on the target server, not on macOS:

```bash
cd qwen_megakernel && NANOVLLM_CUDA_ARCH=sm_80 python -m qwen_megakernel.bench
```

The first integration milestone uses a private contiguous KV cache and
supports one sequence at a time. Batch/paged-KV support remains a follow-up
optimization; use the default backend for sampling or multi-sequence runs.

Currently runnable paths:

- `torch-eager`: standard nano-vLLM PyTorch execution;
- `torch-cudagraph`: standard nano-vLLM CUDA Graph execution when
  `enforce_eager=False`;
- `megakernel`: Qwen3-0.6B BF16, A100-class GPU, greedy, single-sequence
  decode. On V100 or an unsupported request it records fallback and uses the
  standard backend instead of claiming Megakernel performance.

For an end-to-end A/B comparison of eager PyTorch, CUDA Graph, and Megakernel:

```bash
NANOVLLM_CUDA_ARCH=sm_80 python bench_megakernel.py /PATH/TO/Qwen3-0.6B \
  --backends torch-eager,torch-cudagraph,megakernel \
  --prompt-lengths 32,128,512 \
  --max-tokens 128 --warmup 2 --runs 10 \
  --json results/a100.json --markdown results/a100.md
```

The benchmark records TTFT, TPOT, ITL, E2E latency, prefill/decode/output
throughput, peak CUDA memory, optional NVML telemetry, backend fallback reasons,
and greedy-token correctness against the eager baseline. It also emits NVTX
ranges named `nano_vllm_request`, `nano_vllm_prefill`,
`nano_vllm_decode_step`, and `megakernel_decode`.

Use Nsight Systems/Compute separately on the server to collect kernel launches,
memcpy operations, DRAM/L2 throughput, achieved occupancy, and SM utilization:

Before a long benchmark, run the short correctness gate and generate plots from
saved JSON results:

```bash
CUDA_LAUNCH_BLOCKING=1 NANOVLLM_CUDA_ARCH=sm_80 \
python tools/correctness_smoke.py /PATH/TO/Qwen3-0.6B \
  --prompt-lengths 8,32,128 --max-tokens 8 \
  --require-megakernel --json results/a100_correctness.json
python tools/plot_metrics.py results/a100.json --output-dir results/a100-plots
```

Design notes and the complete server procedure are documented in
`docs/megakernel_design.md` and `docs/server_benchmark.md`.


```bash
nsys profile --trace=cuda,nvtx --output=results/a100_megakernel \
  python bench_megakernel.py /PATH/TO/Qwen3-0.6B --backends megakernel
ncu --set full --target-processes all --export results/a100_megakernel \
  python bench_megakernel.py /PATH/TO/Qwen3-0.6B --backends megakernel
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)