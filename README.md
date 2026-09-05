# Megakernel-vLLM

A research implementation that extends nano-vLLM with a Qwen3-0.6B Megakernel
decode backend and reproducible A100/V100 evaluation tooling.

## Current Scope

The repository currently contains three execution paths:

- `torch-eager`: standard nano-vLLM PyTorch execution;
- `torch-cudagraph`: standard nano-vLLM CUDA Graph execution;
- `megakernel`: Qwen3-0.6B BF16 greedy decode on an A100-class GPU, limited to
  one sequence and `max_model_len <= 2048`.

Megakernel requests that are unsupported or run on V100 fall back to the
standard backend and are marked as non-comparable in benchmark reports.

## Repository Layout

```text
nanovllm/                 Core runtime, scheduler, model, and backends
qwen_megakernel/          CUDA extension and fixed-shape Qwen kernel
benchmarks/               Controlled and realistic workload definitions
tests/                    CPU-safe unit tests
tools/                    Correctness, environment, and plotting utilities
docs/                     Kernel design and server runbook
```

## Server Environment

Use Miniconda for Python isolation. The setup script installs Python 3.11, a
CUDA-enabled PyTorch wheel, project dependencies, and the repository in
editable mode. The server must provide `nvidia-smi` and `nvcc`.

```bash
bash setup_server.sh
conda activate nano-vllm-server
python tools/verify_server_env.py
```

Set `NANOVLLM_MODEL_PATH` when the model path should also be checked.

## Quick Start

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/PATH/TO/Qwen3-0.6B", enforce_eager=True)
params = SamplingParams(temperature=0.6, max_tokens=128)
outputs = llm.generate(["Explain KV cache in one paragraph."], params)
print(outputs[0]["text"])
```

For Megakernel decode:

```python
llm = LLM(
    "/PATH/TO/Qwen3-0.6B",
    decode_backend="megakernel",
    max_num_seqs=1,
    max_model_len=2048,
)
params = SamplingParams(temperature=0.0, max_tokens=128)
```

## Correctness and Benchmarking

Run the short correctness gate before long experiments:

```bash
CUDA_LAUNCH_BLOCKING=1 NANOVLLM_CUDA_ARCH=sm_80 \
python tools/correctness_smoke.py /PATH/TO/Qwen3-0.6B \
  --workload-ids short_qa,math_reasoning,code_generation,code_debugging \
  --max-tokens 8 --require-megakernel \
  --json results/a100-correctness.json
```

Run controlled prompt-length and batch experiments:

```bash
NANOVLLM_CUDA_ARCH=sm_80 \
python bench_megakernel.py /PATH/TO/Qwen3-0.6B \
  --suite performance \
  --workloads benchmarks/workloads.json \
  --backends torch-eager,torch-cudagraph,megakernel \
  --batch-sizes 1,8,32 --max-tokens 128 --warmup 2 --runs 10 \
  --json results/a100.json --markdown results/a100.md
```

Run natural-language workloads separately:

```bash
NANOVLLM_CUDA_ARCH=sm_80 \
python bench_megakernel.py /PATH/TO/Qwen3-0.6B \
  --suite realistic --workloads benchmarks/workloads.json \
  --json results/a100-realistic.json --markdown results/a100-realistic.md
```

Generate plots:

```bash
python tools/plot_metrics.py results/a100.json --output-dir results/a100-plots
```

The benchmark records TTFT, TPOT, ITL, E2E latency, prefill/decode/output
throughput, CUDA Event time, peak memory, fallback reasons, and correctness.
Nsight profiling is documented in `docs/server_benchmark.md`.

## Development Checks

```bash
python -m unittest discover -s tests -v
python -m compileall nanovllm qwen_megakernel tools
```

See `docs/megakernel_design.md` for the planned direct paged-KV integration.
