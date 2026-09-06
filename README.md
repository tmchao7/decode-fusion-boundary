# Megakernel-vLLM

A research implementation that extends nano-vLLM with a Qwen3-0.6B Megakernel
decode backend and reproducible A100/V100 evaluation tooling.

## Current Scope

The repository currently contains three execution paths:

- `torch-eager`: standard nano-vLLM PyTorch execution;
- `torch-cudagraph`: standard nano-vLLM CUDA Graph execution;
- `megakernel`: experimental Qwen3-0.6B BF16 greedy decode on an A100-class
  GPU, with paged KV support and an initial batch limit of 32 sequences;
  correctness and performance remain pending server validation.

Megakernel requests that are unsupported or run on V100 fall back to the
standard backend and are marked as non-comparable in benchmark reports.

## Repository Layout

```text
src/nanovllm/             Core runtime, scheduler, model, and public API
src/nanovllm/backends/    Execution backends
  megakernel/             Qwen CUDA extension, wrapper, and kernel sources
benchmarks/               Workload definitions and benchmark entrypoints
tests/                    CPU-safe unit tests
tools/                    Correctness, environment, and plotting utilities
docs/                     Kernel design and server runbook
examples/                 Minimal inference examples
```

## Server Environment

Use `uv` for fast Python environment setup. The setup script creates a
Python 3.11 `.venv`, installs a CUDA-enabled PyTorch wheel and project
dependencies, and installs the repository in editable mode. The server must
provide `nvidia-smi` and `nvcc`.

```bash
bash setup_server.sh
source .venv/bin/activate
python -m tools.verify_server_env
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
python -m tools.correctness_smoke /PATH/TO/Qwen3-0.6B \
  --workload-ids short_qa,math_reasoning,code_generation,code_debugging \
  --max-tokens 8 --require-megakernel \
  --json results/a100-correctness.json
```

Run controlled prompt-length and batch experiments:

```bash
NANOVLLM_CUDA_ARCH=sm_80 \
python -m benchmarks.compare /PATH/TO/Qwen3-0.6B \
  --suite performance \
  --workloads benchmarks/workloads.json \
  --backends torch-eager,torch-cudagraph,megakernel \
  --batch-sizes 1,8,32 --max-tokens 128 --warmup 2 --runs 10 \
  --json results/a100.json --markdown results/a100.md
```

Run natural-language workloads separately:

```bash
NANOVLLM_CUDA_ARCH=sm_80 \
python -m benchmarks.compare /PATH/TO/Qwen3-0.6B \
  --suite realistic --workloads benchmarks/workloads.json \
  --json results/a100-realistic.json --markdown results/a100-realistic.md
```

Generate plots:

```bash
python -m tools.plot_metrics results/a100.json --output-dir results/a100-plots
```

The benchmark records TTFT, TPOT, ITL, E2E latency, prefill/decode/output
throughput, CUDA Event time, peak memory, fallback reasons, and correctness.
Nsight profiling is documented in `docs/server_benchmark.md`.

## Development Checks

```bash
python -m unittest discover -s tests -v
python -m compileall src/nanovllm benchmarks tools
```

## 中文项目说明

- [指标说明](docs/01_指标说明.md)：TTFT、TPOT、ITL、E2E、吞吐、显存和正确性口径；
- [Benchmark 方法](docs/02_Benchmark方法.md)：服务器实验矩阵、运行步骤、结果保存和 Nsight 使用方式；
- [基线与创新](docs/03_基线与创新.md)：baseline、Megakernel 模块、当前已实现能力和后续边界；
- [Megakernel 设计笔记](docs/megakernel_design.md)：paged-KV 直连目标和 kernel 演进路径；
- [服务器运行手册](docs/server_benchmark.md)：A100/V100 环境配置与完整命令；
- [服务器命令清单](SERVER_COMMANDS.txt)：可直接复制执行的服务器命令。
