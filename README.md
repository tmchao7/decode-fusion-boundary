# Megakernel-vLLM

nano-vLLM extended with a Megakernel decode backend for Qwen3-0.6B on A100.
The runtime is a **hybrid dispatch**: batch=1 uses a fused cooperative
Megakernel (device-side EOS, no per-step host launch); batch≥2 falls back to
standard CUDA Graph (full SM + tensor cores, no padding). This split is the
measured conclusion, not a placeholder — see `docs/07` §10.

## Architecture

```
batch == 1  →  Megakernel persistent/paged kernel  (1.31x faster than cudagraph)
batch >= 2  →  torch-cudagraph fallback             (megakernel is 15–135x the device floor)
```

Three backends are available: `torch-eager` (reference), `torch-cudagraph`
(production), and `megakernel` (experimental, batch=1 only by default).
Unsupported requests fall back automatically and are marked in benchmark
output.

## Why This Split (measured on A100)

| fact | value | source |
|---|---:|---|
| attainable HBM bandwidth | 1728 GB/s (85% spec) | `bench_roofline_a100.py` |
| weight floor (tied_embedding=True) | 1.192 GB → 0.77 ms/step | `roofline.py` |
| device lower bound (probe + LM head) | 1.543 ms/step | `bench_burst_lower_bound.py` + verdict |
| cudagraph host-free replay | 0.020 ms/step | `bench_burst_lower_bound.py` |
| host overlay per step | 0.014 ms (not recoverable) | same |
| megakernel batch=1 vs cudagraph | **1.31x E2E win** | `benchmarks.compare` |
| megakernel batch=10 vs cudagraph | 48x the device floor | `account_decode.py` |

The device floor (1.543 ms) is **above** the 1.5 ms recoverable threshold, and
cudagraph's host-free replay (0.020 ms) is far below it — so there is no
device-side or host-side headroom for a batch>1 megakernel to win. The verdict
is `DEVICE_BORDERLINE`; see `docs/07` §10.4 and `docs/06` §7.

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

Use `uv` for fast Python environment setup. The setup script reuses an
existing server Python 3.10–3.12, explicitly avoids downloading a managed
CPython, installs a CUDA-enabled PyTorch wheel and project dependencies, and
installs the repository in editable mode. The server must provide `nvidia-smi`
and `nvcc`.

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
  --max-tokens 8 --require-megakernel --allow-cascade \
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
Nsight profiling and the server workflow are documented in `docs/08_服务器运行手册.md`.

## Development Checks

```bash
python -m unittest discover -s tests -v
python -m compileall src/nanovllm benchmarks tools
```

## 中文项目说明

按阅读顺序：

- [01 项目说明](docs/01_项目说明.md)：**先读这个**。最终形态（batch=1 megakernel / batch>1 cudagraph）、项目定位、集成边界；
- [02 指标说明](docs/02_指标说明.md)：TTFT、TPOT、ITL、E2E、吞吐、显存和正确性口径；
- [03 Megakernel 设计说明](docs/03_Megakernel设计说明.md)：结构、KV cache 契约、算子覆盖、融合边界、已知边界；
- [04 基准测试方法](docs/04_基准测试方法.md)：实验矩阵、运行步骤、结果保存、Nsight 用法；
- [05 对比基线与公平性](docs/05_对比基线与公平性.md)：对照谁、怎么测、三处必须披露的不对称；
- [06 实际运行数据](docs/06_实际运行数据.md)：已实测 / 待标定 / 待跑 三类数字，含 batch 扫描与 host/device 账本；
- [07 架构结论：megakernel 为何打不过 cudagraph](docs/07_架构结论_megakernel为何打不过cudagraph.md)：机制拆解 + 外部印证 + 判定规则；
- [08 服务器运行手册](docs/08_服务器运行手册.md)：A100/V100 环境、RUN_MODE 用法、profiling 注意事项；
- [09 集成问题清单](docs/09_集成问题清单.md)：P0/P1/P2 待办与已验证项；
- [10 交接文档](docs/10_交接文档.md)：**下一个 agent 从这里开始**（做了什么、还剩什么、怀疑什么）；
- [服务器命令清单](SERVER_COMMANDS.txt)：可直接复制执行的服务器命令。

## Host/Device Accounting and the High-Batch Verdict

`tools/account_decode.py` splits each measured decode step into wall time, GPU
time and host residual (and reads both the `benchmarks.compare` and
`bench_burst_batch` JSON schemas). `tools/verdict.py` turns the probe outputs
into a reproducible verdict — `DEVICE_BORDERLINE`, `DEVICE_AT_FLOOR`,
`DEVICE_UNREACHABLE`, `DEVICE_RECOVERABLE`, `NEEDS_CALIBRATION` or
`NEEDS_DATA` — with thresholds locked by `tests/test_verdict.py`. The verdict
includes an LM head stream correction (the probe skips `lm_head`, so the full
device floor = probe + 0.180 ms at 1728 GB/s).

```bash
python -m nanovllm.roofline --batch-sizes 1,8,32 --seq-len 160 --check-kernel
python -m tools.account_decode results/a100_account.json --seq-len 160
python -m tools.verdict --results-dir results
```

The verdict tool deliberately reports `NEEDS_DATA` rather than guessing when a
probe has not run, and `DEVICE_AT_FLOOR` when the measured device time already
sits near the floor (probe fast does not imply a win).
