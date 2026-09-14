# Megakernel-vLLM

A fused persistent **Megakernel** decode backend for Qwen3-0.6B on nano-vLLM, and
the measurement tooling that decided how it should be dispatched.

**Headline result (A100-SXM4-80GB, Qwen3-0.6B, BF16, greedy):** at **batch=1**
the Megakernel is **1.31x faster end-to-end than CUDA Graph** (2.241 vs 2.944
ms/token) by eliminating the per-step host/launch round trip — even though its
*pure kernel* time is 2.6x slower. At **batch≥2** CUDA Graph wins by 1–2 orders
of magnitude, so the runtime keeps it for batch>1. Both halves of that split are
backed by on-machine measurements in [`docs/06`](docs/06_实际运行数据.md) and the
decision rule in [`docs/07` §10](docs/07_架构结论_megakernel为何打不过cudagraph.md).

```text
batch == 1  ->  Megakernel persistent/paged kernel   (one cooperative launch, device-side EOS)
batch >= 2  ->  torch path                           (cudagraph in the A/B; eager inside this engine)
```

> **Status: research project, frozen.** The question it set out to answer is
> settled: batch=1 wins on host-overhead elimination; at batch>1 the device side
> has no headroom left, so the fused batch rewrite is not worth attempting
> (measured verdict `DEVICE_BORDERLINE`, `megakernel_could_win=false`).
> Remaining P1/P2 findings are frozen as documented limitations — read
> [`docs/10`](docs/10_交接文档.md) §1 and §5.9 before changing code.

## Why this split (measured, not assumed)

| fact | value | source |
|---|---:|---|
| batch=1 E2E, megakernel vs cudagraph | **1.31x win** (2.241 vs 2.944 ms/token) | `benchmarks.compare`, `tools/bench_nosync.py` |
| batch=1 pure decode kernel | megakernel 1.89 ms vs cudagraph 0.74 ms (**2.6x slower**) | nsys |
| batch=1 host/launch cost per step | **~2.2 ms** — where the 1.31x actually comes from | host/device ledger, `tools/account_decode.py` |
| batch≥2 megakernel-burst per step | 22.7 ms (b2) … 207.9 ms (b32) vs cudagraph 3.3–3.5 ms | `tools/bench_burst_batch.py` |
| attainable HBM bandwidth | 1699 GB/s (83.3% of the 2039 GB/s A100-80GB spec) | `tools/bench_roofline_a100.py` |
| weight floor (tied embeddings) | 1.192 GB/step → 0.585 ms spec, 0.701 ms measured | `src/nanovllm/roofline.py` |
| full device lower bound (probe + LM head) | **1.548 ms/step** > 1.5 ms threshold | `tools/bench_burst_lower_bound.py` + `tools/verdict.py` |
| cudagraph host-free replay | 0.020–0.029 ms/step | same |
| binary verdict | **`DEVICE_BORDERLINE`**, `megakernel_could_win=false` | `tools/verdict.py` → `results/a100_verdict.json` |

The lower-bound probe streams all 28 layers of weights plus 336 grid syncs and
still takes 1.365 ms/step; adding the LM-head read gives 1.548 ms, above the
1.5 ms "worth rewriting" line. Since cudagraph's host-free replay is only
~0.03 ms, a persistent batch>1 kernel has nothing to win on the device side —
so the project stops there on evidence rather than after a rewrite.

## Install

Local (CPU-only) development needs Python 3.10–3.12:

```bash
python -m pip install -e .
PYTHONPATH=src python -m unittest discover -s tests -v   # 287 tests; a few need torch/transformers
python -m compileall src/nanovllm benchmarks tools
```

On a GPU machine you additionally need an NVIDIA driver, `nvcc`, and a
CUDA-enabled PyTorch. `setup_server.sh` creates a `uv` virtualenv from an
*existing* system Python (it deliberately avoids downloading a managed CPython):

```bash
bash setup_server.sh
source .venv/bin/activate
NANOVLLM_MODEL_PATH=~/models/Qwen3-0.6B python -m tools.verify_server_env
```

## Quick start

```python
from nanovllm import LLM, SamplingParams

model = "~/models/Qwen3-0.6B"          # or set NANOVLLM_MODEL_PATH

# Megakernel decode (batch=1 path)
llm = LLM(model, decode_backend="megakernel", max_num_seqs=1, max_model_len=2048)
params = SamplingParams(temperature=0.0, max_tokens=128)
print(llm.generate(["Explain the KV cache in one paragraph."], params)[0]["text"])
```

```python
# torch path (eager or CUDA Graph), the batch≥2 path
llm = LLM(model, decode_backend="torch", enforce_eager=False)
```

The Megakernel contract is deliberately narrow: Qwen3-0.6B, BF16, greedy
sampling, decode-only, `max_model_len <= 2048`. Anything outside it falls back
to the torch path with an explicit reason (and `LLMEngine` warns at startup if
you set `decode_backend="megakernel"` without `max_model_len=2048`).

## Reproducing the measurements

Every GPU-machine command lives in the single [`SERVER_COMMANDS.txt`](SERVER_COMMANDS.txt).
Run it from the checkout and select a phase:

```bash
bash SERVER_COMMANDS.txt                    # correctness gate (default)
RUN_MODE=build     bash SERVER_COMMANDS.txt # clean CUDA-extension rebuild
RUN_MODE=correctness bash SERVER_COMMANDS.txt
RUN_MODE=benchmark bash SERVER_COMMANDS.txt # eager / cudagraph / megakernel A/B
RUN_MODE=nosync    bash SERVER_COMMANDS.txt # batch=1 burst vs cudagraph
RUN_MODE=account   bash SERVER_COMMANDS.txt # per-step host/device ledger
RUN_MODE=roofline  bash SERVER_COMMANDS.txt # bandwidth + device lower-bound probe + verdict
RUN_MODE=profile   bash SERVER_COMMANDS.txt # short nsys capture
```

Paths and GPU selection come from environment variables (`REPO`,
`NANOVLLM_MODEL_PATH`, `PREFERRED_GPUS`, `NANOVLLM_PYPI_INDEX_URL`), so the file
is portable. `tools/verdict.py` turns the probe JSONs into a reproducible
verdict with thresholds locked by `tests/test_verdict.py`; it reports
`NEEDS_DATA` rather than guessing when a probe has not run.

## Correctness policy

**Do not gate on token equality with torch-eager.** WMMA and scalar kernels
differ from torch's fused bf16 kernels by ~1–2 ULP in reduction order, which
cascades into an occasional token flip at greedy decoding. Use the three-way
gate (`tools/correctness_wmma_vs_scalar.py`, mirrored in
`tools/correctness_smoke.py`): **PASS** (identical), **CASCADE** (divergence only
at decode token ≥ 1 — expected), **FAIL** (prefill-token-0, request-count, or
length mismatch — a real bug). See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Repository layout

```text
src/nanovllm/               Runtime: engine/ (scheduler, blocks, model runner),
                            layers/, models/, backends/megakernel/ (CUDA + bindings),
                            roofline.py, rope_config.py
benchmarks/                 Workload definitions and A/B entrypoints
tools/                      Correctness gates, probes, profiling, plotting
tests/                      Focused unittest suite
docs/                       Ordered documentation set (Chinese)
examples/                   Minimal inference example
SERVER_COMMANDS.txt         The single GPU-machine command file (RUN_MODE phases)
```

## Documentation (中文)

按阅读顺序：

- [01 项目说明](docs/01_项目说明.md)：**先读这个**。最终形态、项目定位、集成边界；
- [02 指标说明](docs/02_指标说明.md)：TTFT、TPOT、ITL、E2E、吞吐、显存与正确性口径；
- [03 Megakernel 设计说明](docs/03_Megakernel设计说明.md)：结构、KV cache 契约、算子覆盖、融合边界；
- [04 基准测试方法](docs/04_基准测试方法.md)：实验矩阵、运行步骤、结果保存、Nsight 用法；
- [05 对比基线与公平性](docs/05_对比基线与公平性.md)：对照谁、怎么测、三处必须披露的不对称；
- [06 实际运行数据](docs/06_实际运行数据.md)：已实测 / 待标定 / 待跑 三类数字，含 batch 扫描与 host/device 账本；
- [07 架构结论](docs/07_架构结论_megakernel为何打不过cudagraph.md)：根因拆解 + 外部印证 + 判定规则；
- [08 服务器运行手册](docs/08_服务器运行手册.md)：环境、RUN_MODE 用法、profiling 注意事项；
- [09 集成问题清单](docs/09_集成问题清单.md)：P0/P1/P2 与已验证项；
- [10 交接文档](docs/10_交接文档.md)：工程日志（每轮改了什么、什么被冻结）。

## Credits

Built on [nano-vLLM](https://github.com/GeeeekExplorer/nano-vLLM) (model
runtime, paged KV cache, scheduler) and [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B).
The fused-kernel direction is inspired by the megakernel / persistent-kernel
literature linked in `docs/07` §9.
