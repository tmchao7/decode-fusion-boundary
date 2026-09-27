# Megakernel-vLLM

**Question.** In LLM decode, how do the returns from *whole-model kernel fusion*
compare with those from *CUDA Graph* as batch size grows — and where does each one
stop paying?

**Answer (measured, on A100-SXM4-80GB).** Kernel fusion wins at batch 1 and only
there, and the win is not compute: it eliminates ~2.2 ms/step of host work while
being **2.6x slower on the device**. At batch ≥ 2 the device gap dominates and the
megakernel loses by up to 133x. An independent 2026 result
([AutoMegaKernel](https://arxiv.org/abs/2606.09682)) localizes the same A100/H100
loss to the cross-SM-sync bottleneck, while the same design wins on
inference-class parts (L4/L40S/A10G/RTX 5090, 1.08–1.33x).

So this repo is **not** a "faster kernel" claim. It is an instrument for locating
the fusion/CUDA-Graph boundary, with the physics (roofline + measured achievable
bandwidth) and the negative results reported alongside the wins.

```text
decode_backend="torch", enforce_eager=False -> torch-cudagraph  (the strong baseline)
decode_backend="megakernel"                 -> task-graph Megakernel (one cooperative launch/step)
```

The previous `kernel.cu` persistent/WMMA implementation is preserved on the
archival branch `codex/archive/frozen-kernel`; it is intentionally not part of
the mainline API, build, scheduler, or benchmark surface. Its numbers are the
evidence base below and are labelled as such.

**文档入口：[docs/README.md](docs/README.md)** —— 21 份文档按"现行 / 历史证据 /
研究资料"三态分类，含按目的选最短路径、关键命令、以及现行 vs 归档的对照表。

## Mainline contract

The active Megakernel is deliberately narrow:

- Qwen3-0.6B geometry: hidden 1024, 28 layers, 16 Q heads, 8 KV heads;
- BF16 weights, greedy sampling, decode-only;
- single-card tensor parallelism;
- `max_model_len <= 2048`;
- batch sizes `1..8` (`TG_MAX_BATCH`);
- one whole-batch cooperative launch per decode step;
- prefill remains on the torch model path; it already uses FlashAttention varlen.
- scheduler-level `max_prefill_tokens_per_step` and `prefill_decode_ratio` control mixed prefill/decode fairness.

The kernel reuses nano-vLLM's paged KV cache. It does not allocate a second KV
cache or replay the prompt. Unsupported requests fall back to the standard torch
path with an explicit reason.

## Results and applicable range

Archived `kernel.cu` line, Qwen3-0.6B, greedy, A100-SXM4-80GB (108 SMs); each row
is end-to-end wall time per token next to the pure-device time per step, so the
host/launch contribution is visible rather than implied.

| batch | backend | E2E ms/tok | device ms/step | host residual | note |
|---|---|---|---|---|---|
| 1 | torch-eager | — | — | — | host-bound floor |
| 1 | torch-cudagraph | 2.944 | **0.74** | ~2.2 ms/step | 1611 GB/s = **95% of measured-achievable 1699 GB/s** |
| 1 | megakernel | **2.241** | 1.89 | ~0.35 ms/step | **1.31x E2E** from launch elimination, 2.6x slower on device |
| 8 | torch-cudagraph | — | 0.77 | — | |
| 8 | megakernel | — | **101.8** | — | **133x slower on device** |

**Physics.** One decode step must stream 1.192 GB of weights
(`tied_embedding=True`): 0.585 ms at the 2039 GB/s spec, 0.702 ms at the measured
1699 GB/s. The CUDA-Graph path at 0.74 ms/step is therefore *at the wall*, and no
fusion of the same arithmetic can beat it — which is exactly what the device
column shows. Applicability: **batch 1, short context, this model class**. The
crossover is not a gentle curve: it is a cliff at batch 1→2.

The task-graph `megakernel` on this branch is a rewrite of that design; its own
numbers are **not measured yet** (`RUN_MODE=taskgraph-baseline`), and until they
are, every figure above belongs to the archived line. The pre-registered
prediction it will be tested against is in
[`docs/18_成本模型.md`](docs/18_成本模型.md).

## What did not work (negative results)

Kept because they are the most useful part of the artifact:

1. **Full-grid barriers are not the bottleneck.** The original hypothesis was
   ~336 `grid.sync()` calls costing 7–8 ms. Measured: **0.547 ms** (1.63 µs each).
   The attribution was wrong, and the measurement that killed it is in
   `docs/07` §3.4.
2. **"Put the GEMMs on tensor cores and it flips" — it did not.** After all FC
   GEMMs moved to WMMA, batch 8 remained **133x** slower: the loss is the wave
   structure and per-SM memory-level parallelism, not the tensor-core coverage.
3. **Deferring work to more SMs does not shorten a wave.** Spreading an
   instruction's tiles across more SMs leaves warps idle; the wave costs
   `(K/16) x k_step_time` regardless (`docs/12` §1.4).
4. **A100 is the wrong target for a full-fusion megakernel**, and an independent
   paper says so too (`docs/07` §9.10).

## Profiling notes

`nsys` works and is how the device/host split above was obtained. **`ncu` does
not**: kernel replay conflicts with `cudaLaunchCooperativeKernel` and hangs the
GPU driver (observed; `nvidia-smi` blocks too). Use
`RUN_MODE=bandwidth` to calibrate achievable bandwidth, `tools/parse_nsys_kernel.py`
for per-kernel decomposition, and `tools/account_decode.py` for the four-way
accounting — see [`docs/04_基准测试方法.md`](docs/04_基准测试方法.md).

## Methodology

- **Correctness gate**: greedy first token must match torch-eager; BF16
  reduction-order divergence at later tokens is classified `CASCADE`, and
  anything else is `FAIL`. A number that does not pass the gate is never reported.
- **Statistics**: timed modes run 5 measured runs and report count/mean/stddev/
  CoV/min/max/p5/p50/p95/p99 plus a bootstrap 95% CI, per
  [docs/02](docs/02_指标说明.md). The archived tables used `runs=3` +
  min-of-runs and are only comparable within themselves.
- **Schedule gate**: `python -m tools.verify_schedule` proves the instruction
  stream cannot deadlock and rejects adversarially mutated schedules (0 false
  accepts over a 282,720-mutant population) before anything is built on the GPU.
- **Numbers come from code, not from reading logs**: byte accounting, roofline
  bounds and the `DEVICE_BORDERLINE` verdict are computed and unit-tested.

## Threats to validity

- Attention and the LM head remained scalar in the archived line, so its device
  time is an upper bound for the megakernel side; the CUDA-Graph side uses
  FlashAttention/cuBLAS. The comparison is therefore *conservative for* the
  megakernel, and stated as such.
- Single card, no NCCL; MoE, expert parallelism and all-to-all are out of scope,
  which removes the cost centre that dominates the multi-GPU literature.
- The cost model is calibrated at batch 1 and does not represent the batch ≥ 2
  collapse; it says so rather than extrapolating.
- The archived and task-graph lines are different kernels. Never mix their
  numbers.

## Install

Local development requires Python 3.10–3.12:

```bash
python -m pip install -e .
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall src/nanovllm benchmarks tools tests
```

On a machine without CUDA, install only the CPU-test dependencies: the
engine/scheduler/contract suites need `numpy` and `xxhash`, and the task-graph
CUDA modules skip themselves instead of erroring, so discovery stays green.

```bash
python -m pip install numpy xxhash
PYTHONPATH=src python -m unittest discover -s tests
```

The same suites also run without `transformers` or a model checkpoint: the
CPU-only fixtures build a real `Config` with `transformers` replaced in
`sys.modules` (`tests/engine_fixtures.py`), so a bad runtime parameter still
fails with the production `ValueError`.

A GPU machine additionally needs an NVIDIA driver, `nvcc`, and CUDA-enabled
PyTorch. `setup_server.sh` creates the server environment from an existing
system Python:

```bash
bash setup_server.sh
source .venv/bin/activate
NANOVLLM_MODEL_PATH=~/models/Qwen3-0.6B python -m tools.verify_server_env --require-megakernel
```

## Quick start

```python
from nanovllm import LLM, SamplingParams

model = "~/models/Qwen3-0.6B"
llm = LLM(
    model,
    decode_backend="megakernel",
    max_num_seqs=8,
    max_model_len=2048,
)
params = SamplingParams(temperature=0.0, max_tokens=128)
print(llm.generate(["Explain the KV cache in one paragraph."], params)[0]["text"])
```

For the torch CUDA-Graph control:

```python
llm = LLM(model, decode_backend="torch", enforce_eager=False)
```

## Code layout

```text
src/nanovllm/
├── engine/                         scheduler, KV blocks, model runner
├── models/                         Qwen3 torch model
└── backends/megakernel/
    ├── backend.py                  admission policy + engine adapter
    ├── taskgraph.py                task-graph decoder and scratch ownership
    ├── schedule.py                 host-side instruction/bar construction
    ├── weights.py                  Qwen weight views and pointer packing
    ├── cache.py                    paged KV-cache contract
    └── csrc/
        ├── taskgraph_kernel.cu     cooperative CUDA kernel
        ├── torch_bindings.cpp      task-graph-only extension bindings
        └── tg_abi.h                layout-locked launch ABI
```

Benchmarks and server commands:

- `/Users/tmchao7/repos/02_personal/MegaKernel-vLLM/benchmarks/compare.py`
- `/Users/tmchao7/repos/02_personal/MegaKernel-vLLM/tools/bench_taskgraph.py`
- `/Users/tmchao7/repos/02_personal/MegaKernel-vLLM/tools/bench_fusion_granularity.py`
- `/Users/tmchao7/repos/02_personal/MegaKernel-vLLM/SERVER_COMMANDS.txt`

Fusion granularity (`Config.fusion_groups`) is a first-class experimental
variable: one decode step can be built as `g` contiguous layer segments launched
`g` times, so "partial fusion" is measured on the same IR as the single
megakernel instead of confounding granularity with implementation language. See
`/Users/tmchao7/repos/02_personal/MegaKernel-vLLM/docs/14_融合粒度实验.md`.

## Prefill/decode scheduling controls

The engine already uses FlashAttention for prefill (`flash_attn_varlen_func`) and
for torch decode (`flash_attn_with_kvcache`). The current Phase 0 scheduler adds
controls for mixed workloads:

```python
LLM(
    model,
    max_prefill_tokens_per_step=512,  # 0 keeps the full token budget
    prefill_decode_ratio=1,            # at most one decode round before prefill
)
```

Runtime reports include waiting/running sequence counts, allocated/free/peak KV
blocks, and prefix-cache hit/miss counters. These controls do not change the
Megakernel CUDA arithmetic; they make the pre-server system behavior measurable
and prevent a long waiting prompt from starving active decode.

## Validation rules

Correctness uses `PASS / CASCADE / FAIL`. A small BF16 reduction-order cascade
starting at decode token 1 is expected; a prefill-token-0 mismatch or length
mismatch is a failure. Do not use Nsight Compute (`ncu`) on the cooperative
kernel: use Nsight Systems (`nsys`) or the checked-in microbenchmarks.

The first GPU validation command for the active line is:

```bash
RUN_MODE=taskgraph-baseline bash SERVER_COMMANDS.txt
```

It produces the correctness log, the torch-eager/torch-cudagraph/Megakernel
comparison table, and the host/GPU accounting table under `results/`.

Historical performance conclusions for the archived implementation remain in
`docs/06_实际运行数据.md` and `docs/07_架构结论_megakernel为何打不过cudagraph.md`; they
must not be read as measurements of the current task-graph implementation.
