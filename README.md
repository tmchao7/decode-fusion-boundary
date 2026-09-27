# Megakernel-vLLM

Qwen3-0.6B decode research backend for nano-vLLM. The **mainline has one
Megakernel implementation**: a persistent task-graph CUDA kernel. Its primary
comparison target is the existing torch CUDA-Graph path.

```text
decode_backend="torch", enforce_eager=False -> torch-cudagraph
decode_backend="megakernel"                 -> task-graph Megakernel
```

The previous `kernel.cu` persistent/WMMA implementation is preserved on the
archival branch `codex/archive/frozen-kernel`; it is intentionally not part of
the mainline API, build, scheduler, or benchmark surface.

**文档入口：[docs/README.md](docs/README.md)** —— 17 个文档按"现行 / 历史证据 /
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
