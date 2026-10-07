# Contributing

Thanks for looking at Megakernel-vLLM. This is a research extension of
[nano-vLLM](https://github.com/GeeeekExplorer/nano-vLLM): a fused persistent
cooperative kernel for the Qwen3-0.6B decode step, plus the measurement
tooling that decided how it should be dispatched.

Start with [`docs/01_项目说明.md`](docs/01_项目说明.md) for what the project is,
[`docs/06_实际运行数据.md`](docs/06_实际运行数据.md) for the measured numbers, and
[`docs/07_融合收益边界与根因.md`](docs/07_融合收益边界与根因.md)
for the root-cause analysis and the decision rule. `docs/10_交接文档.md` is the
engineering log (what each round changed, what is frozen).

## Layout

- `src/nanovllm/` — installable package. `engine/` (scheduler, blocks, model
  runner), `layers/` (operators), `models/` (Qwen3), `backends/megakernel/`
  (CUDA kernel, C++ bindings, build helper, adapter), `roofline.py` +
  `rope_config.py` (torch-free helpers).
- `benchmarks/`, `tools/` — A/B harnesses, correctness gates, profiling and
  visualization utilities.
- `tests/` — focused `unittest` checks; no coverage threshold.
- `docs/` — ordered documentation set.

## Environment

Local (CPU-only) development only needs Python 3.10–3.12:

```bash
python -m pip install -e .
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall src/nanovllm benchmarks tools
```

A GPU machine additionally needs an NVIDIA driver, `nvcc`, and a CUDA-enabled
PyTorch. `bash setup_server.sh` creates a `uv` virtualenv from an *existing*
system Python (it deliberately never downloads a managed CPython) and installs
the runtime requirements:

```bash
bash setup_server.sh
source .venv/bin/activate
python -m tools.verify_server_env          # optional: checks the environment
```

Several tools import `torch` / `transformers`; those tests are skipped or error
out on a dependency-free laptop, which is expected and not a regression.

## Model path

Never hard-code a model path. Point `NANOVLLM_MODEL_PATH` at a local Hugging Face
directory (for example `$HOME/models/Qwen3-0.6B`); it is the default for the
scripts and is also what `tools/verify_server_env` checks.

## Running things on a GPU box

Every command intended for the remote GPU machine lives in the single checked-in
`SERVER_COMMANDS.txt`; select a phase with `RUN_MODE`
(`model`, `verify`, `build`, `correctness`, `benchmark`, `taskgraph`,
`taskgraph-baseline`, `all`). Please update that file instead of
adding another `*_COMMANDS.txt` — `tests/test_server_commands.py` also validates
that every flag it passes exists on the target module.

Paths and GPU selection come from environment variables so the file stays
portable: `REPO` (defaults to the checkout the script lives in),
`NANOVLLM_MODEL_PATH`, `PREFERRED_GPUS` (space-separated list; the script picks
the one with the most free memory), `PYPI_INDEX_URL` for restricted networks.

## Correctness policy (please read before gating anything)

**Do not gate on full-sequence token equality with torch-eager.** The task-graph
kernel and torch's fused BF16 kernels can differ in reduction order; a small
cascade at greedy decode is expected.

Use the three-way gate in `tools/correctness_smoke.py`:

- **PASS** — identical.
- **CASCADE** — divergence only at a decode token (index >= 1) with matching
  lengths; the documented ULP cascade, tolerated.
- **FAIL** — divergence at prefill token 0, a request-count mismatch, or a
  length mismatch. A real bug.

When you need to prove a kernel path is correct, compare it against an *exact
Python reference* for the written K/V at each real decode position rather than
against torch-eager.

## Debugging notes worth knowing

- **RoPE**: always resolve the base frequency through
  `nanovllm/rope_config.py` (`resolve_rope_theta`). The Qwen3-0.6B config.json
  keeps `rope_theta` at the top level with `rope_scaling: null`; some variants
  nest it instead, and newer transformers spells it `rope_parameters`. The
  nano-vLLM model and the Megakernel weight loader must use the same value or
  they build different cos/sin tables. Unsupported scaling types (YaRN / linear /
  dynamic) raise instead of being silently ignored.
- **KV cache length contract**: nano-VLLM stores the current K/V and attends over
  `[0, position]` inclusive, so keep `cache_len = position + 1` in both the paged
  and contiguous paths.
## Commits

Concise, imperative, lowercase subjects with a scope, e.g.
`fix(scheduler): ...`, `docs(06): ...`. Keep each commit focused; put measured
numbers, the GPU/model used, and the validation commands in the body.
