# Repository Guidelines

## Project Structure & Module Organization

- `src/nanovllm/` is the installable Python package. Core runtime flow lives in `engine/` (scheduling, blocks, model execution), reusable operators in `layers/`, model definitions in `models/`, and shared helpers in `utils/`.
- `src/nanovllm/config.py`, `llm.py`, and `sampling_params.py` define the public configuration and inference API.
- `src/nanovllm/roofline.py` is the pure-Python decode-step roofline model (weight bytes, HBM floor, host/device split). It has no torch dependency, is unit tested in `tests/test_roofline.py`, and its model dimensions are contract-tested against `kernel.cu` — change both together or the test fails.
- `examples/basic_generation.py` is the smallest end-to-end inference example; `benchmarks/` contains batch and Megakernel benchmarks.
- `src/nanovllm/backends/megakernel/` contains the Qwen CUDA kernel, C++ bindings, build helper, and backend adapter; changes there require server-side CUDA tooling.
- `docs/` contains the paged-KV design, A100/V100 server runbook, and Chinese metric/baseline/benchmark notes; `tools/` contains correctness and visualization utilities.
- `docs/07_多batch为何打不过cudagraph.md` is the authoritative record for the Megakernel-vs-CUDA-Graph question (currently v6). Before proposing a batch>1 kernel rewrite, read its §10 ledger and §10.4 decision rule: the device side has no headroom to win, so the only recoverable cost is per-step host overhead.

## Build, Test, and Development Commands

This project targets Python 3.10–3.12 and requires a compatible CUDA/PyTorch/Triton environment. For the GPU server, use the checked-in `requirements.server.txt`, `setup_server.sh`, and `SERVER_COMMANDS.txt` workflow; `uv` creates the environment from an existing system Python while the system provides the NVIDIA driver and `nvcc`.

```bash
bash setup_server.sh            # create the uv/CUDA server environment
python -m pip install -e .      # install the package in editable mode
python examples/basic_generation.py       # run local generation; update the model path first
python -m benchmarks.baseline        # benchmark standard nano-vLLM throughput
python -m benchmarks.compare MODEL       # compare backends and write metrics
python -m tools.correctness_smoke MODEL # short greedy correctness gate on GPU
python -m tools.plot_metrics results/a100.json --output-dir results/a100-plots
python -m compileall src/nanovllm benchmarks tools # fast syntax/import compilation smoke check
```

Use a local Hugging Face model directory (for example, `~/huggingface/Qwen3-0.6B/`); do not add model weights or credentials to Git.

## Coding Style & Naming Conventions

Use four spaces, standard Python formatting, and readable type annotations where they clarify interfaces. Keep modules focused and prefer the existing layout over introducing new top-level packages. Use `snake_case` for functions, variables, and modules; `PascalCase` for classes; and uppercase names for constants. Match the current import grouping (standard library, third-party packages, then local modules). No repository formatter or linter is configured, so keep diffs small and manually formatted.

## Server Command File Rule

Every command intended for execution on the remote A100/V100 server must be written to the single checked-in `SERVER_COMMANDS.txt`; do not create additional `*_COMMANDS.txt` files. Select a phase with `RUN_MODE` (`model`, `verify`, `build`, `correctness`, `benchmark`, `nosync`, `account`, `roofline`, `profile`, `sanitize`, or `all`). Before giving a new server command to the user, update, validate, commit, and push this file. Use literal Bash syntax with the required repository/model paths, mirror settings, GPU selection, timeouts, and phase prerequisites. Do not rely on chat-only command blocks.

`RUN_MODE=account` produces the per-step host/device ledger (`tools/account_decode.py`) and `RUN_MODE=roofline` runs the device lower-bound and bandwidth probes (`tools/bench_roofline_a100.py`, `tools/bench_burst_lower_bound.py`). The roofline probe's verdict decides whether a batch>1 megakernel rewrite is attempted at all, so run it before touching `kernel.cu`.

After the user runs `git pull --ff-only` on the server, they should be able to run `bash SERVER_COMMANDS.txt` or `RUN_MODE=<phase> bash SERVER_COMMANDS.txt` directly. Keep server-generated results, logs, model weights, and profiler reports ignored by Git.

## Testing Guidelines

The repository uses focused `unittest` checks under `tests/`; there is no coverage threshold. Run `PYTHONPATH=src python -m unittest discover -s tests -v` for backend/config/metrics logic. For runtime changes, also run `python -m compileall src/nanovllm tools` and validate on the target GPU with `examples/basic_generation.py` or `python -m benchmarks.compare MODEL`; include model, CUDA, GPU, JSON, and Markdown results with performance reports.

## Debugging Lessons (learned from real incidents)

When a correctness/numerical divergence looks deep (RoPE, attention, KV math all
verify fine), **first go back to the most basic parameter/config settings** before
instrumenting kernel internals. More than once the real root cause was a trivial
config-resolution bug, not the fused math.

- **Qwen3 `rope_theta` lives in `rope_scaling`, not as a top-level config field.**
  Qwen3-0.6B stores it as `rope_scaling={"rope_theta": 1000000, "rope_type": "default"}`.
  `getattr(config, "rope_theta", 10000.0)` silently falls back to the default
  `10000.0`, producing wrong cos/sin tables. nano-vLLM's `models/qwen3.py`
  (`Qwen3Attention.__init__`) resolves it via `rope_scaling.get("rope_theta", ...)`;
  the Megakernel weight loader (`backends/megakernel/model.py`) must do the same.
  Fixed by adding `_resolve_rope_theta(config)` there.
- Because Qwen3 `rope_theta=1e6` with `head_dim=128` keeps low-frequency indexes
  near identity (cos≈1, sin≈0), a wrong theta/position shows up as *asymmetric*
  element errors: elements paired with a large-cos index blow up, pairs with
  small-sin stay near-correct. Verify the cos/sin table + a single RoPE application
  directly (see `tools/diag_rope.py`, `tools/diag_pos_search.py`) before suspecting
  RMSNorm/attention/matvec.
- The `cache_len` contract for decode attention must match the torch-eager path:
  nano-vLLM stores the current K/V then calls `flash_attn_with_kvcache` with
  `cache_seqlens=len(seq)=position+1`, so the query attends to `[0, position]`
  inclusive. Keep `cache_len = position + 1` for both paged and contiguous paths.
- **Distinguish a real indexing/config bug from bf16 precision cascade.** To
  tell them apart, verify the megakernel's written K/V against an *exact Python
  reference* at each real decode position (input = the observed token at that
  position; decode positions are `[n_prompt, n_prompt + n_completion - 1)` —
  the last completion token is generated by prefill's first token, so the final
  cached slot `n_prompt + n_completion - 1` is never written and will read
  garbage). If every real decode position matches the reference to 0.0, the
  kernel K/V path is correct; divergence from torch-eager is then just the
  ~1-ulp fused-kernel (reduction-order) difference, which can cascade into an
  occasional token flip or even a repeating-token state at greedy decoding.
  The megakernel matches an exact reference, not torch's fused bf16 kernels.
- **The batched WMMA and scalar megakernel paths are not bit-identical across
  grid/batch configs**, and can't be made token-exact at every batch/length. A
  non-perturbing per-layer hidden dump shows L0 is bit-identical across
  batches; the divergence is a single ~1-2 bf16 ULP at a later layer
  (deterministic in clean runs, but shifted by timing perturbation such as
  device `printf`), which then cascades into an occasional token flip at greedy
  decode. This is the same documented ulp cascade, not memory corruption
  (compute-sanitizer clean, L0 bit-identical, values bounded). A strict
  token-equality gate for WMMA-vs-scalar therefore can never pass at every
  batch; `tools/correctness_wmma_vs_scalar.py` instead classifies each case as
  **PASS** (identical), **CASCADE** (divergence only at a decode token >= 1,
  expected ulp cascade), or **FAIL** (divergence at prefill token 0 or a length
  mismatch = real bug). Use the non-perturbing hidden-state dump
  (`LDG_DBG_HIDDEN` + a per-layer device buffer, see git history) rather than
  device `printf` when verifying a suspected divergence.

## Commit & Pull Request Guidelines

Recent commits use concise, imperative, lowercase subjects, often with a scope such as `fix(scheduler): ...`. Follow that style and keep each commit focused. Pull requests should explain the problem and approach, list validation commands and GPU/model configuration, call out performance or numerical changes, and include relevant logs or screenshots. Avoid unrelated refactors and generated artifacts.
