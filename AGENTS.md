# Repository Guidelines

## Project Structure & Module Organization

- `nanovllm/` is the installable Python package. Core runtime flow lives in `engine/` (scheduling, blocks, model execution), reusable operators in `layers/`, model definitions in `models/`, and shared helpers in `utils/`.
- `nanovllm/config.py`, `llm.py`, and `sampling_params.py` define the public configuration and inference API.
- `example.py` is the smallest end-to-end inference example; `bench.py` contains the throughput benchmark. Static assets are under `assets/`.
- `qwen_megakernel/` is an optional CUDA-focused benchmark project for Qwen3-0.6B, primarily targeting A100 (`sm_80`); changes there require server-side CUDA tooling.
- `docs/` contains the paged-KV design and the A100/V100 server runbook; `tools/` contains correctness and visualization utilities.

## Build, Test, and Development Commands

This project targets Python 3.10–3.12 and requires a compatible CUDA/PyTorch/Triton environment.

```bash
python -m pip install -e .       # install the package in editable mode
python example.py                # run local generation; update the model path first
python bench.py                  # benchmark throughput on the configured GPU
python bench_megakernel.py MODEL # compare backends and write latency/throughput metrics
python tools/correctness_smoke.py MODEL # short greedy correctness gate on GPU
python tools/plot_metrics.py results/a100.json --output-dir results/a100-plots
python -m compileall nanovllm    # fast syntax/import compilation smoke check
```

Use a local Hugging Face model directory (for example, `~/huggingface/Qwen3-0.6B/`); do not add model weights or credentials to Git.

## Coding Style & Naming Conventions

Use four spaces, standard Python formatting, and readable type annotations where they clarify interfaces. Keep modules focused and prefer the existing layout over introducing new top-level packages. Use `snake_case` for functions, variables, and modules; `PascalCase` for classes; and uppercase names for constants. Match the current import grouping (standard library, third-party packages, then local modules). No repository formatter or linter is configured, so keep diffs small and manually formatted.

## Testing Guidelines

The repository uses focused `unittest` checks under `tests/`; there is no coverage threshold. Run `python -m unittest discover -s tests -v` for backend/config/metrics logic. For runtime changes, also run `python -m compileall nanovllm` and validate on the target GPU with `example.py` or `bench_megakernel.py`; include model, CUDA, GPU, JSON, and Markdown results with performance reports.

## Commit & Pull Request Guidelines

Recent commits use concise, imperative, lowercase subjects, often with a scope such as `fix(scheduler): ...`. Follow that style and keep each commit focused. Pull requests should explain the problem and approach, list validation commands and GPU/model configuration, call out performance or numerical changes, and include relevant logs or screenshots. Avoid unrelated refactors and generated artifacts.
