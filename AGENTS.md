# Repository Guidelines

## Project Structure & Module Organization

- `src/nanovllm/` is the installable Python package. Core runtime flow lives in `engine/` (scheduling, blocks, model execution), reusable operators in `layers/`, model definitions in `models/`, and shared helpers in `utils/`.
- `src/nanovllm/config.py`, `llm.py`, and `sampling_params.py` define the public configuration and inference API.
- `examples/basic_generation.py` is the smallest end-to-end inference example; `benchmarks/` contains batch and Megakernel benchmarks.
- `src/nanovllm/backends/megakernel/` contains the Qwen CUDA kernel, C++ bindings, build helper, and backend adapter; changes there require server-side CUDA tooling.
- `docs/` contains the paged-KV design, A100/V100 server runbook, and Chinese metric/baseline/benchmark notes; `tools/` contains correctness and visualization utilities.

## Build, Test, and Development Commands

This project targets Python 3.10–3.12 and requires a compatible CUDA/PyTorch/Triton environment. For the GPU server, prefer the checked-in `environment.server.yml`, `requirements.server.txt`, and `setup_server.sh` workflow.

```bash
bash setup_server.sh            # create the Miniconda/CUDA server environment
python -m pip install -e .      # install the package in editable mode
python examples/basic_generation.py       # run local generation; update the model path first
python -m benchmarks.baseline        # benchmark standard nano-vLLM throughput
python -m benchmarks.compare MODEL       # compare backends and write metrics
python -m tools.correctness_smoke MODEL # short greedy correctness gate on GPU
python -m tools.plot_metrics results/a100.json --output-dir results/a100-plots
python -m compileall src/nanovllm tools   # fast syntax/import compilation smoke check
```

Use a local Hugging Face model directory (for example, `~/huggingface/Qwen3-0.6B/`); do not add model weights or credentials to Git.

## Coding Style & Naming Conventions

Use four spaces, standard Python formatting, and readable type annotations where they clarify interfaces. Keep modules focused and prefer the existing layout over introducing new top-level packages. Use `snake_case` for functions, variables, and modules; `PascalCase` for classes; and uppercase names for constants. Match the current import grouping (standard library, third-party packages, then local modules). No repository formatter or linter is configured, so keep diffs small and manually formatted.

## Testing Guidelines

The repository uses focused `unittest` checks under `tests/`; there is no coverage threshold. Run `PYTHONPATH=src python -m unittest discover -s tests -v` for backend/config/metrics logic. For runtime changes, also run `python -m compileall src/nanovllm tools` and validate on the target GPU with `examples/basic_generation.py` or `python -m benchmarks.compare MODEL`; include model, CUDA, GPU, JSON, and Markdown results with performance reports.

## Commit & Pull Request Guidelines

Recent commits use concise, imperative, lowercase subjects, often with a scope such as `fix(scheduler): ...`. Follow that style and keep each commit focused. Pull requests should explain the problem and approach, list validation commands and GPU/model configuration, call out performance or numerical changes, and include relevant logs or screenshots. Avoid unrelated refactors and generated artifacts.
