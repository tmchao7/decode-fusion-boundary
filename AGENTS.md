# Repository Guidelines

## Project Structure & Module Organization

- `src/nanovllm/` is the installable Python package. Core runtime flow lives in `engine/` (scheduling, blocks, model execution), reusable operators in `layers/`, model definitions in `models/`, and shared helpers in `utils/`.
- `src/nanovllm/config.py`, `llm.py`, and `sampling_params.py` define the public configuration and inference API.
- `examples/basic_generation.py` is the smallest end-to-end inference example; `benchmarks/` contains batch and Megakernel benchmarks.
- `src/nanovllm/backends/megakernel/` contains the Qwen CUDA kernel, C++ bindings, build helper, and backend adapter; changes there require server-side CUDA tooling.
- `docs/` contains the paged-KV design, A100/V100 server runbook, and Chinese metric/baseline/benchmark notes; `tools/` contains correctness and visualization utilities.

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

Every command intended for execution on the remote A100/V100 server must be written to the single checked-in `SERVER_COMMANDS.txt`; do not create additional `*_COMMANDS.txt` files. Select a phase with `RUN_MODE` (`model`, `verify`, `build`, `correctness`, `benchmark`, `profile`, or `all`). Before giving a new server command to the user, update, validate, commit, and push this file. Use literal Bash syntax with the required repository/model paths, mirror settings, GPU selection, timeouts, and phase prerequisites. Do not rely on chat-only command blocks.

After the user runs `git pull --ff-only` on the server, they should be able to run `bash SERVER_COMMANDS.txt` or `RUN_MODE=<phase> bash SERVER_COMMANDS.txt` directly. Keep server-generated results, logs, model weights, and profiler reports ignored by Git.

## Testing Guidelines

The repository uses focused `unittest` checks under `tests/`; there is no coverage threshold. Run `PYTHONPATH=src python -m unittest discover -s tests -v` for backend/config/metrics logic. For runtime changes, also run `python -m compileall src/nanovllm tools` and validate on the target GPU with `examples/basic_generation.py` or `python -m benchmarks.compare MODEL`; include model, CUDA, GPU, JSON, and Markdown results with performance reports.

## Commit & Pull Request Guidelines

Recent commits use concise, imperative, lowercase subjects, often with a scope such as `fix(scheduler): ...`. Follow that style and keep each commit focused. Pull requests should explain the problem and approach, list validation commands and GPU/model configuration, call out performance or numerical changes, and include relevant logs or screenshots. Avoid unrelated refactors and generated artifacts.
