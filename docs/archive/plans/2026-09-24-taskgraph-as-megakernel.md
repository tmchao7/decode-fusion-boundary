# Task-Graph Megakernel as the Sole Mainline Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the new task-graph implementation the only public `megakernel` backend on the mainline while preserving the frozen `kernel.cu` implementation in an archival branch.

**Architecture:** Create `codex/archive/frozen-kernel` at the pre-cleanup commit. On the feature branch derived from main, rename the public task-graph backend from `megakernel-tg` to `megakernel`, remove legacy kernel/burst dispatch from the active API and benchmark surface, and extract shared weight-packing helpers into a task-graph-owned module. Keep the engine's normal prefill/decode flow and KV-cache contract unchanged.

**Tech Stack:** Python 3.10–3.12, unittest, PyTorch/CUDA extension, CUDA C++, task-graph schedule ABI, Markdown documentation, Git worktrees.

**Spec:** Approved chat design: mainline exposes `torch` and task-graph `megakernel`; frozen `kernel.cu` line is preserved in `codex/archive/frozen-kernel`.

## Global Constraints

- `decode_backend` on mainline is exactly `torch` or `megakernel`; `megakernel` means task-graph.
- The task-graph kernel remains decode-only, greedy-only, single-card, Qwen3-0.6B-only, `max_model_len <= 2048`, and `TG_MAX_BATCH = 8`.
- Do not change task-graph CUDA arithmetic, schedule/bar ABI, KV-cache layout, or correctness classification semantics.
- Do not use Nsight Compute (`ncu`) on cooperative kernels; use `nsys` for GPU profiling.
- Legacy `kernel.cu` behavior must remain recoverable from `codex/archive/frozen-kernel`.
- Use test-first changes: each public API/refactor behavior gets a failing test before implementation.

---

### Task 1: Lock the cleaned public backend contract with failing tests

**Files:**
- Modify: `tests/test_config.py`
- Modify: `tests/test_megakernel_backend.py`
- Modify: `tests/test_taskgraph_backend.py`
- Modify: `tests/test_paged_kernel_source.py`
- Modify: `tests/test_server_commands.py`
- Modify: `tests/test_comparison_fairness.py`

**Interfaces:**
- Consumes: existing `Config`, `MegakernelBackend`, benchmark configuration, and task-graph compatibility APIs.
- Produces: tests asserting that `megakernel` is the task-graph backend and legacy names/configuration are no longer part of the mainline public surface.

- [ ] **Step 1: Write failing tests**
  - Assert `VALID_DECODE_BACKENDS == {"torch", "megakernel"}`.
  - Assert `MegakernelBackend.compatibility()` delegates the `megakernel` backend to task-graph admission, including batch sizes up to `TG_MAX_BATCH`.
  - Assert source/configuration no longer exposes `megakernel-tg`, `megakernel-wmma`, or `megakernel-burst` as active mainline backends.
  - Assert `compare.py` defaults to `torch-eager,torch-cudagraph,megakernel` and its megakernel family contains only `megakernel`.
  - Assert server baseline commands invoke the renamed `megakernel` backend.

- [ ] **Step 2: Run the focused tests and verify red**
  - Run: `/Users/tmchao7/miniforge3/bin/python3.13 -m unittest tests.test_config tests.test_megakernel_backend tests.test_taskgraph_backend tests.test_paged_kernel_source tests.test_server_commands tests.test_comparison_fairness -v`
  - Expected: import/dependency failures may occur in this machine, but any runnable contract tests must fail because the repository still exposes `megakernel-tg` and legacy configs.

- [ ] **Step 3: Commit the test-only contract changes**
  - Commit: `test: define taskgraph as the mainline megakernel backend`

---

### Task 2: Rename the public backend and simplify backend dispatch

**Files:**
- Modify: `src/nanovllm/config.py`
- Modify: `src/nanovllm/backends/megakernel/backend.py`
- Modify: `src/nanovllm/backends/megakernel/__init__.py`
- Modify: `src/nanovllm/engine/model_runner.py`
- Modify: `src/nanovllm/engine/llm_engine.py`
- Modify: `src/nanovllm/engine/burst_policy.py`
- Modify: `src/nanovllm/engine/scheduler.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_megakernel_backend.py`
- Modify: `tests/test_taskgraph_backend.py`

**Interfaces:**
- Consumes: `TaskGraphDecoder.decode_rows()` and existing `MegakernelBackend` adapter entrypoints.
- Produces: `MegakernelBackend` that owns only task-graph construction and one-step decode; `ModelRunner.run()` dispatches `decode_backend="megakernel"` to that path.

- [ ] **Step 1: Implement the smallest config rename**
  - Change `VALID_DECODE_BACKENDS` to `{"torch", "megakernel"}`.
  - Remove old-only config fields from `Config` and their validation paths.
  - Keep `megakernel_tg_attn_splits` and `megakernel_tg_verbose`, renaming them to `megakernel_attn_splits` and `megakernel_verbose` for the public API.

- [ ] **Step 2: Make backend dispatch task-graph-only**
  - Remove the `decode_backend == "megakernel-tg"` branch.
  - Construct `TaskGraphDecoder` for `decode_backend="megakernel"`.
  - Make `decode_batch()` call `decode_rows()` directly.
  - Remove legacy `Decoder` construction and no-sync generation methods from the active adapter.
  - Make `supports_no_sync_burst()` unnecessary on mainline and remove old burst gating from the engine.

- [ ] **Step 3: Remove the old batch/burst route from the active engine**
  - Keep `ModelRunner.run()` normal step dispatch intact.
  - Remove `LLMEngine._should_burst()`, `_burst_batch_limit()`, `_step_burst()`, and burst-only metrics flow if no active backend uses it.
  - Remove old burst-only scheduler methods and flags only after confirming no mainline tool/test references remain.
  - Preserve ordinary scheduler prefill/decode/postprocess behavior.

- [ ] **Step 4: Run focused tests and verify green**
  - Run: `/Users/tmchao7/miniforge3/bin/python3.13 -m unittest tests.test_config tests.test_megakernel_backend tests.test_taskgraph_backend -v`
  - Expected: all runnable tests pass; dependency-limited tests are reported separately.

- [ ] **Step 5: Commit**
  - Commit: `refactor(backend): make taskgraph the sole megakernel path`

---

### Task 3: Extract task-graph weight helpers and remove legacy production model code

**Files:**
- Create: `src/nanovllm/backends/megakernel/weights.py`
- Modify: `src/nanovllm/backends/megakernel/taskgraph.py`
- Modify: `src/nanovllm/backends/megakernel/__init__.py`
- Delete or archive-only: `src/nanovllm/backends/megakernel/model.py`
- Delete or archive-only: `src/nanovllm/backends/megakernel/csrc/kernel.cu`
- Modify: `tests/test_megakernel_backend.py`
- Modify: `tests/test_paged_kernel_source.py`

**Interfaces:**
- Consumes: the nano-vLLM model and HF config already passed to `MegakernelBackend`.
- Produces: `weights_from_nano_model()` and `_pack_layer_weights()` in `weights.py`; taskgraph imports them without depending on the legacy decoder module.

- [ ] **Step 1: Add a failing import-boundary test**
  - Assert `taskgraph.py` imports `_pack_layer_weights` from `weights.py`.
  - Assert the active package does not expose `Decoder`, `generate`, `load_weights`, or `generate_paged_batch` as production backend APIs.

- [ ] **Step 2: Move shared helpers**
  - Move the Qwen3 constants needed by taskgraph weight conversion, RoPE table construction, `weights_from_nano_model()`, and `_pack_layer_weights()` into `weights.py`.
  - Keep the output dictionary keys identical so `TaskGraphDecoder` does not need a tensor contract change.

- [ ] **Step 3: Remove the legacy model implementation from the feature branch**
  - Remove the old `Decoder` class and one-shot legacy generation helpers from mainline.
  - Remove `kernel.cu` from the active branch after the archival branch exists.
  - Do not alter `taskgraph_kernel.cu`, `tg_abi.h`, or `torch_bindings.cpp` arithmetic/ABI.

- [ ] **Step 4: Run focused source/import tests**
  - Run: `/Users/tmchao7/miniforge3/bin/python3.13 -m unittest tests.test_megakernel_backend tests.test_paged_kernel_source tests.test_taskgraph_backend -v`
  - Expected: all runnable tests pass and no active import references the removed legacy model.

- [ ] **Step 5: Commit**
  - Commit: `refactor(taskgraph): isolate shared weight packing from legacy decoder`

---

### Task 4: Simplify benchmark/correctness tools to the two-backend mainline

**Files:**
- Modify: `benchmarks/compare.py`
- Modify: `tools/correctness_smoke.py`
- Modify: `tools/correctness_wmma_vs_scalar.py`
- Modify: `tools/correctness_burst_mixed.py`
- Modify: `tools/bench_burst_batch.py`
- Modify: `tools/bench_nosync.py`
- Modify: `tests/test_comparison_fairness.py`
- Modify: `tests/test_workloads.py`
- Modify: `tests/test_server_commands.py`

**Interfaces:**
- Consumes: public backends `torch-eager`, `torch-cudagraph`, and `megakernel`.
- Produces: default benchmark/correctness commands that measure the new Megakernel directly against CUDA Graph without legacy fallback rows.

- [ ] **Step 1: Write failing benchmark contract assertions**
  - Assert `BACKEND_CONFIGS` contains only `torch-eager`, `torch-cudagraph`, and `megakernel` for the active comparison surface.
  - Assert `MEGAKERNEL_FAMILY == ("megakernel",)`.
  - Assert `--backends` default is `torch-eager,torch-cudagraph,megakernel`.
  - Assert taskgraph baseline output names no longer contain `megakernel-tg`.

- [ ] **Step 2: Implement benchmark cleanup**
  - Rename the taskgraph backend config key to `megakernel` with the new config field names.
  - Remove legacy WMMA/burst configs from active defaults and active correctness matrices.
  - Remove obsolete burst-specific benchmark paths from mainline tools, or leave them only as explicit archive-branch utilities if deleting them would make the current branch harder to audit.
  - Keep grid workload rules and fallback-row exclusion intact.

- [ ] **Step 3: Run focused tool/source tests**
  - Run: `/Users/tmchao7/miniforge3/bin/python3.13 -m unittest tests.test_comparison_fairness tests.test_server_commands tests.test_workloads -v`
  - Expected: all runnable tests pass.

- [ ] **Step 4: Commit**
  - Commit: `refactor(benchmarks): compare the single megakernel against cuda graph`

---

### Task 5: Update documentation and server commands

**Files:**
- Modify: `README.md`
- Modify: `docs/01_项目说明.md`
- Modify: `docs/05_对比基线与公平性.md`
- Modify: `docs/08_服务器运行手册.md`
- Modify: `docs/10_交接文档.md`
- Modify: `docs/13_集成与基线规划.md`
- Modify: `docs/taskgraph_progress.md`
- Modify: `SERVER_COMMANDS.txt`

**Interfaces:**
- Consumes: the renamed public API and benchmark commands from Tasks 2–4.
- Produces: one consistent narrative: `torch-cudagraph` versus `decode_backend="megakernel"` (task-graph).

- [ ] **Step 1: Replace transitional naming**
  - Change active examples and commands from `megakernel-tg` to `megakernel`.
  - Mark the old `kernel.cu` line as archived rather than a current backend.
  - Remove descriptions claiming that the active backend uses batch=1 Megakernel and batch>1 torch fallback.

- [ ] **Step 2: State the current limits and benchmark interpretation**
  - Document that the new mainline Megakernel is one task-graph launch per decode step, not the old no-sync burst.
  - Keep the existing correctness, ncu, CUDA, Qwen3, and max-model-length constraints.
  - Keep historical old-kernel numbers labeled as historical/archive data.

- [ ] **Step 3: Update server command names**
  - Make `RUN_MODE=taskgraph-baseline` invoke `megakernel` in the active branch.
  - Preserve the command's correctness gate and four-number accounting.

- [ ] **Step 4: Run documentation/source consistency checks**
  - Run: `git grep -n 'megakernel-tg\|megakernel-wmma\|megakernel-burst' -- ':!docs/superpowers/plans/*'`
  - Expected: no active-code references; any remaining historical reference must explicitly identify the archive branch or historical result.

- [ ] **Step 5: Commit**
  - Commit: `docs: describe taskgraph as the mainline megakernel`

---

### Task 6: Full local verification and handoff

**Files:**
- Modify: `docs/10_交接文档.md` if the handoff status needs a final update.

- [ ] **Step 1: Run Python syntax verification**
  - Run: `/Users/tmchao7/miniforge3/bin/python3.13 -m compileall -q src/nanovllm benchmarks tools tests`
  - Expected: exit code 0.

- [ ] **Step 2: Run the complete unit test command available in the environment**
  - Run: `/Users/tmchao7/miniforge3/bin/python3.13 -m unittest discover -s tests -v`
  - Expected: report exact dependency/environment failures if torch/transformers are unavailable; do not label the suite passing unless exit code is 0.

- [ ] **Step 3: Audit git state and branch preservation**
  - Run: `git status --short --branch`, `git diff --check`, `git worktree list`, and `git show codex/archive/frozen-kernel:src/nanovllm/backends/megakernel/csrc/kernel.cu >/dev/null`.
  - Expected: no whitespace errors, archive branch still contains legacy kernel, feature diff contains only intended cleanup.

- [ ] **Step 4: Produce the final report**
  - Report commits, files changed, tests run/results, environment limitations, and the exact GPU command for A100 validation: `RUN_MODE=taskgraph-baseline bash SERVER_COMMANDS.txt`.
