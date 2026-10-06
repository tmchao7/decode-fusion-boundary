# Taskgraph 5090 Finalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the Megakernel taskgraph locally before the next RTX 5090 session by adding offline ownership/geometry guards, truthful correctness diagnostics, and a reproducible server setup path.

**Architecture:** Keep the CUDA kernel interface unchanged. Add a pure-Python audit layer beside the existing schedule verifier; it will inspect generated instruction ranges and model each instruction's declared producer/consumer ownership without requiring CUDA. Extend the correctness tool's JSON schema with explicit execution/error metadata while preserving PASS/CASCADE/FAIL. Make `setup_server.sh` choose `conda pip` or `uv` explicitly and expose mirror/wheel controls, with post-install checks using the existing verifier.

**Tech Stack:** Python 3.10–3.12, unittest/pytest-compatible tests, Bash, existing taskgraph schedule/ABI modules, CUDA C++ source audits.

**Spec:** The approved design in the preceding conversation: ownership/geometry guard first, DEVICE_ERROR correctness semantics second, server setup reproducibility third.

## Global Constraints

- Do not claim RTX 5090 correctness or grid>1 stability from macOS/offline tests.
- Preserve the existing `PASS / CASCADE / FAIL` classification and make device faults strictly more severe, never less.
- Keep `attn_splits` explicit through process boundaries and record the effective value in result artifacts.
- Keep the active Megakernel contract at Qwen3-0.6B, BF16, decode-only, single-card, batch 1..8.
- Do not add a CUDA dependency to the offline audit tests.
- Keep `setup_server.sh` non-interactive and avoid managed Python downloads by default.

### Task 1: Add offline taskgraph ownership and geometry audit

**Files:**
- Create: `src/nanovllm/backends/megakernel/audit.py`
- Modify: `src/nanovllm/backends/megakernel/__init__.py` only if package export is needed
- Create: `tests/test_taskgraph_audit.py`
- Modify: `tools/verify_schedule.py` to run the audit for each verified configuration

**Interfaces:**
- `audit_schedule(schedule, *, batch_size, num_sms, attn_splits) -> AuditReport`
- `AuditReport.ok: bool`, `AuditReport.failures: tuple[str, ...]`, and `AuditReport.checked_writes: int`
- The audit consumes the existing schedule instruction representation and does not import torch.

- [x] **Step 1: Write failing tests** for packed tile ranges, matrix bounds, one-writer ownership of tile outputs, and split partial bounds across `BS in (1,2,4,8)`, `num_sms in (1,4,108,170)`, and `attn_splits in (1,2)`.
- [x] **Step 2: Run the focused tests** and verify they fail because the audit module does not exist.
- [x] **Step 3: Implement the smallest pure-Python audit** by decoding each instruction's opcode and range fields, checking `tile_n0 + num_tiles <= tiles_N`, packed range contiguity, per-output ownership, and `o_partial/lse_partial` split capacity.
- [x] **Step 4: Integrate the audit into `tools.verify_schedule.verify_matrix`** without changing the existing schedule acceptance logic; a schedule is failed if either schedule simulation or audit fails.
- [x] **Step 5: Run the focused tests and schedule verifier** with the default matrix and confirm zero audit failures and zero false accepts.

### Task 2: Make correctness artifacts truthful and device faults explicit

**Files:**
- Modify: `tools/correctness_smoke.py`
- Modify: `tests/test_correctness_smoke.py`
- Modify: `docs/19_上机清单.md` in the correctness/output section

**Interfaces:**
- `run_backend_isolated(...)` continues to return a JSON-safe result on success and raises a typed `BackendDeviceError` for CUDA/device failures.
- `classify_comparison(...)` keeps returning `PASS`, `CASCADE`, or `FAIL`.
- Result payloads gain `effective_attn_splits`, `backend_errors`, and `classification_details` fields without removing existing fields.

- [x] **Step 1: Add failing unit tests** for classifying a child CUDA fault as `DEVICE_ERROR`, for preserving the exception text, and for recording the effective split value separately from the requested CLI value.
- [x] **Step 2: Run the focused tests** and verify the new assertions fail.
- [x] **Step 3: Implement a narrow exception classifier** that recognizes CUDA/device faults from traceback text and process exit state; never convert a device fault into CASCADE.
- [x] **Step 4: Extend the serialized result and top-level JSON payload** with requested/effective split values and first-divergence details for CASCADE/FAIL.
- [x] **Step 5: Run all correctness-tool unit tests and verify existing `attn_splits` forwarding tests remain green.

### Task 3: Harden server setup reproducibility

**Files:**
- Modify: `setup_server.sh`
- Modify: `requirements.server.txt` only if dependency comments or wheel constraints need alignment
- Modify: `tests/test_server_commands.py` or create `tests/test_setup_server.py`
- Modify: `docs/08_服务器运行手册.md` and `docs/19_上机清单.md`

**Interfaces:**
- `NANOVLLM_INSTALLER=uv|conda` selects the installer, defaulting to `conda` when available and otherwise `uv`.
- `NANOVLLM_PYPI_INDEX_URL` controls the Python mirror.
- `NANOVLLM_FLASH_ATTN_WHEEL` optionally installs a supplied local/remote wheel before falling back to package installation.
- `NANOVLLM_VERIFY_MODEL` optionally supplies the model path to the final environment verifier.

- [x] **Step 1: Add shell-source/AST tests** asserting installer selection, mirror propagation to every pip operation, optional FlashAttention wheel handling, and final `pip check`/environment verification.
- [x] **Step 2: Run those tests and verify they fail against the current uv-only script.
- [x] **Step 3: Implement the smallest branching setup flow** with shared package lists and no automatic uv-managed Python download.
- [x] **Step 4: Update the runbook with the known 5090 recipe as an example, without hard-coding a site-specific mirror or credential.
- [x] **Step 5: Run `bash -n setup_server.sh`, shell tests, and a dry-run/static command audit.

### Task 4: Full offline verification and handoff

**Files:**
- Modify: `docs/19_上机清单.md` with the exact local validation commands and remaining server-only matrix
- No CUDA claims are added from offline results.

- [x] **Step 1: Run compileall and all unittest/pytest-compatible offline tests available in the local environment.
- [x] **Step 2: Run `tools.verify_schedule` with the default and expanded matrix.
- [x] **Step 3: Run `git diff --check` and `bash -n setup_server.sh`.
- [x] **Step 4: Review the diff for accidental changes to the active kernel contract and report remaining RTX 5090-only validation steps.
