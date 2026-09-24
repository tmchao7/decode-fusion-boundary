# Task-Graph Megakernel 进展报告

**日期**: 2026-09-23
**硬件**: A100-SXM4-80GB (sm_80, 108 SM)
**模型**: Qwen3-0.6B (28 层, hidden=1024, intermediate=3072, vocab=151936)

---

## 1. 已完成

### 1.1 确定性修复

发现并修复了 **3 处跨 warp global memory race**——根因相同：`bar.sync` 只保证
shared memory 可见性，global store 需要 `fence.acq_rel.gpu` 才能跨 warp 可见。

| 位置 | race 描述 | 修复 |
|---|---|---|
| `tg_ffn_upgate_op` (~L1448) | gate/up GEMM → SiLU 读 `g.silu_out`/`g.up_out`，不同 warp 负责不同 tile，无 fence | GEMM 后加 `tg_fence_gpu()` + `tg_worker_sync_inner()` |
| `tg_attn_decode_op` (~L1113) | warp 0 写 KV cache → flash attention 读，`bar.sync` 不保证 global 可见 | 写后 `tg_fence_gpu()` + sync + `tg_fence_gpu()` |
| `tg_lm_argmax_op` (~L1612) | LM_HEAD GEMM 写 `lm_logits` → argmax 扫 vocab，跨 tile 不可见 | `tg_wait` 后加 `tg_fence_gpu()` + sync |

**修复效果**: 30/30 次运行全确定性（token=14582），修复前约 60-70% 正确。

### 1.2 测试修复

- `inst_buf` 用 `uint32` 创建再 `.view(int32)` — barrier count 可超过 2^31
- BS=2 测试在 BS=1 run 时用 BS=1 schedule（原来用 BS=2 schedule 导致越界写）
- RoPE 参考 `unsqueeze(1)` 广播修复
- `tg_zero_regions` CPU dispatch + smoke test skip/uint32 修复

### 1.3 测试结果

| 测试套件 | 结果 |
|---|---|
| `test_taskgraph_smoke.py` | 3 passed |
| `test_taskgraph_abi.py` (attn_splits=1) | 37 passed, 5011 subtests |
| `test_taskgraph_positions.py` (attn_splits=1) | 3 passed (max logit diff 0.21) |
| `test_taskgraph_positions.py` (attn_splits=2) | FAILED — schedule deadlock（预存问题） |
| `test_taskgraph_bs2.py` | 4 passed (BS=2 与独立 BS=1 结果一致) |
| `test_taskgraph_multi_sm.py` | passed |
| `compute-sanitizer --tool memcheck` | 无内核内存错误 |
| `compute-sanitizer --tool racecheck` | 无 shared memory race |

### 1.4 端到端文本生成

```
prompt: The capital of France is
output: The capital of France is Paris. The capital of Italy is Rome.
        The capital of Spain is Madrid. The capital of Portugal is Lisbon.
        The capital of Germany is Berlin.

prompt: Hello, how are you? I am
output: Hello, how are you? I am a student, and I am studying for the final exam.
        I am also a member of the school's club.
```

1 SM 和 4 SM 均输出相同文本，multi-SM 端到端正确。

### 1.5 性能数据 (ms/step)

全 SM (108 SM)，attn_splits=1，50 步平均：

| BS | ctx=1 | ctx=128 | ctx=512 | ctx=2048 |
|---|---|---|---|---|
| 1 | 10.8 ms (111 GB/s) | 11.5 ms (104 GB/s) | 15.9 ms (76 GB/s) | 25.8 ms (47 GB/s) |
| 2 | 11.1 ms (108 GB/s) | 11.9 ms (101 GB/s) | 15.4 ms (78 GB/s) | 26.1 ms (46 GB/s) |
| 4 | 12.5 ms (96 GB/s) | 13.4 ms (90 GB/s) | 16.5 ms (73 GB/s) | 32.4 ms (37 GB/s) |
| 8 | 13.6 ms (88 GB/s) | 14.4 ms (83 GB/s) | 17.5 ms (69 GB/s) | 33.6 ms (36 GB/s) |

单 SM: 275 ms/step (4 GB/s) — 确认是 weight-bandwidth bound。

### 1.6 Kernel 资源使用

- 110 regs / 0 spill / 64B stack / 21920B static shared + 64KB dynamic shared
- 7 worker warps (224 threads) + 1 controller warp = 256 threads/CTA

---

## 2. 多 Batch Decode 支持情况

**已支持**: BS=1,2,4,8 全部通过正确性测试。

- BS=2 测试验证：两行独立解码，结果与各自 BS=1 运行一致
- `row_active` 支持：inactive 行不写 KV cache（连续批处理基础）
- 多 SM 端到端：4 SM 生成文本与 1 SM 一致

---

## 3. 与 CUDA Graph PK 的现状

### 3.1 架构差距

- **CUDA Graph 路径**: 集成在引擎内 (`model_runner.py`)，通过 `graph.replay()` 重放捕获的 `Qwen3ForCausalLM` 前向
- **Task-graph 路径**: 尚未集成进引擎，只能通过 `tools/bench_taskgraph.py` / `generate_taskgraph.py` 直接调用 `tg_decode_step`

### 3.2 已有性能数据

Task-graph (108 SM, 50 步):
- BS=1 ctx=512: 15.9 ms/step
- BS=4 ctx=512: 16.5 ms/step

引擎 CUDA Graph 数据待测。

### 3.3 PK 方案

**已改为扩 `benchmarks/compare.py`，不再写独立脚本**（`docs/13 §2`）：该脚本已经是
"一个 case 一个 `LLM()`，串行跑"的结构，天然支持两条路径各自建引擎，并且自动继承
全部公平性规则与 correctness gate。新增 `RUN_MODE=taskgraph-baseline` 一次跑完
"correctness → 主表 → 四口径账"。

原计划的 `tools/compare_tg_vs_cudagraph.py` **已废弃**，避免第二套计时/公平性逻辑漂移。

---

## 4. 已知问题

1. ~~**`attn_splits=2` schedule deadlock**~~ **已失效**：该结论记录于 `f6cd3a8`
   （`184/10264`），但**用同一份 `simulate_execution` 在 `f6cd3a8` / `1a7539b` 上
   复现不出来**（`10264/10264` 全部可执行）。当时的 `184` 与模拟器的实际输出不符，
   应视为**未经验证的记录**而非已确认缺陷。`f3d8114` 之后指令数降到 9870，ABI 测试
   现覆盖 `attn_splits ∈ {1,2}` × `BS ∈ {1..8}` × `num_sms ∈ {1,4,8,108}` 全部无死锁；
   新加的 `tests/test_taskgraph_backend.py` 把 `BS 1..8` 也纳入了模拟器覆盖。
   **数值侧仍需服务器**（`test_taskgraph_positions::test_attn_splits_matches_causal_reference`）。
2. **性能远未优化**:
   - `__syncthreads` dispatch (~2µs/SM 开销)，ring buffer 仍是死代码
   - scalar attention 未向量化
   - 权重带宽利用率低 (111 GB/s vs A100 HBM 峰值 1.5 TB/s)
3. ~~**Task-graph 未集成引擎**~~ **已完成**（本轮）：`decode_backend="megakernel-tg"`
   走 `backends/megakernel/taskgraph.py`，`benchmarks/compare.py` 的
   `BACKEND_CONFIGS` 已加该 backend 并成为默认 `--backends`。

---

## 5. 下一步

1. ~~写 `tools/compare_tg_vs_cudagraph.py`~~ → 改为扩 `compare.py`（已完成）
2. 服务器跑 `RUN_MODE=taskgraph-baseline` 取主表 + 四口径账
3. ~~调查 `attn_splits=2` schedule deadlock~~ → 记录已失效，改为服务器数值验证
4. 性能优化: 启用 ring buffer dispatch、向量化 attention（`docs/12 §3`）
5. ~~集成 task-graph 到引擎 `MegakernelBackend`~~（已完成）

---

## 6. 提交记录

| Commit | 描述 |
|---|---|
| (本轮) | feat(taskgraph): 阶段 0 集成——`megakernel-tg` backend + compare.py 主表 + taskgraph-baseline 模式 |
| `ef4e1ab` | docs: integration + baseline plan; fix the ncu-on-cooperative-kernel hazard |
| `b46d23f` | build: make TG_B_STAGES env-overridable, add taskgraph-perf mode |
| `1a7539b` | docs: fill §10.4 with A100 real-hardware results |
| `9c7b707` | fix: use uint32 for inst_buf in tools (barrier count overflow) |
| `c8dbb33` | fix: eliminate cross-warp global memory races with fence.acq_rel.gpu |
| `e0c50df` (origin) | feat: honor row_active in the attention KV write |
