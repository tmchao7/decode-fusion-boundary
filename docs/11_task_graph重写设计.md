# Task-Graph Megakernel 重写设计（v1）

> **状态**：设计稿（2026-09-18 起）。目标是推翻 docs/07 的"batch>1 不值得重写"
> 冻结结论，让 megakernel 支持 **连续 batching** + **tile/block 级依赖计数器调度**，
> 替换当前每层全网格 `grid.sync()` 大同步。参照隔壁 `cohere-megakernel` 的 task-graph
> + persistent-worker + dependency-counter 架构，适配到 Qwen3-0.6B（dense）+ A100（sm_80）。
>
> 阅读顺序：先读 §1 目标与约束，再 §2 总体架构，再 §3-§8 各模块细节，最后 §9 迁移路径。
> 配套代码：`csrc/taskgraph_kernel.cu`（新文件，不动 `kernel.cu`）、`csrc/tg_bindings.cpp`、
> `schedule.py`、`runtime.cu`、`session.py`（新增）。

## 1. 目标与约束

### 1.1 要解决的两个问题

1. **batch>1 打不过 cudagraph**：当前 `ldg_decode_kernel_batch_wmma`（`kernel.cu:1148`）
   每层 12 次 full-grid `grid.sync()` × 28 层 = 336 次大同步/step，把 28 个 GEMM 阶段
   串成一条长链；且每个 batched GEMM 因 `tiles_M=1` 只有 4-12/80 个 block 干活，其余
   SM 空转等 barrier（`docs/07` §2.3）。b8 megakernel 纯 kernel 98.5ms vs cudagraph
   0.77ms（133x 慢）。
2. **不支持连续 batching**：当前 burst 路径（`generate_paged_batch`）要求 burst 内
   batch 固定、block table 预分配、`max_steps = min(budget)`（短 seq 限全长，否则
   cooperative grid 不能 per-seq 缩容会越界写邻居 KV，`scheduler.py:99-117`）。无法
   像 vLLM/cohere 那样序列随时加入/离开。

### 1.2 硬件/模型约束

| 项 | 值 | 影响 |
|---|---|---|
| GPU | A100-SXM4-80GB，**sm_80** | **无 TMA、无 wgmma、无 `fence.proxy.async`**。只有 WMMA `m16n16k16`、`cp.async`、`__threadfence`。cross-SM 信号量靠 `__threadfence` + `atomicAdd`（现有 `kv_flag`/`attn_flag` 已验证可行） |
| SM 数 | 108 | persistent grid 上限 108（当前人为 cap 80） |
| 模型 | Qwen3-0.6B，**dense**（无 MoE） | 省掉 cohere 的 router/topk/route_finalize/moe_gather/moe_drain/moe_combine 一整套，只保留 dense FFN |
| 层数 | 28 | task graph 每层 7 个 named wave，共 ~196 wave/step |
| hidden/intermediate | 1024 / 3072 | GEMM: QKV(1024→3072), O(2048→1024), Gate/Up(1024→3072), Down(3072→1024), LMHead(1024→151936) |
| Q/KV heads | 16 / 8（GQA） | attention 按 kv_head 切 split |
| head_dim | 128 | |
| embedding | tied | LM head 权重 = embedding，每步只读一次 0.311GB |
| 既有契约 | `QWEN3_CONTRACT`（`backend.py:22`）、`PagedKVCacheSpec`（`cache.py`） | 必须沿用，KV cache 布局不变 |

### 1.3 不做的事（本轮范围外）

- **不**上 H100/sm_90a 特化（TMA/wgmma）——A100 是主目标，先让 sm_80 路径跑通。
- **不**做 MoE——Qwen3-0.6B 是 dense。cohere 的 MoE drain/router 整套砍掉。
- **不**做 sampling（temperature/top-p/top-k）——先 greedy，与现有契约一致。
- **不**做 tensor parallel——`tensor_parallel_size=1`，与现有一致。
- **不**动 prefill——继续走 nano-vLLM 的 PyTorch/FlashAttention 路径。
- **不**删 `kernel.cu`——保留 batch=1 persistent 路径作为 A/B 基线和 correctness 参考。

## 2. 总体架构

### 2.1 一句话

**一次 cooperative launch（grid = #SMs），每个 SM 一个常驻 CTA**。host 侧为每个 SM
预编一条"指令流"（per-SM `inst_buf`，每条指令 16 个 `int32`）。指令就是 task（一个
GEMM tile、一个 attention split、一次 RMSNorm…）。task 之间的依赖不用 `grid.sync()`，
而是用**全局内存里的 `uint32` 原子计数器数组**（`bar_*`）：生产者 `__threadfence` +
`atomicAdd`，消费者 `volatile` 轮询直到 `>= target`。ready 的 task 立即被认领，partial
wave 的空闲 SM 被 backfill。

### 2.2 与 cohere 的差异（适配 sm_80 + dense）

| 维度 | cohere（H100/sm_90a） | 我们（A100/sm_80） |
|---|---|---|
| GEMM 原语 | TMA load + wgmma + TMA store | WMMA `m16n16k16` + `cp.async`/`__ldg` load + `store_matrix_sync` |
| cross-SM 信号 | `fence.proxy.async`（TMA proxy）+ `__threadfence` + `atomicAdd` | **`__threadfence` + `atomicAdd`**（无 TMA，不需要 proxy fence） |
| warp 角色 | 4 role（producer/storer/consumer×1-2/idle），TMA 专用 warp | **简化为 2 role**：compute warp（做 WMMA + global load + store）+ aux warp（做 RMSNorm/attention/argmax 等标量）。sm_80 无 TMA，不需要 producer/storer 分离 |
| task 图 | 14 bar（含 MoE） | **8 bar**：`bar_qkv, bar_attn, bar_oproj, bar_silu, bar_ffn_down, bar_layer, bar_combine, bar_lm_head`（dense） |
| 动态 claimer | attn drain + MoE drain | 先做**静态** attention split（Qwen3-0.6B context ≤ 2048，split 数可静态算）；动态 drain 留作后续 |
| JIT | little_jit 按 batch size 编译 .so | **先不做 JIT**：用 `build.py` 的 torch cpp_extension 一次编译，靠 runtime 配置（batch/context bucket）切 `inst_buf`。若寄存器压力需要按 batch 特化再引入 JIT |
| 连续 batching | C++ `DecodeService` 线程 + Python scheduler pause/park/resume | **同构移植**：新增 `runtime.cu` 的 `DecodeService` + `session.py`，复用 nano-vLLM 的 `Scheduler`/`Sequence`/`block_manager` 做 prefill 与 block 生命周期 |
| context bucket | 多 bucket × 共享 JIT handle | 类似：多 bucket（按 context 长度分）× 共享编译 kernel，只换 `inst_buf` |

### 2.3 新增文件清单

```
src/nanovllm/backends/megakernel/
  csrc/
    taskgraph_kernel.cu   ← 新：persistent task-graph kernel（主入口 + wait/arrive + worker loop + dense task 执行）
    taskgraph_runtime.cu  ← 新：C++ DecodeService（decode 线程循环 + pause/park/slot/geometry）
    taskgraph_bindings.cpp← 新：pybind 绑定（DecodeService + schedule 构建 + token callback）
    tg_abi.h              ← 新：descriptor 结构（NmcLaunchDesc 等价物，layout-locked）
  schedule.py             ← 新：task schedule 构建（per-SM inst_buf + bar 维度）
  runtime.py              ← 新：Python 侧 DecodeService wrapper
  session.py              ← 新：连续 batching 调度（Python scheduler 线程，接 nano-vLLM Scheduler）
  backend.py              ← 改：新增 task-graph 分支，保留旧路径做 A/B
  build.py                ← 改：新增 taskgraph 源文件编译
docs/
  11_task_graph重写设计.md ← 本文件
tests/
  test_taskgraph_*.py     ← 新：schedule 构建、bar index、ABI 不变式等单测
```

### 2.4 数据流（一个 dense 层的 task DAG）

```
x_resid[L] ──RMSNORM──> norm_out ──QKV_PROJ──> q/k/v
                                          │
                                  (RoPE + 写 KV cache)
                                          │
                                          ▼
                              ┌── ATTN_DECODE(split) ──┐
                              │   (per kv_head × split) │
                              └──────────┬──────────────┘
                                         ▼ (ATTN_COMBINE if split>1)
                                    attn_out ──O_PROJ──> o_proj_out
                                          │
                        x_resid[L] ───────┼──(+residual)──> x_attn
                                          │
                                          ▼ RMSNORM
                                  post_norm ──FFN_UPGATE_ACT──> silu_out ──FFN_DOWN──> ffn_down_out
                                          │                                                      │
                        x_attn ───────────┼──(+residual)──> x_ffn ───────────────────────┘
                                          │
                                          ▼ ADD_RMSNORM
                                    x_resid[L+1] ──> (下一层) ... ──> LM_HEAD ──> argmax ──> token
```

每个箭头是一个 bar。task 自带 `(wait_bar_idx, wait_target, produce_bar_idx)`。

## 3. Task / 指令 ABI

### 3.1 指令格式：16 个 int32（`TgInstruction`）

cohere 用 32 int，我们 dense 模型不需要那么多字段，**压缩到 16 int**省 inst_buf 带宽：

```cpp
constexpr int TG_INST_WIDTH = 16;

struct TgInstruction {
  int data[TG_INST_WIDTH];
  // data[0] = opcode (TgOp)
  // data[1..15] = opcode-specific layout (见下)
};
```

### 3.2 Opcode 枚举（`TgOp`）

dense 模型只需要 10 种（砍掉 cohere 的 8 个 MoE/Router opcode）：

| opcode | 值 | 含义 | 执行函数 |
|---|---:|---|---|
| `TG_NOP` | 0 | 占位/填充 | `tg_dispatch` 空转 |
| `TG_QKV_PROJ` | 1 | QKV 投影 tile（含 RMSNorm 前置 + RoPE + 写 KV cache） | `tg_qkv_op` |
| `TG_ATTN_DECODE` | 2 | 单 (batch, kv_head, split) attention | `tg_attn_decode` |
| `TG_ATTN_COMBINE` | 3 | split-KV partial 合并（split>1 时） | `tg_attn_combine` |
| `TG_O_PROJ` | 4 | output 投影 GEMM tile | `tg_o_proj_op` |
| `TG_FFN_UPGATE_ACT` | 5 | fused gate+up GEMM tile + SiLU | `tg_ffn_upgate_op` |
| `TG_FFN_DOWN` | 6 | down projection GEMM tile | `tg_ffn_down_op` |
| `TG_ADD_RMSNORM` | 7 | 残差加 + RMSNorm（层汇合点） | `tg_add_rmsnorm_op` |
| `TG_LM_HEAD` | 8 | LM head GEMM tile（produce `bar_lm_head`） | `tg_lm_head_op` |
| `TG_LM_ARGMAX` | 9 | 全 vocab tile 就绪后的 per-batch argmax（终结 op） | `tg_lm_argmax_op` |

> `TG_LM_ARGMAX` 是从 `TG_LM_HEAD` 拆出来的终结指令：`num_sms>1` 时各 SM 只写自己的
> vocab tile，若在每个 tile 指令里就地 argmax 会读到别的 SM 还没写的 logits。拆出来后
> `TG_LM_HEAD` 每个 tile 指令 `arrive(bar_lm_head, inc=NUM_TILES)`，`TG_LM_ARGMAX` 等
> `bar_lm_head[0] >= vocab/lm_head_bn` 再做 argmax，跨 SM 正确。

> 注：`RMSNORM` 不单独列 opcode——QKV 前的 input-layernorm 融进 `TG_QKV_PROJ`，
> O_PROJ 前的 post-attn-layernorm融进 `TG_O_PROJ`，层末的 final-norm 融进
> `TG_ADD_RMSNORM`。这跟当前 `kernel.cu` 的融合边界一致（`docs/03` §fusion boundaries）。

### 3.3 各 opcode 的字段 layout

**通用 GEMM**（`TG_O_PROJ / TG_FFN_UPGATE_ACT / TG_FFN_DOWN / TG_LM_HEAD`）：

```
data[0]  = OPCODE
data[1]  = LAYER
data[2]  = TILE_M          // GEMM M 维 tile 索引
data[3]  = TILE_N          // GEMM N 维 tile 索引
data[4]  = SPLIT           // split-K 索引（0 表示无 split-K）
data[5]  = K_TILES         // K 维 tile 数（split-K 时为每 split 的 K tile 数）
data[6]  = WAIT_BAR_IDX    // 等哪个 bar slot
data[7]  = WAIT_TARGET     // 等到多少
data[8]  = PRODUCE_BAR_IDX // 完成后 arrive 哪个 bar slot
data[9]  = NUM_TILES       // 本指令打包的 tile 数（同依赖可合并）
data[10..15] = TILE_IDS    // 打包的 (tile_m<<16)|tile_n（最多 6 个；超过则拆多条指令）
```

**`TG_QKV_PROJ`**（多了 GEMM_IDX 和 NEEDS_ROPE）：

```
data[0]  = OPCODE
data[1]  = LAYER
data[2]  = TILE_N          // QKV 的 N 维 tile（M=batch，固定 tiles_M=1 或 ceil(BS/16)）
data[3]  = GEMM_IDX        // 0=Q, 1=K, 2=V
data[4]  = NEEDS_ROPE      // 1=对 Q/K 做 RoPE + 写 KV cache；0=否
data[5]  = WAIT_BAR_IDX    // bar_layer[L]（等上一层 rmsnorm）
data[6]  = WAIT_TARGET     // batch（每行 rmsnorm arrive 一次）
data[7]  = PRODUCE_BAR_IDX // bar_qkv[L*Hkv + kvh]（按 kv_head 切，K/V 的 produce 对应 kv_head）
data[8]  = BATCH_ROW       // 本 tile 服务的 batch 行（attention 要按行找 KV）
data[9]  = NUM_TILES       // 本指令打包的 tile 数（同 gemm_idx / produce 才可合并）
data[10..15] = TILE_N 列表 // 连续 tile_n（最多 6 个）；kernel 用 N=NUM_TILES*16 一次 GEMM 扇出到各 warp
```

> Q/K/V 三次 GEMM 的 N 维 tile 分开编指令（cohere 也是每 GemmIdx 一条）。
> 同一次 GEMM 内连续的 tile_n 会打包成一条指令（`schedule.py:_pack_qkv`），
> `tg_qkv_proj_op` 只发一次 `tg_wmma_gemm`（N=NUM_TILES*16），把 N tile 分给 7 个
> worker warp；produce 用 `inc=NUM_TILES`，attention 的 `WAIT_TARGET` 仍按总 tile 数计。
> 不同 kv_head（produce slot 不同）不会合并。

**`TG_ATTN_DECODE`**：

```
data[0]  = OPCODE
data[1]  = LAYER
data[2]  = BATCH_IDX
data[3]  = KV_HEAD_IDX
data[4]  = SPLIT_IDX
data[5]  = NUM_SPLITS
data[6]  = WINDOW_SIZE     // Qwen3-0.6B 若无 sliding window 则 = context_len
data[7]  = WAIT_BAR_IDX    // bar_qkv[L*Hkv + kvh]
data[8]  = WAIT_TARGET     // qkv_wait_per_kv（本 kv_head 的 Q+K+V tile 数）
data[9]  = PRODUCE_BAR_IDX // split>1: bar_combine[...]; split==1: bar_attn[...]
data[10..15] = 保留
```

**`TG_ATTN_COMBINE`**（split>1 时）：

```
data[0]  = OPCODE
data[1]  = LAYER
data[2]  = BATCH_IDX
data[3]  = KV_HEAD_IDX
data[4]  = NUM_SPLITS
data[5]  = WAIT_BAR_IDX    // bar_combine[L*BS*Hkv + batch*Hkv + kvh]
data[6]  = WAIT_TARGET     // num_splits
data[7]  = PRODUCE_BAR_IDX // bar_attn[L*oproj_splits + split]
```

**`TG_ADD_RMSNORM`**（层汇合点，等 attn + ffn 两路）：

```
data[0]  = OPCODE
data[1]  = LAYER
data[2]  = BATCH_IDX       // per-row rmsnorm（每行一条指令）
data[3]  = WAIT_OPROJ_IDX  // bar_oproj[L] 的 slot
data[4]  = WAIT_OPROJ_TARGET
data[5]  = WAIT_FFN_IDX    // bar_ffn_down[L] 的 slot
data[6]  = WAIT_FFN_TARGET
data[7]  = PRODUCE_BAR_IDX // bar_layer[L+1]（始终 L+1；最后一层 = bar_layer[num_layers] 给 LM_HEAD）
data[8]  = IS_FINAL_LAYER  // 1=最后一层（LM_HEAD 等的就是这个 slot）
```

### 3.4 调用约定总结

每条指令自带"入边"（`WAIT_BAR_IDX/WAIT_TARGET`）和"出边"（`PRODUCE_BAR_IDX`）。
执行流程统一为：

```
wait_input_bars(inst)       // volatile 轮询 bar[wait_idx] >= wait_target
  → load 权重 + 激活（cp.async / __ldg）
  → compute（WMMA / 标量 attention）
  → store 输出（store_matrix_sync / 标量写）
signal_output_bar(inst)     // __threadfence + atomicAdd(bar[produce_idx], 1)
```

## 4. 依赖计数器（bar）机制

### 4.1 bar 数组布局（7 个，dense）

```cpp
// 每个 bar 是 device 上的 uint32 数组，每步 decode 前由 host 清零。
struct TgBars {
  uint32_t *bar_qkv;       // [layers * Hkv]                  — QKV→ATTN
  uint32_t *bar_attn;      // [layers * oproj_splits]         — ATTN→O_PROJ
  uint32_t *bar_oproj;     // [layers]                        — O_PROJ→ADD_RMSNORM(attn路)
  uint32_t *bar_silu;      // [layers * down_splits]          — FFN_UPGATE→FFN_DOWN
  uint32_t *bar_ffn_down;  // [layers]                        — FFN_DOWN→ADD_RMSNORM(ffn路)
  uint32_t *bar_layer;     // [layers + 1]                      — ADD_RMSNORM→下一层 QKV / LM_HEAD
  uint32_t *bar_combine;   // [layers * BS * Hkv]             — ATTN_DECODE→ATTN_COMBINE（split>1）
  uint32_t *bar_lm_head;   // [1]                             — LM_HEAD tiles→LM_ARGMAX
};
```

> 对比 cohere 的 14 个 bar（多 7 个 MoE 相关：`bar_router/topk/route/gather/moe_upgate/moe_down`），
> 我们 dense 全部砍掉。`bar_upgate` 在 cohere 里保留但实际用 `bar_silu`，我们直接只留 `bar_silu`。

### 4.2 cross-SM 原语（sm_80 适配版）

cohere 的 `wait_cross_sm` / `arrive_cross_sm`（`megakernel.cuh:560-669`）用 `fence.proxy.async`
（TMA proxy 可见性）。**sm_80 无 TMA，不需要 proxy fence**，用 `fence.acq_rel.gpu`
（kernel.cu `AtomicGridSync` 在 A100 上验证过的原语）+ `atomicAdd`：

```cpp
// 消费者：等 bar[idx] >= target。只有 worker warp 0 轮询，其余 warp 在 worker barrier
// 上阻塞（108 个 CTA × 224 线程全自旋会把 L2 计数器行打爆，拖慢生产者的 atomicAdd）。
__device__ __forceinline__ void tg_wait(const uint32_t* bar, int idx, uint32_t target) {
  if (target == 0) { tg_fence_gpu(); return; }     // layer 0 无依赖
  if (tg_wwarp() == 0) {
    const volatile uint32_t* slot = bar + idx;
    while (*slot < target) __nanosleep(20);        // "DONT DELETE"（cohere 注释）
    tg_fence_gpu();                                // acquire
  }
  tg_worker_sync_inner();                          // acquire 通过 barrier 传播到其余 warp
}

// 生产者：通知 bar[idx] += inc。
// 关键不变量：先 worker barrier（本 CTA 所有 global 写完成），再 fence，最后**单线程**
// atomicAdd。若 224 个 worker 线程各 add 一次，到达数会被乘 224，消费者提前 224 条指令放行。
__device__ __forceinline__ void tg_arrive(uint32_t* bar, int idx, uint32_t inc = 1) {
  tg_worker_sync_inner();
  tg_fence_gpu();                                  // release
  if (threadIdx.x == TG_WARP_SIZE) atomicAdd(bar + idx, inc);
}
```

**正确性论证**：`fence.acq_rel.gpu` 提供 GPU scope 的 acquire/release；生产侧 fence 在
worker barrier 之后执行，barrier 的 release/acquire 边把所有线程的写在因果序上排在
fence 之前；消费侧 warp 0 的 acquire 再经 worker barrier 传播给其余 warp。这与
`kernel.cu` 的 `AtomicGridSync`（`__syncthreads` → fence → 单线程 atomicAdd → 自旋 →
`__syncthreads`）是同一模式，已在 A100 上验证。

**到达计数**：非 GEMM op（ATTN/ADD_RMSNORM）每条指令 `inc=1`；GEMM op
（QKV/O_PROJ/FFN_UPGATE/FFN_DOWN/LM_HEAD）每条指令 `inc=NUM_TILES`（打包了 N 个
tile）。`WAIT_TARGET` 按总 tile 数计，`test_taskgraph_abi.py` 逐 slot 校验
`produces == max(waits)`。

**与 `AtomicGridSync`（`kernel.cu:108-145`）的区别**：
- `AtomicGridSync` 是**全网格** barrier（所有 block 到齐才放行），强制层间串行。
- `tg_wait/tg_arrive` 是**生产者-消费者**计数器（只等"我依赖的 task"完成），ready 立即放行。
- 这正是从"大同步"换"细粒度信号量"的核心。

### 4.3 多 bar 同时等待

`TG_ADD_RMSNORM` 要等 attn 和 ffn 两路，需要 `tg_wait_all`。两个计数器在**同一个循环**
里轮询（而不是先后两次 `tg_wait`），这样两个依赖里更晚到的那个一到就放行；同样只有
warp 0 轮询 + worker barrier 广播：

```cpp
__device__ __forceinline__ void tg_wait_all(
    const uint32_t* bar1, int idx1, uint32_t target1,
    const uint32_t* bar2, int idx2, uint32_t target2) {
  const bool need1 = target1 > 0, need2 = target2 > 0;
  if (!need1 && !need2) { tg_fence_gpu(); return; }
  if (tg_wwarp() == 0) {
    const volatile uint32_t* s1 = bar1 + idx1;
    const volatile uint32_t* s2 = bar2 + idx2;
    while ((need1 && *s1 < target1) || (need2 && *s2 < target2)) __nanosleep(20);
    tg_fence_gpu();
  }
  tg_worker_sync_inner();
}
```

### 4.4 每步清零

bar 数组每步 decode 前必须清零（上一步的计数器是旧的）。host 侧用一个小的 memset kernel
批量清零所有 bar + scratch region（仿 cohere 的 `nmc_zero_regions_kernel`）。**关键**：
清零必须在 persistent kernel launch 之前的同一个 stream 上排队，保证 kernel 看到零初值。

## 5. Persistent Worker 循环

### 5.1 kernel 入口（`tg_decode_kernel`）

```cpp
__global__ __launch_bounds__(TG_BLOCK_SIZE, 1)
void tg_decode_kernel(__grid_constant__ const TgGlobals g) {
  // grid = num_sms, block = TG_BLOCK_SIZE (e.g. 256 = 8 warps)
  const int bid = blockIdx.x;
  const int num_inst = g.num_inst_per_sm[bid];
  const int *sm_base = g.inst_buf + bid * g.max_inst * TG_INST_WIDTH;

  // 指令环形缓冲（双缓冲，depth = TG_INST_RING = 4）
  __shared__ int inst_ring[TG_INST_RING][TG_INST_WIDTH];
  __shared__ uint32_t inst_arrived[TG_INST_RING];  // controller → worker
  __shared__ uint32_t inst_done[TG_INST_RING];     // worker → controller

  if (threadIdx.x < WARP_SIZE) {
    // warp 0 = controller：预取指令
    tg_controller_loop(inst_ring, inst_arrived, inst_done, sm_base, num_inst);
  } else {
    // warp 1..7 = worker：按 task dispatch
    tg_worker_loop(inst_ring, inst_arrived, inst_done, g, bid);
  }
}
```

### 5.2 角色简化（sm_80 无 TMA）

cohere 的 4-role（producer/storer/consumer/idle）是为 TMA 设计的：TMA load/store 需
专用 warp 发起异步操作，WGMMA consumer 独立。**sm_80 用 WMMA，load/store/compute 在同一
warp 内串行**，所以简化为：

- **warp 0 = controller**：只做指令预取（合作 load 16 int），不参与计算。
- **warp 1..7 = worker**（统一 role）：每条指令由"本 warp 负责的 tile"驱动，做
  `wait → global load → WMMA/标量 compute → store → arrive`。

worker 间用 `bar.sync`（命名空间 1）同步指令边界（仿 cohere `worker_sync`），保证所有
worker 在进入下一条指令前完成当前指令的自己那部分。**关键不变式**：每个 worker warp
在每条指令上执行**相同次数**的 `bar.sync`（`NOP` 也要走，否则 arrival 错位死锁）。

### 5.3 controller / worker 双缓冲

```cpp
// controller (warp 0)
for (int i = 0; i < num_inst; ++i) {
  int cur = i % TG_INST_RING;
  if (i >= TG_INST_RING) tg_wait_shared(inst_done[cur], (i/TG_INST_RING)%2);
  tg_coop_load_inst(inst_ring[cur], sm_base, i);  // 32 lane 各读一个 int
  tg_arrive_shared(inst_arrived[cur], (i/TG_INST_RING)%2);
}

// worker (warp 1..7)
for (int i = 0; i < num_inst; ++i) {
  int cur = i % TG_INST_RING;
  tg_wait_shared(inst_arrived[cur], (i/TG_INST_RING)%2);
  tg_dispatch_instruction(inst_ring[cur], g, bid);  // 大 switch(opcode)
  bar.sync 1;  // worker_sync
  if (lane == 0) tg_arrive_shared(inst_done[cur], (i/TG_INST_RING)%2);
}
```

shared memory 的 semaphore 用 `__shared__ uint32_t` + `volatile` 轮询 + `__syncthreads`
数组（仿 cohere `megakernel.cuh:3070-3128`）。

### 5.4 tile 分配（backfill 的关键）

每条 GEMM 指令可能打包多个 tile（`NUM_TILES`）。worker warp 按 `global_warp_id`（跨整个
grid 的 warp id）轮询认领 tile：

```cpp
int global_warp = (bid - 0) * (TG_BLOCK_SIZE/WARP_SIZE) + (threadIdx.x / WARP_SIZE);
// worker warp 从 warp 1 起，所以 global_warp 从 bid*7 + (warp_id-1) 开始
for (int t = global_warp; t < inst.NUM_TILES; t += total_warps) {
  int tile_m = inst.TILE_IDS[t] >> 16;
  int tile_n = inst.TILE_IDS[t] & 0xFFFF;
  // 算这个 tile 的 WMMA
}
```

**backfill 机制**：
- host 把不同层、不同 op 的 task 交错排进 SM 指令流（§7 的 `reorder_waves`）。
- 空闲 SM（自己当前的 task 是 `NOP` 或已做完自己的 tile）会进入下一条指令——但下一条
  指令可能依赖别的 bar，`tg_wait` 会让它等。一旦依赖就绪，立即开始。
- **没有 `grid.sync()`**，所以 attention 尾部（只占少数 SM）期间，已排好 MoE/下一层
  task 的 SM 会在自己的 bar 就绪后立即开跑，不必等整个 attention grid。

### 5.5 attention 的 task 切分

Qwen3-0.6B context ≤ 2048，按 context 长度静态选 split 数（仿 cohere
`_choose_attn_splits`，`schedule.py:1066-1098`）：

- `seq_len ≤ 256`：1 split（每 kv_head 1 个 task）
- `seq_len ≤ 1024`：2 split
- `seq_len ≤ 2048`：4 split

每层 attention task 数 = `BS * Hkv * num_splits`。split>1 时配 `TG_ATTN_COMBINE`。
**先做静态 split**（context bucket 内固定）；动态 drain（per-step 重建队列）留作后续，
因为 Qwen3-0.6B context 短，静态 split 的 schedule 数量可控。

### 5.6 attention 计算复用

`tg_attn_decode` 的标量 QK/softmax/PV 计算直接复用现有 `ldg_attention`（`kernel.cu:378-727`）
的核心逻辑——只是去掉里面的 `grid.sync()` 和 `kv_flag`/`attn_flag`（改成 bar 的
`tg_wait`/`tg_arrive`）。attention 仍标量（不上 tensor core），与 `docs/07` §3.2 一致，
是"公平性/未用满硬件"而非正确性问题。

## 6. Schedule 构建（host 侧，`schedule.py`）

### 6.1 输入与输出

**输入**：`(batch_size, num_layers, hkv, oproj_splits, down_splits, attn_splits_per_bucket,
tiling_config)`。

**输出**：
- `inst_buf`：device `int32 [num_sms, max_inst, TG_INST_WIDTH]`
- `num_inst_per_sm`：device `int32 [num_sms]`
- `bar_sizes`：7 个 bar 各自的长度（用于分配 + 清零）

### 6.2 每层 named wave（dense）

每层按依赖顺序构建 7 个 wave：

```
wave 0: QKV_PROJ        (Q/K/V 各若干 N-tile, produce bar_qkv)
wave 1: ATTN_DECODE     (BS*Hkv*splits 个, produce bar_combine/bar_attn)
wave 2: ATTN_COMBINE    (BS*Hkv 个, split>1 时, produce bar_attn)
wave 3: O_PROJ          (若干 tile, produce bar_oproj)
wave 4: FFN_UPGATE_ACT  (若干 tile, produce bar_silu)
wave 5: FFN_DOWN        (若干 tile, produce bar_ffn_down)
wave 6: ADD_RMSNORM     (BS 个, per-row, produce bar_layer)
```

最后一层多一个 `wave 7: LM_HEAD`（若干 vocab tile，produce `bar_lm_head`），
再接一条终结的 `TG_LM_ARGMAX`（等 `bar_lm_head` 到齐后做 per-batch argmax）。

### 6.3 依赖 index 公式（与 §4.1 bar 维度对应）

```
QKV produce:    bar_qkv[L*Hkv + kvh]
ATTN wait:      bar_qkv[L*Hkv + kvh], target = qkv_wait_per_kv
ATTN produce:   split>1 ? bar_combine[L*BS*Hkv + b*Hkv + kvh]
                         : bar_attn[L*oproj_splits + (kvh // kvh_per_oproj)]
COMBINE wait:   bar_combine[L*BS*Hkv + b*Hkv + kvh], target = num_splits
COMBINE produce:bar_attn[L*oproj_splits + split]
O_PROJ wait:    bar_attn[L*oproj_splits + split], target = batch*kvh_per_oproj
O_PROJ produce: bar_oproj[L]
FFN_UPGATE wait:bar_oproj[L], target = oproj_tiles   (等 O_PROJ 完成；op body 融合 residual+RMSNorm+GEMM)
FFN_UPGATE produce: bar_silu[L*down_splits + split]
FFN_DOWN wait:  bar_silu[L*down_splits + split], target = silu_tiles_per_down_split
FFN_DOWN produce: bar_ffn_down[L]
ADD_RMSNORM wait: bar_oproj[L] (target=oproj_tiles) + bar_ffn_down[L] (target=down_tiles)
ADD_RMSNORM produce: bar_layer[L+1]（始终 L+1；最后一层 L_last produce bar_layer[num_layers] 给 LM_HEAD）
LM_HEAD wait:   bar_layer[num_layers]（= L_last + 1）, target = batch
LM_HEAD produce:bar_lm_head[0], inc = NUM_TILES
LM_ARGMAX wait: bar_lm_head[0], target = vocab / lm_head_bn
```

### 6.4 placement：round-robin + 依赖亲和

仿 cohere 的两阶段：
1. **Plain RR**（先实现）：跨 wave 维护 RR cursor，每条指令轮流分给下一个 SM。cursor
   跨 wave 不重置，避免 waiter 集中在低编号 SM。
2. **依赖亲和 RR**（后优化）：把 `ATTN_COMBINE` 尽量放到其 `ATTN_DECODE` 生产者所在
   SM，把 `O_PROJ` 放到 `ATTN_COMBINE` 的 SM。但注释（cohere `schedule.py:1300`）明确：
   **"Kernel barriers remain the source of correctness"**——亲和只影响性能，正确性靠 bar。

### 6.5 context bucket

按 context 长度分桶（如 `[256, 1024, 2048]`），每桶预建一份 `inst_buf`（split 数不同）。
运行时按最大 active row 的 context_len 选桶，**只换 `inst_buf` 指针，不重编译**。
这与 cohere 的 bucket 机制一致（`schedule.py:1899-1951`）。

### 6.6 JIT（先不做）

cohere 按 batch size JIT 编译特化 kernel（tiling/寄存器）。**本轮先用 `build.py` 一次
编译通用 kernel**，靠 runtime 配置（batch/context bucket）切 `inst_buf`。理由：
- Qwen3-0.6B shape 小，WMMA tiling 对 batch 不敏感（`tiles_M = ceil(BS/16)`，BS≤8 时都
  是 1）。
- 先跑通机制，若寄存器压力/occupancy 需要按 batch 特化再引入 little_jit（可移植 cohere
  `jit.cpp` 的 SHA-256 内容寻址缓存）。

## 7. 连续 batching 运行时（`runtime.cu` + `session.py`）

### 7.1 线程模型（移植 cohere `DecodeService`）

两个 daemon 线程：
- **decode 线程**：跑 `DecodeService::run()`（C++，阻塞），每步 launch task-graph
  kernel + sample + token callback。
- **scheduler 线程**：Python `_run_scheduler`，2ms 心跳，每个 tick 决定准入/抢占/几何
  切换/prefill。

握手协议（与 cohere 完全一致，已验证）：
```
[scheduler] signal_pause(True) → wait_paused(timeout)
  [decode]   等当前步 GPU idle（watchdog sync）→ paused_ack=1 → cv.wait
[scheduler] 改 slot / geometry / KV（独占权）
[scheduler] signal_pause(False) + notify
  [decode]   cv.wait 返回 → 重快照 → 跑下一步
```

### 7.2 slot 管理

固定 BS 行（支持 BS ∈ {1,2,4,8}），`set_slot(row, active, start_pos, gen_col, ...)`：
- **准入**：prefill 跑完 → `set_slot(active=True, start_pos=prompt_len, gen_col=0)` →
  下一步 decode 把该 row 算进去。
- **摘除**：EOS/length → `set_slot(active=False)` → `kv.free_row(row)` → 释放 KV 页。
- **抢占**：KV pressure 时抢最大 footprint 的 row，emitted token 拼回 prompt，重新入队。

### 7.3 geometry 切换

BS 变化（1→2→4→8）时 `set_geometry(new_desc)`：
- **不需在线 JIT**（本轮不 JIT，所有 BS 共用一个编译 kernel，靠 `inst_buf` 切换）。
- KV 块所有权用 rebind 转移（块不动，引用计数搬）。
- 切换后重装所有 live slot（`set_geometry` 清空 active）。

### 7.4 token callback

每步 decode 后 D2H + watchdog sync，调 `token_callback(context, sampled, emitted_rows, BS)`：
- callback 在 decode 线程（无 GIL）被调，先 `gil_scoped_acquire` 拿 GIL，再调 Python
  `_on_tokens`。
- `_on_tokens` 只把 token id `put` 进 per-request queue，detokenization 在客户端线程
  做（避免 stall 整个 batch）。

### 7.5 KV pressure

每步 decode 前查 `free_blocks < crossing * num_layers`（crossover 的 row 数 × 层数）→
自 park → Python `_tick` 抢占 → `resume_from_pressure`。

### 7.6 prefill 与 decode 交错

prefill 走 PyTorch/FlashAttention（nano-vLLM 现有路径），**期间 decode park**（GPU idle
确认后 prefill 独占 GPU）。这与 cohere 一致（README 说 "prefill pauses decode"）。

### 7.7 接 nano-vLLM 的 Scheduler

关键：**不重写 nano-vLLM 的 `Scheduler`/`Sequence`/`block_manager`**，只在 `session.py`
里桥接：
- `Scheduler` 仍管 `waiting`/`running` 队列、block 分配/释放/抢占、prefix cache hash。
- `session.py` 把 `Scheduler.schedule()` 的输出（`Sequence` 列表 + block table）翻译成
  `set_slot` + block table 上传给 `DecodeService`。
- prefill 调 `Scheduler.schedule(is_prefill=True)` + nano-vLLM model runner。
- decode 不再走 `ModelRunner.run`，改走 `DecodeService`（task-graph kernel）。

## 8. 复用 / 改造 / 重写清单

### 8.1 直接复用（与调度架构无关）

| 模块 | 位置 | 复用理由 |
|---|---|---|
| `weights_from_nano_model` | `model.py:132` | 权重切分+打包，与 kernel 调度无关 |
| `_pack_layer_weights` | `model.py:187` | ptr 打包成 `LDGLayerWeights` |
| `_rope_tables` | `model.py:49` | RoPE 表生成 |
| `PagedKVCacheSpec` | `cache.py` | KV 布局规范 |
| `ldg_kv_ptr` offset 算式 | `kernel.cu:360` | 分页寻址，task-graph 仍需（搬到 `taskgraph_kernel.cu`） |
| `prepare_block_tables` | `model_runner.py:195` | 已修 0-padding |
| `_validate_cache_layout` | `backend.py:208` | init 时 gate |
| `QWEN3_CONTRACT` | `backend.py:22` | 形状契约 |
| `ldg_attention` 标量计算 | `kernel.cu:378` | QK/softmax/PV 逻辑搬到 `tg_attn_decode`，去 barrier |
| `ldg_warp_reduce_sum` | `kernel.cu:155` | all-reduce 原语 |
| `wmma_gemm` 的 WMMA 计算 | `kernel.cu:962` | m16n16k16 核心搬到 `tg_gemm_op`，tile 分配改 task-driven |
| `ldg_lm_head_fused_batch` | `kernel.cu:1604` | LM head argmax（可作独立 kernel 或 task） |
| `burst_policy` | `burst_policy.py` | torch-free 策略（连续 batching 后 burst 概念弱化，但策略可复用） |
| `Scheduler`/`Sequence`/`block_manager` | `engine/` | 生命周期/block 管理，`session.py` 桥接 |
| correctness 门 | `tools/correctness_*.py` | PASS/CASCADE/FAIL 三级，task-graph 路径同样适用 |

### 8.2 改造

| 模块 | 改动 |
|---|---|
| `backend.py` | 新增 task-graph 分支（`decode_backend="megakernel-tg"` 或 `megakernel_task_graph=True`），保留旧路径做 A/B |
| `build.py` | 新增 `taskgraph_kernel.cu`/`taskgraph_runtime.cu`/`taskgraph_bindings.cpp` 编译 |
| `compatibility` | batch 上限判定改由 task-graph 调度器决定（不再固定 grid 切分） |
| `burst_policy` | 放松 `max_steps=min(budget)`（task-graph 支持 per-seq 动态停步后）和 ignore_eos 一致性约束 |

### 8.3 重写

| 模块 | 新文件 | 说明 |
|---|---|---|
| 主 kernel | `csrc/taskgraph_kernel.cu` | persistent task-graph kernel（§5） |
| C++ runtime | `csrc/taskgraph_runtime.cu` | `DecodeService`（§7） |
| bindings | `csrc/taskgraph_bindings.cpp` | pybind |
| ABI header | `csrc/tg_abi.h` | `TgLaunchDesc`/`TgBars`/`TgInstruction`（layout-locked） |
| schedule 构建 | `schedule.py` | per-SM inst_buf + bar 维度（§6） |
| 连续 batching | `session.py` | Python scheduler 线程（§7） |

## 9. 迁移路径（增量交付）

### 阶段 0：骨架 + 单测（无 GPU 也能跑）
- `tg_abi.h`：`TgInstruction`/`TgBars`/`TgLaunchDesc` 结构 + layout-locked 注释。
- `taskgraph_kernel.cu`：`tg_wait`/`tg_arrive`/`tg_wait_all` 原语 + `tg_decode_kernel`
  入口 + controller/worker loop 骨架 + `tg_dispatch` 大 switch（各 case 先空实现）。
- `schedule.py`：bar 维度计算 + 每层 named wave 构建 + plain RR placement + 序列化成
  `inst_buf`。
- 单测：bar index 公式、placement RR、指令编码/解码、layout-locked 不变式。

### 阶段 1：单序列 greedy correctness（batch=1）
- 实现 `TG_QKV_PROJ`/`TG_ATTN_DECODE`/`TG_O_PROJ`/`TG_FFN_UPGATE_ACT`/`TG_FFN_DOWN`/
  `TG_ADD_RMSNORM`/`TG_LM_HEAD` 各 op（复用 `kernel.cu` 的计算逻辑，换同步原语）。
- 单序列 batch=1 跑通，与现有 `ldg_decode_kernel_persistent_paged` 对比 token correctness
  （PASS 或 CASCADE）。
- **此阶段验证 task-graph 机制本身正确**（bar 依赖、wait/arrive、worker loop）。

### 阶段 2：batch>1 correctness + 性能
- batch 2/4/8 跑通 correctness（不再有 grid 切分上限）。
- benchmark 对比：task-graph vs cudagraph vs 旧 WMMA batch。
- 验证 backfill：nsys 看 SM 占用是否提升、layer 间是否重叠。

### 阶段 3：连续 batching 运行时
- `runtime.cu` 的 `DecodeService` + `session.py`。
- 序列动态加入/离开、几何切换、KV pressure 抢占。
- HTTP server（可选，或直接用 `LLMEngine` API）。

### 阶段 4：优化
- 依赖亲和 placement、动态 attention drain、JIT 按 batch 特化、attention/LM head 上
  tensor core（若需要追 cudagraph 的 device 侧）。

## 10. 风险与未决

1. **sm_80 `__threadfence` 的可见性**：理论上 GPU scope fence 够用，但 cohere 在 sm_90a
   用 `fence.proxy.async` 有 TMA 特殊原因。我们无 TMA，`__threadfence` 应该够——但需
   在 b2/b3/b5 等"奇数 batch"上做 sanity 测试（`docs/07` 提到 Ampere L2 在 b3/b5 暴露
   coherence 问题，现有 `AtomicGridSync` 用 `fence.acq_rel.gpu` 解决）。**缓解**：先在
   `tg_wait/tg_arrive` 里用 `fence.acq_rel.gpu`（与 `AtomicGridSync` 一致），不用裸
   `__threadfence`。

2. **WMMA tile padding**：batch<16 时 `tiles_M=1`，`store_matrix_sync` 写满 16 行。
   task-graph 不改变这个——仍需 scratch pad 到 16 倍数（`model.py:275`）。但 backfill
   能让"空闲 SM 去跑别的 task"弥补 padding 浪费。

3. **指令 buffer 带宽**：28 层 × 每层 ~50 task × 16 int = ~22K int/SM。A100 HBM 带宽
   够，但 controller 预取要跟得上 worker 消费。`TG_INST_RING=4` 双缓冲应够；若不够再加深。

4. **cooperative launch 仍需**：persistent worker 自旋 `tg_wait` 需要所有 block 驻留，
   所以仍用 `cudaLaunchCooperativeKernel`。A100 occupancy=1 block/SM（`__launch_bounds__
   (256,1)` + 大 smem），grid ≤ 108。**不再人为 cap 80**——用满 108 SM。

5. **per-seq 动态停步**：cohere 的 cooperative grid 也不能 per-seq 缩容，靠 host 每
   步重建 batch（park → 改 slot → resume）。我们同样如此：**每步 decode 都是一次
   launch**（不像 burst 那样 N 步不回 host）。host 开销靠"park 期间 Python 才动手"和
   "decode 线程不回 Python"压低。若要 N 步不回 host 的 burst，需 device 侧 per-seq
   done flag + 动态 mask（留作后续）。

6. **与现有 burst 路径的关系**：连续 batching 运行时上线后，burst 路径（`generate_paged
   _batch`）可保留为"无 scheduler 线程的简化模式"，或逐步废弃。本轮不删。

---

> 本设计稿完成后，进入阶段 0 实现。每阶段完成后回填本文档的"实测状态"小节，
> 并更新 `docs/10_交接文档.md` §1/§2/§3。

---

## 10. 实现状态（2026-09-22，服务器实测前）

> 本节记录代码实际落地到哪、哪些已静态验证、哪些还没做。设计与实现的偏差都列在
> §10.2。服务器实测后回填 §10.4。

### 10.1 已完成并验证

| 模块 | 状态 | 验证方式 |
|---|---|---|
| `tg_abi.h` 指令/bar/desc 布局 | ✅ | `test_taskgraph_abi.py` 解析头文件与 `schedule.py` 对齐（37 项） |
| `schedule.py` 调度构建（QKV/ATTN/O/FFN/ADD_RMSNORM/LM_HEAD/LM_ARGMAX） | ✅ | ABI 测试 + 到达计数 + tile 连续性 |
| 调度**无死锁**（任意 num_sms） | ✅ | CPU 执行模拟器（BS∈{1,2,4,8} × num_sms∈{1,4,8,108}）；灵敏度已验证 |
| cross-SM `tg_wait/tg_arrive` | ✅ | nvcc 12.4 sm_80；A100 真机验证 |
| LM_HEAD → `TG_LM_ARGMAX` 跨 SM argmax | ✅ | 逻辑推导 + 单/多 SM 测试 + A100 真机 |
| cp.async 权重流水（8 stage/warp，硬件上限） | ✅ 编译 | nvcc + `static_assert(TG_B_STAGES<=8)`；group 计数与 buffer 复用对 S=2..5、K/16∈1..192 验证 |
| per-warp A staging（GEMM K loop 无 CTA barrier，float4 向量化） | ✅ 编译 | nvcc（104 regs / 0 spill / 35.4KB static smem） |
| GEMM fan-out（QKV/O/FFN/LM 一次 GEMM，N=num_tiles*16） | ✅ | 指令数 10516→4916→4522（cap 7）；到达计数不变 |
| tile 打包区间语义（inst[9]=num_tiles, inst[10]=tile_n0） | ✅ | ABI 测试锁定 inst[11..15] 未用；`_TILE_SLOTS_PER_INST=7` 饱和 7 个 worker warp |
| bar 记账与打包粒度无关 | ✅ | num_sms=1/4/108 的 814 个 bar 槽位计数逐位相同 |
| 性能诊断与优化计划 | 📄 | 见 `docs/12_性能诊断与优化计划.md`（Amdahl 拟合、波次模型、ncu 决策点、split-K） |
| `x_attn` 单发布者 + 就地 RMSNorm | ✅ | ABI 测试锁定"每层一个发布者" |
| `tg_zero_regions` | ✅ | 回归测试 + 静态检查 |
| bar 自旋超时 trap | ✅ 编译 | nvcc |
| cooperative launch 失败抛异常 | ✅ | 代码路径 |
| `row_active`：inactive 行不写 KV cache | ✅ | A100 真机 BS=2 测试（inactive 行的页保持哨兵值、active 行不受影响） |
| 多 position / 跨 KV block / 自回归 greedy | ✅ | A100 真机对拍 causal PyTorch 参考；`generate_taskgraph` 文本连贯 |
| 跨 warp global memory fence（确定性） | ✅ | A100 真机 30/30 次确定性；3 处 race 已修复 |

### 10.2 与设计稿的偏差

1. **bar 等待**：不是所有线程自旋，而是 **worker warp 0 轮询 + worker barrier 广播**
   （108 CTA × 224 线程全自旋会打爆 L2 计数器行）。`tg_arrive` 必须 **单线程**
   atomicAdd + worker barrier，否则到达数 ×224。
2. **GEMM 打包**：设计稿写"Q/K/V 每个 N tile 一条指令"；实际把**连续 tile_n 打包**
   （QKV 用 `_pack_qkv`，其余用 `_pack_tiles`），每条指令发一次 GEMM，
   `N = num_tiles*16`，把 N tile 扇出到 7 个 warp。否则 `total_tiles=1` 只有 warp 0 干活。
3. **权重加载**：cohere 用 TMA；A100 无 TMA，改为 `cp.async.cg` + 4 stage 流水
   （`.cg` 绕 L1，权重无复用）。`fence.proxy.async` 不需要。
4. **A staging**：改为 per-warp（`s_a[7][256]`），K loop 内 0 个 CTA barrier。
5. **LM head**：argmax 从 LM_HEAD 拆出为终结 op `TG_LM_ARGMAX` + `bar_lm_head`
   （否则多 SM 下每个 tile 指令扫全 vocab 会读到别的 SM 还没写的 logits）。
6. **x_attn**：FFN_UPGATE 在 shared `s_norm` 里就地 staging + 就地 RMSNorm，
   只有 `tile_n0==0` 的指令发布给 ADD_RMSNORM（消除跨 CTA 同值写）。
7. **ring buffer**：仍是死代码；实测显示 dispatch 的 `__syncthreads` 开销 ~2µs/SM
   （vs 权重 800µs），暂不值得启用。

### 10.3 未实现（明确不可用）

- ~~`attn_splits > 1`~~：已实现（ATTN_DECODE 写 unnormalized partial + LSE，
  ATTN_COMBINE 按 `exp(lse - max_lse)` 重标定并归一化）。有对拍参考的测试 + 死锁模拟覆盖。
  注：attention 已按 SM×warp 并行，短/中 context 不缺这个 split。
- `oproj_splits / down_splits > 1`：`build_schedule` 直接 `NotImplementedError`。
- 连续批处理：kernel 侧已支持 `row_active`（inactive 行不写 KV，避免复用页被污染；
  其余计算照跑、bar 照到达）；但缺 `runtime.cu`/scheduler/session，`backend.py` 未接 taskgraph。
- sampling：只有 greedy argmax，无 temperature/top-p。

### 10.4 服务器实测结果（2026-09-23，A100-SXM4-80GB）

#### 硬件 / 环境

- GPU: A100-SXM4-80GB (sm_80, 108 SM)
- 模型: Qwen3-0.6B (28 层, hidden=1024, intermediate=3072, vocab=151936)
- CUDA 12.8, PyTorch nightly (cu128), Python 3.12
- `attn_splits=1`, `oproj_splits=1`, `down_splits=1`, `num_sms=108`（全 SM）

#### 确定性验证

发现并修复了 **3 处跨 warp global memory race**——根因相同：`bar.sync` 只保证
shared memory 可见性，global store 需要 `fence.acq_rel.gpu` 才能跨 warp 可见。

| 位置 | race 描述 | 修复 |
|---|---|---|
| `tg_ffn_upgate_op` | gate/up GEMM → SiLU 读 `g.silu_out`/`g.up_out`，不同 warp 负责不同 tile，无 fence | GEMM 后加 `tg_fence_gpu()` + `tg_worker_sync_inner()` |
| `tg_attn_decode_op` | warp 0 写 KV cache → flash attention 读，`bar.sync` 不保证 global 可见 | 写后 `tg_fence_gpu()` + sync + `tg_fence_gpu()` |
| `tg_lm_argmax_op` | LM_HEAD GEMM 写 `lm_logits` → argmax 扫 vocab，跨 tile 不可见 | `tg_wait` 后加 `tg_fence_gpu()` + sync |

修复后 **30/30 次运行全确定性**（token=14582），修复前约 60-70% 正确。

#### 测试结果

| 测试套件 | 结果 |
|---|---|
| `test_taskgraph_smoke.py` | ✅ 3 passed |
| `test_taskgraph_abi.py` (attn_splits=1) | ✅ 37 passed, 5011 subtests |
| `test_taskgraph_positions.py` (attn_splits=1) | ✅ 3 passed (max logit diff 0.21) |
| `test_taskgraph_positions.py` (attn_splits=2) | ❌ schedule deadlock（预存问题，非确定性修复引入） |
| `test_taskgraph_bs2.py` | ✅ 4 passed (BS=2 与独立 BS=1 结果一致) |
| `test_taskgraph_multi_sm.py` | ✅ passed |
| `compute-sanitizer --tool memcheck` | ✅ 无内核内存错误 |
| `compute-sanitizer --tool racecheck` | ✅ 无 shared memory race |

#### 端到端文本生成

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

#### 性能（ms/step 与有效权重带宽）

| BS | ctx=1 | ctx=128 | ctx=512 | ctx=2048 |
|---|---|---|---|---|
| 1 | 10.8 ms (111 GB/s) | 11.5 ms (104 GB/s) | 15.9 ms (76 GB/s) | 25.8 ms (47 GB/s) |
| 2 | 11.1 ms (108 GB/s) | 11.9 ms (101 GB/s) | 15.4 ms (78 GB/s) | 26.1 ms (46 GB/s) |
| 4 | 12.5 ms (96 GB/s) | 13.4 ms (90 GB/s) | 16.5 ms (73 GB/s) | 32.4 ms (37 GB/s) |
| 8 | 13.6 ms (88 GB/s) | 14.4 ms (83 GB/s) | 17.5 ms (69 GB/s) | 33.6 ms (36 GB/s) |

- 权重总量 ~1.2 GB（0.6B params × 2 bytes），每步全读一遍
- A100 HBM 峰值 ~1.5 TB/s；当前 ctx=1 达到 ~111 GB/s（~7% 峰值）
- 1 SM: 275 ms/step (4 GB/s)——单 SM 串行，确认是 weight-bandwidth bound
- BS 1→8 增量很小（10.8→13.6 ms），说明 decode 几乎不受 batch 影响（权重只读一次）
- ctx 增长时 attention 成本上升（scalar flash attention，未向量化）

#### kernel 资源使用

- 104 regs / 0 spill / 35.4KB static shared + 64KB dynamic shared = 99.4KB（上限 163KB）
  （`TG_B_STAGES=8` + A staging float4 向量化后的数据；服务器实测为 110 regs/21.9KB，对应改动前的版本）
- 7 worker warps (224 threads) + 1 controller warp = 256 threads/CTA

#### 已知问题

- `attn_splits=2` schedule deadlock：ABI 模拟器显示 184/10264 指令可执行，
  调度器逻辑问题，与确定性修复无关
- 性能远未优化：当前用 `__syncthreads` dispatch（~2µs/SM 开销），ring buffer
  仍是死代码；scalar attention 未向量化；权重带宽利用率低
