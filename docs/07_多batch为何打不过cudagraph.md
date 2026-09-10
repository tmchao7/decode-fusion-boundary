# 多 batch 场景为何打不过 cudagraph：证据与根因分析

> 更新日期：2026-09-10
>
> 本文针对「单 batch 能击败 cudagraph，多 batch 并发为何反输」这一问题，
> 逐条核验代码与服务器实测，给出结论。所有结论都带 `file:line` 锚点；
> 性能数字来自 `results/a100_nosync_bench.log`、`docs/06` §0 的 A100 实测
> （`tools/bench_burst_batch.py`，`ignore_eos=False`，burst 真实触发）。
>
> 前置阅读：[docs/04](04_Megakernel集成问题清单.md)、
> [docs/05](05_Megakernel收益与风险分析.md)、[docs/06](06_Megakernel后续演进路线.md)。

## 0. 一句话结论

用户的三个猜测**都命中了真实瓶颈，但权重排序与原话不同**：

1. **显卡资源问题（决定性根因，权重最高）**。cooperative megakernel 是一块
   **固定大小、被切分的** GPU 资源——~80 个 block 的常驻 grid 要按序列数等分，
   每序列至少 16 个 block，把并行 batch 硬卡在 ≤5；而 cudagraph 的独立 kernel
   launch 让整块 GPU 服务于每个 batched GEMM，batch 越大越高效。
2. **多 batch 下 tensor core 覆盖严重不足（核心算力根因）**。比用户猜的更糟：
   当前 `ldg_decode_kernel_batch_wmma` 里**只有 QKV 三个 GEMM 用了 WMMA**，
   占算力大头的 O/Gate/Up/Down（MLP）和 Attention（QK+PV）**全是 scalar FMA**。
   而单 batch 那个「赢」的路径压根没用 tensor core。
3. **持续批处理（continuous batching）未真正支持（架构根因）**。burst 路径要求
   block table 在整段 burst 内固定、无 waiting 序列、全 greedy——是「静态定长
   burst」，不是动态进出序列的 continuous batching；能做 continuous batching 的
   step-wise 路径又被每步 `.cpu().tolist()` 主机同步钉死。

三者相互叠加：资源切分让 batch 上限低 + 每序列分到的 block 少 → tensor core
没覆盖的 scalar attention 在更少的 block 上更慢 → tile padding 浪费算力 →
每步 3 个 launch 开销 → 最终多 batch 既无算力优势也无资源优势。

## 1. 先把「单 batch 击败 cudagraph」的真相摆清楚

「单 batch 击败」**只在特定口径下成立**，理解这个口径是分析多 batch 的前提。

### 1.1 真实 burst 路径（`ignore_eos=False`）—— 确实赢

`results/a100_nosync_bench.log`（A100-SXM4-80GB，prompt_len=32，200 tokens）：

```text
megakernel sync (.item())             32.27 ms/token
megakernel no-sync                     1.99 ms/token
megakernel burst (engine E2E)          2.71 ms/token
torch-cudagraph                        2.82 ms/token
no-sync vs cudagraph: 1.42x (BEATS)
burst vs cudagraph:   1.04x (BEATS)
```

`docs/06` §0 的 `tools/bench_burst_batch.py` 实测（batch=1 行）：

| batch | cudagraph (ms/tok) | burst (ms/tok) | 倍率 |
|------:|-------------------:|---------------:|-----:|
| 1 | 2.95 | **2.53** | **1.16x BEATS** |

### 1.2 标准 benchmark（`ignore_eos=True`）—— 单 batch 也输

`results/a100_nosync_compare.md`、`results/a100_quick_b1.md`（`benchmarks/compare.py`，
`ignore_eos=True`，compare.py:489）：

| Workload | cudagraph decode tok/s | megakernel-burst decode tok/s | Status |
|---|---:|---:|---|
| perf_p32_b1 | 377.89 | 31.36 | **fallback_or_incorrect** |
| perf_p128_b8 | 2480.25 | 85.42 | **fallback_or_incorrect** |

**关键**：`_should_burst()`（`llm_engine.py:95-98`）**拒绝 `ignore_eos` 序列**
（设备侧 EOS 无法 per-seq 关闭），所以 `compare.py` 里 `megakernel-burst` 名字
虽在，实际**回退到 step-wise 逐 token 同步路径**，根本没跑 burst。Status
`fallback_or_incorrect` 就是这个回退的标记（`compare.py:351-359`）。

> 所以「单 batch 击败 cudagraph」的准确表述是：**在 `ignore_eos=False` 的 burst
> 路径上，batch=1 靠 hybrid dispatch 走 persistent 单序列 kernel + 设备侧 EOS，
> 1.04-1.16x 微胜**。换到标准 benchmark 口径，单 batch 也是大输。**

### 1.3 单 batch 赢的机制：launch 消除，不是算力

`backend.py:250-262` 的 hybrid dispatch：batch=1 走 `generate_paged` →
`ldg_decode_kernel_persistent_paged`（`kernel.cu:1893`）→ `ldg_decode_one`
（`kernel.cu:1697`）。这条路径：

- **一次 cooperative launch 排队 N 步**，`ldg_update_step`（`kernel.cu:1928`）
  在 device 上做 argmax→token 反馈，**零每 token `.item()` 同步**；
- **全 scalar FMA，无 tensor core**（`ldg_matvec_qkv` kernel.cu:190、
  `ldg_attention` kernel.cu:355 都是 `__bfloat162float` × float 标量点积）；
- 赢 cudagraph 靠的是「消除 ~170 次 inter-kernel launch 间隙 + 主机同步」
  （docs/05 §4.1），**不是单 kernel 算力更强**。docs/04 §P1 实测纯 GPU kernel
  megakernel 1.86ms 仍慢于 cudagraph 0.96ms。

> 这条事实直接预示了多 batch 会输：单 batch 的赢点是 launch 消除，而 batch 越大
> launch 开销占比越低、算力占比越高——megakernel 在算力侧（kernel 质量）本来就
> 落后，batch 放大只会暴露这个落后。

## 2. 根因一：显卡资源被「固定 grid 切分」卡死（决定性）

### 2.1 cooperative grid 是固定大小、必须等分的资源

- 常驻 grid 上限 `LDG_NUM_BLOCKS = 80`（`kernel.cu:34-35`），由
  `ldg_resident_grid_blocks`（`kernel.cu:2096`）按
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor × multiProcessorCount` 取
  `min(requested, resident)`。kernel 带 `__launch_bounds__(512, 1)`
  （`kernel.cu:973`、`1592` 等），「,1」意味着**每 SM 最多 1 个 block**。
  A100-SXM4-80GB 有 108 SM，故 resident ≈ 80（被 `LDG_NUM_BLOCKS` 钳住）。
- batch kernel 把这块 grid **按序列等分**：
  ```cpp
  int seq_id = blockIdx.x / blocks_per_seq;          // kernel.cu:1870, 1001
  int blocks_per_seq = grid_blocks / batch_size;     // kernel.cu:2202, 2267, 2568
  if (blocks_per_seq < LDG_ATTN_BLOCKS) { return; }  // kernel.cu:2203, 2268
  ```
  `LDG_ATTN_BLOCKS = NUM_Q_HEADS = 16`（`kernel.cu:49-50`）——每序列至少要 16 个
  block 才能分给 16 个 Q head 做 attention。

### 2.2 并行 batch 硬上限 ≈ 5

由 `blocks_per_seq = 80 / batch_size >= 16`：

| batch | blocks_per_seq | 能否启动 |
|------:|--------------:|:--------:|
| 1 | 80 | ✓（实际走 persistent 全 grid） |
| 2 | 40 | ✓ |
| 4 | 20 | ✓ |
| 5 | 16 | ✓（下限） |
| 6 | 13 | ✗ `< 16`，kernel 直接 return |
| 8 | 10 | ✗ |

`benchmarks/compare.py:166-167` 因此硬编码 `batch_size > 10` 报错；`backend.py:15`
`MAX_BATCH_SIZE = 32` 是 scratch buffer 上限，**不是真能并行的 batch**。`docs/04`
§P1 已明确「默认 80-block grid 把并行 batch 上限卡在约 8-10 条」。

> **对照 cudagraph**：cudagraph replay 的是 cuBLAS/flash-attn 独立 kernel，每个
> batched GEMM（M=batch）可以**用满 108 SM 的几百个 threadblock**，batch 越大
> 算术强度越高、SM 占用越满。megakernel 是「80 个 block 分给 batch 个序列」，
> 每序列分到的 block 随 batch 线性下降。这是**资源模型上的根本不对称**。

### 2.3 实测体现：cudagraph 随 batch 飞速变快，megakernel 反而变慢

`docs/06` §0（`tools/bench_burst_batch.py`，burst 真实触发）：

| batch | cudagraph (ms/tok) | burst (ms/tok) | 倍率 |
|------:|-------------------:|---------------:|-----:|
| 1 | 2.95 | 2.53 | 1.16x BEATS |
| 2 | 1.63 | 5.67 | 0.29x 输 |
| 4 | 0.85 | 3.69 | 0.23x 输 |
| 8 | 0.43 | 5.07 | 0.08x 输 |

- cudagraph：b1→b8 从 2.95 降到 0.43，**~6.9x 加速**（batched GEMM 摊薄权重带宽 +
  占满 SM）。
- megakernel burst：b1→b8 从 2.53 升到 5.07，**反而变慢 ~2x**。b2 突跳到 5.67 是
  hybrid dispatch 切到 WMMA batch kernel 的拐点（`backend.py:250` batch==1 走
  persistent，`backend.py:264` batch>1 走 WMMA batch）。

> **结论**：batch 越大，cudagraph 的「独立 launch + 全 GPU 服务每个 GEMM」模型越
> 占优；megakernel 的「固定 grid 等分」模型越劣势。这是资源问题，不是调参问题。

## 3. 根因二：多 batch 下 tensor core 覆盖严重不足

用户的猜测「多 batch 下未使用 tensor core」**方向对，但实际比猜的更严重**。

### 3.1 batch>1 的 WMMA kernel 里，只有 QKV 用了 tensor core

`ldg_decode_kernel_batch_wmma`（`kernel.cu:974`）中 `wmma_gemm` 调用只有 3 处
（`grep -n wmma_gemm` 全文件仅 kernel.cu:1048-1050）：

```cpp
wmma_gemm<H, Q>(..., w.q_proj_weight, g_q, ...);   // kernel.cu:1048  Q
wmma_gemm<H, K>(..., w.k_proj_weight, g_k, ...);   // kernel.cu:1049  K
wmma_gemm<H, K>(..., w.v_proj_weight, g_v, ...);   // kernel.cu:1050  V
```

而后续步骤**复用的是 scalar 实现**：

- **O projection + Gate + Up + Down MLP**：调 `ldg_o_proj_postnorm_mlp`
  （`kernel.cu:1069` → `kernel.cu:693`），内部是 `ldg_load_weight_u4` + 标量
  `__bfloat162float(w) * a` FMA 累加（`kernel.cu:728-742`、`826-852`、`885-900`），
  **无任何 `wmma::` / `mma_sync`**。
- **Attention（QK score + softmax + PV）**：调 `ldg_attention`（`kernel.cu:1058`
  → `kernel.cu:355`），QK 是 `q_local[i] * __bfloat162float(k_ptr[i])` 标量点积
  （`kernel.cu:564-567`），PV 是 `weight * __bfloat162float(v_ptr[i])` 标量累加
  （`kernel.cu:579-582`），**无 tensor core**。

> `docs/04` §P1 自己也承认：「下一步：把 O projection / Gate / Up / Down 也从
> scalar 换成整批 WMMA GEMM（代码已写好，逐步集成验证）」——即 O/MLP 的 WMMA
> **尚未集成进当前 kernel**。

### 3.2 没覆盖的恰恰是算力大头

Qwen3-0.6B 每层 GEMM 的 FLOPs 量级（M=batch，weights BF16）：

| 算子 | 形状 | 相对算力 |
|---|---|---|
| QKV proj | (batch,1024)×(1024,2048+1024+1024) | 1×（已 WMMA） |
| O proj | (batch,2048)×(2048,1024) | ~0.5×（**scalar**） |
| Gate/Up | 2×(batch,1024)×(1024,3072) | ~3×（**scalar**） |
| Down | (batch,3072)×(3072,1024) | ~1.5×（**scalar**） |
| Attention QK+PV | (batch,16,128)×kv | 随 context 增长（**scalar**） |

**MLP（Gate/Up/Down）是绝对算力大头，全部 scalar**。所以「tensor-core batch
kernel」这个名字有误导性：真正用上 tensor core 的只有 QKV 一小部分。

### 3.3 单 batch 那个「赢」的路径，根本没用 tensor core

如 §1.3 所述，batch=1 走 `ldg_decode_kernel_persistent_paged` → `ldg_decode_one`，
全 scalar FMA（`ldg_matvec_qkv` kernel.cu:190、`ldg_attention` kernel.cu:355）。
**单 batch 的赢与 tensor core 无关**，纯靠 launch 消除。

### 3.4 WMMA tile padding 雪上加霜

`wmma_gemm` 用 `m16n16k16` tile（`kernel.cu:920` `WMMA_TILE=16`），batch 维
`store_matrix_sync` 无条件写满 16 行（`kernel.cu:963`）。因此 scratch buffer
把 batch 维补齐到 16 的倍数（`model.py:271` `pad_batch = ((max_batch+15)//16)*16`）：

| 实际 batch | padded batch | 算力浪费 |
|----------:|-------------:|---------:|
| 1 | 16 | 16×（但 batch=1 走 persistent，不走这里） |
| 2 | 16 | 8× |
| 4 | 16 | 4× |
| 8 | 16 | 2× |

`docs/06` §1.3 明确点名：「batch=2 实测 5.67 ms/tok vs persistent batch=1 2.53
ms/tok」，tile padding 是主因之一。QKV 即使用了 tensor core，算力也被 padding
吃掉大半。

## 4. 根因三：没有真正的 continuous batching

### 4.1 burst 路径是「静态定长 burst」，不是 continuous batching

`_should_burst()`（`llm_engine.py:75-99`）的触发条件全部是「**静态**」：

- `not self.scheduler.waiting`（无等待 prefill 的序列）——`llm_engine.py:88`；
- `self.scheduler.running` 非空——`llm_engine.py:90`；
- 每条 seq `is_greedy`——`llm_engine.py:93`；
- 每条 seq **不** `ignore_eos`——`llm_engine.py:95-98`；
- `schedule_burst`（`scheduler.py:83-115`）要求 `block_manager.can_allocate_burst`
  能一次性预分配 `budget` 个 block，**block table 在整段 burst 内固定**（R4，
  `scheduler.py:108-110`）。

这意味着：序列不能中途加入、不能动态变长进出、batch 组成在 burst 内不变。
**这不是 vLLM 意义上的 continuous batching**（每步可加入新请求、踢出完成请求、
batch 大小动态变化），而是「选一批 running 序列、锁定 block table、跑 K 步、
一次性回收」的静态 burst。

### 4.2 能做 continuous batching 的 step-wise 路径被同步钉死

`scheduler.schedule()`（`scheduler.py:26-74`）本身**支持**每步从 running 队列动态
挑序列（这才是 continuous batching 的调度语义），对应的 `ModelRunner.run()`
多序列分支调 `megakernel.decode_batch`（`model_runner.py:313-318` →
`backend.py:216-231`）。但 `Decoder.step_batch` 末尾：

```python
return self._out_token[:batch_size].cpu().tolist()   # model.py:415
```

**每步一次 `.cpu().tolist()` 主机同步**。`docs/04` §P1 实测这条路径 ~13-15x 慢于
cudagraph，瓶颈就是逐 token 主机往返，而非 kernel 本身。

> 于是陷入两难：
> - 走 burst（快）→ 放弃 continuous batching（静态定长）；
> - 走 step-wise（支持 continuous batching）→ 被主机同步钉死。
>
> 当前没有「既 continuous batching 又 no-sync」的路径。`docs/06` §2.3 的 R3
> （单 launch 内多 token 循环 + LM head 融合）是解决这个问题方向，但尚未实现。

## 5. 三大根因如何叠加

把多 batch 的开销拆解（以 batch=8 为例，对照 cudagraph 0.43 ms/tok）：

```text
megakernel batch=8 burst 5.07 ms/tok
├─ 资源切分：80 block / 8 seq = 10 block/seq < 16  → 实际 batch≤5，b8 已是
│   「blocks_per_seq 取整后 grid 缩水」的退化态（kernel.cu:2273 grid_blocks=blocks_per_seq*batch）
├─ tensor core 缺位：QKV 用 WMMA 但被 16-row padding 浪费 2×；
│   O/Gate/Up/Down/Attention 全 scalar FMA，无 tensor core 加速
├─ per-step 3 launch：burst 每步 = cooperative kernel + lm_head_fused_batch
│   + update_step_batch（kernel.cu:2587-2622），N 步 = 3N launch，没合并成单 launch
└─ attention block 不足：每序列 ~10-16 block 做 16 head 的 attention + KV 读，
    scalar QK/PV 在更少 block 上更慢，且与 GEMM 串行（grid.sync 等待）
```

对照 cudagraph batch=8：cuBLAS batched GEMM 用满 108 SM、tensor core 全开、
无 padding、无跨序列资源切分。**每一条都是 cudagraph 的优势项**，叠加后就是
0.43 vs 5.07（~12x 差距）。

## 6. 验证清单（把上述结论从代码证据变成实测结论）

下列实验多数尚未跑，跑完可把本文从「代码分析」升级为「实测结论」：

| 实验 | 命令 | 期望数据 | 判定 |
|---|---|---|---|
| 真实 burst 多 batch A/B | `python tools/bench_burst_batch.py --prompt-length 128 --max-tokens 128` | b1/2/4/8 的 ms/tok | 复现 docs/06 §0 表，确认 b>1 全输 |
| ncu kernel 时间分解 | `ncu --set roofline ... megakernel vs cudagraph` | WMMA QKV vs scalar O/MLP/Attn 的 kernel 时间、tensor core 占用率 | 预期 O/MLP/Attn 占 kernel 时间 >70%，tensor core 利用率低 |
| ncu SM 占用 | `ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active` | batch=1/4/8 的 occupancy | 预期 megakernel 随 batch 下降，cudagraph 上升 |
| batch 上限实测 | 逐 batch 跑直到 kernel return | 最大可启动 batch | 预期 ≈5（80/16），验证 §2.2 |
| O/MLP 接 WMMA 后 A/B | 改 kernel.cu:1069 调 wmma 版 O/MLP，再跑 bench_burst_batch | b8 ms/tok 变化 | 验证 tensor core 覆盖是否是主因（docs/06 §5 P1） |
| continuous batching 交错负载 | 构造「序列错峰到达」workload | burst 是否回退 step-wise、吞吐对比 | 验证 §4 的两难 |

## 7. 修复方向（与 docs/06 路线图对齐，按本文根因排序）

1. **资源切分（根因一）**：
   - 增大常驻 grid（提高 `LDG_NUM_BLOCKS`、降低每 block 的 SMEM/寄存器以提 occupancy），
     或改成「不按序列等分、而是全 grid 协作所有序列的 GEMM」的 batched GEMM 模型
     （即让多序列共享全 grid 算 GEMM，只在 attention 阶段按序列分）——本质是向
     cudagraph 的「全 GPU 服务每个 GEMM」靠拢。
   - 这是 P0，但和「megakernel 单 launch 融合」的架构卖点有张力，需权衡。
2. **tensor core 全覆盖（根因二）**：
   - 把 O/Gate/Up/Down 从 scalar 换成 `wmma_gemm`（docs/06 §5 P1，docs/04 §P1
     已标「代码已写好，逐步集成」）；
   - Attention 的 QK/PV 改 WMMA fragment（docs/06 §5 P1，独立工作项）；
   - 解决 16×16 tile padding（batch<16 的浪费）：用更小 tile 或拼 batch 到 16。
3. **单 launch 多 token 循环（根因三 + per-step launch）**：
   - docs/06 §2.4 R3：把 K 步 × batch 循环搬进一个 persistent cooperative kernel，
     LM head 融合进 grid，消除每步 3 launch（`kernel.cu:2587-2622`）。
4. **continuous batching 与 no-sync 共存**：
   - 让 burst 路径支持「burst 边界上动态加入/踢出序列」——burst 内固定，burst 之间
     重新 `schedule_burst`。当前 `schedule_burst` 已是每 burst 重选
     （`scheduler.py:83`），但 `_should_burst` 拒绝 waiting 序列
     （`llm_engine.py:88`）导致有 prefill 待调度时整体回退 step-wise。
     可改为「waiting 走 prefill、running 走 burst」的混合调度。

## 8. 不要做的事

- **不要**在跑完 §6 的 ncu 实测前，把「tensor core 没用」当成已证实的头号根因——
  代码层面 O/MLP 确实 scalar，但占比需 ncu 量化（attention 随 context 增长可能反超）。
- **不要**把 `results/a100_nosync_compare.md` 的数字当作 burst 性能——那是
  `ignore_eos=True` 下的 step-wise 回退（Status `fallback_or_incorrect`），真实
  burst 数字只在 `tools/bench_burst_batch.py` / `tools/bench_nosync.py`。
- **不要**期望通过单一改动反超——三大根因叠加，只解一个（如只接 O/MLP WMMA）
  仍会被资源切分和 per-step launch 拖住。
- **不要**在 batch≥32 上承诺 speedup：资源切分让并行 batch ≤5，batch 32 只能靠
  串行多批，架构上没有优势（docs/05 §9、docs/06 §7 已述）。
