# 多 batch 场景为何打不过 cudagraph：证据与根因分析（第 5 版）

> 更新日期：2026-09-11（第 5 版）
>
> - 第 5 版新增 **§3.5 nsys 实测纯 GPU kernel 时间分解**（4 组配置 b1/b8 × mega/cg），
>   用四个实测比值坐实根因：b1 mega 纯 kernel 慢于 cg 2.6x 但 E2E 反超（launch 消除）；
>   b8 mega 纯 kernel 慢 133x；b8/b1 mega kernel 增长 54x vs cg 1.04x（SM 空转）。
>   发现 ncu 不能用于 cooperative kernel（driver hang），改用 nsys。§6 同步更新。
> - 第 4 版把 **§9.6 crossover 实验从「待跑」升级为「A100 已跑 + 实测数据」**：
>   batch=1/2/3/…/32 全扫描，拐点实测在 batch=1→2（悬崖式，无打平区），
>   b1 1.31x BEATS、b≥2 单调劣化到 b32 0.02x，MegaKernel 在 b6 触底后平台化
>   于 ~6.5 ms/tok。数据存 `results/crossover_scan.json` +
>   `results/crossover_plot.png`，工具 `tools/bench_burst_batch.py` +
>   `tools/plot_crossover.py`。§6 验证清单同步更新。
> - 第 3 版在第 2 版结论之上，补充 **§9 外部公开实现/研究的交叉印证**
>   （megaQwen、Ada-MK、vLLM CUDA Graph 设计），并新增 **§6 验证清单中的
>   crossover 实验与 batch-aware 审计**两个待办。
>
> 第 2 版相对第 1 版纠正了两处过时/错误判断：
>
> 1. **张量核覆盖**：第 1 版称「只有 QKV 用 WMMA，O/Gate/Up/Down 与 Attention 全是
>    scalar」——这条**已过时**。当前 `ldg_o_proj_mlp_wmma`（`kernel.cu:1008-1122`）
>    已经用 WMMA 处理 **O projection + Gate + Up + Down**（`wmma_gemm` 调用见下文 §3）。
>    批处理 WMMA 路径的**全部 FC GEMM 已上张量核**，只剩 Attention(QK/PV) 与 LM head
>    仍 scalar。
> 2. **同步(barrier)归因**：第 1 版/此前 agent 推测「~336 次全网格 barrier ≈ 7-8ms
>    主导」。**本机实测推翻了这一量级**：纯 336 次 barrier 仅 **0.547 ms/token**
>    （1.63 µs/barrier，`tools` 微基准，见 §3.4）。barrier 不是毫秒级主开销。

## 0. 一句话结论

多 batch 打不过 cudagraph，且随着 WMMA 逐步覆盖 FC GEMM，**已经不能再用「没上
张量核」来解释了**。当前真正的瓶颈是**架构性的「每层全网格 barrier 强制层间全串行，
而每个批处理 GEMM 只有极少数 block 真正干活、其余 SM 全部空转等待」**，再叠加：

- LM head（151936×1024，每 token 一次）仍是**纯 scalar**，无张量核；
- Attention(QK+PV) 仍是 scalar；
- per-step 多 launch 开销；
- `LDG_ATTN_BLOCKS=8`（build 默认）把并行 batch 卡在 ≈10，不能随负载增长。

cudagraph 的「独立 kernel launch + 整块 GPU(108 SM 全部)服务每个 batched GEMM」在
batch>1 下把 SM 用满、张量核全开，且每个 GEMM 独立、不被层间 barrier 串行化。
这是**资源使用模型**的根本不对称，不是调参能解决的。

## 1. 「单 batch 击败 cudagraph」的真相（不变）

「单 batch 击败」仅存在于 `ignore_eos=False` 的 burst 路径，且是靠**消除 launch
间隙**，不是算力更强：

```text
megakernel burst (engine E2E)  2.71 ms/token   （batch=1, persistent kernel）
torch-cudagraph                2.82 ms/token
burst vs cudagraph:            1.04x (BEATS)
```

- batch=1 走 `backend.py:250-262` hybrid dispatch → `generate_paged` →
  `ldg_decode_kernel_persistent_paged`（`kernel.cu`），**全 scalar**（
  `ldg_matvec_qkv` kernel.cu:190、`ldg_attention` kernel.cu:355）。
- 赢点 = 「一次 cooperative launch 排队 N 步 + 设备侧 EOS，零每 token `.item()`
  同步」，**与张量核无关**（docs/05 §4.1）。

> 关键预示：batch 越大，launch 消除的收益占比越低、GEMM 算力占比越高，而 megakernel
> 在 GEMM 质量/利用率侧本就落后，batch 放大只会暴露。

## 2. 根因一：固定 grid 按序列切分 + 张量核利用率低（决定性）

### 2.1 cooperative grid 是固定大小、必须等分的资源

- 常驻 grid 上限 `LDG_NUM_BLOCKS = 80`（`kernel.cu:34-35`）。grid 取
  `ldg_resident_grid_blocks(...)`（`kernel.cu:2272`）
  = `cudaOccupancyMaxActiveBlocksPerMultiprocessor × multiProcessorCount`，
  然后 `min(requested, resident)`。
- 批处理 kernel 把 grid **按序列等分**：
  ```cpp
  int seq_id = block_id / blocks_per_seq;      // kernel.cu:1163
  int blocks_per_seq = grid_blocks / batch_size;
  if (blocks_per_seq < LDG_ATTN_BLOCKS) return; // kernel.cu:1100 附近 guard
  ```
- `LDG_ATTN_BLOCKS` **编译期由 build.py 传入**：`build.py:57`
  `-DLDG_ATTN_BLOCKS={env或8}`（默认 **8**，不是 kernel.cu 里 `#ifndef→NUM_Q_HEADS=16`
  的兜底值）。因此并行 batch 上限 ≈ `80/8 = 10`，不是第 1 版说的 5。

### 2.2 并行 batch 上限 ≈ 10

| batch | blocks_per_seq (grid=80) | 能否启动 |
|------:|--------------------------:|:--------:|
| 1 | 80 | ✓（persistent 全 grid） |
| 2 | 40 | ✓ |
| 4 | 20 | ✓ |
| 8 | 10 | ✓（≥8） |
| 10 | 8 | ✓（下限） |
| 16 | 5 | ✗ `<8`，kernel 直接 return（实测一致：agent 报告 batch=16 blocks_per_seq=5<8 不跑） |

> cudagraph 对照：cuBLAS batched GEMM 用满 108 SM 的几百个 threadblock，batch 越大
> 算术强度越高、SM 占用越满。megakernel 是「80 block 分给 batch 个序列」，
> **资源模型根本不对称**。

### 2.3 真正致命：批处理 GEMM 只有极少 block 干活，其余 SM 空转

批处理 WMMA kernel 的每个 GEMM 是**全 grid 协作**，但 `wmma_gemm` 的 tile 总量很小，
`total_tiles = tiles_M × tiles_N`，M=batch≤16 时 `tiles_M=1`：

- QKV(Q)：N=2048 → tiles_N=128 → total_tiles=128 → 80*16=1280 warps 里只有
  **128/16=8 个 block 的 128 个 warp** 分到 tile，其余 72 个 block 空转到 barrier；
- O/Down：total_tiles=64 → 只有 **4 个 block**；
- Gate/Up：total_tiles=192 → 只有 **12 个 block**。

每个 GEMM 开头都要等前面 `grid.sync()` 放行，做的小矩阵（M=batch，行数只有 batch
不是 16）算力远低于 SM 能力，做完又必须等后面 `grid.sync()` 收拢。**28 层 × 每层约
12 次 barrier 把 28 个 GEMM 阶段串行成一条长链，每步只有一小撮 SM 在算、绝大多数
SM 在等**。这是「固定 ~9ms/token、不随 batch 摊薄」的真正来源。

## 3. 根因二：张量核覆盖（第 2 版修正）——FC 已全覆盖，剩 Attention/LM head

### 3.1 已用张量核的 FC GEMM（与第 1 版不同）

当前批处理 WMMA 路径 `ldg_decode_kernel_batch_wmma`（`kernel.cu:1136-1253`）全部
FC GEMM 都走 `wmma_gemm`（m16n16k16，`kernel.cu:919-992`）：

```cpp
// QKV（kernel.cu:1217-1219）
wmma_gemm<H, Q>(g_activations, q_proj, g_q, ...);
wmma_gemm<H, K>(g_activations, k_proj, g_k, ...);
wmma_gemm<H, K>(g_activations, v_proj, g_v, ...);
// O / Gate / Up / Down（ldg_o_proj_mlp_wmma, kernel.cu:1030/1088/1093/1110）
wmma_gemm<Q, H>(g_attn_out, o_weight, g_activations, ...);
wmma_gemm<H, M>(g_activations, gate_weight, g_mlp_inter, ...);
wmma_gemm<H, M>(g_activations, up_weight,   g_mlp_up, ...);
wmma_gemm<M, H>(g_mlp_inter,   down_weight, g_activations, ...);
```

所以**不再是「tensor core 覆盖不足」的问题**——QKV+O+MLP 全是 WMMA。

### 3.2 仍未用张量核的两处

- **Attention(QK + softmax + PV)**：`ldg_attention`（`kernel.cu:367`），QK 是
  `q_local[i] * __bfloat162float(k_ptr[i])` 标量点积（kernel.cu:564-567），PV 是
  `weight * __bfloat162float(v_ptr[i])` 标量累加（kernel.cu:579-582）。**无张量核**。
  decode 场景 context 短，此部分受内存/延迟约束，张量核收益有限。
- **LM head**：`ldg_lm_head_fused_batch`（`kernel.cu:1586`），是**独立 kernel**，
  `151936×1024` 每 token 一次，用 scalar FMA（`ldg_load_weight_u4` + 标量累加），
  **无张量核**。LM head 是单 token 最大的 GEMM，scalar 实现是明显短板
  （cudagraph 的 LM head 走 cuBLAS）。

### 3.3 WMMA tile padding 仍造成浪费

`wmma_gemm` 用 16 行 tile，M=batch<16 时 `store_matrix_sync` 无条件写满 16 行
（`kernel.cu:989-992`），scratch 按 `pad_batch = ((max_batch+15)//16)*16`
（`model.py:271`）补齐：

| 实际 batch | padded batch | 算力浪费 |
|----------:|-------------:|---------:|
| 2 | 16 | 8× |
| 4 | 16 | 4× |
| 8 | 16 | 2× |

即便上了 WMMA，M 侧利用率也只有 `batch/16`。这是「已用张量核但用不满」的浪费。

### 3.4 实测：barrier 不是 7-8ms（修正归因）

本机 A100-SXM4-80GB，写了一个与 `AtomicGridSync`（`kernel.cu:102`）同结构的纯
barrier 微基准（`/tmp/barrier_cost.cu`，grid=80，512 线程，cooperative launch）：

```text
device=NVIDIA A100-SXM4-80GB SMs=108 blocks/SM=4 -> grid=80
launch(336 barriers): avg 0.5468 ms  -> 1.63 us/barrier
=> 28 layers * 12 sync = 336 barriers/token => 0.547 ms/token pure barrier
```

结论：
- **336 次全网格 barrier 的总成本只有 ~0.55 ms/token**，不是此前推测的 7-8ms。
- 但 **~9ms/token 的恒定开销 ≠ barrier 本身**，而是 barrier **强制层间全串行 + 每个
  GEMM 只有 4-12/80 block 干活、其余 SM 空转**导致的「潜伏期累加」——这是延迟问题，
  不是 barrier 的原始执行成本。

### 3.5 nsys 实测：纯 GPU kernel 时间分解（坐实 SM 空转 + 资源模型不对称）

> ncu（Nsight Compute）**不能用于 cooperative kernel**：其 kernel-replay 机制与
> `cudaLaunchCooperativeKernel` 冲突，会导致 GPU driver hang（本机实测，连
> `nvidia-smi` 都被阻塞，需等 driver TDR 恢复）。改用 nsys（只 trace 不 replay）
> 采集 4 组配置的 GPU kernel 时间，结果存 `results/nsys_b*_*.sqlite`，
> 解析工具 `tools/parse_nsys_kernel.py`。

**实测**（A100-SXM4-80GB, Qwen3-0.6B, prompt_len=32, max_tokens=50, greedy；
nsys `--trace=cuda --sample=none`；per-step = 纯 decode kernel ÷ 49 decode steps，
prefill 占比 <4% 已分摊掉）：

| 配置 | 纯 decode kernel ms/step | 主 kernel | E2E ms/tok |
|------|------------------------:|-----------|-----------:|
| b1 cudagraph | **0.74** | cuBLAS gemm + triton softmax/rmsnorm（33 种 kernel） | 2.94 |
| b1 megakernel | **1.89** | `ldg_decode_kernel_persistent_paged` 1.71ms + `ldg_lm_head` 0.18ms | 2.24 |
| b8 cudagraph | **0.77** | 同 b1 cuBLAS（batched gemm，kernel 时间几乎不变） | 0.43 |
| b8 megakernel | **101.8** | `ldg_decode_kernel_batch_wmma` 98.5ms + `ldg_lm_head_batch` 3.38ms | 6.50 |

**四个关键比值（全部 nsys 实测，非推理）**：

1. **b1 megakernel 纯 kernel 比 cudagraph 慢 2.6x**（1.89 vs 0.74 ms/step），
   但 E2E **反超 1.31x**（2.24 vs 2.94）——因为 megakernel 把 cudagraph 的
   **2.2ms host/launch 开销**消到 0.35ms。**这量化了 §1 的「赢点 = launch 消除，
   不是算力更强」**：megakernel 的 scalar kernel 本身就比 cuBLAS 慢，是 cooperative
   一次 launch + 设备侧 EOS 把 host 开销压掉才赢的。

2. **b8 megakernel 纯 kernel 比 cudagraph 慢 133x**（101.8 vs 0.77 ms/step）——
   host 消除的 2.2ms 收益 vs kernel 多花的 101ms，**完全无法补偿**。E2E 14x 慢
   （6.50 vs 0.43 ms/tok）的根源就在 kernel 本身，不是 host。

3. **b8/b1 megakernel kernel 时间增长 54x**（98.5 vs 1.71 ms/step），而 batch
   只大 8x——每多一个序列 kernel 时间涨 ~7x，远超计算量增长（batch GEMM 的
   计算量只随 batch 线性增长）。**坐实「每 GEMM 仅 4-12/80 block 干活 + barrier
   串行」：加 batch 不加并行度，只加等待时间。**

4. **b8/b1 cudagraph kernel 时间增长 1.04x**（0.77 vs 0.74 ms/step，几乎不变）——
   batched cuBLAS GEMM 用满 108 SM，算力富余，batch 从 1→8 计算量 8x 但 kernel
   时间不变。**这是资源模型的根本不对称**：cudagraph 的 kernel 天生能吃满 SM，
   megakernel 的固定 grid 吃不满。

> `ldg_decode_kernel_batch_wmma` 在 b8 时极稳定（min 96.6 / avg 98.5 / max 101.5 ms，
> 9 次采样），说明 ~98ms 是确定性的「层间串行潜伏期」而非随机噪声。
> `ldg_decode_kernel_persistent_paged` 在 b1 同样极稳（min 1.707 / avg 1.711 / max 1.724 ms，
> 49 次采样），确定性 scalar matvec。

## 4. root cause 三：没有真正的 continuous batching（不变）

- burst 路径是「静态定长 burst」：`_should_burst`（`llm_engine.py:75-99`）要求无
  waiting 序列、block table 在整段固定、全 greedy、非 ignore_eos。不能动态进出。
- 支持动态进出的 step-wise 路径又被 `model.py` 末尾 `.cpu().tolist()` 每次主机同步
  钉死。
- 结论：当前没有「既 continuous batching 又 no-sync」的路径。

## 5. 三层根因叠加（以 batch=8 为例）

```text
megakernel batch=8   ~9.66 ms/token   （agent 实测；cudagraph 0.461 = 21x 慢）
├─ 每层 ≈12 次全网格 barrier × 28 层 = 336 次，强制层间全串行（实测 pure ≈0.55ms，
│   但串行潜伏期近 9ms）
├─ 每个批处理 GEMM 仅 4-12/80 block 干活，其余 SM 空转等 barrier
├─ LM head 151936×1024 纯 scalar + Attention 纯 scalar
├─ 16 行 tile padding：M=batch<16 → 算力利用 batch/16
└─ 并行 batch 上限 ≈10（LDG_ATTN_BLOCKS=8）
```

对照 cudagraph batch=8：cuBLAS 用满 108 SM、张量核全开、无 padding、每 GEMM 独立不
被 barrier 串行。**每一项都是 cudagraph 的优势项**，叠加成 ~21x。

## 6. 验证清单（已跑 / 待跑）

| 实验 | 状态 | 结果 |
|---|---|---|
| 纯 barrier 成本（336 次全网格） | ✅ 本机已跑 | 0.547 ms/token（1.63 µs/barrier） |
| WMMA tile 覆盖（QKV/O/Gate/Up/Down） | ✅ 代码确认 | 已全 WMMA（§3.1） |
| 剩余 scalar（Attention/LM head） | ✅ 代码确认 | 仍 scalar（§3.2） |
| batch 上限（LDG_ATTN_BLOCKS=8） | ✅ build/agent 确认 | ≈10，batch=16 不跑 |
| E2E 多 batch A/B（bench_burst_batch.py） | 服务器已跑 | batch=8 9.66 vs 0.461（21x） |
| ncu kernel 时间分解 / SM 占用 | ✅ nsys 替代（ncu 对 cooperative kernel 致 driver hang） | b8 mega kernel 133x 慢于 cg；b8/b1 增长 54x vs cg 1.04x（§3.5） |
| LM head 接 WMMA 后 A/B | 待跑 | 验证 scalar LM head 占比 |
| **crossover 扫描**（batch 1/2/3/4/5/6/7/8/10/16/32 双轨 ms/tok + tok/s） | ✅ A100 已跑 | 拐点在 batch=1→2（§9.6）；b1 1.31x BEATS，b≥2 单调劣化到 b32 0.02x |
| **batch-aware 审计**（batch 是否直映 CTA/warp、tile/KV/mapping） | 部分完成 | 已确认每 GEMM 仅 4-12/80 block 干活（§9.7） |
| 四象限对照（eager / CG / mega / mega+CG） | 待跑 | 区分 launch 消除 vs kernel 内效率（§9.8） |

## 7. 结论（更新为可执行判语）

1. **多 batch 的败局是资源使用模型问题，不是调参/张量核覆盖问题**。即便 QKV+O+MLP
   全上张量核，fixed-grid 按序列切分 + 每层全网格 barrier + 每 GEMM 少数 block 干活，
   决定了 batch>1 无法通过「优化某个 kernel」翻盘。
2. **唯一可持续的战略胜利点是 batch=1**：走 persistent kernel + 设备侧 EOS，靠消除
   launch 间隙 1.04-1.16x 微胜（docs/06 §0 实测）。
3. **务实集成（✅ 已落地）**：batch=1 走 megakernel persistent；batch>1 回退 cudagraph
   （megakernel 批处理路径不占优，不值得付出层间串行 + 多 launch 的代价）。WMMA batch
   保留为显式开关 `megakernel_batch_decode=True` 供正确性研究。`MegakernelBackend.
   compatibility()`（`backend.py:93-100`）默认对 batch>1 返回 not supported →
   `model_runner.run` 自动回退 torch+cudagraph；`_should_burst`（`llm_engine.py:99-103`）
   同步拒绝 batch>1 burst。`megakernel-wmma` / `megakernel-burst` 后端（`compare.py`、
   `correctness_smoke.py`）显式开启该开关以走 megakernel batch 路径。67 unittest 通过 +
   GPU 实测：默认 batch=2 FALLBACK、batch=1 supported、开启后 batch=2 走 megakernel。
4. 若要彻底翻盘只能**去掉逐层 grid.sync**、改成每层软件流水/隐式同步 + 让每个 batched
   GEMM 用满全 grid（向 cudagraph 靠拢）——工作量大且不确定（docs/06 R3 的方向）。

## 8. 不要做的事

- **不要**再以「多 batch 没用张量核」作为根因——现已证明 FC GEMM 全 WMMA 仍输，
  这是本轮审查最有价值的纠正。
- **不要**把 barrier 原始成本当成 ~7-8ms 主开销——实测 pure barrier 仅 0.55ms，
  真正成本是它强制的层间串行潜伏期 + SM 空转。
- **不要**在 batch≥8 上承诺 speedup——并行 batch 上限 ≈10，架构上没有吞吐优势。
- **不要**把 `results/a100_nosync_compare.md` 的数字当 burst 性能（那走的是
  `ignore_eos=True` 的 step-wise 回退）。

## 9. 外部交叉印证：公开实现 / 研究与我们的结论一致（第 3 版新增）

以下来自公开的 megaQwen 实现、Ada-MK 论文与 vLLM 官方文档（§9.9 列引用），
**独立佐证**了本文 §0/§2 的结论。注意这些是外部项目的**口头/报告性结论**，非本仓库
实测；我们用它做「方向印证」，不放进来当本机证据。

### 9.1 megaQwen 与我们的现象几乎完全吻合

公开的 Qwen3-0.6B **megaQwen** 项目对这个问题有明确表述：

- megakernel 在 **batch=1、短 context** 下明显超过 vLLM/SGLang；
- 生产框架针对 **dynamic batch / variable batch size / long context /
  continuous batching** 做了优化，而它自己的 benchmark 是 **single model +
  single GPU + no continuous batching**；
- 作者把它归纳为「**batch=1、short context、static shapes**」的窄场景优化。

它的 kernel 结构与我们一致：**整个 Transformer block 融合进一个 persistent
kernel，用 cooperative groups 做 grid-wide synchronization**。其作者进一步实测：

- memory bandwidth 只用了约 **5%**；
- 每个 token 有 **140+ 次 `grid.sync()`**、单次约 **0.7 µs**；
- 最终 **batch=1 的性能已经被同步开销卡住**。

对照我们的 A100 实测：336 次 barrier / token、1.63 µs/barrier（§3.4）——量级吻合
（都是 ~1µs 级、每 token 上百次）。**双方共同说明：同步次数是 megakernel 的结构性
特征，且 batch=1 就已被此卡住。**

### 9.2 二者的优劣机制（与 §1-§2 一致）

> megakernel 的优势**不是「计算更快」**，而是「把大量 kernel launch + 中间
> HBM round-trip 消掉」。

- **batch=1**：launch overhead 是主要矛盾（占时比高），megakernel 消除它 → 赢；
- batch 增大后，launch overhead 被摊薄，真正的瓶颈变成 **GEMM 效率 / 内存带宽 /
  occupancy / 同步 / KV cache 访问 / 并行度** → 成熟的 batched cuBLAS/FMHA +
  CUDA Graph 开始占优。

这与 docs/07 第 1 版 §1.3 及本文 §2 的结论一致：**单序列 win 点 = launch 消除，
batch 放大后 GEMM 算力/利用率侧 `crossover` 反超**。

### 9.3 vLLM 的 CUDA Graph 已远超「普通 graph」

vLLM 现在是 `CudagraphDispatcher + BatchDescriptor`，按 `num_tokens/num_reqs/
uniform` 选 graph；graph mode 分 `NONE / PIECEWISE / FULL / FULL_DECODE_ONLY /
FULL_AND_PIECEWISE`，默认 `FULL_AND_PIECEWISE`（decode→Full，prefill/mixed→
Piecewise）。它预捕获 1/2/4/8/16/…/256 等 batch shape，按实际 batch 就近 replay
（不足则 padding）。这正说明：

> **CUDA Graph 不是「只适合 batch=1」，而是天生搭配 batched kernel 用于 decode。**
> 我们拿 batch>1 去和它比，是在 cudagraph 的强项上比。

### 9.4 生产框架已转向 hybrid，而非「megakernel 万能」

- vLLM：batching + optimized kernels + CUDA Graph + runtime dispatch；
- Ada-MK（arXiv:2605.11581）：把 MegaKernel 定位在 decode（消除 kernel launch
  与 inter-operator HBM round-trip），但**不停留在固定 megakernel**——做 shared-memory
  constraint modeling、K-dimension splitting、offline DAG search、运行时分支提前到
  compile time，作为 TensorRT-LLM 插件用于低延迟 decode。

方向已经从「MegaKernel vs CUDA Graph」转向「**MegaKernel + 成熟 batched kernel +
CUDA Graph + runtime scheduling**」的 hybrid 架构——与 docs/06 R3 的演进方向一致。

### 9.5 判定：我们的结果**合理，不是反常**，但要确认「真 batch-aware」还是「扩展」

行为：

```
B=1   MegaKernel > CUDA Graph
B↑    MegaKernel << CUDA Graph
```

大概率合理。但真正要查的是下面 §9.7 的 batch-aware 审计：megakernel 到底
是「真 batched megakernel」，还是「batch=1 kernel 被硬扩展到 batch=N」——
两者的性能含义完全不同。**我们的代码证据（每 GEMM 仅 4-12/80 block 干活、
tile padding 使用率 batch/16）指向后者**，这从内部机制上印证了外部项目
「megakernel 是 batch=1 窄场景优化」的说法。

### 9.6 crossover 实验（✅ A100 已跑，从「demo」升级成「可发表结论」）

不要问「megakernel 能不能打败 vLLM」，改问：

> **在 LLM decode 中，Kernel Fusion 与 CUDA Graph 的收益随 batch size 如何变化？
> MegaKernel 的最优适用区间在哪里？**

**实测**（A100-SXM4-80GB, Qwen3-0.6B, prompt_len=32, max_tokens=128, greedy,
warmup=2, runs=3, 主指标 = min-of-runs 滤干扰；`tools/bench_burst_batch.py`，
结果存 `results/crossover_scan.json` + `results/crossover_plot.png`）：

| batch | CG ms/tok | MegaKernel ms/tok | speedup | verdict |
|------:|----------:|------------------:|--------:|:--------|
| 1 | 2.944 | 2.241 | **1.31x** | **BEATS** |
| 2 | 1.634 | 11.352 | 0.14x | loses |
| 3 | 1.120 | 8.236 | 0.14x | loses |
| 4 | 0.840 | 6.631 | 0.13x | loses |
| 5 | 0.680 | 5.902 | 0.12x | loses |
| 6 | 0.566 | 5.621 | 0.10x | loses |
| 7 | 0.487 | 5.979 | 0.08x | loses |
| 8 | 0.429 | 6.495 | 0.07x | loses |
| 10 | 0.347 | 7.478 | 0.05x | loses (serial×2) |
| 16 | 0.224 | 6.500 | 0.03x | loses (serial×2) |
| 32 | 0.111 | 6.496 | 0.02x | loses (serial×4) |

> batch=1 权威复现（`tools/bench_nosync.py`, max_num_seqs=1, tokens=200）：
> burst 2.62 vs cudagraph 2.85 = **1.09x BEATS**；no-sync raw kernel 2.01 = 1.42x。
> 两个工具量级一致，crossover 工具用 max_num_seqs=32（噪声略大）+ min 指标后
> batch=1 为 1.31x，均落在交接记录的 1.04–1.16x 区间之上沿（本次 GPU 较空闲）。

**实测拐点结论**：

- **crossover 发生在 batch=1 → 2**，且是「悬崖式」而非「平缓过渡」：batch=1
  MegaKernel 赢 1.31x，batch=2 立刻跌到 0.14x（7x 慢），没有「打平区」。
  原因是 batch=1 的赢点 = 消除 launch 间隙（cooperative 一次 launch 排队 N 步），
  一旦 batch>1，固定 grid 按序列切分 + 每层 barrier 串行 + 每 GEMM 仅少数 block
  干活的三重根因（§2/§5）立即主导，launch 消除的收益被淹没。
- **CUDA Graph 平滑扩展**：ms/tok 从 2.94（b1）单调降到 0.11（b32），近似
  线性摊薄——每个 batched GEMM 用满 108 SM、张量核全开、无 padding。
- **MegaKernel 在 b6 触底（5.62）后平台化于 ~6.5 ms/tok**：b6→b32 几乎不变，
  证实 §2.3「每 GEMM 仅 4-12/80 block 干活，其余 SM 空转」是恒定开销而非
  随 batch 摊薄。b>8 因 `LDG_ATTN_BLOCKS=8` 上限拆成串行 burst（note 列标
  serial×N），但单 burst 内部开销不变，所以 b8/b16/b32 都停在 ~6.5。
- **b32 时 MegaKernel 慢 ~50x**（0.111 vs 6.496），与 §1.1 交接记录的 batch=8
  21x 量级一致（batch 更大差距更大）。

这把 §9.5 的「B≤B1 赢、B1<B<B2 打平、B≥B2 输」三段假设**修正为实测**：
真实形态是「B=1 赢、B≥2 立输」，没有打平区——因为 MegaKernel 的优势
（launch 消除）是 batch=1 独占的，batch>1 时 cudagraph 在每一项都是强项（§5），
叠加成单调扩大而非平缓过渡。

> 复现：`python -m tools.bench_burst_batch --batch-sizes 1,2,3,4,5,6,7,8,10,16,32 \
> --max-tokens 128 --runs 3 --warmup 2 --json results/crossover_scan.json`，
> 然后 `python -m tools.plot_crossover results/crossover_scan.json \
> --output results/crossover_plot.png`。本仓库并行 batch 上限 ≈10，
> batch>8 标注为 serial×N（hybrid dispatch 拆成多次 ≤8 的串行 burst）。

### 9.7 batch-aware 审计（最关键的自查清单）

检查以下字段，判断 batch 是否**直接映射到 CTA/warp**，还是把 batch=1 的 tile
分解硬套到 batch=N：

- `grid.x / grid.y、block.x`、batch 维、seq 维
- KV cache indexing、Q/K/V tile mapping、warp assignment

结合我们的代码：
- batch 进 `wmma_gemm` 的 M 维（`tiles_M`），**每个 GEMM 只用 4-12 个 block**，
  没有按 batch 扩展并行度 → 判定为「batch=1 tile 分解硬套 batch=N」；
- `LDG_ATTN_BLOCKS=8` 又把 grid 按 seq 切分，进一步限制 batch 上限 ≈10。

这基本等于坐实了 §9.5 的第二种解释。

### 9.8 四象限对照（避免误判）

只比「megakernel vs CUDA Graph」容易误判。应区分：

```
A. eager + normal kernels
B. normal kernels + CUDA Graph
C. MegaKernel
D. MegaKernel + CUDA Graph   ← 二者不互斥：CG 消 CPU launch，mega 消 kernel-to-kernel
```

理论上 D = CUDA Graph 包一层 megakernel 是可行的（megakernel 已只有一次 launch，
此收益主要来自执行路径稳定化 + graph replay 对 host scheduling 的优化）。跑满四个
象限能干净地区分「launch 消除」与「kernel 内效率」各自贡献多少。

### 9.9 引用

- megaQwen（Qwen3-0.6B custom CUDA kernel）：
  `github.com/NoneBone/megaQwen`
- Ada-MK: Adaptive MegaKernel Optimization via Automated DAG-based Search for LLM
  Inference（TensorRT-LLM 插件，L20 single-batch）：`arXiv:2605.11581`
- vLLM CUDA Graph 设计：`docs.vllm.ai/en/latest/design/cuda_graphs/`
- vLLM `CudagraphDispatcher`/`BatchDescriptor`/graph mode：vLLM 最新文档
- vLLM batch=12/20 graph capture 收益（~7x decode）：
  `github.com/vllm-project/vllm-ascend/issues/16049`

> 注：以上引用为 2026-09-11 检索的公开资料，链接文本与 arXiv 编号按检索当时为准；
> 引入本仓库时以「方向印证」看待，不作为本机实测依据。
