# 多 batch 场景为何打不过 cudagraph：证据与根因分析（修订版）

> 更新日期：2026-09-11（第 2 版）
>
> 第 2 版基于**当前代码**与**本机 A100 实测**修订。第 1 版（2026-09-10）结论方向
> 正确，但两处关键判断已过时需要纠正：
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
| ncu kernel 时间分解 / SM 占用 | 待跑 | 预期 GEMM 只用少数 SM |
| LM head 接 WMMA 后 A/B | 待跑 | 验证 scalar LM head 占比 |

## 7. 结论（更新为可执行判语）

1. **多 batch 的败局是资源使用模型问题，不是调参/张量核覆盖问题**。即便 QKV+O+MLP
   全上张量核，fixed-grid 按序列切分 + 每层全网格 barrier + 每 GEMM 少数 block 干活，
   决定了 batch>1 无法通过「优化某个 kernel」翻盘。
2. **唯一可持续的战略胜利点是 batch=1**：走 persistent kernel + 设备侧 EOS，靠消除
   launch 间隙 1.04-1.16x 微胜（docs/06 §0 实测）。
3. **务实集成**：batch=1 走 megakernel persistent；batch>1 回退 cudagraph
   （megakernel 批处理路径不占优，不值得付出层间串行 + 多 launch 的代价）。WMMA batch
   保留为显式开关供正确性研究。当前 hybrid dispatch（backend.py:250-274）已接近此
   形态，仅需把 batch>1 默认策略从 WMMA batch 调整为回退 cudagraph。
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
