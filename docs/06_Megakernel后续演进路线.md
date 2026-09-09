# Megakernel 后续演进路线（给后续 agent 的方向参考）

> 更新日期：2026-09-09
>
> 本文是**面向改造的方向性指南**，不是结论报告。所有「当前状态」均带 `file:line`
> 锚点，方便后续 agent 直接定位。性能数字与 correctness 状态以
> [docs/04](04_Megakernel集成问题清单.md) 与 [docs/05](05_Megakernel收益与风险分析.md)
> 为准；本文只在此之上补充「要击败 cudagraph 还需要改哪里、怎么改」。
>
> 状态含义沿用 docs/04：**已接入**=engine 实际调用路径里在用；**已写未接入**=
> op/Python wrapper 都存在但 `ModelRunner.run()` 没调；**未完成**=尚未进入可靠路径。

## 0. 一句话现状

Megakernel 已经**正确**地直接复用 nano-vLLM 的 paged KV cache 和 scheduler
block table（不创建第二套 cache），单序列/batch 的 token correctness 已通过
（匹配精确参考）。但要「在 batch 场景用 tensor-core 击败 cudagraph」，
当前**两个关键能力虽然写好却没接进 engine**：WMMA tensor-core batch decode、
和 no-sync 设备侧调度。把它们接进来并扩展到 batch，是后续的主线。

## 1. 当前代码真实状态（带锚点）

### 1.1 已经在 engine 实际调用路径里的（已接入）

- 后端选择与 KV 绑定：`model_runner.py:63-78` 在 `decode_backend="megakernel"`
  时创建 `MegakernelBackend`，`set_kv_cache(self.kv_cache)` 把 nano-vLLM 的
  `[2, layers, blocks, block_size, kv_heads, head_dim]` cache 直接交给 megakernel。
- 兼容性门禁：`backend.py:71-107` `MegakernelBackend.compatibility()` 守住
  decode-only / TP=1 / greedy / Qwen3-0.6B 固定 shape / batch≤32。
- engine 分流：`model_runner.py:294-339` `run()` 在 `decision.supported` 时走
  megakernel，单序列调 `megakernel.decode(seq, block_table[0])`
  (`backend.py:222-237` → `Decoder.step` → `_decode_paged`)，多序列调
  `megakernel.decode_batch(input_ids, positions, block_tables)`
  (`backend.py:216-220` → `Decoder.step_batch` → `_decode_paged_batch`，**scalar 版**)。
- block table 直读：`model_runner.py:194-198` `prepare_block_tables()` 把
  scheduler 的 `seq.block_table` 装成 device int32 tensor 传进去；megakernel
  按 `block_table[position//block_size]` 算物理块，不重建 cache。
- 权重复用（不二次加载 HF 模型）：`model.py:129-180` `weights_from_nano_model()`
  从已加载的 nano-vLLM `model.model` 拆 qkv/gate_up，并 `_resolve_rope_theta`
  修 Qwen3 `rope_scaling` 内 `rope_theta=1e6` 的坑
  （`model.py:28-44`，AGENTS.md 已记录该事故）。
- 失败硬停（不在 kernel 已写 cache 后静默 fallback）：`model_runner.py:319-336`
  一旦 `decoder` 已初始化后 kernel 报错，置 `_cuda_poisoned` 并抛错，不回退 torch。

### 1.2 已写好但 engine 没调用的（已写未接入 —— 这是后续最大杠杆）

| 能力 | op / wrapper | engine 是否调用 | 位置 |
|---|---|---|---|
| WMMA tensor-core batch decode | `decode_paged_batch_wmma` / `Decoder.decode_paged_batch_wmma` | **否** | `model.py:417-441`、`444-468`（重复定义，需去重）；kernel `kernel.cu:974-1165`、launch `kernel.cu:2278-2335` |
| no-sync 单序列 N 步生成 | `generate_nosync_paged` / `Decoder.generate_paged` | **否**（仅 `tools/bench_nosync.py`） | `model.py:523-571`；kernel `kernel.cu:2487-2565`；设备侧反馈 `ldg_update_step` `kernel.cu:2001` |
| 持久 paged 单序列 kernel | `ldg_decode_kernel_persistent_paged` | 被 no-sync 复用，未独立暴露 | `kernel.cu:1969` |

> **结论**：当前 engine 的 batch 路径走的是 **scalar** `decode_paged_batch`，
> 并没有用 tensor-core；no-sync 也只测过单序列独立 bench，没进 scheduler 循环。
> 「tensor-core batch 击败 cudagraph」所需的两个零件都还在 bench 工具里，没上流水线。

### 1.3 仍缺的零件（未完成）

- **batch + no-sync**：`ldg_update_step` (`kernel.cu:2001`) 只反馈单序列
  `d_output_token → d_mutable_token_id / d_mutable_position`；没有 batch 版
  （per-seq token/position 数组 + per-seq EOS 掩码 + per-seq 是否还在生成的 flag）。
- **设备侧 EOS / 提前结束**：`generate_paged` 假设 `num_tokens` 固定、block 全预分配，
  没有在 device 上识别 EOS 后停止单条序列。
- **LM head 仍在 cooperative kernel 之外**：每步是 `cooperative kernel + lm_head
  kernel + (no-sync 时) update_step` 三个 launch，per-step launch 开销没完全消除
  （`kernel.cu:2450-2480` 循环里每步 3 个 launch）。
- **burst 预分配 block**：scheduler 现在一步一分配（`block_manager.may_append`
  `scheduler.py:70`），no-sync 多步生成需要生成区间内的 block 一次性预分配好。

## 2. 要在 batch 场景用 tensor-core 击败 cudagraph：差距分解

### 2.1 差距来源（按影响排序）

实测（A100, 单序列 200 tok, 见 docs/04 §P1）：

```
路径                        E2E ms/tok   纯 GPU kernel
megakernel .item() 同步      ~29          —
megakernel no-sync          ~3-4.5       1.86
torch-cudagraph             ~2.8         0.96
```

差距拆成三层：

1. **host 同步层**（已基本解决）：`.item()`/`.cpu().tolist()` 每 token 主机往返。
   no-sync 已把单序列从 15x 慢压到 1.3-1.5x。**batch 路径还没享受这个收益**
   （`step_batch` 末尾 `model.py:415` 仍 `.cpu().tolist()`）。
2. **per-step launch 层**（部分解决）：no-sync 把 N 步 back-to-back 排队，但每步
   仍是 `cooperative + lm_head + update` 三个 launch（`kernel.cu:2450-2480`）。
   单 launch 内循环多 token 才能彻底消除。
3. **kernel 质量层**（核心待解决）：纯 GPU 内核 megakernel 1.86ms vs cudagraph
   0.96ms。瓶颈是 scalar FMA matvec + 自研 `ldg_attention`。**WMMA 把 QKV/O/Gate/
   Up/Down 改成整批 GEMM 已数值验证（batch≤8），但还没接进 engine、还没做最终
   perf 对比、attention 仍是 scalar**。

> 击败 cudagraph 的充分条件（不是必要）：layer 1+2 在 batch 上落地 + layer 3
> 把 attention 也纳入 tensor-core 或至少消除其与 GEMM 的串行等待。

### 2.2 改造点 R1：把 WMMA batch 接进 engine（P0，低风险高收益）

当前 `ModelRunner.run()` 多序列分支只调 scalar `decode_batch`。第一步**不引入
no-sync**，仅把 batch 路径从 scalar 换成 WMMA，验证 batch perf 是否追上 cudagraph。

具体改动：

- `model_runner.py:313-318`：多序列分支改为调一个新的
  `megakernel.decode_batch_wmma(input_ids, positions, block_tables)`（语义同
  `decode_batch` 但走 WMMA op）。
- `backend.py:216-220`：`decode_batch` 内按 `config` 或环境开关选择 WMMA op；
  或新增 `decode_batch_wmma` 方法。建议加一个 `megakernel_batch_kernel` config
  （`scalar`/`wmma`）便于 A/B，不要硬编码。
- `model.py:417-468`：**先删除重复定义的第二个 `decode_paged_batch_wmma`**
  （`model.py:444-468` 是 `417-441` 的逐字复制，纯死代码，会误导后续 agent）。
  把保留下来的方法签名改成接受 tensor（`input_token_ids`/`positions` 为 device
  tensor），与 `step_batch` 对齐，避免 `model.py:427-428` 现在每次 `torch.tensor(...,
  device="cuda")` 的 host→device 上传开销。
- 验收门槛：batch=1/2/4/8、prompt 32/128/512 下，WMMA batch 的 E2E decode tok/s
  ≥ scalar batch，且与 cudagraph 差距 <1.3x（用 `benchmarks/compare.py`）。

### 2.3 改造点 R2：no-sync 扩展到 batch + WMMA（P0，主线）

这是「真正击败 cudagraph」的核心。目标：一次 host 提交，device 上跑 K 步，
每步对所有序列各产 1 token，per-seq EOS 后该序列停跑，最后单次同步回传。

需要的新零件：

1. **batch 版 `ldg_update_step`**（`kernel.cu:2001` 旁新增）：
   - 输入：`d_output_tokens[batch]`（LM head 输出）、`d_mutable_tokens[batch]`、
     `d_mutable_positions[batch]`、`d_active[batch]`（per-seq 是否还在生成）、
     `d_eos_token`、`output_log[batch][max_steps]`、`d_step_counter`。
   - 每序列独立：若 `active[s]`，写 `output_log[s][step] = out_tok`，检查 EOS /
     `max_tokens`，命中则 `active[s]=0`；否则 `mutable_token[s]=out_tok`、
     `mutable_position[s]++`。
   - 单线程单 block 即可（开销可忽略），但要 per-seq 独立判断。
2. **batch 版 `generate_nosync_paged`**（`kernel.cu:2487` 旁新增）：
   - 复用 `ldg_decode_kernel_batch_wmma`（已是 sequence-level 并行 + WMMA）。
   - 循环 `for step in [0, max_steps)`：launch WMMA batch kernel → batch LM head
     → batch `ldg_update_step`。**所有序列共享同一个 step 计数**，但 `active[]`
     决定每条是否真的前进；`max_steps = max(max_tokens over batch)`。
   - block_tables 在整个 burst 内固定（见 R4）。
3. **早停**：所有序列 `active==0` 时提前退出。简单实现：host 侧用
   `cudaMemcpyAsync(d_active, ...)` + `cudaStreamQuery` 不行（会同步）；
   最简方案是 host 始终排满 `max_steps`，让 inactive 序列的 kernel 步变成
   「读 active flag 后立刻 return」的轻量空转（每步 cooperative launch 仍要付，
   但算力空转极小）。彻底方案是把 step 循环也搬进 persistent kernel（见 R3）。

验收门槛：batch=8、各 128 token，E2E decode tok/s **≥ cudagraph**
（用 `tools/bench_nosync.py` 的 batch 版扩展）。

### 2.4 改造点 R3：单 launch 内多 token 循环（P1，决定性优势）

R2 仍每步付 3 个 launch（cooperative + lm_head + update）。把整个「K 步 × batch」
循环**搬进一个 persistent cooperative kernel**：kernel 内部 `for (step=0;
step<max_steps; step++)`，每步末尾在 device 上做 argmax→token 反馈 + grid.sync()，
LM head 也融合进同一个 grid（已有 `ldg_lm_head_fused` 逻辑，改为 grid 内调用而非
新 launch）。这是 megakernel 相对 cudagraph 的**决定性架构优势**：cudagraph 再
replay 也仍是 N 个 kernel 的图，persistent megakernel 是 1 个 kernel 内 N 步。

风险：persistent kernel 占满所有 SM，LM head 的 vocab=151936 归约在 grid 内做
需要重新分块；occupancy/寄存器压力要重新评估（见 docs/05 §6.2）。

验收门槛：纯 GPU kernel 时间 < cudagraph 各 kernel 时间总和（用 ncu）；
E2E decode tok/s > cudagraph 1.1x 以上。

### 2.5 改造点 R4：burst block 预分配（P0，R2 的前置）

no-sync 多步要求生成区间内 block table 固定。当前 scheduler 一步一分配
（`scheduler.py:67-71` `may_append`）。需要：

- 新增 `BlockManager.allocate_burst(seq, max_new_tokens)`：一次性把
  `ceil(max_new_tokens / block_size)` 个新 block 挂到 `seq.block_table`，
  返回 burst 内 block table 不变的承诺。
- scheduler 在把序列交给 megakernel no-sync 路径前调用它；burst 结束后
  回到正常一步一分配的 cadence（处理 EOS 提前结束未用满的 block —— 释放即可，
  因为 burst 内写过的 slot 之外都是干净的）。
- 内存压力：`max_new_tokens` 不能取 `config.max_model_len`（会瞬间耗尽 pool）。
  建议固定 burst 上限（如 64 或 128），超过则分多段 burst；或按剩余 pool 动态算。

验收门槛：burst 期间无 `block_manager` 调用；EOS 提前结束后未用 block 正确释放
（`tools/correctness_smoke.py` + 交错完成用例）。

## 3. 更无缝地接入 nano-vLLM 的 decode 阶段

### 3.1 当前 cadence vs 目标 cadence

当前（`llm_engine.py:75-107` `step()` → `model_runner.run()`）：

```
每 token：
  scheduler.schedule()              CPU
  prepare_block_tables/decode       CPU + 小 H2D
  megakernel.decode_batch()         1 GPU step
  .cpu().tolist()                   H2D 同步
  scheduler.postprocess()           CPU（append_token、EOS、deallocate）
```

megakernel 被钉在「与 cudagraph 完全相同的逐步 cadence」上，只是把 N 个小 kernel
换成 1 个大 kernel。**这不是 megakernel 的目标形态**。

目标 cadence（R2/R4 落地后）：

```
每 burst（K token × batch）：
  scheduler.schedule_burst()        CPU：选序列、预分配 burst block
  megakernel.generate_batch_nosync(seqs, block_tables, max_steps)  GPU K 步
  单次同步回传 output_log[batch][K]  H2D
  scheduler.postprocess_burst()     CPU：逐 seq 扫 output_log，append、判 EOS、释放
```

### 3.2 三方契约改动

- **Scheduler**：新增 `schedule_burst()` —— 选 running 序列 + 算每序列 burst 内
  `max_new_tokens = min(seq.max_tokens - seq.num_completion_tokens, BURST_CAP)`；
  调 `block_manager.allocate_burst`；返回 `(seqs, max_steps, block_tables)`。
  `postprocess_burst()` 按 `output_log` 一次 append 多 token、判 EOS、释放未用 block。
- **ModelRunner**：`run()` 在 `decode_backend=="megakernel"` 且 batch 全 greedy 时
  走 burst 分支；否则保留现行逐步路径作为 fallback（兼容 sampling / TP>1）。
- **MegakernelBackend**：新增 `generate_batch_nosync(seqs, block_tables,
  max_steps) -> list[list[int]]`，内部调 R2 的 batch no-sync op。

### 3.3 不要破坏的现有契约

- `cache_len = position + 1`（AGENTS.md 调试记录）：decode 写当前 token 的 K/V 后，
  attention 要 attend `[0, position]` 闭区间。no-sync 多步里每步仍要守这条。
- block table 在 burst 内固定；burst 之外 scheduler 仍可正常抢占/释放。
- prefill 仍走 torch/FlashAttention，**不进 megakernel**（保持 `backend.py:73`）。
- 失败硬停策略（`model_runner.py:319-336`）保留：kernel 已写 cache 后不静默 fallback。

## 4. 丝滑复用 nano-vLLM KV cache（澄清 + 强化）

### 4.1 澄清：megakernel 已经在复用，没有「中间再生成」

用户原话「而不是由 megakernel 中间再生成」可能源于误解。当前实现
（`backend.py:109-114`、`model.py:239-255`、`cache.py`）：

- megakernel **直接持有** `self.kv_cache[0]` / `[1]` 的 view，就是 nano-vLLM
  `ModelRunner.allocate_kv_cache` (`model_runner.py:164-192`) 分配的那块；
- 它**不**另外 malloc 任何 K/V pool（standalone `load_weights` 的 contiguous cache
  仅用于离线参考工具，不进 engine 路径）；
- prefill 的 K/V 由 nano-vLLM 原模型 + FlashAttention 写入同一块 cache；
  decode 时 megakernel 按 `block_table` 读这些 slot，并把自己新算的 K/V 写进
  `position` 对应的追加 slot —— 这是**必要且正确**的写，不是「再生成一套」。

所以 KV 侧没有「去掉中间再生成」的工作可做。**真正要强化的**是下面这些契约点。

### 4.2 需要强化的 KV 契约点

1. **layout 守护已就位但要随 R2 更新**：`backend.py:132-174`
   `_validate_cache_layout()` 在 init 时校验 shape/contiguous/layer_stride。
   R2 的 batch no-sync op 要复用同一份 cache view，**不要在新 op 里重新解析
   layout**，直接传 `k_cache`/`v_cache` 指针 + `PagedPVCacheSpec` 的 stride 常量。
2. **64 位 offset 不可回退**：`kernel.cu` 里 `kvls` 等用 `int64_t`（已修过 32 位
   溢出，见 docs/04 §P0）。R2/R3 的新 kernel 在写 batch 版 offset 时同样要 `int64_t`，
   不要图省事用 `int`。
3. **写契约对齐 nano-vLLM**：每步写 `position` 的 K/V，attention attend `[0, position]`。
   no-sync 多步里，第 `i` 步写 `start_position + i` 的 slot，attend 到
   `start_position + i`。`ldg_decode_kernel_persistent_paged` 已守这条
   （`kernel.cu:1969` 起），batch 版要逐序列守。
4. **prefix cache 共享 block 的读安全**：megakernel 只写自己 `position` 的追加 slot，
   不覆盖共享 prefix slot（docs/04 §P1 已验证）。R2 burst 期间同理：burst 内每步
   只追加，不动共享 block。`hash_blocks` (`block_manager.py:110-119`) 在 burst 结束
   后由 `postprocess_burst` 统一做，不要在 burst 中途 hash（会读到半写状态）。
5. **不要让 megakernel 调用 nano-vLLM 的 FlashAttention**：cooperative kernel 内
   跨 block 同步与 FA 的 launch 模型不兼容；自研 `ldg_attention` 是正确选择。
   若 attention 想用 tensor-core，方向是把 attention 的 QK/softmax/PV 改成 WMMA
   fragment（独立工作项），而不是引入 FA kernel。

### 4.3 不要做的事

- **不要**在 megakernel 里 malloc 第二份 KV pool（会翻倍显存、破坏 prefill/decode
  共享）。
- **不要**把 nano-vLLM 的 `slot_mapping`/`cu_seqlens` 机制搬进 megakernel decode
  （那是 prefill 的变长批语义；decode 是固定 M=1/batch 的 GEMM + paged 读，用
  block_table 就够）。
- **不要**为了「复用」而让 megakernel 反向调 `model.forward()` —— 那会回到逐 kernel
  launch，丢掉融合收益。

## 5. 优先级路线图

| 优先级 | 改造点 | 主要文件 | 验收 |
|---|---|---|---|
| P0 | 去掉 `model.py:444-468` 重复 `decode_paged_batch_wmma` | `model.py` | compileall + 现有 unittest |
| P0 | R1：WMMA batch 接进 engine（逐步 cadence，先不上 no-sync） | `model_runner.py:313-318`、`backend.py:216-220`、`model.py:417-441` | batch≤8 perf ≥ scalar batch，与 cudagraph <1.3x |
| P0 | R4：burst block 预分配 | `block_manager.py`、`scheduler.py` | burst 内无 may_append；交错完成正确 |
| P0 | R2：batch + WMMA no-sync（含 batch `ldg_update_step`、设备侧 EOS） | `kernel.cu:2001`/`2487` 旁、`model.py`、`backend.py`、`model_runner.py` | batch=8 E2E decode tok/s ≥ cudagraph |
| P1 | R3：单 launch 内多 token 循环（LM head 融合进 grid） | `kernel.cu` | 纯 GPU kernel 时间 < cudagraph kernel 总和 |
| P1 | attention 用 tensor-core（QK/PV WMMA） | `kernel.cu` `ldg_attention` | ncu memory throughput ≥ 70% 峰值 |
| P2 | sampling（temperature>0）接入 | `backend.py:86-91`、kernel | 与 torch sampler 数值一致 |
| P2 | V100 FP16 路径（sm_70 WMMA 是 fp16） | `kernel.cu`、`build.py` | V100 correctness + perf |
| P2 | tensor_parallel>1 | `backend.py:83`、weight 切分 | 与 nano-vLLM TP 对齐 |

## 6. 验证与回归基线（每个 P0 改造都要跑）

1. **fast smoke**：`python -m compileall src/nanovllm benchmarks tools` +
   `PYTHONPATH=src python -m unittest discover -s tests -v`。
2. **correctness**：`python -m tools.correctness_smoke MODEL`（greedy）+ 255/256/257
   边界 + 长序列（len=1000 跨 block）+ batch=2/4/8 mixed-length + 交错完成。
   口径：匹配**精确 Python 参考**（非逐 bit torch，见 AGENTS.md 调试记录）。
3. **perf A/B**：`python -m benchmarks.compare MODEL`（torch-eager / cudagraph /
   megakernel 三路）+ `tools/bench_nosync.py`（batch 版扩展）。
   记录：batch=1/2/4/8、prompt 32/128/512、decode_tok_s、tpot_ms、纯 GPU kernel
   时间（ncu）。
4. **server 落地**：按 AGENTS.md「Server Command File Rule」只更新
   `SERVER_COMMANDS.txt`，`RUN_MODE=benchmark` / `profile`，A100 跑完写
   `results/a100.json` + `python -m tools.plot_metrics`。

## 7. 风险与不要做的事（汇总）

- **不要**在没有跑 P0 验证基线的前提下声称「tensor-core batch 已击败 cudagraph」。
  当前 WMMA batch 只有 batch≤8、参考位置的数值验证，**没有 perf 数字**。
- **不要**把 `max_steps` 取成 `max_model_len`（会瞬间耗尽 KV pool）。burst 必须有上限。
- **不要**在 cooperative kernel 已写 cache 后静默 fallback torch
  （`model_runner.py:319-336` 的硬停策略要保留）。
- **不要**为了消除 per-step launch 而牺牲 correctness：persistent kernel 内多步循环
  时，每步的 grid.sync() 和 KV 写位置必须严格守 `cache_len = position + 1`。
- **occupancy 风险**（docs/05 §6.2）：R3 单 launch 全融合会把 LM head 的 vocab
  归约塞进同一个 grid，寄存器/SMEM 压力可能爆。先 ncu 量 occupancy 和 spill，再决定
  是否拆分。
- **甜点区预期**（docs/05 §9）：小 batch（1-8）才有正向收益；batch 越大权重带宽越
  摊薄、cudagraph 越难超过。不要在 batch=32 上承诺 speedup。
- **bf16 级联**（AGENTS.md + docs/04 §P1）：WMMA 与 scalar 归约顺序不同，~1 ulp
  差异随 28 层放大，个别位置 greedy 偶发翻转是**预期**，按「与 nano-vLLM/torch 参考
  比较」口径评估，不要按 scalar-megakernel 逐 bit 对齐去 debug。

## 8. 给后续 agent 的「第一刀」建议

如果只做一件事，做 **R1 + 删重复定义**：把 WMMA batch 接进 `ModelRunner.run()` 的
多序列分支，并删掉 `model.py:444-468` 的死代码。这一刀零风险（数值已验证）、立刻让
engine 的 batch 路径用上 tensor-core，并产出第一个「WMMA batch vs cudagraph」的
真实 perf 数字，作为 R2/R3 是否值得投入的决策依据。在拿到这个数字之前，R2/R3 都是
基于「WMMA 应该更快」的假设，不要直接跳进 R3。
