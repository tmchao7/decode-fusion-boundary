# Megakernel 与 nano-vLLM 集成问题清单

本文档记录当前 Megakernel 接入 nano-vLLM 后，与原有 runtime、KV cache 和 scheduler 之间的集成状态。问题按依赖关系排序，状态分为：

- **已实现，待服务器验证**：代码路径和本地测试已完成，但需要 A100/V100 真实 CUDA correctness 或 profiling；
- **未实现**：尚未进入集成路径；
- **已关闭**：有完整测试和服务器证据。

当前没有任何问题可以标记为“已关闭”，因为 paged kernel 尚未在 A100 上编译并完成 token-level correctness。

## 当前数据流（开发分支目标路径）

```text
nano-vLLM prefill
    ↓
写入 nano-vLLM paged KV cache
    ↓
Megakernel decode adapter
    ↓
decode_paged 直接读取/写回同一份 cache
```

旧的 private contiguous cache 和 prompt replay 只保留在 reference 代码中，
不应再被 nano-vLLM 集成路径使用。A100 上的 CUDA correctness 仍需服务器验证。

最近服务器暴露的非法显存访问已完成第一轮根因修复：Megakernel 的跨 block
barrier/flag 改为由 host 在同一 CUDA stream 上初始化，移除了 block 0 在 kernel
内部进行全局重置的竞态；同时增加了 transformer decode 与 LM head 的分阶段
CUDA 错误检查。该修复仍必须通过服务器重编译和 token-level correctness，暂不
标记为“已关闭”。

理想数据流应为：

```text
nano-vLLM scheduler
    ↓
同一份 paged KV cache
    ↓
Megakernel 直接读取和写入
```

## P0：统一 cache contract（已建立测试基线，CUDA 待验证）

`src/nanovllm/backends/megakernel/cache.py` 定义了 nano-vLLM KV cache 的 shape、element stride、物理 block 映射和 flat element offset 规则。当前开发分支已经增加 `decode_paged` operator 和 kernel paged 地址计算；是否在 A100 上正确执行仍需服务器验证。

## P0：重复 KV cache 与重复计算

### 问题 1：Megakernel 固定申请最大长度

**状态：已实现，待服务器验证。** 集成路径不再创建独立 cache；只有 standalone reference 模式仍会创建 contiguous cache。

当前 paged 集成路径使用 `src/nanovllm/engine/model_runner.py` 分配的 cache：

```text
K/V cache view: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
combined cache:  [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
```

`num_blocks` 由 nano-vLLM 的 GPU memory budget 决定，sequence 通过 block table 按需占用 block；固定 2048 contiguous cache 只存在于 standalone reference 模式。

**影响：**

- standalone reference 模式显存利用率低；
- reference 模式最大长度被硬编码为 2048；
- 集成 paged path 已能复用 nano-vLLM 的 KV cache pool；
- 多请求场景仍受 Megakernel batch 限制。

**当前进展：**backend 已接收并绑定 nano-vLLM 的 `kv_cache`，paged decoder 不再创建独立 K/V cache；需要通过服务器 correctness 后才能关闭此问题。

### 问题 2：prompt 被重复计算

**状态：已实现，待服务器验证。** nano-vLLM prefill 后，Megakernel 首次 decode 直接接收共享 paged cache；旧的 prompt replay 仅保留在 contiguous reference 代码路径。

**影响：**

- TTFT 被额外 replay 时间拉高；
- prefill 计算结果没有被复用；
- 端到端 benchmark 不再反映纯 decode 优化；
- 可能掩盖 kernel 本身的收益。

**当前进展：**decode 路径已改为接管 prefill 产出的 paged KV cache，并移除 adapter 内的 prompt replay；需要 A100 token-level correctness 和 TTFT 对比来关闭此问题。

## P1：无法顺承原有 scheduler

### 问题 3：不能消费 nano-vLLM 的 batch sequence 整体计算

**状态：代码已实现，待服务器验证和性能优化。**

当前 backend 的初始 paged batch 限制是：

```text
1 <= len(seqs) <= 32
max_num_seqs >= len(seqs)
tensor_parallel_size == 1
greedy sampling
```

原始 nano-vLLM 在 decode 阶段会由 scheduler 将多条 running sequence 组成一个 batch：每条 sequence 提供一个 `last_token`，并携带自己的 `context_len`、`block_table` 和 `slot_mapping`。当前 Megakernel 已能接收最多 32 条 sequence，并通过 `decode_paged_batch` 在一次 PyTorch operator 调用中处理整个 batch；为降低初版同步风险，kernel 内部暂时按 sequence 依次处理，尚未做到 sequence 级并行。

**影响：**

- 当前 batch kernel 还没有发挥 sequence 级并行度；
- 无法在服务器验证前确认真实并发场景收益；
- batch workload 需要检查 `backend_actual`，避免把 fallback 当成 Megakernel 性能；
- 多 sequence 的 KV cache 读写虽已进入 Megakernel operator，但生命周期和性能仍待验证。

**后续处理：**在现有 `decode_paged_batch()` 接口上完成 correctness，并逐步并行化 kernel，传入：

```text
input_ids[batch]
positions[batch]
context_lens[batch]
block_tables[batch, max_blocks]
slot_mapping[batch]
k_cache
v_cache
output_token_ids[batch]
```

先完成 batch=2/8 的 correctness，再扩大到 mixed-length batch 和 batch=32。

### 问题 4：不能复用 prefix cache

**状态：未实现。** 虽然 kernel 已经读取 `block_table`，但 prefix-cache 命中、共享 block 和 ref-count 生命周期还没有服务器 correctness 覆盖。

nano-vLLM 的 prefix cache 通过 block hash 和 block table 管理。当前 kernel 已能接收单序列 `block_table`，但 prefix cache 命中、共享 block 和 ref-count 生命周期还没有完整 GPU correctness 覆盖。

**影响：**

- 相同前缀会被再次计算；
- prefix cache 命中率无法转化为 Megakernel 性能收益；
- cache 生命周期由两套系统分别管理，存在状态不一致风险。

**后续处理：**统一以 nano-vLLM 的 `block_table` 为 cache 真相源，让 Megakernel 按逻辑位置映射到物理 block；先覆盖无 prefix cache，再覆盖部分 prefix 命中和跨 block 场景。

## P1：KV cache 布局不一致

### 问题 5：连续布局与 paged 布局不一致

**状态：代码已实现，待服务器验证。** `decode_paged` 已增加 paged 地址计算；旧 contiguous 访问仍保留用于 reference benchmark。

当前开发分支保留两种访问模式：

```text
reference path: kv_head × max_seq_len × head_dim + position × head_dim
paged path:     [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
```

集成路径的 `decode_paged` 已按 nano-vLLM paged layout 计算地址；reference path 仅用于独立 contiguous benchmark。

**后续处理：**继续完善并验证 kernel metadata 和地址计算：

```text
logical_block = position // block_size
block_offset = position % block_size
physical_block = block_tables[seq_id, logical_block]
```

kernel 必须覆盖 block 边界、不同 sequence 长度和最后一个不完整 block。

## P2：采样、并行和生命周期问题

### 问题 6：只支持 greedy decode

**状态：未实现。**

当前 kernel 直接完成 argmax，无法直接承载 `temperature` sampling。非 greedy 请求会回退标准路径。

### 问题 7：暂不支持 tensor parallel

**状态：未实现。**

Megakernel 假设完整权重位于单卡，当前 backend 要求 `tensor_parallel_size=1`。nano-vLLM 原本支持多卡权重切分和通信。

### 问题 8：两套状态生命周期不同

**状态：部分处理，待服务器验证。** 集成 decoder 已不再持有 sequence-specific KV cache 和 prompt replay state，但 sequence 抢占、block 复用和异常路径仍未完成 GPU 测试。

nano-vLLM scheduler 负责 sequence 的等待、运行、抢占、完成和 block 释放；Megakernel 当前只保留共享 cache view 和 scratch buffer，position/block table 来自当前 sequence/context。请求抢占、切换或异常复用仍未完成 GPU 验证。

**后续处理顺序：**

1. 单序列 paged KV 直连（代码已完成，服务器待验证）；
2. cache 写入和读取 correctness；
3. sequence 完成/释放同步；
4. batch decode；
5. prefix cache；
6. sampling；
7. tensor parallel。

## 当前阶段完成情况

### 已完成的代码工作

- 统一了 `PagedKVCacheSpec` 的 shape、stride 和物理 block 映射契约；
- 增加 `decode_paged` PyTorch operator；
- CUDA kernel 增加单序列 paged KV 地址计算；
- backend 绑定 nano-vLLM 的 K/V cache view；
- 首次 decode 移除 prompt replay；
- 保留 torch eager、CUDA Graph 和 contiguous reference 作为对照/回退路径；
- 增加本地 contract、backend 和 CUDA source tests。

### 尚未完成的真实验证

- A100 上编译 `decode_paged`；
- A100 上确认 occupancy-limited grid 不再触发跨 block barrier 死锁；
- block 边界 `255/256/257` token correctness；
- prefill 写入 cache 后 Megakernel 直接读取 correctness；
- 新 token 写回 `slot_mapping` 对应物理位置；
- 多次请求后的 block 释放和复用；
- Nsight launch/memory 指标；
- batch paged KV 的 A100 correctness 和 sequence 级并行性能；
- prefix cache、V100 FP16、sampling 和 tensor parallel。

## 每个问题的关闭标准

问题只有在以下证据齐全后才能标记为完成：

- 有对应单元测试或 GPU correctness case；
- A100 上与 torch baseline token-by-token 一致；
- 结果 JSON 记录实际 backend，没有 fallback 伪装；
- Nsight 能确认 kernel launch 和内存访问路径；
- 至少记录变更前后的 TTFT、TPOT、ITL、decode tok/s 和显存；
- 文档中的当前数据流与代码实现一致。

## 当前项目描述边界

当前可以准确描述为：

> 在 nano-vLLM 的单序列 decode 路径中加入了一个初步的 paged-KV Megakernel 接口，复用 scheduler 分配的 K/V cache；该路径已完成本地静态验证，A100 CUDA correctness 待验证。

当前不能描述为：

> 已经完全复用 nano-vLLM 的 paged attention/KV cache，并在 batch decode 场景完成端到端加速。
