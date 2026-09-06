# Megakernel 与 nano-vLLM 集成问题清单

本文档记录当前 Megakernel 直接接入 nano-vLLM 后，与原有 runtime、KV cache 和 scheduler 不顺承的地方。当前版本优先用于建立单序列 correctness 和 kernel 性能基线；以下问题需要后续逐项处理，完成前不能把项目描述为“完整复用 nano-vLLM paged KV cache 的 Megakernel”。

## 当前数据流

```text
nano-vLLM prefill
    ↓
写入 nano-vLLM paged KV cache
    ↓
Megakernel decode adapter
    ↓
重新 replay prompt
    ↓
写入 Megakernel private contiguous KV cache
    ↓
Megakernel 后续 decode
```

理想数据流应为：

```text
nano-vLLM scheduler
    ↓
同一份 paged KV cache
    ↓
Megakernel 直接读取和写入
```

## P0：统一 cache contract（已建立测试基线）

`src/nanovllm/backends/megakernel/cache.py` 现在定义了 nano-vLLM KV cache 的 shape、element stride、物理 block 映射和 flat element offset 规则。它只定义契约，不代表 CUDA kernel 已经完成 paged KV 直连；后续 kernel 必须遵守该契约。

## P0：重复 KV cache 与重复计算

### 问题 1：Megakernel 固定申请最大长度

当前 `src/nanovllm/backends/megakernel/model.py` 创建：

```text
K cache: [28, 8, 2048, 128], BF16
V cache: [28, 8, 2048, 128], BF16
```

两份 cache 合计约 224 MiB。即使请求只包含几十个 token，也会预留完整的 2048 token capacity，无法按 block 按需分配。

**影响：**

- 显存利用率低；
- 最大长度被硬编码为 2048；
- 无法复用 nano-vLLM 的 KV cache pool；
- 不适合多请求和高并发场景。

**后续处理：**让 Megakernel 接收 nano-vLLM 已经分配的 `kv_cache`，删除 Decoder 内部独立创建的 K/V cache。

### 问题 2：prompt 被重复计算

nano-vLLM 已经执行了 prefill，但 Megakernel 首次 decode 时会重新遍历 `seq.prompt_token_ids`，用 `decoder.step()` replay 整个 prompt。

**影响：**

- TTFT 被额外 replay 时间拉高；
- prefill 计算结果没有被复用；
- 端到端 benchmark 不再反映纯 decode 优化；
- 可能掩盖 kernel 本身的收益。

**后续处理：**先让 Megakernel 支持“接管 prefill 产出的 KV cache”，然后再评估是否需要 fused prefill；在此之前，报告中必须单独区分 kernel decode latency 和端到端 TTFT。

## P1：无法顺承原有 scheduler

### 问题 3：只能支持单序列

当前 backend 要求：

```text
len(seqs) == 1
max_num_seqs == 1
tensor_parallel_size == 1
greedy sampling
```

nano-vLLM scheduler 原本可以在一个 decode step 中调度多条 sequence，但 Megakernel 直接拒绝 batch，并回退标准路径。

**影响：**

- 无法评估 Megakernel 在真实并发场景的收益；
- batch workload 实际测到的是 fallback，而不是 Megakernel；
- 无法复用 scheduler 的连续 batch 调度能力。

**后续处理：**设计 `decode_batch()` 接口，传入每条 sequence 的 `context_lens`、`block_tables` 和 `slot_mapping`，先完成 batch=2/8 的 correctness，再扩大规模。

### 问题 4：不能复用 prefix cache

nano-vLLM 的 prefix cache 通过 block hash 和 block table 管理。当前 Megakernel 不读取已有 `block_table`，也不理解 prefix cache 命中后的逻辑位置。

**影响：**

- 相同前缀会被再次计算；
- prefix cache 命中率无法转化为 Megakernel 性能收益；
- cache 生命周期由两套系统分别管理，存在状态不一致风险。

**后续处理：**统一以 nano-vLLM 的 `block_table` 为 cache 真相源，让 Megakernel 按逻辑位置映射到物理 block；先覆盖无 prefix cache，再覆盖部分 prefix 命中和跨 block 场景。

## P1：KV cache 布局不一致

### 问题 5：连续布局与 paged 布局不一致

当前 Megakernel 按如下方式访问 cache：

```text
kv_head × max_seq_len × head_dim + position × head_dim
```

nano-vLLM 使用：

```text
[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
```

二者不能直接共享同一个 tensor 指针。

**后续处理：**统一 kernel metadata 和地址计算：

```text
logical_block = position // block_size
block_offset = position % block_size
physical_block = block_tables[seq_id, logical_block]
```

kernel 必须覆盖 block 边界、不同 sequence 长度和最后一个不完整 block。

## P2：采样、并行和生命周期问题

### 问题 6：只支持 greedy decode

当前 kernel 直接完成 argmax，无法直接承载 `temperature` sampling。非 greedy 请求会回退标准路径。

### 问题 7：暂不支持 tensor parallel

Megakernel 假设完整权重位于单卡，当前 backend 要求 `tensor_parallel_size=1`。nano-vLLM 原本支持多卡权重切分和通信。

### 问题 8：两套状态生命周期不同

nano-vLLM scheduler 负责 sequence 的等待、运行、抢占、完成和 block 释放；Megakernel Decoder 维护自己的 position、KV cache 和 scratch buffer。请求抢占、切换或异常时，两套状态不能自动同步。

**后续处理顺序：**

1. 单序列 paged KV 直连；
2. cache 写入和读取 correctness；
3. sequence 完成/释放同步；
4. batch decode；
5. prefix cache；
6. sampling；
7. tensor parallel。

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

> 在 nano-vLLM 中集成了一个面向 Qwen3-0.6B 单序列 greedy decode 的 CUDA Megakernel 原型，并保留标准 PyTorch/CUDA Graph 路径用于 correctness、fallback 和性能对比。

当前不能描述为：

> 已经完全复用 nano-vLLM 的 paged attention/KV cache，并在 batch decode 场景完成端到端加速。
