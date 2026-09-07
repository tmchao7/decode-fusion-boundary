# Megakernel 与 nano-vLLM 集成问题清单

> 更新日期：2026-09-07
>
> 本文档只记录当前代码和服务器证据，不把“已经写了接口”误记为“已经完成”。
> 状态含义：**已验证**=有代码和实际服务器证据；**部分完成**=主路径已写但仍有
> correctness、生命周期或性能缺口；**未完成**=尚未进入可靠的集成路径。

## 当前准确结论

当前项目已经实现了 Qwen3-0.6B 的单序列 decode Megakernel，并直接复用
nano-vLLM 的 paged KV cache 和 scheduler block table。Megakernel 自己实现
paged attention；它没有调用 vLLM/PagedAttention kernel，也没有复用 nano-vLLM
原有的 FlashAttention 实现。

一次集成 decode step 当前实际是：

```text
nano-vLLM prefill / FlashAttention 写 KV
    ↓
prepare_decode 生成 token、position、block table
    ↓
Megakernel cooperative Transformer kernel
    ├── embedding
    ├── 28 层 Qwen Transformer
    ├── Q/K RMSNorm + RoPE
    ├── paged KV attention（自研 ldg_attention）
    ├── O projection + MLP
    └── final norm
    ↓
独立 LM-head kernel + argmax
    ↓
.item() 返回 CPU，scheduler 进入下一步
```

因此当前还不是“整个 decode step 一个 CUDA kernel、无 CPU 同步”的最终形态。

## 服务器最新证据

2026 年 9 月 7 日 A100 服务器结果：

- 环境、Qwen3-0.6B 配置、Megakernel extension 编译和加载通过；
- Python 测试已达到 45 项并通过；
- sanitizer 的 KV cache 非法写已经从报告中消失，最近一次 sanitizer 报告为
  `ERROR SUMMARY: 0 errors`；
- paged metadata 显示 `block_table[0]=0`、`num_cache_blocks=2516`，说明基础
  block table 传递正常；
- token-level correctness 仍失败：`synthetic_p8_b1` 的第 2 个输出 token
  baseline 为 `279`，Megakernel 为 `498`；
- exact-math 构建已提交，但还没有新的服务器 correctness 证据。

所以目前不能计算 Megakernel speedup，也不能把单序列 paged 集成标记为关闭。

## P0：统一 cache contract

**状态：部分完成，单序列 memory safety 已验证。**

nano-vLLM 分配的 cache 是：

```text
[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
```

Megakernel 通过 `self.kv_cache[0]` / `self.kv_cache[1]` 取得同一份 K/V
view，不创建第二套集成 cache。kernel 用：

```text
logical_block = position // block_size
block_offset  = position % block_size
physical_block = block_table[logical_block]
```

并直接读写对应 physical block。此前的 32 位 layer-stride 溢出已经改为 64 位，
A100 sanitizer 已验证没有该处非法访问。

仍需验证：长序列、255/256/257 边界、prefix 命中、block 释放和复用。

## P0：prefill 与 decode 连接

**状态：主路径已实现，token correctness 未通过。**

prefill 继续走 nano-vLLM 原有模型和 FlashAttention 路径；Megakernel 只在
`is_prefill=False` 且请求满足 Qwen3-0.6B、BF16、greedy、单卡等条件时进入。
Megakernel decode 不 replay prompt，直接读取 prefill 写入的 nano-vLLM paged
cache。服务器已证明这条路径可以安全完成 kernel，但输出数值仍不一致。

下一步必须逐层比较 baseline 和 Megakernel 的 Q/K/V、attention output、MLP、
final norm 和 LM-head top-k，而不是直接继续跑 benchmark。

## P1：Megakernel 是否真正替代 CUDA Graph

**状态：已修复初始化路径，decode 仍保留 CPU 驱动。**

当前 `ModelRunner` 只在标准 torch backend 下 capture CUDA Graph；Megakernel
backend 不再无条件 capture 一套无用 graph。Megakernel decode 本身不调用
`graph.replay()`。

但每个 token 仍会：

```text
launch cooperative Transformer
→ launch LM head
→ _out_token.item()
→ CPU scheduler append token
```

因此 CUDA Graph replay 开销被绕开了，但 CPU/GPU 往返和 token-level host sync
还没有消除。`generate_nosync` 目前只适用于 contiguous reference path，尚未
扩展到 paged KV 集成路径。

## P1：batch decode

**状态：接口存在，但不应视为完成。**

`decode_paged_batch` 可以接受最多 32 条 sequence、各自的 `positions` 和
`block_tables`，并使用同一个 paged cache pool。但当前 kernel 是：

```cpp
for (int seq_id = 0; seq_id < batch_size; ++seq_id) {
    ldg_decode_one(...);
}
```

即一次 launch 内顺序处理 sequence，不是 sequence-level parallel。当前已补充
每条 sequence 的 barrier generation 和 per-sequence flags 重新初始化逻辑，但
仍需 A100 上 batch=2/8/32、mixed-length 和 scheduler lifecycle correctness。

## P1：prefix cache

**状态：未完成。**

kernel 能按 block table 读取共享物理 block，但尚未验证：

- prefix 命中后的 K/V 数值一致性；
- 共享 block 是否只读且不会被错误覆盖；
- block ref count、抢占和释放是否一致；
- 部分 prefix 命中及跨 block 边界是否正确。

## P1：连续布局与 paged 布局

**状态：单序列 memory safety 已验证，数值 correctness 未通过。**

集成路径使用 paged cache；contiguous cache 只保留给 standalone reference。64 位
offset 已修复，但仍需完成 token-level correctness 和边界测试。

## P2：采样、并行和生命周期

### Sampling

**状态：未完成。** Megakernel 只支持 greedy argmax；temperature/sampling
请求必须 fallback 到 torch backend。

### Tensor parallel

**状态：未完成。** Megakernel 要求 `tensor_parallel_size=1`，没有接入 nano-vLLM
的多卡权重切分和通信。

### 生命周期

**状态：部分完成。** scheduler 仍然拥有 sequence、block allocation、释放和
抢占；Megakernel 只持有 shared cache view 和 scratch buffer。正常单序列路径的
cache 写入已经通过 sanitizer，但异常退出、block reuse、prefix cache 和 batch
生命周期仍未通过 GPU 测试。

## 当前代码审查结果

### 已完成且证据充分

- Megakernel backend 通过 `decode_backend="megakernel"` 独立选择；
- prefill 不进入 Megakernel；
- 集成路径不加载第二份 HF 模型，权重来自已加载的 nano-vLLM model；
- 直接绑定 nano-vLLM KV cache view；
- 直接读取 scheduler block table；
- paged KV offset 使用 64 位；
- 单序列 kernel 使用 cooperative launch 和 occupancy 限制；
- Megakernel 不满足条件时保留 torch fallback；
- CUDA 错误和 sanitizer 诊断链路已建立。

### 仍然不能宣传为已完成

- token-level correctness；
- 整个 decode step 单 kernel；
- 无 CPU `.item()` 同步；
- sequence-level parallel batch；
- prefix cache；
- V100 FP16；
- sampling、tensor parallel；
- 任何性能 speedup 结论。

## 下一步顺序

1. 用 exact math 重新验证单序列 token correctness；
2. 逐层 dump baseline/Megakernel 中间张量，定位数值分歧；
3. 通过后再验证 255/256/257 和长序列；
4. 验证 batch barrier、mixed-length 和 block lifecycle；
5. 增加 paged no-sync generation，减少 `.item()` 和 Python loop；
6. 最后运行 A100 benchmark、Nsight 和 V100 fallback 报告。

## 当前项目描述边界

可以这样描述：

> 在 nano-vLLM decode 路径中集成了一个面向 Qwen3-0.6B 的 CUDA Megakernel，
> 将主要 Transformer decode 计算融合到 cooperative kernel，并直接复用
> nano-vLLM scheduler 分配的 paged KV cache 和 block table。当前 A100 上单序列
> paged cache memory safety 已通过 sanitizer，但 token correctness、batch 并行
> 和端到端性能收益仍在验证。

不能这样描述：

> Megakernel 已经完全替代 CUDA Graph，复用 vLLM PagedAttention，并在 batch
> decode 上完成性能加速。
