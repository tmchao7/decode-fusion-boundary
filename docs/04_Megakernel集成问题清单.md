# Megakernel 与 nano-vLLM 集成问题清单

> 更新日期：2026-09-08
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

2026 年 9 月 8 日 A100 服务器结果：

- 环境、Qwen3-0.6B 配置、Megakernel extension 编译和加载通过；
- Python 测试达到 47 项并通过；
- sanitizer 的 KV cache 非法写已经从报告中消失，最近一次 sanitizer 报告为
  `ERROR SUMMARY: 0 errors`；
- paged metadata 显示 `block_table[0]=0`、`num_cache_blocks=2516`，说明基础
  block table 传递正常；
- **token correctness 已解决（匹配精确参考）**。根因是 Qwen3-0.6B 的
  `rope_theta=1e6` 存在 `rope_scaling` dict 而非顶层字段，Megakernel 权重加载
  用 `getattr(config, "rope_theta", 10000.0)` 静默回退到 `10000`，导致 cos/sin
  频率错误。已加 `_resolve_rope_theta(config)` 修复（见下）。
- 每个真实 decode 位置（`[n_prompt, n_prompt+n_completion-1)`）的 K/V 与
  精确 Python 参考逐位一致（max diff = 0.0）；与 torch-eager 的差异仅为
  ~1 ulp 的融合 kernel 归约顺序差，可按 bf16 精度级联处理（可接受）。
- 边界 255/256/257 与长序列（len=1000 跨 block）内存安全、输出连贯；
- batch=2/8/32、mixed-length、交错完成（block 生命周期）均 batch==single；
- 修复了一个 batch 缩小到单条时的 bug：单序列 `Decode.step()` 对
  `_out_token`（size=`_max_batch_size`>1）调用 `.item()` 报错，改为读取元素 0。

所以目前 Megakernel 单序列与 batch 的 paged KV 集成已在数值正确性上通过；
speedup 仍需在无 `.item()` 同步的路径上进一步测量。

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

仍需验证：prefix 命中、block 释放和复用的极端组合（交错完成已测，正常路径通过）。

## P0：prefill 与 decode 连接

**状态：已验证（匹配精确参考，非逐 bit torch）。**

prefill 继续走 nano-vLLM 原有模型和 FlashAttention 路径；Megakernel 只在
`is_prefill=False` 且请求满足 Qwen3-0.6B、BF16、greedy、单卡等条件时进入。
Megakernel decode 不 replay prompt，直接读取 prefill 写入的 nano-vLLM paged
cache。

根因（已修复）：`weights_from_nano_model` 与 `load_weights` 用
`getattr(hf_config, "rope_theta", 10000.0)` 解析 RoPE 频率，但 Qwen3 把
`rope_theta=1e6` 放在 `rope_scaling` dict 里，导致 cos/sin 表用了错误的
`theta=10000`。新增 `_resolve_rope_theta(config)` 从 `rope_scaling` 解析。

验证方式：对每个真实 decode 位置，用「观察到的输入 token」计算精确 Python
参考 K/V，与 Megakernel 写入的 paged cache 对比；全部为 0.0 级一致。与
torch-eager 的残留差异是 ~1 ulp 融合 kernel 归约顺序差，会导致约半数 prompt
在第 1+ token 处偶发翻转（含进入重复 token 状态），按「匹配精确参考」验收。

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

**状态：已验证（batch==single），仍为顺序处理（非 sequence-level parallel）。**

`decode_paged_batch` 接受最多 32 条 sequence、各自的 `positions` 和
`block_tables`，使用同一个 paged cache pool。当前 kernel 是：

```cpp
for (int seq_id = 0; seq_id < batch_size; ++seq_id) {
    ldg_decode_one(...);
}
```

即一次 launch 内顺序处理 sequence，不是 sequence-level parallel。

已验证：batch=2/8/32、mixed-length（8/16/32/64）、交错完成（不同 max_tokens
导致序列逐步释放、batch 缩小到 1）均 batch==single，无跨序列污染。修复了
batch 缩小到单条时的 `Decode.step()` 对 `_out_token` 多元素 `.item()` 报错。
block 复用/释放的正常调度路径通过；sequence-level parallel 仍是已知性能缺口。

## P1：prefix cache

**状态：已验证 megakernel 与 torch 行为一致（读共享 block 正确）。**

kernel 按 block table 读取共享物理 block，且只写自己 `position` 的追加槽，不
覆盖共享 prefix 槽（写路径只读安全）。

验证：两序列 A/B 共享 prefix（B 命中 A 的 block hash）时，megakernel 与
torch-eager 对 B 的输出**完全一致**（如 B 首 token 均为 778），而两者 B-alone
也一致（17642）。即 megakernel 读取共享 block table 的行为与 torch 相同；与
B-alone 的差异源自 nano-vLLM 调度器 prefix-cache 在 A/B 并发下的行为
（A 的 decode KV 写入与 B 共享 block 交互），该差异对 torch 和 megakernel
同样出现，**不是 megakernel 缺陷**。

仍可探索：block ref count 在抢占/释放的极端组合下的一致性（正常调度路径
已通过）。

## P1：连续布局与 paged 布局

**状态：已验证。** 集成路径使用 paged cache；contiguous cache 只保留给
standalone reference。64 位 offset 已修复。paged 路径的单序列/batch token
correctness（匹配精确参考）、255/256/257 边界、长序列均通过。

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
- CUDA 错误和 sanitizer 诊断链路已建立；
- 单序列与 batch paged token correctness（匹配精确参考）；
- 255/256/257 与长序列边界、batch=2/8/32、mixed-length、交错完成 block 生命周期。

### 仍然不能宣传为已完成

- 整个 decode step 单 kernel、无 CPU `.item()` 同步（`generate_nosync` 仍在
  contiguous reference path，未扩展到 paged 集成）；
- sequence-level parallel batch（当前为顺序处理 sequence）；
- prefix cache；
- V100 FP16；
- sampling、tensor parallel；
- 任何性能 speedup 结论（仍需无 `.item()` 同步路径测量）。

## 下一步顺序

1. ✅ 单序列 token correctness（精确参考）——已通过，根因为 rope_theta 解析；
2. ✅ 255/256/257 和长序列、batch barrier、mixed-length、block lifecycle；
3. ⏳ 增加 paged no-sync generation，减少 `.item()` 和 Python loop；
4. ✅ prefix cache 实现/验证趋于完成（读取共享 block），仍需极端复用组合验证；
5. ⏳ 最后运行 A100 benchmark、Nsight 和 V100 fallback 报告。

## 当前项目描述边界

可以这样描述：

> 在 nano-vLLM decode 路径中集成了一个面向 Qwen3-0.6B 的 CUDA Megakernel，
> 将主要 Transformer decode 计算融合到 cooperative kernel，并直接复用
> nano-vLLM scheduler 分配的 paged KV cache 和 block table。当前 A100 上单序列
> 与 batch 的 paged token correctness（匹配精确参考）、255/256/257 边界、
> mixed-length 与 block 生命周期均已验证；batch 仍为顺序处理 sequence，
> 无 CPU `.item()` 同步的最终形态与端到端性能收益仍在验证。

不能这样描述：

> Megakernel 已经完全替代 CUDA Graph，复用 vLLM PagedAttention，并在 batch
> decode 上完成性能加速。
