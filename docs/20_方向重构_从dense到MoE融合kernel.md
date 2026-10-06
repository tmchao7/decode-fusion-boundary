# 方向重构：从 dense megakernel 到 MoE + 低比特融合 kernel

> **状态**：**方向决议**（2026-10-01 起）。这是方向文档，**不是测量报告**。
> 文内每个外部数字都标了证据等级（沿用 [16](16_MoE_A100可行性调研.md) 的标签体系）；
> 标 **[推测]** 的不得当作结论引用。本文不引入新的 `RUN_MODE`，也不承诺任何尚未实测的数字。

## 0. 一句话

`docs/14` 与一个外部独立实现**各自**证明了同一件事：**dense 小模型在 batch=1 的 device 侧
已经贴着权重墙，融合再深也赢不了**。因此本项目的 device 侧收益只可能来自两处：
**消除 host 边界（已做）** 与 **streaming fewer bytes**。后者把方向指向 **MoE + 低比特**。

---

## 1. 触发这次重构的证据

### 1.1 我们自己的（A100 / dense）—— [measured]

摘自 [14 融合粒度实验](14_融合粒度实验.md) §1：

```text
batch=1  megakernel E2E     2.241 ms/tok  vs cudagraph 2.944 ms/tok   -> 1.31x 胜
batch=1  megakernel device  1.89  ms/step vs cudagraph 0.74  ms/step  -> 慢 2.6x
cudagraph 0.74 ms/step = 1611 GB/s = 实测可达带宽(1699 GB/s) 的 95%
```

**结论一**：batch=1 的胜利全部来自消除 host/launch（~2.2 ms/step），不是算得更快。
**结论二**：cuBLAS + CUDA Graph 已经吃到实测带宽的 **95%** —— device 侧没有可赚的空间。

### 1.2 外部独立实现（AutoMegaKernel）—— [measured, 作者自报, 未经同行评审]

[AutoMegaKernel (AMK)](https://github.com/RightNow-AI/AutoMegaKernel)，
[arXiv:2606.09682](https://arxiv.org/abs/2606.09682)，开源 MIT。它把 HuggingFace Llama 系
模型编译成**一个 persistent cooperative kernel**，同一份源码重定向 `sm_75 → sm_120`。

它自己的表（比值 = cuBLAS / AMK，**>1 表示 AMK 更快**）：

| GPU | 类别 | int8 (W8A16) vs CUDA-graphed cuBLAS **bf16** | 结论 |
|---|---|---:|---|
| L4 (sm_89, 300 GB/s) | inference | 1.18× → 1.33× | ✅ 赢，随模型变大而增大 |
| L40S (sm_89, 864 GB/s) | inference | 1.25× → 1.27× | ✅ 赢 |
| A10G (sm_86, 600 GB/s) | inference | 1.04× → 1.08× | ✅ 大规模下赢 |
| **RTX 5090 (sm_120)** | **consumer** | **1.19× → 1.23×** | ✅ **赢** |
| A100 (sm_80, 1.4 TB/s) | **training** | 0.79× → 0.55× | ✗ 输，且随模型变大更输 |
| H100 (sm_90, 3.1 TB/s) | **training** | 0.72× → 0.60× | ✗ 输 |

**并且他们在同一份 README 里主动写了同精度对照：**

> 在**同精度 bf16** 路径上 AMK *输给* cuBLAS。bf16 kernel 跑 **~1.38 ms/token，
> 比 CUDA-graphed cuBLAS 慢 ~1.24×**；只跑到 **~451 GB/s = 规格的 51% /
> 实测 HBM 峰值的 63%**（cuBLAS 天花板 ~90%）。
> **所以 int8 的胜利来自 streaming fewer bytes，不是来自更快的 kernel。**
> **"We do not beat cuBLAS/vLLM at bf16 batch-1, and we say so."**

### 1.3 我们自己的（RTX 5090 / MoE）—— [measured, 本项目 LeanMoE-Quant]

在 5090 上对 `allenai/OLMoE-1B-7B-0924` fp16、batch=1、ctx=1024 的实测：

```text
roofline 理论下限      1.598 ms/token
实测 wall            100.10  ms/token        （慢 62.6 倍）
GPU 真正在算           10.49  ms （10.5%）
host / launch          89.62  ms （89.5%）
每 token kernel 数      8,277 个
其中 routing/packing    占 GPU 时间的 44.8%
```

根因：HF 的 `OlmoeSparseMoeBlock` 对**全部 64 个专家**做 Python 循环，每 token 只激活 8 个。
单步恰好 1,024 次 `Memcpy DtoH`，而 `64 experts × 16 layers = 1,024`（完全吻合）。

---

## 2. 核心推论

### 2.1 dense + batch=1：device 侧是死路

两处独立证据（1.1 的 95% roofline、1.2 的 bf16 慢 1.24×）指向同一结论。
**这不是"融合得还不够深"，是数学上没有再赚的余地。**

dense 上 megakernel 能做的只剩两件事，而且都已经或可以达成：
1. 消除 host 边界（~2.2 ms/step）——**已做**，这是现有 1.31× 的全部来源；
2. streaming fewer bytes ——**这正是 AMK 唯一赢的原因**。

### 2.2 MoE：唯一同时满足三个条件的战场

| 条件 | dense | MoE |
|---|---|---|
| 进不了 CUDA Graph | ✗ 能进 | ✅ **进不了**（动态路由 → 数据依赖形状） |
| device 侧远离 roofline | ✗ 已 95% | ✅ **GPU 只忙 10.5%** |
| 量化收益大 | 一般 | ✅ **experts 占 `[W]` 的 68.3%** |

三条**必须同时成立**才值得做。dense 只在第 1 条上就否了。

> **【推测】** MoE 进入 CUDA Graph 的阻塞点是 `torch.where(expert_mask[e])`：`nonzero`
> 的输出长度依赖数据，必须先把 count 拷回 host 才能分配输出张量。这是在 LeanMoE-Quant
> 侧静态分析得出的，**尚未在本仓库上实测**。若成立，则 MoE 的 decode 必须靠融合 kernel
> 把 routing + expert + 依赖全吃进一个 launch —— 不是"更快"，是**唯一选项**。

### 2.3 于是两个个人项目的关系不是重复，是依赖链

```text
megakernel 想赢  →  必须 streaming fewer bytes
                 →  必须先回答"减字节能赚多少、精度撑不撑得住"
                 →  那正是 LeanMoE-Quant 在测量的东西
                 →  且 MoE 是唯一战场（2.2）
```

| 项目 | 回答什么问题 | 手段 |
|---|---|---|
| **LeanMoE-Quant** | 天花板在哪？量化值不值得？ | 测量方法学、roofline、字节账 |
| **MegaKernel-vLLM** | 一个 kernel 能不能够到天花板？ | CUDA/Triton kernel、同步、occupancy |

**分工纪律**：不要两个项目都变成"我证明了 MoE 很吃 launch"。那是同一句话讲两遍。

---

## 3. 你描述的"细粒度 tile 通知-消费"已有成熟形式

两份可读实现，机制一致，**都不是 grid.sync**。

### 3.1 AutoMegaKernel 的 VM（计数器 DAG）—— [measured, 作者自报]

> 一次前向是一个 **DAG**。Producer 只 **increment 计数器**；consumer 只 **wait 在静态
> 已知的阈值**上；执行是带单调计数器的拓扑遍历：**没有锁，没有任意 signal**。
>
> VM **拒绝加载任何不是合法 DAG 的 schedule** —— 非法 schedule 变成干净的 `REJECTED`，
> 而不是挂死 GPU。

配套的**静态验证器**：7,160 个对抗性 schedule（其中 6,091 个不安全），**零误收**。

### 3.2 Cohere megakernel（arrive/wait 分离）—— [measured, 读本机源码]

`cohere-megakernel/src/decode/megakernel.cuh` 的角色划分：

```cpp
enum class NmcRole : int {
    PRODUCER = 0,   // WG0 warp 1 — TMA-loads, sem inits, cross-SM input bar waits
    STORER   = 1,   // WG0 warp 2 — TMA-stores + cross-SM output bar arrives
    CONSUMER = 2,   // WG1, WG2
    IDLE     = 3,
};
```

**cross-SM barrier 的 `wait` 与 `arrive` 是分开的** —— producer 可以跑在前面，这是
数据流而不是屏障。其 README 原话：**"no full-grid barriers between ops"**。

### 3.3 他们的度量工具（"SM 有没有吃满"）

Cohere 自带 `src/sm-profiler/`：

- 按 **SM id** 而不是 block index 记录事件 —— 匹配"一个 SM 一个 block"的 persistent kernel；
- 一次 profiler 调用 **~几十纳秒**（上游是几百），所以能放在单条 megakernel 指令周围
  而不扰动被测量对象；
- `sm_profiler_swap` 在单次时间戳读取里结束上一段并开始下一段，消除背靠背事件间的空隙。

**已知局限**（他们自己标的）：`__syncthreads` 之后**紧邻**的事件测不准，读数偏短 —— 这是
硬件计时器的性质，不是实现缺陷。

### 3.4 与外部实现的差距对照

| 项目 | 目标硬件 | 模型 | MoE | 状态 |
|---|---|---|---|---|
| AutoMegaKernel | sm_75→sm_120 | HF Llama 系（dense） | **roadmap placeholder** | 开源 |
| Cohere | H100 **sm_90a**（需 wgmma/TMA） | North Mini Code（含 MoE） | ✅ 有 | 开源 |
| MoK | **SM100/103**（NVL72） | — | ✅ 但**训练用** | 开源 |
| **本项目** | **RTX 5090 sm_120** | Qwen3-0.6B（dense） | ✗ | 主线 task-graph |

**空隙很清楚**：**sm_120 上的 MoE 推理融合 kernel，目前没有公开实现。**
AMK 的 MoE 是 roadmap；Cohere 有 MoE 但只跑 H100 且是商业 checkpoint；MoK 是训练且要
SM100/103。

---

## 4. 5090 独有的杠杆：MX block-scale

`sm_120` 是消费级里**唯一**有 MX block-scale（MXFP4 / NVFP8）硬件块缩放 MMA 的
（LLVM 门控：`SM120f`，家族变体）。

**AMK 用的是 int8（W8A16），没有用 FP4。** 而 5090 有硬件支持。

> **【推测】** 在同一个融合 kernel 里用 MXFP4 取代 int8，字节缩减比 W8A16 更激进，
> 且由硬件块缩放承担精度代价。**精度是否撑得住必须由 LeanMoE-Quant 的
> `EXP-p1-01`（W8/W4 精度门）回答，不能假定。**

这是本项目可以差异化、而 AMK 没做的点。

---

## 5. 预注册的假设与反证信号

> 按 [17](17_调度验证器.md) 的纪律：**先写下来，被否证是结果不是事故。**

### H1：MoE 的 decode 无法进入 CUDA Graph

- **推测**：阻塞点是 `torch.where` → `nonzero` 的数据依赖输出长度。
- **反证信号**：在 5090 上成功的 capture + 通过 KL/top-1 正确性门 → H1 被否证。
- **为什么重要**：若 H1 不成立，融合 kernel 就从"唯一选项"降级为"其中一个选项"，
  本方向的价值需要重新论证。

### H2：融合 kernel 在 MoE 上的收益显著大于 dense

- **推测**：dense 上 device 侧已无空间（§2.1），而 MoE 的 GPU 忙占比只有 10.5%，
  routing 占 GPU 时间的 44.8% —— 融合能同时消掉 launch 与 routing 浪费。
- **反证信号**：融合后 device 时间没有改善，或改善远小于 launch 部分 → H2 被否证。
- **量化预测**：现行 100.1 ms/token，roofline 1.598 ms。dense 那套已经把 host 消掉了，
  所以 MoE 上融合的现实靶子是 **2–4 ms/token**（**预测值，待实测**）。

### H3：MXFP4 在这个模型上精度可接受

- **推测**：5090 的硬件块缩放能承担 FP4 的精度代价。
- **反证信号**：KL / top-1 门不过。
- **依赖**：必须等 LeanMoE-Quant 的 `EXP-p1-01` 出数，**不得提前假定成立**。

---

## 6. 本轮明确**不**做的事

1. **不在 dense 上继续压 device 时间** —— §2.1 已证明无空间。
2. **不重做 AMK 已经做完的事** —— 它开源、更通用（同源重定向 sm_75→sm_120）、带静态验证器
   与自搜索循环。重做没有增量。
3. **不引入需要 SM90a / SM100 的特性** —— 5090 没有 `wgmma`（`SM90a` 变体）、没有
   `tcgen05`（`SM100a`）。但**有** thread-block cluster（base `SM90`，LLVM:
   `hasClusters() = hasFeature(SM90)`，而 sm_120 蕴含 sm_90）、TMA、FP8、MX block-scale。
4. **不把 [推测] 当结论** —— §5 的三条假设在实测前不得写进任何对外材料。

---

## 7. 可借鉴的具体设计（按优先级）

| # | 借什么 | 从哪 | 为什么现在就需要 |
|---|---|---|---|
| 1 | **计数器 DAG 取代 grid.sync** | AMK `vm/`；本仓 [11](11_task_graph重写设计.md) 已设计 | 现有 336 次大同步/step 是 batch>1 输掉 133× 的直接原因 |
| 2 | **静态 schedule 验证器** | AMK（本仓 [17](17_调度验证器.md) 已有雏形） | **这是快速迭代的前提** —— 非法 schedule 在启动前被拒，不会挂 GPU 烧机时 |
| 3 | **arrive / wait 分离的 cross-SM barrier** | Cohere `megakernel.cuh` | 让 producer 跑在前面，而不是等全局 |
| 4 | **按 SM id 的 per-SM profiler** | Cohere `src/sm-profiler/` | 度量"SM 是否吃满"的唯一手段，且开销低到可放进指令之间 |

---

## 8. 与 LeanMoE-Quant 的接口

LeanMoE-Quant 侧已产出、本项目可直接引用（**引用时须标注出处与卡片**）：

| 数字 | 值 | 卡片 |
|---|---|---|
| 实测可达带宽（只读） | **1559.2 GB/s** | RTX 5090, sm_120 |
| OLMoE `[W]` fp16 | 2,357,723,136 B | batch=1 |
| OLMoE `[T]` fp16 @ctx=1024 | 2,491,940,864 B | |
| `[W]` 分项 | experts 68.3% / attention 22.8% / lm_head 8.7% / router 0.2% | |
| roofline floor_W / floor_T | 1.5121 / 1.5982 ms | 用实测带宽 |
| 实测 per-token p50 | 100.10 ms | HF eager, ctx=1024 |
| 模型权重字节 | 13,838,721,960 B（已校验） | ModelScope 下载 |

**注意口径**：LeanMoE 的 `[W]` 是**每 token 读到的权重字节**（top-k 专家），与 checkpoint
全量 13.84 GB 不是一回事。混用会得出错误的 roofline。

---

## 9. 下一步（按顺序，一次一个变量）

1. **先复现 dense 的 5090 结果** —— 把 `docs/14` 的粒度实验在 5090 上跑一遍。
   只换机器一个变量。产出：A100 输 / 5090 赢 的两机对照。
   依据：AMK 报 5090 上 int8 赢 1.19–1.23×（§1.2），而我们的 dense 结果是 A100 上输。
2. **再验 H1** —— 在 5090 上尝试 MoE 的 CUDA Graph capture。失败本身就是结论。
3. **然后才动 MoE 融合 kernel** —— 此时才有明确的对手（HF eager / vLLM）与动机数字。
4. **最后做 MXFP4 臂** —— 且只在 LeanMoE 的精度门通过之后。

**不要**跳过 1 直接做 3：那会让"我的实现对不对"和"MoE 是不是更值得"两个问题搅在一起。

---

## 参考

- [AutoMegaKernel](https://github.com/RightNow-AI/AutoMegaKernel) / [arXiv:2606.09682](https://arxiv.org/abs/2606.09682) —— 开源，sm_75→sm_120，int8 在 5090 上赢 1.19–1.23×
- [Mixture-of-Kittens](https://github.com/cursor/mixture-of-kittens) / [arXiv:2609.36070](https://arxiv.org/abs/2609.36070) —— NVL72 训练 megakernel，需 SM100/103
- [Cohere megakernel](https://github.com/cohere-ai/megakernel) / [blog](https://cohere.com/blog/megakernels) —— H100 sm_90a，1.58× vLLM @BS=1
- [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/pdf/Blackwell_Tuning_Guide.pdf)
- 本机：`~/repos/02_personal/cohere-megakernel`（§3.2/§3.3 的源码依据）
