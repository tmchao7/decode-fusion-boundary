# 归档说明（2026-10-06）

**本仓库已归档，不再作为简历主线项目。** 代码、文档与实测记录保留，作为方法资产与证据留存。

## 1. 这个项目做了什么

在单张 RTX 5090（sm_120）上研究：**decode 阶段把整个前向融进一个持久化 kernel（megakernel），相对 CUDA Graph 的收益边界在哪。**

建立了：instruction-stream ABI + cooperative 持久化 kernel、bar-DAG 调度器、对抗性调度验证器（误接受 0）、roofline/可达带宽标定、accounting 与配对的正确性门。

## 2. 得出的边界结论（本质检结果）

**dense 融合在 5090 上不成立，原因是机制消失了，不是实现不够好：**

1. **host 残差已归零**：`wall 624.1 / host 603.4 / device 601.7 ms`（128 token）→ 差 0.3%。
   megakernel 当初的立论是"少一次 host 边界"，而 **CUDA Graph 已经把这个边界消掉了**。
2. **device 侧离墙 5.81×**：ctx=1（完全无 attention）时 4.180 ms vs 权重地板 0.719 ms，仅 **17.2%** 可达带宽。
3. **瓶颈是同步链，不是带宽**：SM 从 16→170，每 SM 的活减少 10.5×，总时间只降 1.5×，而**单条指令耗时反涨 7×**（22→154.9 µs）。单 SM 独跑 16 GB/s，而峰值只需每 SM 9.75 GB/s —— **单卡能力够，合起来反而慢**。可并发指令仅 ≈55（vs 170 SM），与 64 SM 的饱和拐点一致。

即：**排名第一的瓶颈是"依赖/同步链"，不是 HBM，也不是 launch。** 这个结论在 MoE 上同样成立（见 §4）。

## 3. 为什么不继续

- dense 的收益来源（host 边界）已被 CUDA Graph 消除，device 侧又贴墙 → **没有可赚的东西**
- 大 batch 下 megakernel 的经济性反向：kernel 变大、launch 摊薄、grouped GEMM 已足够高效
- 项目本身的结论是**负面的，且已被完整记录**——这是有价值的结局，不是失败

## 4. 可迁移资产（本项目真正值得留下的部分）

| 资产 | 内容 | 对新项目的用处 |
|---|---|---|
| **测量方法论** | 分母只用**本机实测**可达带宽；它是**尺寸敏感的**（0.3 GB→1477、1.2 GB→1600、4 GB→1683 GB/s，拟合截距 −3.9 µs 即无固定开销） | 任何 roofline/达成率 |
| **profiling 陷阱** | **NCU 不能 profile cooperative kernel**，其 kernel-replay 与 `cudaLaunchCooperativeKernel` 冲突，实测会 hang 掉 GPU driver（连 `nvidia-smi` 都被阻塞），改用 nsys | 直接对应"NYS/NVU 硬要求" |
| **拆解顺序** | 先 nsys 定位"时间花在哪"，再 ncu 问"这个 kernel 为什么慢" | 通用 |
| **同步链诊断法** | 扫 SM 数并观察**单条指令耗时是否随 SM 数上升**——上升即时间花在等待而非计算 | 任何 persistent/调度系统 |
| **sm_120 实测** | 无 wgmma（ptxas 明确拒绝）、无 tcgen05/TMA；原生路径是 `mma.sync` + block-scaled MX；cluster/DSMEM 可用 | 消费级 Blackwell |
| **叙事纪律** | 把结论写成**划定适用域**（"收益随 batch 如何变化"），而不是"我打不过 cudagraph" | 面试表达 |

## 5. MoE 侧已实测的对外基线（OLMoE-1B-7B-0924，RTX 5090，b=1，ctx=1024，fp16）

由 [MoE-Flow] 在**本机实测**，两次独立复测相差 0.13%：

| 指标 | 值 |
|---|---|
| per-token p50 | **5.2701 ms**（复测 5.2637） |
| roofline 地板 | 1.5955 ms |
| **slack** | **3.3×** |
| **带宽达成率** | **30.3%** |
| **host 占比** | **0.5%** |
| kernels/step | **1073** |
| 最大单项 | **gather 30.78%**（1.6137 ms，32 次 × 50.4 µs） |
| 其次 | attention 23.23%、gemvx 12.09%、cutlass wmma 9.71% |

**判读**：CUDA Graph 已经消掉 launch（host 0.5%），但达成率仍只有 30.3%；那 30.8% 的 gather 是"为进图而引入的真实拷贝"，可由**设备侧指针解析**（grouped GEMM）消除。这条留给后续项目作为起点基线。

## 6. 本仓库不再更新的内容

`docs/21`（5090 dense 复现）与 `docs/22`（小 batch MoE 架构）保留为**设计/测量记录**，其中 `docs/22` 的 MoE opcode 契约与不变量未被实现，**不得引用为"已完成"**。

---
*归档决定由仓库作者作出：转向多卡推理系统方向。*
