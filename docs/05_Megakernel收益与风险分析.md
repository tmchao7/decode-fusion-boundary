# Megakernel 相对 CUDA Graph 的收益来源、风险与验证框架

> 更新日期：2026-09-08
>
> 本文是**分析框架与假设**，不是实测结论。所有收益/风险条目都标注了验证方法；
> 当前实测状态以 [docs/04](04_Megakernel集成问题清单.md) 为准：
> **token correctness 已通过，但 A/B benchmark 显示 decode 比 torch-cudagraph
> 慢 ~13-15x**（decode_tok_s ≈ 0.07x，tpot_ms ≈ -1300%）。
> 在第 7 节前置条件落地之前，不应基于本文做任何“megakernel 更快”的承诺。

## 1. 对照层级

- vs **torch eager**（`enforce_eager=True`）：0.6B decode 在 eager 下是
  host-bound（Python 开销 + 逐算子 launch）。cudagraph 与 megakernel 都会
  超过它，**eager 不是主对照，只是下界参照**。
- vs **torch-cudagraph**：主对照。graph replay 已经把 host launch 开销拿掉，
  剩下的差距只能来自：kernel 间 GPU 侧开销、访存流水线质量、host 同步路径。

## 2. 当前实测差距（必须最先摆出来）

当前 13-15x 的差距**是路径差距，不是 kernel 差距**：

```text
megakernel 路径（现状）            cudagraph 路径
────────────────────────          ─────────────────────
CPU 驱动循环                      一次 graph.replay()
每 token .item() 主机同步         无逐 token 同步
batch 内逐 sequence 串行          单 token 直接 replay
```

因此当前数字回答的是“CPU 驱动调度 vs graph replay”，
**还没有回答“megakernel kernel 的 GPU 时间 vs torch 各 kernel GPU 时间总和”**。
后者才是第 4-6 节分析的对象，需要先做第 7 节前置条件才测得出来。

## 3. 共同天花板：权重带宽

0.6B BF16 权重 ≈ 1.2GB，batch=1 的 decode 每 token 都要从 HBM 完整读一遍，
**这个地板两条路线都要付**，理论上限相同。真正的问题是“离上限多近”：

- megakernel 的收益空间 = 把非权重开销（kernel 间隙、激活往返、host 同步）
  压到最小；
- 一旦权重带宽成为绝对瓶颈，继续调 kernel 的收益趋近于 0；
- 推论：**batch 越大权重摊薄，天花板收益越小；甜点区在小 batch（1-8）**。

## 4. 收益来源（假设，待 ncu 验证）

1. **消除 inter-kernel 间隙**。graph replay 下每个 kernel 仍有 GPU 侧固定
   launch 延迟 + SM drain 尾巴；28 层 × 每层 ~6-7 个 kernel ≈ 每 token 170+ 次。
   persistent megakernel 一次 launch 全部消掉。batch 越小占比越高。
2. **权重流式流水线**。带宽受限阶段的目标是把 HBM bus 打满；独立小 kernel
   各有 ramp-up/ramp-down，persistent kernel 可跨层 pipeline load 让 bus
   持续饱和。
3. **激活 HBM 往返**。逐 kernel 是写回 HBM 再读回；megakernel 留在
   寄存器/SMEM 可省。但 0.6B 的中间激活（1024/3072 维）相对 1.2GB 权重
   流量占比小，此项收益真实但量级有限。

## 5. 架构优势（比 µs 级收益更稳的卖点）

- **形状无关**：cudagraph 需按固定 batch/seq 桶 capture，running set 变化要
  pad 或重 capture；megakernel 直接吃 block table 指针 + 可变 context_lens，
  无 re-capture、无 padding 浪费。
- **天然接 prefix cache / mixed-length**：paged block table 就是输入，与
  nano-vLLM scheduler 无缝。“不过多计算”是公平对照的前提，本身不是
  加速来源（graph 做同样 forward 也不会多算）。
- **单 launch 简单性**：无 graph capture 生命周期管理。

## 6. 风险（可能反而更慢）

1. **融合 GEMM 打不过 cuBLAS 带宽**。0.6B 的 GEMM 全是 M=1、带宽受限，
   手写融合版未必追得上厂商调优 matmul 的 bus 利用率。单点最大风险，
   必须用 ncu 的 memory throughput % 判定，不能靠总时间猜。
2. **寄存器压力 / occupancy**。persistent kernel 要装下最大中间量，
   spill 到 local memory 会灾难性变慢；occupancy 低则 latency hiding 变差。
3. **“并发”是 persistent kernel 的弱点而非优点**。cooperative launch 占满
   所有 SM，很难与其他 kernel 真并发（采样、另一 stream 的 prefill 都会
   抢 SM）；多 stream overlap 恰是独立小 kernel + graph 的强项。
   - 若“并发”指“一次 launch 内多 sequence”：是 megakernel 的地盘，但当前
     实现是串行，需要 sequence 级并行（见第 7 节）；
   - 若指“与别的 GPU 工作同时跑”：graph 路线反而更强。

## 7. 前置条件（不落地，kernel 再快也被同步吃掉）

性能对比有意义之前，必须先（顺序见 docs/04）：

1. **去 `.item()`**：device 侧调度或异步回传，消除每 token 主机同步；
2. **sequence 级并行 batch**：当前一次 launch 内逐 sequence 串行，batch
   收益没有兑现；
3. （可选）LM head/argmax 融合进主 kernel 或独立异步 kernel。

## 8. 验证方案（correctness 已通过，可以开始）

目标：把第 4-6 节从假设变成结论。

| 实验 | 数据 | 判定 |
|---|---|---|
| A/B TPOT，batch=1/4/8，prompt 32/128/512 | megakernel vs cudagraph vs eager，decode_tok_s | 前置条件落地后预期甜点区快 10-40%；**不承诺 2-3x** |
| ncu kernel 时间分解 | megakernel 单 kernel 时间 vs cudagraph 各 kernel 时间总和 | 若 megakernel kernel 自身仍 > torch 总和，瓶颈在 kernel 质量（6.1/6.2），不在 launch |
| ncu memory throughput % | 两条路线的 DRAM 读带宽利用率 | 接近 A100 峰值（>80%）则权重带宽已是瓶颈，继续调 kernel 收益趋零；差距大则还有空间 |
| host 时间分解 | python 侧 step() 耗时占比 | 确认 `.item()` 移除后 host 不再是瓶颈 |

**判定门槛（建议随 benchmark PR 一起记录）**：

- megakernel TPOT ≤ cudagraph TPOT → 收益成立，可叠加第 5 节架构优势叙事；
- 慢但在 1.3x 以内 → 性能收益不成立，但架构优势（第 5 节）独立成立；
- 慢 >1.3x 且 ncu 显示 kernel 质量差距 → 回到 kernel 调优，不扩大模型范围。

## 9. 当前判断

- 甜点区（小 batch + 前置条件落地）预期 **10-40%**；上限被权重带宽钉死，
  收益来自“离顶多近 + 形状无关”，不是绝对算力。
- 最稳的对外叙事：**单 launch、形状无关、天然吃 prefix cache / mixed-length**，
  小模型小 batch 甜点区叠加实测 µs 级收益——而不是“比 graph 快 X 倍”。
- 当前 13-15x 是路径差距不是 kernel 差距；第 7 节落地前，任何
  “megakernel 更快”的说法都不成立。
