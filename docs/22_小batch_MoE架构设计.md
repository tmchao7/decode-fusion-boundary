# 小 batch MoE：megakernel 与推理系统的整体架构（设计决议）

> **状态**：现行 —— **设计决议**，**未实测**。这是方向文档，不是测量报告。
> 本文每个数字要么有出处、要么写出算式、要么标 **[推测]** / **[未测]**。
> **未实测的数字不得当结论引用**，也不得写进任何对外材料。

> 依赖关系：[20 方向重构](20_方向重构_从dense到MoE融合kernel.md) §9 第 1 步（5090 dense 复现，
> 见 [21](21_5090复现_融合粒度实验.md)）→ **本页（第 3 步的架构前置）** → MoE kernel 实现。
> 量化侧（W8/W4/MXFP4 精度门）归 LeanMoE-Quant（`EXP-p1-01`），本文**不假定**其结论。

---

## 0. 先算三个数：它们决定整个架构

`src/nanovllm/moe_budget.py` 是本节算式的**唯一可执行来源**，
`tests/test_moe_budget.py` 是它的锁。输入是 OLMoE-1B-7B-0924，**已用真实 checkpoint 逐张量核对**：

```text
safetensors 头里 3,219 个张量，parameters.BF16 = 6,919,161,856
MoEDims.total_params()                     = 6,919,161,856   ← 完全相等
```

核对过程纠正了四处记录（详见 `moe_budget.py` 的模块说明与
`~/repos/02_personal/LeanMoE-Quant/docs/MODELS.md` §3.1）：

| # | 事实 | 为什么重要 |
|---|---|---|
| 1 | OLMoE 有 **per-head QK-norm**（`self_attn.q_norm`/`k_norm`，每层各 2048 参数） | 每层多读 2 个张量、且位于投影与 attention 之间的关键路径上 |
| 2 | `num_attention_heads == num_key_value_heads == 16` → **MHA 不是 GQA** | KV 宽度 = hidden，KV 读是 GQA(16/8) 的 2 倍 |
| 3 | `norm_topk_prob = false` | top-k 权重**不做**归一化，combine 直接吃 softmax 原值 |
| 4 | `config.json` 里**没有** `num_expert_groups` | OLMoE 是普通 top-k，**不是**分组路由 |

### 0.1 每-token 字节预算（bf16，batch=1）

| 项 | 字节/token | 占 `[W]` |
|---|---:|---:|
| attention（q/k/v/o 各 2048×2048 ×16 层）+ 层内 norm | 537,133,056 | 22.78% |
| router（64×2048 ×16 层） | 4,194,304 | 0.18% |
| **top-8 专家**（8×16 个专家各 3×2048×1024） | **1,610,612,736** | **68.30%** |
| lm_head（untied，50304×2048） | 206,045,184 | 8.74% |
| **`[W]` 合计** | **2,357,985,280** | 100% |
| KV（ctx=1024，MHA） | 134,217,728 | 5.4% of `[T]` |
| **`[T]` 合计** | **2,492,203,008** | |

### 0.2 三个结论

```text
floor_W = 1.512 ms/token      2.358 GB ÷ 1559.2 GB/s（本机实测带宽）
floor_T = 1.598 ms/token      ctx=1024
ctx=4096 时 floor_T = 1.857 ms（KV 增至 537 MB，占 [T] 18.5%）

算力：2.49 GFLOP/token ÷ 209.5 TFLOP/s = 0.0119 ms = 地板的 0.74%
      同一份 FLOP 若落在 CUDA core（~8 TFLOP/s FP32）= 0.31 ms，仍低于地板
```

**结论 1 —— b=1 是纯流式问题，不是 GEMM 问题。** 算力占地板不到 1%，
所以架构建模的单位不是 tile 而是**"字节被读了几遍"**。tensor core 是必需的
（否则 0.31 ms 的 CUDA-core 路径会顶到地板），但它永远不是瓶颈。

**结论 2 —— b=1 不需要 gather，也不需要 token-major 重排。** 每个被激活的专家
只服务 1 个 token，所以"权重主序、token 落在 N 维"是**天然形态**，不是需要构造的东西。
对照 LeanMoE-Quant `MOE_CUDAGRAPH.md` §3.3：那条"固定形状 + CUDA Graph"路线为了
进图要付 `gate_up[ids]` 的**真实 gather 拷贝**，约 **+2.07 ms/token**（本模块算得
`active_weight_floor_ms = 3.664 ms` vs `floor_T = 1.598 ms`）。**megakernel 把整个
step 放进一个 launch 后就不需要这笔拷贝**——这就是本仓相对那条路线的真实增量。

**结论 3 —— 真正会杀死实现的是 MLP 不足，不是 tiling。** 按 Clark 定律，
达到 `B` 的持续带宽需要 `in-flight ≈ B × latency`：

```text
每 SM 目标带宽 = 1559.2 GB/s ÷ 170 SM = 9.17 GB/s
所需 in-flight/SM = 9.17 GB/s × 5.5 µs ≈ 50 KB     (HBM 延迟取 5–6 µs)

现状（taskgraph_kernel.cu）：每个 K 步每 warp 一个 16×16 bf16 tile = 512 B，
7 个在途 stage group ⇒ 3.5 KB/warp，×7 worker warp ⇒ 约 24.5 KB/SM
⇒ 约 50% 的目标在途量
```

**[推测]** 这是当前 kernel 只跑到 10.8 ms/step（A100，`docs/12`）而非 ~1 ms 的主因之一；
**必须上机测**（§4 P0.2）。注意这条与"融合粒度"是**不同**的旋钮：粒度改的是
launch/barrier 边界，MLP 改的是同一个 kernel 内部的在途字节量。

---

## 1. 四层架构

```text
L0 控制面   Python engine：prefill 走 torch/FA，decode 每步一次 cooperative launch
             └─ 不变量 D1「一步一 launch」  不变量 D2「decode 内零 .item()」
L1 执行引擎 持久 task graph：grid=#SM、每 SM 一条预编指令流、计数器依赖（非 grid.sync）
             └─ 不变量 E1「指令流与数据无关」 E2「指针表 + 运行期基址」 E3「16 字 ABI」
L2 数据面   MoE 权重布局、4 个新 opcode、bar 维度、combine 语义
L3 分工     本仓只管「设备侧实现能不能够到天花板」；量化与精度门归 LeanMoE-Quant
```

### L0 控制面 —— 已有，只做收敛

- **D1「一步一 launch」**：dense 上 1.31× 的胜利**全部**来自这里（[14](14_融合粒度实验.md) §1）。
  MoE 同理，且更极端：当前 HF eager 每 token **8,277 个 kernel + 1,024 次 `Memcpy DtoH`**（LeanMoE
  `EXP-p0-02`）。**强制点**：`TaskGraphDecoder.decode_rows()` 只发 `g` 次 `tg_decode_step`。
- **D2「decode 内零 host 同步」**：唯一允许的 D2H 是每步一次 token-id 回读
  （`taskgraph.py:441`）。**任何新增 op 若需要 host 知道路由结果，就是架构违规。**
- prefill 继续走 torch/FlashAttention；采样保持 greedy、device 侧 argmax。

### L1 执行引擎 —— 复用，扩展而非重写

现状（`docs/11`）：每 SM 一个常驻 CTA，host 为每个 SM 预编一条
`inst_buf`（每条指令 16×`int32`），依赖用全局 `uint32` 原子计数器，
生产者 `__threadfence + atomicAdd`、消费者 `volatile` 轮询。**MoE 不改变这一层**，
只是往里加 opcode 和 bar 维度——前提是下面三条不变量成立：

| 不变量 | 内容 | 现状强制点 |
|---|---|---|
| **E1 指令流与数据无关** | 每个 CTA 的指令条数、依赖边、bar 槽位只由 `(BS, ctx, tiling)` 决定，**不由路由结果决定** | `schedule.build_schedule(geo, num_sms, …)`；`_TILE_SLOTS_PER_INST` 的连续 tile 打包 |
| **E2 指针表 + 运行期基址** | 指令只写 `layer`，内核用权重指针数组取地址；MoE 需要的"多一层间接"加在这个数组上 | `weights.py:_pack_layer_weights`（每层 11 个指针） |
| **E3 16 字 ABI** | `TG_INST_WIDTH = 16` 是**测试强制的头文件契约** | `csrc/tg_abi.h` + `tests/test_taskgraph_abi.py` |

> **E1 为什么在 MoE 上仍然成立（关键论证）**：指令流里**不携带专家 id**。
> 指令说"处理专家槽 `s` 的第 `[n0, n0+count)` 个 N tile"；专家 id 在设备侧从
> `expert_ids[s]` 读出，再用它索引 `expert_weight_ptrs[layer][expert][0/1/2]` 取基址。
> CTA 内所有线程取到**同一个地址**，所以既没有 warp 分歧，也不需要 host 参与。
> **路由是数据，不是形状。**

### L2 数据面 —— 本次设计的主体

#### L2.1 权重布局（装权重时一次性做，图外）

```text
gate_up_experts[layer][expert] : [2048, 2048] bf16   (gate ‖ up)
down_experts[layer][expert]    : [1024, 2048] bf16
router_gate[layer]             : [64, 2048]   bf16   (256 KB/层，每步必读)
expert_weight_ptrs[layer][expert][3] : int64        (16×64×3×8 B = 24 KB)
```

**为什么 `torch.stack` 成连续张量在 b=1 是白拿的优化**：LeanMoE 的 gather 之所以贵，
是因为 `gate_up[ids]` 要做**拷贝**；把指针堆成表之后，内核只是**换一个基址**去流同样的
12.58 MB/专家。指针表 24 KB，L2 常驻。**这一步把"gather 拷贝"变成"多一层指针间接"，
省下结论 2 里的 ~2.07 ms/token。**

#### L2.2 新增 bar 维度（加在 `TgBarDims` 上，不改老字段）

| bar | 长度 | 生产 → 消费 |
|---|---|---|
| `bar_router` | `L` | ROUTER → EXPERT_GATEUP（可读 `expert_ids`） |
| `bar_silu` | `L × slots` | gate/up tile 的 SiLU 结果 → 该槽的 EXPERT_DOWN |
| `bar_expert_out` | `L × BS × top_k` | 某槽 down 结果 → 该 token 的 COMBINE |
| `bar_moe_layer` | `L + 1` | 一层 MoE 完成 → 下一层（与现有 `bar_layer` 同构） |

#### L2.3 串行层级与并行度（b=1，`plan_work()` 输出）

```text
每层串行层级 = ATTENTION → ROUTER → EXPERT_GATEUP → EXPERT_DOWN → COMBINE = 5
串行层级总数 = 16×5 + LM_HEAD + LM_ARGMAX = 82
每层的 K 步   = max(hidden, expert_ffn) / 16 = 128
```

对照 [18](18_成本模型.md)：当前 dense taskgraph 是 170 波 / 11292 K 步 = **66 K 步/波**；
MoE 是 82 层 / 128 K 步 = **0.64 K 步/层**（因为工作被摊到 170 个 SM 上，每层的 K 循环很短）。
**串行深度不是 MoE 的问题**——dense 上 `g` 曲线平坦已经证明 barrier 在被带宽主导的区间里
很便宜（[18](18_成本模型.md) §4.1）。

**并行度够不够**（这是 MoE 比 dense 更难的地方：工作单元更碎）

| 工作单元 | 数量 | 单位 |
|---|---:|---|
| QKV 输出列 tile | 384 | 16 列 |
| O_PROJ | 128 | |
| router | 4 | |
| **EXPERT_GATEUP** | **1024 / 层**（8 专家 × 2×1024/16） | 4 KB 权重/tile |
| **EXPERT_DOWN** | **512 / 层**（8 专家 × 1024/16） | |
| LM_HEAD | 3144 | |
| **全步合计** | **36,249** | |

对 170 个 SM：专家 gate-up 波每 SM **8 条指令**、down 波 **8 条**、lm_head 波 **3 条**，
其余各 1 条。**每条指令要流 4 KB×打包数 的权重；8 条指令/波意味着 8 次独立的
k 循环** —— 这正是 MLP 的用武之地，也是 §4 P0.2 必须先测的原因。

#### L2.4 combine 语义（直接决定 correctness 门）

`norm_topk_prob = false`，所以：

```text
x_ffn[b] = Σ_{j=1..8} routing_w[b][j] · expert_out[slot(b,j)]
```

**不做归一化**：直接乘 softmax 原值。这一条如果按"常规 MoE 先归一化"实现，
首 token 门就会挂——把它写进这里，避免实现时凭印象。

### L3 与 LeanMoE-Quant 的分工（不许两个项目讲同一句话）

| 问题 | 归谁 |
|---|---|
| 减字节能赚多少、精度撑不撑得住（W8/W4、MXFP4） | **LeanMoE-Quant**（本仓只搬结论） |
| 一个 kernel 能不能够到那个天花板 | **本仓** |
| 路由开销 / 专家粒度 / 动态负载 | 本仓只碰**设备侧实现**；LeanMoE 只碰**量化敏感度** |

---

## 2. 明确**不**做的事（每条附证据）

1. **不做 token-major grouped GEMM + gather**。那是为"多个 token 命中同一专家"摊销构造开销的；
   b=1 时每个专家只 1 个 token，构造开销 100% 是浪费（§0 结论 2）。MonoMoE 的
   *诊断*（token-major 在 decode 下错）与*解法*（N 维放 token tile、CTA 按专家权重 tile 切）
   与本文一致——但那个"构造开销"在 b=1 已经不存在，我们不需要它的 FP8 部分。
2. **不做动态 work-stealing / 设备侧队列**（Cohere 的 drain）。b=1 时 `b*topk = 8` 个槽、
   每槽恰好 1 个 token，**负载天然均衡**（`expert_token_load(1)` 断言 min == max == 1）；
   动态认领只增加原子流量。**留到 b>1**（§5 阶段 D）。
3. **不引入需要 sm_90a / sm_100 的写法**（wgmma / tcgen05）。sm_120 **有** TMA、
   thread-block cluster（base SM90）、FP8、MX block-scale，但这些只在阶段 D 的量化臂才动。
4. **不重测量化精度**（归 LeanMoE）。
5. **不用规格带宽**（1792）**也不用 A100 的 1699**：一切 roofline 用本机实测
   （引用值 1559.2，上机必须 `RUN_MODE=bandwidth` 复核；本仓所有 accounting 工具缺
   `results/bandwidth.json` 就报错退出）。
6. **不把 [推测] 当结论**：§5 的 H2 预测值（2–4 ms/token，来自 `docs/20` §5）标 **[推测]**。

---

## 3. 新增 opcode 契约（ABI 冻结）

在 `TgOp` 尾部追加，**不改动已有 0–9 的编号**：

```text
ROUTER         = 10   # layer
                      # 读 hidden → router_gate 投影 → softmax → top-8（不归一化）
                      # 写 expert_ids[slot], routing_w[slot]
EXPERT_GATEUP  = 11   # layer, slot, tile_n(range), k_tiles, wait/produce
                      # base = expert_weight_ptrs[layer][expert_ids[slot]][0 or 1]
                      # 输出写 gateup_out[slot]，并原地 SiLU + 乘 routing 前不归一化
EXPERT_DOWN    = 12   # base = ...[2]；消费 bar_silu → 写 expert_out[slot]
EXPERT_COMBINE = 13   # 每 token 读 top_k 个 expert_out，乘 routing_w 求和 → x_ffn
```

字段落在 16 字内（沿用现有 `_inst` 布局，**`inst[11..15]` 保持空闲**）：

```text
[0]=opcode [1]=layer [2]=tile_m [3]=tile_n [4]=slot/split [5]=k_tiles
[6]=wait_idx [7]=wait_target [8]=produce_idx [9]=num_tiles [10]=packed(tile_m,tile_n)
```

**边界**：`num_tiles` 的打包上限是 `_TILE_SLOTS_PER_INST = 7`（指令流编码的上限），
但**内核的 shared memory 预算更紧**——现有 `tg_s_b[7][8][16*16]` 是每 warp 4 KB ×8 stages；
MoE 的 tile 比 dense 大，实际可打包数必须在内核侧重新推导并写成断言，**不得照抄 7**
（见 §5 阶段 A1 的门）。

---

## 4. 实现路径：先 b=1，再扩 b>1

### 阶段 P0 —— 上机前必须先清掉"会不会白干"

| # | 动作 | 判决门 |
|---|---|---|
| **P0.1** | 把 [19](19_上机清单.md) §3c.3 的 **grid>1 崩溃** 做成完整矩阵（`sms=1 → 4 → 170`，先 `TG_NUM_SMS=1`） | 170 CTA cooperative launch 能稳定跑 ≥100 步 → 持久化路线可行；**否则本方向必须改为"少量 CTA + 多 kernel"**，本页 §1 的 L1 需重写 |
| **P0.2** | 带宽探针加臂：1 CTA/SM（现状）+ 2 CTA/SM + 每 CTA ≥8 stages，量"每 SM 要多少 in-flight bytes 才到 90% 可达带宽" | 得出这个数字；它直接决定 kernel 的 stage / warp / CTA 配置 |
| **P0.3** | 纯 CPU：`moe_budget.py` + `tests/test_moe_budget.py` | ✅ **本次已完成**（48 项） |

> P0.1/P0.2 **需要 GPU**。本次未做，已登记进 [19](19_上机清单.md) 的执行顺序。

### 阶段 A —— b=1 的 MoE kernel（主目标 2–4 ms/token，地板 1.598 ms）

| # | 交付 | 门 |
|---|---|---|
| A1 | `tg_abi.h` 加 4 opcode；`schedule.py` 加 4 个 builder + 4 个 bar + `build_moe_layer_waves`；内核侧重推 shared memory 打包上限 | 头文件对拍绿；`verify_schedule` 在 `g∈{1,2,4,7,14,28}`×`sms∈{1,4,108,170}` 全 PASS、误接受 0 |
| A2 | `router_reference.py`（torch，纯 CPU 可测）：softmax → top-8 → **不归一化** | 与 HF `OlmoeSparseMoeBlock` 逐元素对拍，含"全部 token 挤到同一专家"的极端路由 |
| A3 | 4 个 op body；先 `--dry-run` 验 schedule 合法，再实装 | 扩展编译过；`TG_NUM_SMS=1` 能跑完不崩 |
| A4 | 端到端正确性（首 token 对 HF greedy） | PASS 或 CASCADE；`DEVICE_ERROR` 是硬失败 |
| A5 | 取数：`runs=5` + CoV + bootstrap CI + roofline 距离列 | 记录"离 1.598 ms 多远"；**负面结果照报** |

### 阶段 D —— 扩到 b>1（按你要的"扩展着做"）

| # | 内容 | 触发条件 |
|---|---|---|
| D1 | **静态多槽**：`batch × top_k` 个静态槽，负载仍接近均衡 | b≤2–4，`expert_token_load` 仍可用 |
| D2 | **动态认领**：一旦某专家分到的 token 变多，runtime 由 `max_e` 主导 | `docs/19` §4 的"BS 1→8 device 显著变化"判据触发 |

---

## 5. 预注册（取数之前写下）

| # | 假设 | 反证信号 | 记录方式 |
|---|---|---|---|
| **H-MoE-1** | b=1 字节账完整：`[W]=2.358 GB`、专家 68.3%、**gather 可完全省掉** | 实测 device ms/token 显著高于 `[T]/实测带宽`，且 nsys 显示额外大块 memcpy | 修正 `[W]` 口径 |
| **H-MoE-2** | 融合在 MoE 上的收益**远大于** dense（dense 已贴 95% 墙，MoE 只忙 10.5%） | 融合后 device 时间没改善，或改善全在 launch 侧 | [20](20_方向重构_从dense到MoE融合kernel.md) §5 的 H2 被否证 |
| **H-MoE-3** | 串行波深（82 层）不是瓶颈 | 实测层间 drain 占总时间 > 20% | 需要 arrive/wait 分离的 cross-SM barrier |
| **H-MoE-4** | 170 CTA persistent grid 在 sm_120 上可用 | P0.1 崩溃 | 架构改为多 kernel + 少量 CTA，本页 §1 重写 |
| **量化门** | W8/W4 使 `[W]` 降到 1.18/0.59 GB → floor 降到 ~0.76/0.38 ms | — | **依赖 LeanMoE `EXP-p1-01`，不假定** |

**可复现的判读**：`python -m nanovllm.moe_budget` 的输出（或 `render_report()`）给出
地板与 tile 数；实测 per-token 时间除以地板就是"离墙多远"，必须与正确性结论同时出现。

---

## 6. 风险（按会杀死项目的程度排序）

| 风险 | 后果 | 缓解 |
|---|---|---|
| **P0.1：170 CTA cooperative grid 不可用** | 持久化 task graph 的整个前提没了 | 先做 P0.1；有明确 fallback（多 kernel + 少量 CTA） |
| **MLP 不足** | 跑到地板的 3–5 倍，看起来像"融合没用" | P0.2 先定配置；必要时 2 CTA/SM 或加 stage |
| 专家 tile 太小、指令太碎 | 每条指令流不到足够字节，barrier 占比上升 | §2.3 的并行度表已给出每 SM 指令数；A1 时用真实 smem 预算重推 |
| 权重 bf16↔fp16 落差 | 精度门误判 | 内核走 bf16 流式；落差必须记录（不是"顺手转 fp16"） |
| `[W]` 与实测不符 | 所有"离墙多远"都错 | H-MoE-1 的修正路径；不得当 kernel 快 |

---

## 7. 验证与验收（本轮已完成的离线部分）

```bash
PYTHONPATH=src python -m unittest tests.test_moe_budget      # 48 项
python -c "from nanovllm.moe_budget import *; print(render_report(decode_budget(OLMOE_1B_7B, bandwidth_gbps=1559.2), plan_work()))"
```

**已锁定的量**（容差见测试）：`[W]=2,357,985,280`；专家占 68.3%；
`floor_W=1.512 ms`、`floor_T(ctx=1024)=1.598 ms`、`floor_T(ctx=4096)=1.857 ms`；
gather 路径地板 3.664 ms（省下 2.066 ms）；算力占地板 0.74%；
`total_params=6,919,161,856` 与 safetensors 完全相等；b=1 槽负载 min==max==1；
36,249 tiles/step、82 串行层级、专家波每 SM 8 条指令。

**未做（需 GPU）**：P0.1、P0.2、阶段 A 全部、阶段 D 全部。

---

## 8. [未测] / [推测] 清单（不得清空前不得对外）

```text
[未测] P0.1 170 CTA cooperative launch 是否稳定（H-MoE-4）
[未测] P0.2 达到 90% 可达带宽所需的 in-flight bytes/SM
[未测] 在 5090 上实装 MoE kernel 的 device ms/token、implied GB/s、离 1.598 ms 的倍数
[未测] H-MoE-1/2/3 的裁决
[推测] 目标 2–4 ms/token（docs/20 §5 的 H2 预测值）
[推测] MLP 不足是当前 10.8 ms/step 的主因之一（§0 结论 3）
[依赖] W8/W4 的精度门（LeanMoE-Quant EXP-p1-01）
```
