# AGENTS.md

给在**本仓库**工作的 AI agent（以及新加入的人）的操作约定。人类贡献者同样适用。

---

## 1. 这个项目在做什么

测量 **batch 增大时，整前向融合（megakernel）相对 CUDA Graph 的收益边界及其机制**，
并把所用 profiling 方法作为产物。

**定位与结论见 [docs/00_定位与方法论.md](docs/00_定位与方法论.md) —— 动手前先读它。**

一句话记住：**本项目的主张是"划定了适用域"，不是"做出了更快的 kernel"。**

---

## 2. 方向边界（防止走偏）

**本节是操作约束。范围的*定义*见 [docs/00_定位与方法论.md](docs/00_定位与方法论.md)，
本节只写"做什么 / 不做什么 / 怎么算走偏"。**

### 2.1 在范围内

**小 batch（`B = 1..8`，即 `TG_MAX_BATCH`）的 decode 推理**，做三件事：

1. **找瓶颈** —— 用 nsys 定位"时间花在哪"，用 **SM 数扫描**判定"在等待还是在干活"，
   而不是先猜（方法见 [docs/00](docs/00_定位与方法论.md) §3）；
2. **给出机制** —— 说清**为什么**，而不只是"慢了多少"；
3. **得出可复现的结论** —— 每个数字都能追到 raw artifact + commit。

`B ≤ 8` 不是随意定的：与参考实现 cohere 的 `BS ≤ 8` 同区间，且 `TG_MAX_BATCH = 8` 是编译期约束。

### 2.2 参考实现：cohere-megakernel —— 借什么 / 不借什么

[cohere-ai/cohere-megakernel](https://github.com/cohere-ai/cohere-megakernel)
（`src/decode/megakernel.cuh` 3233 行）是**与小 batch decode 同一问题的工业级公开实现**，
自报 **BS=1 达 H100 speed-of-light 的 62%**。它的 `AGENTS.md` 也值得读（环境约束的写法可借鉴）。

| 借（架构无关，可迁移） | 不借（架构绑定，会走偏） |
|---|---|
| **依赖计数器 DAG**：`arrive_cross_sm` = fence + `__threadfence` + `atomicAdd`，`wait_cross_sm` = 自旋轮询。我们已有同构的 `tg_arrive` / `tg_wait` | **wgmma / TMA / tcgen05 代码**。⚠️ **本仓已实测：sm_120 的 ptxas 直接拒绝 `wgmma`**（`Instruction 'wgmma.fence' not supported on .target 'sm_120'`）。移植必然失败；我们的原语是 WMMA `m16n16k16` + `cp.async` |
| **每 SM 各自的 host 预构建指令表**，op 之间不做 grid barrier（我们已有同构设计） | **`GRID_SYNC` op**：cohere 只留作 ablation，生产 schedule **不发射**。若我们的调度需要全网格同步，那本身就是要修的问题，不是要引入的机制 |
| **speed-of-light 达成率口径**：他们把性能表达为"物理极限的百分比"，与本仓 roofline 纪律同源 | **`NMC_*` 硬编码模型**：cohere 的 `_validate_nmc_model_abi()` 硬门只认 North Mini Code，整体 fork 会立刻卡死 |
| **per-SM profiler 的思路**（`src/sm-profiler/`）。⚠️ **我们目前没有 per-SM profiler**，只有 `tools/parse_nsys_kernel.py` —— 这是真实缺口，值得补 | **CUDA 13 + ThunderKittens/nanobind submodule 构建链**：本仓是 CUDA 12.8 + `build.py`（torch cpp_extension），引入会摧毁现有构建 |
| **连续 batching / paged KV / prefix caching 的架构取舍**（作为对照，**不照抄**） | **little_jit 按 batch size 编译 .so**：本仓明确"先不做 JIT"，靠 runtime 配置切 `inst_buf` |

> 逐维度对照表在 [docs/11_task_graph重写设计.md](docs/11_task_graph重写设计.md) §2.2。

### 2.3 不做什么（每条都是已识别过的漂移）

| 漂移 | 为什么算漂移 |
|---|---|
| 把 cohere 的 wgmma/TMA 代码搬过来 | sm_120 编译不过（§2.2）；第一轮评估已判定"移植是**改动最大**的选项，不是最小" |
| 追求 `B > 8`、大规模、分布式、生产 serving | 超出范围；本仓测量已显示 `B ≥ 2` 融合输给 CUDA Graph，继续加 batch 不是在解决问题 |
| 复活 MoE 融合方向 | 已废弃，见 `docs/archive/{16,20,22}` |
| 复活 A100 取数 | 已由 5090 取代 |
| 把项目讲成"更快的 kernel" | 定位是**边界测量**。声称"更快"会被一次 batch 扫描推翻 |
| 为"参考"而整体 fork cohere | 它的模型与构建链都是硬绑定的（§2.2） |
| 引入新的量化 / 训练工作 | 不在范围内 |
| 把"调度层 batch"写成"完整 fused GPU batch execution" | 二者不是一回事（§3 硬纪律第 6 条已列，此处重申） |

### 2.4 动手前必须回答（任何性能类任务）

1. **基线是谁？** 必须是 **CUDA Graph**，不是 eager
2. **输入、dtype、layout、模型、硬件、batch 各是什么？**
3. **什么结果会推翻我的假设？** 没有可证伪点 = 没有实验
4. **怎么确认输出没有错？** 配对 correctness，见 §3
5. **原始结果存在哪？** `results/`，且可复现
6. **哪些 shape / 负载会更慢？** 边界必须报，不能只报最好的那个

### 2.5 方向变更必须改本节

**改研究方向 = 改 §2**，并在 [docs/00_定位与方法论.md](docs/00_定位与方法论.md) 同步。
只改代码不改本节，视为**未完成**。

---

## 3. 不可放宽的硬性纪律

违反任何一条，产物即为**无效**，不得提交：

1. **没有配对 correctness 结果的延迟数字，不许写出来。** 门为 PASS / CASCADE /
   `DEVICE_ERROR`（**硬失败，不可降级**），并叠加 `LOGIT_ATOL=1.0` 的严格 logit 对拍。
2. **roofline 分母只用本机实测可达带宽**，不许用规格值、不许从别的机器继承。
   可达带宽**尺寸敏感**（0.3 GB→1477、4 GB→1683 GB/s），报数时必须声明用了哪一行。
3. **负结果照报。** 本项目的核心结论就是负面的；删掉它等于删掉项目。
4. **`[推测]` / `[未测]` 不是结论。** 标了就必须保持标注，清空之前不得对外。
5. **不改已有 opcode 编号。** `TgOp` 0–9 已冻结，新增只能追加。
6. **不把"调度层 batch"写成"完整 fused GPU batch execution"。** 二者不是一回事。

---

## 4. 改代码前必须跑的门

```bash
# 1) CPU 套件（约 600 项）
PYTHONPATH=src python -m unittest discover -s tests

# 2) 文档护栏（3 道，强制 docs 编号与产物名一致）
PYTHONPATH=src python -m unittest tests.test_server_commands

# 3) 改 schedule.py 之前/之后**必须**过调度验证器
PYTHONPATH=src python -m tools.verify_schedule
#   期望：192/192 通过、误接受 0
```

**改 `schedule.py` 必须先过 `verify_schedule`。** 这不是建议——调度一旦死锁，
GPU 会挂死到进程被杀。

---

## 5. 命令的唯一来源

**所有 GPU 命令只在 [`SERVER_COMMANDS.txt`](SERVER_COMMANDS.txt) 里定义。**
不要在文档、脚本或对话里另写一套 `RUN_MODE`。可用模式见该文件头部注释。

---

## 6. 已知陷阱（踩过的坑，不要重复）

| 陷阱 | 后果 | 正确做法 |
|---|---|---|
| 用 `ncu` profile cooperative kernel | kernel-replay 与 `cudaLaunchCooperativeKernel` 冲突，**GPU driver hang，连 `nvidia-smi` 都阻塞** | 用 `nsys` |
| 想把 `TG_B_STAGES` 加到 8 以上 | `static_assert` 封死（cp.async commit-group 硬件上限） | 别试；改并发度 |
| 靠"加 SM"提速 | 实测 170 SM 比 64 SM 更慢 | 先修依赖链 |
| 把 GPU 满载当成"带宽没问题" | 本仓实测满载但只 17.2% 可达带宽 | 看 roofline 达成率 |
| 用 "megakernel" 描述本项目 | 会被归为窄域构造 claim | 本项目是**边界测量** |
| 把厂商规格带宽当分母 | 所有"离墙多远"失真 | 用 `results/bandwidth.json` |

---

## 7. 文档维护规则

- **编号唯一**：`docs/NN_名称.md`，新文档取未用编号；**本仓库已用 00–22 与 99**
- **必须在 `docs/README.md` 注册**，否则 `tests/test_server_commands.py` 的护栏会失败
- 三态标注：**现行** / **历史证据** / **研究资料**。历史文档不得当作当前测量引用
- **CLI 里的文件路径与文档声明的产物名必须一致**（同一道护栏强制）
- 产物统一写到 `results/`，**不要带机器前缀**（`a100_`、`rtx_` 之类已被禁止）

---

## 8. 提交约定

- **一个提交做一件事**，提交信息写清"为什么"，不只是"做了什么"
- 结论性改动（结论、数字、边界）必须在提交信息里给出**复现路径**
- 归档/废弃方向时**不要删文件**，写清归档理由与可迁移资产（参见 `ARCHIVE.md`）

---

## 9. 当前状态

**重新激活（2026-10-07）**：dense 融合的边界结论已定（见 [docs/00](docs/00_定位与方法论.md) §2），
现按 **§2 的范围**继续做**小 batch（`B=1..8`）的瓶颈定位与结论**。

第一轮已归档的证据与可迁移资产见 [ARCHIVE.md](ARCHIVE.md) ——
**其中的方法论（可达带宽标定、nsys 优先、SM 数扫描）仍然有效，继续用。**

**历史方向已废弃，不要复活**：MoE 融合 kernel（`docs/archive/{16,20,22}`）、
A100 取数（已由 5090 取代）、Qwen3-0.6B 之外的 dense 扩展。
