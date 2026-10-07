# 文档索引

> **`docs/` 的唯一入口。** 每份文档按三态分类，读之前先确认状态：
>
> - **现行**：随代码更新，可作为操作与结论依据
> - **历史证据**：归档线的实测记录，只用于追溯结论来源，**不得当作当前主线的测量**
> - **研究资料**：外部调研，不随代码变
> - **已废弃**：方向已终止，仅留档，**不要复活**
>
> **先读 [00 定位与方法论](00_定位与方法论.md)。** 它定义本项目在问什么、用什么方法答、
> 以及结论不得外推的部分。README 是门面，本页是索引，**00 是定义**。

---

## 1. 按目的选最短路径

| 你想 | 读这些（按顺序） |
|---|---|
| **30 秒了解结论** | [00 定位与方法论](00_定位与方法论.md) §2 → 仓库根 [README](../README.md) |
| **了解 profiling 方法**（本项目的核心产物） | **[00](00_定位与方法论.md) §3** → [04 基准测试方法](04_基准测试方法.md) → [02 指标说明](02_指标说明.md) |
| **复现取数** | [19 上机清单](19_上机清单.md) → [08 服务器运行手册](08_服务器运行手册.md) → [04 基准测试方法](04_基准测试方法.md) |
| **改 kernel / 调度** | [11 task-graph 设计](11_task_graph重写设计.md) → **[17 调度验证器](17_调度验证器.md)（改 `schedule.py` 必须先过）** → [18 成本模型](18_成本模型.md) → [12 性能诊断](12_性能诊断与优化计划.md) |
| **看边界结论的完整推导** | **[07 融合收益边界与根因](07_融合收益边界与根因.md)** → [14 融合粒度实验](14_融合粒度实验.md) → [21 5090 复现](21_5090复现_融合粒度实验.md) |
| **接手项目** | [00](00_定位与方法论.md) → [10 交接文档](10_交接文档.md) → [01 项目说明](01_项目说明.md) → [AGENTS.md](../AGENTS.md) |

---

## 2. 状态表

### 现行（可作为依据）

| 文档 | 一句话 |
|---|---|
| [00 定位与方法论](00_定位与方法论.md) | **定位单一来源**：在问什么、结论、方法（含 NCU/cooperative 陷阱、SM 数扫描）、效力边界 |
| [01 项目说明](01_项目说明.md) | 主线范围与保留的 backend |
| [02 指标说明](02_指标说明.md) | 指标定义 + 统计口径（所有数字的解释依据） |
| [04 基准测试方法](04_基准测试方法.md) | 三类 workload 的流程与产物 |
| [05 对比基线与公平性](05_对比基线与公平性.md) | 三条对比路径的口径与公平性规则 |
| [07 融合收益边界与根因](07_融合收益边界与根因.md) | **核心分析**：边界在哪、为什么；§9.10 有外部独立复现 |
| [08 服务器运行手册](08_服务器运行手册.md) | GPU 操作的唯一手册 |
| [10 交接文档](10_交接文档.md) | 交接入口；§9 记录方向决策 |
| [11 task-graph 设计](11_task_graph重写设计.md) | 834 行设计稿；§11 为实现状态 |
| [12 性能诊断与优化计划](12_性能诊断与优化计划.md) | 优化计划与优先级 |
| [13 集成与基线规划](13_集成与基线规划.md) | 集成范围与基线定义 |
| [14 融合粒度实验](14_融合粒度实验.md) | 融合粒度扫描 |
| [17 调度验证器](17_调度验证器.md) | 死锁/到达计数验证器 + 对抗性变异（纯 CPU） |
| [18 成本模型](18_成本模型.md) | 结构式 decode 成本模型 + 预注册预测 |
| [19 上机清单](19_上机清单.md) | 离线已做 vs 必须上机；执行顺序与决策门 |
| [21 5090 复现](21_5090复现_融合粒度实验.md) | 5090 上的口径、预注册假设、已测结果 |

### 历史证据（只用于追溯，不得当当前测量）

| 文档 | 说明 |
|---|---|
| [03 Megakernel 设计说明](03_Megakernel设计说明.md) | **已归档**：旧 `kernel.cu` 线的英文设计说明 |
| [06 实际运行数据](06_实际运行数据.md) | 归档线实测数字（`runs=3` + min-of-runs，口径与现行不同） |
| [09 集成问题清单](09_集成问题清单.md) | 2026-09 的集成问题清单 |
| [taskgraph_progress](taskgraph_progress.md) | 2026-09-23 的优化进展快照 |
| [99 变更历史](99_变更历史.md) | 各文档抽出的版本变更日志 |

### 研究资料（外部调研，不随代码变）

| 文档 | 说明 |
|---|---|
| [15 招聘门槛调研](15_招聘门槛调研_推理优化与AI编译器实习.md) | 岗位硬门槛、面试题分布、作品集信号（含证据分级） |

### 已废弃（方向终止，不要复活）

| 文档 | 为什么废弃 |
|---|---|
| [16 MoE 可行性调研](16_MoE_A100可行性调研.md) | 调研本身仍有价值（外部工作索引齐全），但 **MoE 方向未推进** |
| [20 方向重构](20_方向重构_从dense到MoE融合kernel.md) | 曾计划转向 MoE；**已放弃**（见 [`ARCHIVE.md`](../ARCHIVE.md)） |
| [22 小 batch MoE 架构设计](22_小batch_MoE架构设计.md) | 设计决议**未实现**；其中 opcode 契约**不得引用为"已完成"** |

---

## 3. 关键命令

**唯一来源：[`SERVER_COMMANDS.txt`](../SERVER_COMMANDS.txt)。** 不要在此另抄一套。

```bash
RUN_MODE=verify      bash SERVER_COMMANDS.txt   # 环境 + 扩展 import
RUN_MODE=bandwidth   bash SERVER_COMMANDS.txt   # 可达带宽标定（roofline 分母，必须先跑）
RUN_MODE=build       bash SERVER_COMMANDS.txt   # 干净重建 CUDA 扩展
RUN_MODE=correctness bash SERVER_COMMANDS.txt   # PASS/CASCADE/FAIL 门
RUN_MODE=benchmark   bash SERVER_COMMANDS.txt   # eager / cudagraph / megakernel A/B
RUN_MODE=taskgraph   bash SERVER_COMMANDS.txt   # kernel 微基准 + ABI 测试
```

### 3b. 离线门（不需要 GPU）

```bash
PYTHONPATH=src python -m unittest discover -s tests   # 约 600 项，11 项需 torch 自动 skip
PYTHONPATH=src python -m tools.verify_schedule        # 调度验证器：192/192，误接受 0
```

**改动 `schedule.py` 之前必须过 `tools.verify_schedule`**（见 [17](17_调度验证器.md)）。

## 4. 现行 vs 归档（不要混）

| | 现行主线 | 归档（`codex/archive/frozen-kernel`） |
|---|---|---|
| decode 实现 | `csrc/taskgraph_kernel.cu` + `backends/megakernel/` | `csrc/kernel.cu` |
| 工具 | `tools/*.py` | `tools/bench_burst_*`、`bench_roofline_a100`、`verdict` |
| 数字口径 | `runs=5` + stddev/CoV + bootstrap 95% CI | `runs=3` + min-of-runs |
| 结论 | 边界结论已定（见 [00](00_定位与方法论.md)） | 06/07 部分数字只属于这一线 |

## 5. 维护规则（有护栏）

1. **`RUN_MODE`**：文档里出现的任何模式必须在 `SERVER_COMMANDS.txt` 有 dispatch 分支
2. **工具引用**：`tools.x` / `tools/x.py` 必须在本分支存在（历史文档、glob、删除线豁免）
3. **命令参数**：文档里的 `python -m …` 所带参数必须是该模块 argparse 接受的
4. **输出文件名**：`docs/08` 承诺的文件必须真的由 `SERVER_COMMANDS.txt` 写出
5. **新增文档**：按编号命名 + 在本页登记 + 首行加状态说明。**已用编号 00–22 与 99**

以上 1–4 由 `tests/test_server_commands.py` 强制：

```bash
PYTHONPATH=src python -m unittest tests.test_server_commands
```
