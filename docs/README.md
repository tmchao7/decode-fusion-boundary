# 文档索引

> **`docs/` 的唯一入口。** 这里只保留**新一轮开发真正要读的文档**。
> 历史/过期/废弃材料已移到 [`archive/`](archive/README.md)，见该目录的说明。
>
> **先读 [00 定位与方法论](00_定位与方法论.md)。** 它定义本项目在问什么、用什么方法答、
> 以及结论不得外推的部分。README 是门面，本页是索引，**00 是定义**。

---

## 1. 按目的选最短路径

| 你想 | 读这些（按顺序） |
|---|---|
| **30 秒了解结论** | [00 定位与方法论](00_定位与方法论.md) §2 → 根 [README](../README.md) |
| **了解 profiling 方法**（核心产物） | **[00](00_定位与方法论.md) §3** → [04 基准测试方法](04_基准测试方法.md) → [02 指标说明](02_指标说明.md) |
| **复现取数** | [19 上机清单](19_上机清单.md) → [08 服务器运行手册](08_服务器运行手册.md) → [04 基准测试方法](04_基准测试方法.md) → [02 指标说明](02_指标说明.md) |
| **改 kernel / 调度** | [11 task-graph 设计](11_task_graph重写设计.md) → **[17 调度验证器](17_调度验证器.md)（改 `schedule.py` 必须先过）** → [18 成本模型](18_成本模型.md) → [12 性能诊断](12_性能诊断与优化计划.md) |
| **看边界结论的完整推导** | **[07 融合收益边界与根因](07_融合收益边界与根因.md)** → [14 融合粒度实验](14_融合粒度实验.md) → [21 5090 复现](21_5090复现_融合粒度实验.md) |
| **找历史材料** | [archive/](archive/README.md) |

---

## 2. 工作文档（14 份）

| 文档 | 作用 | 什么时候读 |
|---|---|---|
| [00 定位与方法论](00_定位与方法论.md) | **定位单一来源**：问什么、结论、方法、效力边界 | 动手前**必读** |
| [01 项目说明](01_项目说明.md) | 主线范围与保留的 backend | 确认边界 |
| [02 指标说明](02_指标说明.md) | 指标定义 + **统计口径**（`runs=5` + CoV + bootstrap CI） | 报任何数字前 |
| [04 基准测试方法](04_基准测试方法.md) | 三类 workload 的流程与产物 | 取数前 |
| [05 对比基线与公平性](05_对比基线与公平性.md) | 对比口径与公平性规则 | 做对比前 |
| [07 融合收益边界与根因](07_融合收益边界与根因.md) | **核心分析**（含 §9.9 外部证据） | 理解结论 |
| [08 服务器运行手册](08_服务器运行手册.md) | GPU 操作的唯一手册 | 上机前 |
| [11 task-graph 设计](11_task_graph重写设计.md) | 内核设计稿；§11 为实现状态 | 改内核 |
| [12 性能诊断与优化计划](12_性能诊断与优化计划.md) | 优化计划与优先级 | 想提速 |
| [14 融合粒度实验](14_融合粒度实验.md) | 融合粒度扫描的定义与验证 | 复现实验 |
| [17 调度验证器](17_调度验证器.md) | 死锁/到达计数验证器 + 对抗性变异 | **改 `schedule.py` 前必读** |
| [18 成本模型](18_成本模型.md) | 结构式 decode 成本模型 + 预注册预测 | 做 roofline 推算 |
| [19 上机清单](19_上机清单.md) | 离线已做 vs 必须上机；执行顺序与决策门 | 规划取数 |
| [21 5090 复现](21_5090复现_融合粒度实验.md) | 5090 上的口径、预注册假设、已测结果 | 看当前机器结论 |

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

---

## 4. 现行 vs 归档（不要混）

| | 现行主线 | 归档（`codex/archive/frozen-kernel` 分支 + [archive/](archive/README.md)） |
|---|---|---|
| decode 实现 | `csrc/taskgraph_kernel.cu` + `backends/megakernel/` | `csrc/kernel.cu` |
| 数字口径 | `runs=5` + stddev/CoV + bootstrap 95% CI | `runs=3` + min-of-runs |
| 文档 | 本页 §2 的 14 份 | `docs/archive/` |

> ⚠️ **归档线的数字（`runs=3` + min-of-runs）与现行口径不同，引用时必须声明。**

---

## 5. 维护规则（有护栏）

1. **`RUN_MODE`**：文档里出现的任何模式必须在 `SERVER_COMMANDS.txt` 有 dispatch 分支
2. **工具引用**：`tools.x` / `tools/x.py` 必须在本分支存在
3. **命令参数**：文档里的 `python -m …` 参数必须是该模块 argparse 接受的
4. **输出文件名**：`docs/08` 承诺的文件必须真由 `SERVER_COMMANDS.txt` 写出
5. **新增文档**：按编号命名 + 在本页 §2 登记 + 首行加状态说明。**已用编号 00–22 与 99**
6. **减负优先**：不写"以防万一"的文档。新的历史材料直接进 `docs/archive/`，不进 `docs/`

以上 1–4 由 `tests/test_server_commands.py` 强制：

```bash
PYTHONPATH=src python -m unittest tests.test_server_commands
```
