# 文档索引

> **本页是 `docs/` 的唯一入口。** 24 份文档按三种状态分类，读之前先确认状态：
>
> - **现行**：随代码更新，可以作为操作与结论依据；
> - **历史证据**：归档 `kernel.cu` 线的实测记录，只用于追溯结论来源，**不得当作当前主线的测量**；
> - **研究资料**：外部调研，不随代码变。
>
> 维护规则见文末，`tests/test_server_commands.py` 有三道护栏强制它们成立。

## 1. 按目的选最短路径

| 你的目的 | 读这些（按顺序） |
|---|---|
| 刚接手这个项目 | [10 交接文档](10_交接文档.md) → [01 项目说明](01_项目说明.md) → [05 基线与公平性](05_对比基线与公平性.md) |
| 要复现/取数 | **[19 上机清单](19_上机清单.md)** → [08 服务器运行手册](08_服务器运行手册.md) → [04 基准测试方法](04_基准测试方法.md) → [02 指标说明](02_指标说明.md) |
| 要改 kernel / 调度 | [11 task-graph 设计](11_task_graph重写设计.md) → [17 调度验证器](17_调度验证器.md)（改 `schedule.py` 必须先过） → [18 成本模型](18_成本模型.md) → [12 性能诊断与优化计划](12_性能诊断与优化计划.md) → [14 融合粒度实验](14_融合粒度实验.md) |
| 要做 MoE（小 batch） | **[22 小 batch MoE 架构](22_小batch_MoE架构设计.md)** → [16 MoE 可行性](16_MoE_A100可行性调研.md) → [20 方向重构](20_方向重构_从dense到MoE融合kernel.md) §9 |
| 要看融合/CUDA Graph 的**收益边界** | [07 融合收益边界与根因](07_融合收益边界与根因.md) §0 → §9.10（外部独立复现）→ [06 实际运行数据](06_实际运行数据.md) |
| 要决定做什么/投什么方向 | **[20 方向重构](20_方向重构_从dense到MoE融合kernel.md)** → [16 MoE 可行性](16_MoE_A100可行性调研.md) → [15 招聘门槛调研](15_招聘门槛调研_推理优化与AI编译器实习.md) |
| 想知道"配置为什么这么定" | [13 集成与基线规划](13_集成与基线规划.md) → [99 变更历史](99_变更历史.md) |

## 2. 状态表

| 文档 | 状态 | 一句话 |
|---|---|---|
| [01 项目说明](01_项目说明.md) | 现行 | 主线是什么、只保留哪些 backend |
| [02 指标说明](02_指标说明.md) | 现行 | 指标定义 + **统计口径**（所有数字的解释依据） |
| [03 Megakernel 设计说明](03_Megakernel设计说明.md) | **已归档** | 旧 `kernel.cu` 线的英文设计说明，未随重写更新 |
| [04 基准测试方法](04_基准测试方法.md) | 现行 | 三类 workload 的运行流程与产物 |
| [05 对比基线与公平性](05_对比基线与公平性.md) | 现行 | 三条对比路径的口径与公平性规则 |
| [06 实际运行数据](06_实际运行数据.md) | **历史证据** | 归档线实测数字（`runs=3` + min-of-runs） |
| [07 融合收益边界与根因](07_融合收益边界与根因.md) | **历史分析** | 证据与根因；结论是适用区间，§9.10 有外部独立复现 |
| [08 服务器运行手册](08_服务器运行手册.md) | 现行 | GPU 操作的唯一手册 |
| [09 集成问题清单](09_集成问题清单.md) | **历史** | 2026-09-08/09 的集成问题清单 |
| [10 交接文档](10_交接文档.md) | 现行 | 交接入口；§9 记录三个方向决策 |
| [11 task-graph 设计](11_task_graph重写设计.md) | 设计稿 | 834 行设计；§11 是实现状态 |
| [12 性能诊断与优化计划](12_性能诊断与优化计划.md) | 现行计划 | 基线待 `taskgraph-baseline` 刷新 |
| [13 集成与基线规划](13_集成与基线规划.md) | 现行 | 主线集成范围与基线定义 |
| [14 融合粒度实验](14_融合粒度实验.md) | 现行 | Phase 1 实验；CPU 侧已验证，待取数 |
| [15 招聘门槛调研](15_招聘门槛调研_推理优化与AI编译器实习.md) | 研究资料 | 岗位硬门槛、面试题分布、作品集信号（含证据分级） |
| [16 MoE 可行性调研](16_MoE_A100可行性调研.md) | 研究资料 | 单卡 MoE 选型 + per-token 成本模型（英文） |
| [17 调度验证器](17_调度验证器.md) | 现行 | 死锁/到达计数的离线验证器 + 对抗性变异（纯 CPU） |
| [18 成本模型](18_成本模型.md) | 现行 | 结构式解码步成本模型 + 预注册的粒度预测（纯 CPU） |
| [19 上机清单](19_上机清单.md) | 现行 | 离线已做完 vs 必须上机；执行顺序、决策门、回传文件 |
| [20 方向重构](20_方向重构_从dense到MoE融合kernel.md) | 现行 | **方向决议**：dense device 侧无空间 → MoE + 低比特；含预注册假设与反证信号 |
| [21 RTX 5090 复现：融合粒度](21_5090复现_融合粒度实验.md) | 现行 | `docs/20` §9 第 1 步：5090 上的口径、预注册假设、命令序列；**上机结果待填** |
| [22 小 batch MoE 架构](22_小batch_MoE架构设计.md) | 现行 | **设计决议（未实测）**：字节账/四层架构/不变量/新 opcode 契约；先 b=1 再扩 b>1；含判决门 |
| [99 变更历史](99_变更历史.md) | 历史 | 各文档抽出的版本变更日志 |
| [taskgraph_progress](taskgraph_progress.md) | **历史快照** | 2026-09-23 的优化进展报告 |

## 3. 关键命令（唯一来源：`SERVER_COMMANDS.txt`）

```bash
RUN_MODE=verify            bash SERVER_COMMANDS.txt   # 环境 + 扩展 import
RUN_MODE=bandwidth         bash SERVER_COMMANDS.txt   # 可达带宽标定（roofline 分母）
RUN_MODE=build             bash SERVER_COMMANDS.txt   # 干净重建 CUDA 扩展
RUN_MODE=correctness       bash SERVER_COMMANDS.txt   # PASS/CASCADE/FAIL 门
RUN_MODE=benchmark         bash SERVER_COMMANDS.txt   # eager / cudagraph / megakernel A/B
RUN_MODE=taskgraph         bash SERVER_COMMANDS.txt   # kernel 微基准 + ABI 测试
RUN_MODE=taskgraph-baseline bash SERVER_COMMANDS.txt  # correctness + A/B + 四口径账
RUN_MODE=taskgraph-fusion  bash SERVER_COMMANDS.txt   # 融合粒度扫描
```

## 3b. 开发机（CPU）命令

```bash
python -m pip install numpy xxhash                             # 缺 xxhash 时 4 个 scheduler 模块 import 失败
PYTHONPATH=src python -m unittest discover -s tests          # 全套 CPU 测试（499 项，11 项需 torch 自动 skip）
python -m tools.verify_schedule --json results/schedule_verification.json  # 调度验证器
python -m tools.bench_fusion_granularity --help              # 粒度扫描计划（不需要 GPU）
```

上面这些**不需要 GPU**；改动 `schedule.py` 之前必须跑 `tools.verify_schedule`（见
[17](17_调度验证器.md)）。

## 4. 现行 vs 归档（不要混）

| | 现行主线 | 归档（`codex/archive/frozen-kernel`） |
|---|---|---|
| decode 实现 | `csrc/taskgraph_kernel.cu` + `backends/megakernel/` | `csrc/kernel.cu` |
| 工具 | `tools/*.py`（除下列项） | `tools/bench_burst_*`、`bench_roofline_a100`、`verdict`、`plot_crossover` |
| 测试 | `tests/`（320 项，CPU 可跑） | `tests/test_verdict.py` |
| 数字口径 | `runs=5` + stddev/CoV + bootstrap 95% CI | `runs=3` + min-of-runs |
| 结论 | 待 `taskgraph-baseline` 刷新 | 06/07 的数字只属于这一线 |

## 5. 维护规则（有护栏）

1. **`RUN_MODE`**：文档里出现的任何模式必须在 `SERVER_COMMANDS.txt` 有 dispatch 分支。
2. **工具引用**：`tools.x` / `tools/x.py` 必须在本分支存在（历史文档、glob、删除线豁免）。
3. **命令参数**：文档里的 `python -m …` 所带参数必须是该模块 argparse 接受的。
4. **输出文件名**：`docs/08` 承诺的文件必须真的由 `SERVER_COMMANDS.txt` 写出。
5. **新增文档**：按编号命名 + 在本页登记 + 在正文首行加状态说明。

以上 1–4 由 `tests/test_server_commands.py` 强制，改动文档后跑
`PYTHONPATH=src python -m unittest tests.test_server_commands` 即可。
