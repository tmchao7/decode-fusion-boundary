# docs/archive/ —— 隔离区

**这里的东西不属于当前工作集。** 2026-10-06 从 `docs/` 移入，目的是让 `docs/` 只保留
新一轮开发真正要读的文档。

## 为什么是"隔离"而不是"删除"

- 仓库自己的纪律是**归档不删**（见根目录 `ARCHIVE.md`），
- 隔离是**可逆**的：`git mv docs/archive/X.md docs/X.md` 即可拉回；
- 这些文件里有**真实的实测记录与外部调研**，删掉只能翻 git 历史去找。

## 目录内容与为什么被隔离

| 文件 | 为什么隔离 |
|---|---|
| `06_实际运行数据.md` | 归档线（`kernel.cu` 线）实测数字，口径是 **`runs=3` + min-of-runs**，与现行 `runs=5` + CoV + bootstrap CI **不同**。留在工作集会**混口径**，这是最危险的一份 |
| `03_Megakernel设计说明.md` | 旧 `kernel.cu` 线的英文设计说明，已被 taskgraph 重写取代 |
| `09_集成问题清单.md` | 2026-09 的问题清单，已过期 |
| `10_交接文档.md` | 历次开发轮次的工程日志。**若要接手历史，读这份**；新一轮开发不需要 |
| `13_集成与基线规划.md` | 早期集成范围与基线定义，已被 `01` / `05` 覆盖 |
| `15_招聘门槛调研_推理优化与AI编译器实习.md` | **不是项目文档**，是求职调研。内容仍有价值，但不该占 `docs/` 的位置 |
| `16_MoE_A100可行性调研.md` | MoE 方向的可行性调研。外部工作索引（UniEP / MonoMoE / Ada-MK / AutoMegaKernel 等）**仍然有用**，但该方向**未推进** |
| `20_方向重构_从dense到MoE融合kernel.md` | 曾计划转向 MoE。**已放弃** |
| `22_小batch_MoE架构设计.md` | 设计决议，**未实现**。其中的 opcode 契约**不得引用为"已完成"** |
| `99_变更历史.md` | 各文档抽出的版本变更日志 |
| `taskgraph_progress.md` | 2026-09-23 的优化进展快照 |
| `plans/2026-09-24-taskgraph-as-megakernel.md` | 9 月的实施计划，已执行完毕 |
| `plans/2026-09-30-taskgraph-finalization.md` | 9 月的收尾计划，已执行完毕 |

## 引用规则

1. **`docs/` 里的现行文档不得把这里的内容当作当前测量引用**
2. 引用 `06` 的数字时**必须同时声明它的口径**（`runs=3` + min-of-runs）
3. `16` / `20` / `22` 涉及的方向**已终止**，不要据此复活 MoE 工作
4. 本目录**不参与** `tests/test_server_commands.py` 的文档护栏扫描（那些护栏只扫 `docs/*.md`，非递归）

## 如何拉回

```bash
git mv docs/archive/10_交接文档.md docs/10_交接文档.md
# 然后在 docs/README.md 的状态表里登记
```

拉回后**必须**重跑护栏：

```bash
PYTHONPATH=src python -m unittest tests.test_server_commands
```
