# AGENTS.md

给在**本仓库**工作的 AI agent（以及新加入的人）的操作约定。人类贡献者同样适用。

---

## 1. 这个项目在做什么

测量 **batch 增大时，整前向融合（megakernel）相对 CUDA Graph 的收益边界及其机制**，
并把所用 profiling 方法作为产物。

**定位与结论见 [docs/00_定位与方法论.md](docs/00_定位与方法论.md) —— 动手前先读它。**

一句话记住：**本项目的主张是"划定了适用域"，不是"做出了更快的 kernel"。**

---

## 2. 不可放宽的硬性纪律

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

## 3. 改代码前必须跑的门

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

## 4. 命令的唯一来源

**所有 GPU 命令只在 [`SERVER_COMMANDS.txt`](SERVER_COMMANDS.txt) 里定义。**
不要在文档、脚本或对话里另写一套 `RUN_MODE`。可用模式见该文件头部注释。

---

## 5. 已知陷阱（踩过的坑，不要重复）

| 陷阱 | 后果 | 正确做法 |
|---|---|---|
| 用 `ncu` profile cooperative kernel | kernel-replay 与 `cudaLaunchCooperativeKernel` 冲突，**GPU driver hang，连 `nvidia-smi` 都阻塞** | 用 `nsys` |
| 想把 `TG_B_STAGES` 加到 8 以上 | `static_assert` 封死（cp.async commit-group 硬件上限） | 别试；改并发度 |
| 靠"加 SM"提速 | 实测 170 SM 比 64 SM 更慢 | 先修依赖链 |
| 把 GPU 满载当成"带宽没问题" | 本仓实测满载但只 17.2% 可达带宽 | 看 roofline 达成率 |
| 用 "megakernel" 描述本项目 | 会被归为窄域构造 claim | 本项目是**边界测量** |
| 把厂商规格带宽当分母 | 所有"离墙多远"失真 | 用 `results/bandwidth.json` |

---

## 6. 文档维护规则

- **编号唯一**：`docs/NN_名称.md`，新文档取未用编号；**本仓库已用 00–22 与 99**
- **必须在 `docs/README.md` 注册**，否则 `tests/test_server_commands.py` 的护栏会失败
- 三态标注：**现行** / **历史证据** / **研究资料**。历史文档不得当作当前测量引用
- **CLI 里的文件路径与文档声明的产物名必须一致**（同一道护栏强制）
- 产物统一写到 `results/`，**不要带机器前缀**（`a100_`、`rtx_` 之类已被禁止）

---

## 7. 提交约定

- **一个提交做一件事**，提交信息写清"为什么"，不只是"做了什么"
- 结论性改动（结论、数字、边界）必须在提交信息里给出**复现路径**
- 归档/废弃方向时**不要删文件**，写清归档理由与可迁移资产（参见 `ARCHIVE.md`）

---

## 8. 当前状态

**已归档（2026-10-06）**：dense 融合方向已给出边界结论，不再继续开发。
文件保留作为方法与证据。若重新激活，先读 `ARCHIVE.md` 与 `docs/00`。

**历史方向已废弃，不要复活**：Qwen3-0.6B 之外的 dense 扩展、MoE 融合 kernel（docs/archive/16/20/22）、
A100 取数（已由 5090 取代）。
