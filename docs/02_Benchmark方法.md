# Benchmark 方法与运行流程

本项目将 benchmark 分为三类：标准 baseline、可控性能 workload 和真实语义 workload。所有 CUDA 命令都在 A100/V100 服务器执行，不在本地 Mac 执行。

## 目录与环境

从仓库根目录执行：

```bash
bash setup_server.sh
source .venv/bin/activate
python -m tools.verify_server_env
```

确认代码可导入并通过纯 Python 检查：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall src/nanovllm benchmarks tools
```

## 第一步：短 correctness gate

A100：

```bash
CUDA_LAUNCH_BLOCKING=1 \
NANOVLLM_CUDA_ARCH=sm_80 \
python -m tools.correctness_smoke /PATH/TO/Qwen3-0.6B \
  --workloads benchmarks/workloads.json \
  --workload-ids short_qa,math_reasoning,code_generation,code_debugging \
  --max-tokens 8 \
  --require-megakernel \
  --json results/a100-correctness.json
```

V100：

```bash
CUDA_LAUNCH_BLOCKING=1 \
NANOVLLM_CUDA_ARCH=sm_70 \
python -m tools.correctness_smoke /PATH/TO/Qwen3-0.6B \
  --workloads benchmarks/workloads.json \
  --workload-ids short_qa,math_reasoning,code_generation,code_debugging \
  --max-tokens 8 \
  --json results/v100-correctness.json
```

V100 当前预期是验证 baseline 和 fallback；没有 `--require-megakernel` 时，Megakernel 未执行不会被误报为失败。

## 第二步：可控性能测试

可控 workload 使用固定 token IDs、固定 seed 和 `ignore_eos=True`，主要研究 prompt 长度、batch size 和 decode 延迟的变化：

```bash
NANOVLLM_CUDA_ARCH=sm_80 \
python -m benchmarks.compare /PATH/TO/Qwen3-0.6B \
  --suite performance \
  --workloads benchmarks/workloads.json \
  --backends torch-eager,torch-cudagraph,megakernel \
  --prompt-lengths 32,128,512,1024 \
  --batch-sizes 1,8,32 \
  --max-tokens 128 \
  --warmup 2 \
  --runs 10 \
  --json results/a100-performance.json \
  --markdown results/a100-performance.md
```

其中：

- `torch-eager` 是 PyTorch eager baseline；
- `torch-cudagraph` 是最重要的优化对照 baseline；
- `megakernel` 当前只对单序列 greedy decode 有效；
- batch 8/32 只比较标准 nano-vLLM 路径，Megakernel batch 仍是后续工作。

## 第三步：真实语义 workload

workload 定义位于：

```text
benchmarks/workloads.json
```

当前包含 QA、数学、代码生成、代码 debugging、摘要、长上下文问答和多轮对话：

```bash
NANOVLLM_CUDA_ARCH=sm_80 \
python -m benchmarks.compare /PATH/TO/Qwen3-0.6B \
  --suite realistic \
  --workloads benchmarks/workloads.json \
  --backends torch-eager,torch-cudagraph,megakernel \
  --json results/a100-realistic.json \
  --markdown results/a100-realistic.md
```

真实 workload 主要用于检查不同语义请求是否保持 token-level correctness；性能结论仍应以可控 workload 为主。

## 第四步：结果可视化

```bash
python -m tools.plot_metrics \
  results/a100-performance.json \
  --output-dir results/a100-performance-plots
```

输出包括：

- throughput 对比；
- TTFT/TPOT/ITL p50；
- 峰值显存；
- baseline speedup；
- 首个请求的 token 时间线。

## 第五步：Nsight profiling

只对短 workload 做 profiling，不把 profiler 运行时间用于 latency 表：

```bash
NANOVLLM_CUDA_ARCH=sm_80 \
nsys profile --trace=cuda,nvtx \
  --output=results/a100_nsys \
  python -m benchmarks.compare /PATH/TO/Qwen3-0.6B \
    --suite performance --backends megakernel \
    --prompt-lengths 32 --max-tokens 32 --warmup 1 --runs 1
```

重点查看 `nano_vllm_decode_step` 和 `megakernel_decode` 范围内的 kernel launch、CUDA API、memcpy 和 GPU 时间。

解析与 roofline 核对（`--roofline` 会把每步 kernel 时间换算成"隐含权重带宽"，
用来判断这个时间在物理上是否可能覆盖整个 decode step）：

```bash
python -m tools.parse_nsys_kernel results/a100_nsys.sqlite \
  --decode-steps 49 --memory --roofline
```

注意 `nsys profile --trace=cuda` **不采集 DRAM 计数器**；需要 GPU metrics 采样
（`--gpu-metrics-device` / `--enable-gpu-metrics`，视 nsys 版本）。工具在缺表时
明确输出 `UNAVAILABLE` 并提示替代方案（`tools.bench_roofline_a100 --sections bandwidth`），
不会静默报 0。

## 第六步：host/device 账本与 roofline（判定"能不能更快"）

性能表只给"多快"，不给"还能快多少"。第六步把每个 decode step 拆成 device 时间与
host 残差，并和权重带宽下界对比：

```bash
python -m tools.account_decode results/a100_account.json --seq-len 160 \
  --json results/a100_account_breakdown.json
python -m nanovllm.roofline --batch-sizes 1,2,4,8,16,32 --seq-len 160 --check-kernel
```

口径（务必区分，两个字段的命名历史上有坑）：

- `summary.decode_time_ms_total`：所有请求的 **decode 墙钟之和**（毫秒）。一个 decode
  step 给每条序列各出 1 个 token，且所有序列共享同一步，所以
  **每步墙钟 = decode_time_ms_total / (request_count × decode_steps)**。
- `summary.cuda_decode_time_ms_total`：同上口径的 **纯 GPU 时间**（CudaEvent 计时）。
- `tpot_ms`：**每序列**每 token 时间，即 TPOT；因此 `tpot_ms × batch == 每步墙钟`，
  可以与上一条交叉验证（差异 >5% 时 `account_decode` 会 WARNING）。
- 注意 `summary.decode_time_ms` / `cuda_decode_time_ms` 这两个字段在
  `aggregate_requests` 里被 `summarize(..., scale=1000.0)` 乘了 1000（历史遗留），
  **不要**拿它们当毫秒用；用 `*_total` 字段。

`account_decode` 能直接吃两种输入，自动识别：

| 输入 | 来源 | device 列 |
|---|---|---|
| `experiments` schema | `benchmarks.compare --json` | **实测** `cuda_decode_time_ms_total / decode_steps` |
| `records` schema | `tools/bench_burst_batch.py --json`（crossover 扫描） | **roofline 权重下界**（标 `(floor)`），host 残差是下界、GPU 占比是上界；`note` 里的 `serial xN` 会被读成 `serial_chunks` |

两类输入可以一次传多个文件，`render_comparison` 每个 batch 取最快的后端，device
anchor 优先用实测 GPU 时间，没有则退回 0.967 ms/step 的权重下界。

**标定告警（必须处理，不是工具报错）**：若实测每步 GPU 时间低于权重下界
（例如 0.77 ms < 0.967 ms，隐含 1952 GB/s > A100 规格 1555 GB/s），工具会打
`WARNING: ... Calibrate with RUN_MODE=roofline`。这表示该行的 headroom 数字
**在标定前不能信**（docs/07 §10.4 的矛盾）。`--warnings-as-errors` 只对 cadence
交叉验证不一致返回非零，标定告警不会让命令失败。

完整的"跑什么 / 看什么 / 怎么判"见 [docs/09 高并发判定任务书](09_高并发判定任务书.md)；
它把 `RUN_MODE=roofline` 与 `RUN_MODE=account` 的产出文件、判据阈值和停止规则写在
一处，是决定是否投入 R3 的唯一入口。判定本身由 `tools/verdict.py` 机械计算
（输入上面三个 JSON，输出 `results/a100_verdict.json` + `.md`），阈值与分支由
`tests/test_verdict.py` 锁定，因此结论可复现、不依赖人工读日志。

## 报告要求

每个 A100/V100 报告至少包含：

1. correctness 是否通过；
2. backend_requested 与 backend_actual；
3. TTFT、TPOT、ITL、E2E；
4. prefill/decode/output tok/s；
5. peak memory；
6. fallback 原因；
7. A/B 的 absolute delta、improvement percentage 和 speedup；
8. Nsight 采集的 kernel launches/token 和内存系统指标；
9. **每步墙钟 / 每步 GPU / host 残差 / GPU 占比**（`tools.account_decode`）；
10. **相对 `roofline.py` 权重下界的差距**（`tools.account_decode` 的 comparison 表
    自动产出；与物理规格冲突时报告必须显式标注，而不是当成功）。

