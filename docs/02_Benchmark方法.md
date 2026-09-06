# Benchmark 方法与运行流程

本项目将 benchmark 分为三类：标准 baseline、可控性能 workload 和真实语义 workload。所有 CUDA 命令都在 A100/V100 服务器执行，不在本地 Mac 执行。

## 目录与环境

从仓库根目录执行：

```bash
bash setup_server.sh
conda activate nano-vllm-server
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

## 报告要求

每个 A100/V100 报告至少包含：

1. correctness 是否通过；
2. backend_requested 与 backend_actual；
3. TTFT、TPOT、ITL、E2E；
4. prefill/decode/output tok/s；
5. peak memory；
6. fallback 原因；
7. A/B 的 absolute delta、improvement percentage 和 speedup；
8. Nsight 采集的 kernel launches/token 和内存系统指标。
