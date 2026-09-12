"""Calibrate the decode roofline on the target GPU (A100).

`docs/07` §10 shows that the whole "can a megakernel beat CUDA Graph at large
batch" question reduces to two numbers:

* the machine's *attainable* HBM bandwidth, which sets the weight-streaming
  floor (1.503 GB/step for Qwen3-0.6B bf16), and
* the per-step *host* cost, which is what a persistent kernel can remove.

This tool measures both, plus the grid-wide barrier cost that the megakernel
pays every layer, so the roofline model can be fitted to the actual machine
instead of to the A100 spec sheet.  It is deliberately standalone: it does not
import the megakernel backend, so it can run before/independently of a build.

Sections (run all, or pick with ``--sections``):

``bandwidth``  vectorized read of a ~1.5 GB buffer, reporting achieved GB/s as
               a fraction of the device's spec bandwidth;
``weights``    the same read applied to the real checkpoint's layer + LM head
               tensors, which is the exact traffic a decode step must pay;
``host``       fixed host cost per decode step, measured with a small GEMM at
               batch 1/8/32 using CUDA events (the gap between the wall time
               and the kernel time is host);
``barrier``    grid-wide AtomicGridSync cost at the production grid size
               (default 80 blocks x 512 threads), matching ``kernel.cu``.

Example::

    python -m tools.bench_roofline_a100 --model ~/huggingface/Qwen3-0.6B \
        --json results/roofline_probe.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nanovllm.roofline import (  # noqa: E402
    HardwareSpec,
    ModelDims,
    RooflineModel,
)


def _require_cuda():  # pragma: no cover - GPU only
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device is required for this tool")
    return torch


def device_summary() -> dict:  # pragma: no cover - GPU only
    import torch

    props = torch.cuda.get_device_properties(0)
    spec_bw = props.memory_clock_rate * 1e3 * (props.memory_bus_width / 8) * 2 / 1e9
    return {
        "name": props.name,
        "sm_count": props.multi_processor_count,
        "total_memory_bytes": props.total_memory,
        "memory_clock_rate_khz": props.memory_clock_rate,
        "memory_bus_width_bits": props.memory_bus_width,
        "spec_bandwidth_gbps": spec_bw,
        "l2_cache_bytes": getattr(props, "L2_cache_size", None),
    }


def mean_of_runs(fn, runs: int, warmup: int) -> tuple[float, float, float]:  # pragma: no cover
    values: list[float] = []
    for _ in range(max(0, warmup)):
        fn()
    for _ in range(max(1, runs)):
        values.append(float(fn()))
    return statistics.fmean(values), min(values), max(values)


# --------------------------------------------------------------------------
# bandwidth
# --------------------------------------------------------------------------
def bench_bandwidth(  # pragma: no cover - GPU only
    target_bytes: float, runs: int, warmup: int
) -> dict:
    import torch

    torch.cuda.synchronize()
    elements = int(target_bytes // 2)  # bf16
    # One 1-D buffer; every read touches a distinct 128-bit sector.
    buf = torch.empty(elements, dtype=torch.bfloat16, device="cuda")
    buf.normal_(mean=0.0, std=0.05)

    def read_once() -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        # A full reduction over the buffer reads every byte exactly once.
        torch.sum(buf)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    mean_ms, min_ms, max_ms = mean_of_runs(read_once, runs, warmup)
    bytes_read = buf.numel() * buf.element_size()
    del buf
    torch.cuda.empty_cache()
    return {
        "buffer_bytes": bytes_read,
        "mean_ms": mean_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "mean_gbps": bytes_read / (mean_ms / 1000.0) / 1e9,
        "best_gbps": bytes_read / (min_ms / 1000.0) / 1e9,
    }


# --------------------------------------------------------------------------
# real checkpoint weights
# --------------------------------------------------------------------------
def _load_weight_tensors(model_path: str):  # pragma: no cover - GPU only
    import torch
    from safetensors import safe_open

    index_path = Path(model_path) / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = sorted(set(index["weight_map"].values()))
    else:
        shards = [path.name for path in sorted(Path(model_path).glob("*.safetensors"))]

    tensors = []
    for shard in shards:
        with safe_open(Path(model_path) / shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.endswith("embed_tokens.weight"):
                    # A decode step only gathers a row, not the whole table.
                    continue
                tensor = handle.get_tensor(key)
                if tensor.dtype in (torch.bfloat16, torch.float16) and tensor.dim() == 2:
                    tensors.append(tensor.to("cuda"))
    return tensors


def bench_weight_stream(  # pragma: no cover - GPU only
    model_path: str, runs: int, warmup: int
) -> dict:
    import torch

    try:
        tensors = _load_weight_tensors(model_path)
    except Exception as exc:  # pragma: no cover - depends on local files
        return {"error": f"could not load weights: {exc}"}

    total_bytes = sum(t.numel() * t.element_size() for t in tensors)

    def read_once() -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for tensor in tensors:
            torch.sum(tensor)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    mean_ms, min_ms, max_ms = mean_of_runs(read_once, runs, warmup)
    summary = {
        "tensor_count": len(tensors),
        "weight_bytes": total_bytes,
        "mean_ms": mean_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "mean_gbps": total_bytes / (mean_ms / 1000.0) / 1e9,
        "best_gbps": total_bytes / (min_ms / 1000.0) / 1e9,
    }
    del tensors
    torch.cuda.empty_cache()
    return summary


# --------------------------------------------------------------------------
# host cost per step
# --------------------------------------------------------------------------
def bench_host_step(  # pragma: no cover - GPU only
    dims: ModelDims, batches: list[int], runs: int, warmup: int
) -> list[dict]:
    import torch

    results: list[dict] = []
    weight = torch.randn(
        dims.hidden_size, dims.hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    for batch in batches:
        activation = torch.randn(
            batch, dims.hidden_size, dtype=torch.bfloat16, device="cuda"
        )

        def run() -> tuple[float, float]:
            wall_start = torch.cuda.Event(enable_timing=True)
            wall_end = torch.cuda.Event(enable_timing=True)
            kernel_start = torch.cuda.Event(enable_timing=True)
            kernel_end = torch.cuda.Event(enable_timing=True)
            wall_start.record()
            kernel_start.record()
            for _ in range(4):  # mimic a short chain of dependent kernels
                activation = (activation @ weight).contiguous()
            kernel_end.record()
            wall_end.record()
            torch.cuda.synchronize()
            return (
                wall_start.elapsed_time(wall_end),
                kernel_start.elapsed_time(kernel_end),
            )

        for _ in range(max(0, warmup)):
            run()
        walls, kernels = [], []
        for _ in range(max(1, runs)):
            wall, kernel = run()
            walls.append(wall)
            kernels.append(kernel)
        wall_ms = statistics.fmean(walls)
        kernel_ms = statistics.fmean(kernels)
        results.append(
            {
                "batch_size": batch,
                "wall_ms": wall_ms,
                "kernel_ms": kernel_ms,
                "host_residual_ms": wall_ms - kernel_ms,
                "kernel_fraction": kernel_ms / wall_ms if wall_ms else 0.0,
            }
        )
        del activation
    del weight
    torch.cuda.empty_cache()
    return results


# --------------------------------------------------------------------------
# barrier
# --------------------------------------------------------------------------
_BARRIER_SOURCE = r"""
#include <cuda_runtime.h>

// Structurally identical to the production AtomicGridSync in kernel.cu:
//  - acquire/release fences at GPU scope (fence.acq_rel.gpu), not __threadfence,
//    because __threadfence only orders the executing thread's accesses and the
//    production barrier relies on the acquire/release pair to publish WMMA
//    store_matrix_sync output across SMs;
//  - the *local* generation is captured before arriving and compared with `<=`,
//    which tolerates a fast peer advancing the generation more than once
//    between a waiter's read and its comparison (the `==` form can miss an
//    increment entirely).
// Keep this in sync with kernel.cu; tests/test_probe_fidelity.py guards it.
__device__ __forceinline__ void atomic_grid_sync(
    unsigned int *counter, unsigned int *generation, unsigned int nblocks,
    unsigned int local_gen) {
  __syncthreads();
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
  if (threadIdx.x == 0) {
    unsigned int arrived = atomicAdd(counter, 1u);
    if (arrived == nblocks - 1) {
      *counter = 0u;
      asm volatile("fence.acq_rel.gpu;" ::: "memory");
      atomicAdd(generation, 1u);
    } else {
      volatile unsigned int *vgen = (volatile unsigned int *)generation;
      while (*vgen <= local_gen) {
      }
      asm volatile("fence.acq_rel.gpu;" ::: "memory");
    }
  }
  __syncthreads();
}

// One block per SM, matching the production decode kernel's occupancy contract
// (kernel.cu declares __launch_bounds__(LDG_BLOCK_SIZE, 1)), so the cooperative
// limit and the register pressure match production.
__global__ void __launch_bounds__(512, 1) barrier_probe(
    unsigned int *counter, unsigned int *generation, unsigned int nblocks,
    int iterations) {
  unsigned int local_gen = 0u;
  for (int i = 0; i < iterations; i++) {
    atomic_grid_sync(counter, generation, nblocks, local_gen);
    local_gen += 1u;
  }
}

extern "C" double run_barrier_probe(int grid_blocks, int block_size,
                                    int iterations) {
  if (grid_blocks <= 0 || block_size <= 0 || iterations <= 0) return -1.0;
  unsigned int *counter = nullptr, *generation = nullptr;
  if (cudaMalloc(&counter, sizeof(unsigned int)) != cudaSuccess) return -2.0;
  if (cudaMalloc(&generation, sizeof(unsigned int)) != cudaSuccess) {
    cudaFree(counter);
    return -3.0;
  }
  cudaMemset(counter, 0, sizeof(unsigned int));
  cudaMemset(generation, 0, sizeof(unsigned int));

  void *args[] = {&counter, &generation, (void *)&grid_blocks, &iterations};
  dim3 grid(grid_blocks), block(block_size);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  cudaError_t err = cudaLaunchCooperativeKernel(
      (void *)barrier_probe, grid, block, args, 0, nullptr);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaFree(counter);
  cudaFree(generation);
  if (err != cudaSuccess) return -4.0;
  return (double)ms;
}
"""


def bench_barrier(  # pragma: no cover - GPU only
    grid_blocks: int,
    block_size: int,
    iterations: int,
    warmup: int,
    runs: int,
    *,
    module: Any = None,
    torch_module: Any = None,
) -> dict:
    """Time the grid-wide barrier at the production grid shape.

    ``module`` and ``torch_module`` are injection seams: on the server they are
    built here, but tests pass fakes so the measurement loop, error handling and
    statistics are exercised without a GPU.
    """

    torch = torch_module
    if torch is None:
        import torch as _torch

        torch = _torch
    preflight = None
    if module is None:
        # Fail before compiling if the cooperative grid is not launchable.
        from tools.bench_burst_lower_bound import cooperative_preflight

        preflight = cooperative_preflight(
            block_size, grid_blocks, torch_module=torch
        )
        if not preflight["ok"]:
            raise RuntimeError(f"cooperative preflight failed: {preflight['error']}")
        from torch.utils.cpp_extension import load_inline

        module = load_inline(
            name="ldg_barrier_probe",
            cpp_sources="double run_barrier_probe(int, int, int);",
            cuda_sources=_BARRIER_SOURCE,
            functions=["run_barrier_probe"],
            extra_cuda_cflags=["-O3", "-arch=sm_80"],
            verbose=False,
        )

    def run() -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        kernel_ms = module.run_barrier_probe(grid_blocks, block_size, iterations)
        end.record()
        torch.cuda.synchronize()
        if kernel_ms < 0:
            raise RuntimeError(f"barrier probe failed with cuda status {kernel_ms}")
        # Prefer the host-side event delta; the in-kernel value is a check.
        return float(start.elapsed_time(end))

    samples = []
    for _ in range(max(0, warmup)):
        run()
    for _ in range(max(1, runs)):
        samples.append(run())
    mean_ms = statistics.fmean(samples)
    return {
        "grid_blocks": grid_blocks,
        "block_size": block_size,
        "iterations": iterations,
        "mean_ms": mean_ms,
        "us_per_barrier": mean_ms * 1000.0 / iterations,
        "samples_ms": samples,
        "warmup_runs": max(0, warmup),
        "preflight": preflight,
    }


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def build_report(args) -> dict:  # pragma: no cover - GPU only
    dims = ModelDims()
    sections = set(args.sections.split(","))
    report: dict = {"sections": sorted(sections)}

    if "bandwidth" in sections or "weights" in sections:
        report["device"] = device_summary()
        spec_bw = report["device"]["spec_bandwidth_gbps"]
        report["hardware"] = {"name": report["device"]["name"], "spec_bandwidth_gbps": spec_bw}

    if "bandwidth" in sections:
        report["bandwidth"] = bench_bandwidth(
            args.target_bytes, args.runs, args.warmup
        )
        if report["device"]["spec_bandwidth_gbps"] > 0:
            report["bandwidth"]["spec_fraction"] = (
                report["bandwidth"]["best_gbps"]
                / report["device"]["spec_bandwidth_gbps"]
            )

    if "weights" in sections and args.model:
        report["weights"] = bench_weight_stream(args.model, args.runs, args.warmup)
        weight_bytes = report["weights"].get("weight_bytes")
        if weight_bytes:
            model = RooflineModel(dims, HardwareSpec())
            report["weights"]["weight_roofline_bytes"] = model.weight_bytes()
            report["weights"]["weight_bytes_ratio_vs_model"] = (
                weight_bytes / model.weight_bytes()
            )

    if "host" in sections:
        report["host"] = bench_host_step(
            dims, args.batches, args.runs, args.warmup
        )

    if "barrier" in sections:
        report["barrier"] = bench_barrier(
            args.grid_blocks,
            args.block_size,
            args.barrier_iterations,
            args.warmup,
            args.runs,
        )
        # Per-step barrier budget used by the roofline model: 28 layers x 12.
        if report["barrier"]["us_per_barrier"] > 0:
            report["barrier"]["ms_per_step_estimate"] = (
                report["barrier"]["us_per_barrier"] * 28 * 12 / 1000.0
            )

    if "bandwidth" in sections and report.get("bandwidth", {}).get("best_gbps"):
        measured = report["bandwidth"]["best_gbps"]
        model = RooflineModel(dims, HardwareSpec(bandwidth_gbps=measured))
        report["roofline_with_measured_bandwidth"] = {
            "bandwidth_gbps": measured,
            "weight_bytes": model.weight_bytes(),
            "weight_floor_ms": model.weight_bytes() / (measured * 1e9) * 1000.0,
            "b1_attainable_tok_s": model.attainable_tok_s(1, args.seq_len),
            "b8_attainable_tok_s": model.attainable_tok_s(8, args.seq_len),
            "compute_bound_batch": model.compute_bound_batch(args.seq_len),
        }
    return report


def parse_batches(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=None, help="local Qwen3-0.6B dir")
    parser.add_argument(
        "--sections",
        default="bandwidth,weights,host,barrier",
        help="comma list of bandwidth,weights,host,barrier",
    )
    parser.add_argument("--target-bytes", type=float, default=1.503e9)
    parser.add_argument("--batches", type=parse_batches, default=[1, 8, 32])
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=224)
    parser.add_argument("--grid-blocks", type=int, default=80)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--barrier-iterations", type=int, default=336)
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args(argv)

    _require_cuda()
    import torch

    if args.barrier_iterations < 1:
        parser.error("--barrier-iterations must be >= 1")

    report = build_report(args)
    print(json.dumps(report, indent=2))
    if args.json_path:
        Path(args.json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_path).write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(f"wrote {args.json_path}", file=sys.stderr)
    del torch
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
