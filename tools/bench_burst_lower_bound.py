"""Measure the two bounds that decide the high-batch megakernel question.

`docs/07` §10.4 lists three cheap, falsifying experiments.  This tool covers A
and C:

**A. device-side lower bound probe.**
   A cooperative kernel that does *no math*: it streams every transformer
   weight through the memory system once per layer (exactly the traffic a real
   decode step must pay), and executes the same number of grid-wide barriers
   per layer as the production kernel.  Its per-step time is the floor a
   "perfect" fused megakernel could reach on this machine at this batch size.
   *Decision rule:* if the floor is <= ~1.5 ms/step the device side is
   recoverable and the R3 rewrite is worth attempting; if it is > ~3 ms/step
   the high-batch crossover is not achievable and the rewrite should be
   abandoned.

**C. host-free graph replay.**
   Capture one decode-shaped step into a CUDA Graph and replay it N times
   without any host round trip.  This measures how fast the cudagraph path
   would be if per-step host overhead were removed entirely, i.e. the ceiling
   that a persistent kernel is competing against.  It also makes the point
   that a CUDA Graph cannot queue N steps inside one graph.

Both sections are standalone (no megakernel build needed) so they can run
before any kernel work starts::

    python -m tools.bench_burst_lower_bound --model ~/huggingface/Qwen3-0.6B \
        --batches 1,8,32 --json results/lower_bound.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nanovllm.roofline import HardwareSpec, ModelDims, RooflineModel  # noqa: E402

# Grid-wide barrier count per layer used by the roofline/barrier budget model.
BARRIERS_PER_LAYER = 12


def _require_cuda():  # pragma: no cover - GPU only
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device is required for this tool")
    return torch


# --------------------------------------------------------------------------
# A. device-side lower bound probe (CUDA source kept inline for portability)
# --------------------------------------------------------------------------
_PROBE_SOURCE = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Grid-wide barrier identical in structure to the production AtomicGridSync.
__device__ __forceinline__ void atomic_grid_sync(
    unsigned int *counter, unsigned int *generation, unsigned int nblocks) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned int gen = *generation;
    unsigned int arrived = atomicAdd(counter, 1u);
    if (arrived == nblocks - 1) {
      *counter = 0u;
      __threadfence();
      atomicAdd(generation, 1u);
    } else {
      while (*generation == gen) { __nanosleep(64); }
    }
  }
  __syncthreads();
}

// One cooperative pass over `nlayers` layers' worth of weights, with
// `barriers_per_layer` grid syncs per layer and no arithmetic.
__global__ void __launch_bounds__(512, 1) lower_bound_probe(
    const char *__restrict__ weights, unsigned long long bytes_per_layer,
    unsigned long long buffer_bytes, int nlayers, int barriers_per_layer,
    unsigned int *counter, unsigned int *sync_generation,
    unsigned long long *out, int iterations) {
  const int block_id = blockIdx.x;
  const int nblocks = gridDim.x;
  const int block_size = blockDim.x;
  float sink = 0.0f;
  const unsigned long long step_clock_start = clock64();
  for (int iter = 0; iter < iterations; ++iter) {
    for (int layer = 0; layer < nlayers; ++layer) {
      // Wrap so that no read runs past the allocation: the same weight bytes
      // are streamed once per layer, which is exactly a decode step's traffic.
      const unsigned long long base_offset = (unsigned long long)layer * bytes_per_layer;
      const char *base = weights + (base_offset % buffer_bytes);
      unsigned long long remaining = bytes_per_layer;
      unsigned long long off = (unsigned long long)(block_id * block_size + threadIdx.x) * 16ull;
      const unsigned long long grid_step = (unsigned long long)nblocks * block_size * 16ull;
      for (; off + 16ull <= remaining; off += grid_step) {
        const uint4 raw = *reinterpret_cast<const uint4 *>(base + off);
        sink += (float)(raw.x ^ raw.y ^ raw.z ^ raw.w);
      }
      for (int b = 0; b < barriers_per_layer; ++b) {
        atomic_grid_sync(counter, sync_generation, (unsigned int)nblocks);
      }
    }
  }
  const unsigned long long elapsed = clock64() - step_clock_start;
  // One global write per block: keeps `sink` alive, costs nothing measurable.
  if (threadIdx.x == 0) {
    out[block_id] = (unsigned long long)(sink * 1e-30f) + elapsed;
  }
}

extern "C" int run_lower_bound_probe(
    const void *weights, unsigned long long bytes_per_layer,
    unsigned long long buffer_bytes, int nlayers, int barriers_per_layer,
    int grid_blocks, int block_size, int iterations, double *ms_per_step_out) {
  if (weights == nullptr || nlayers <= 0 || grid_blocks <= 0 || buffer_bytes == 0)
    return 1;
  unsigned int *counter = nullptr, *generation = nullptr;
  unsigned long long *out = nullptr;
  if (cudaMalloc(&counter, sizeof(unsigned int)) != cudaSuccess) return 2;
  if (cudaMalloc(&generation, sizeof(unsigned int)) != cudaSuccess) return 3;
  if (cudaMalloc(&out, sizeof(unsigned long long) * grid_blocks) != cudaSuccess)
    return 4;
  cudaMemset(counter, 0, sizeof(unsigned int));
  cudaMemset(generation, 0, sizeof(unsigned int));

  void *args[] = {(void *)&weights, &bytes_per_layer, &buffer_bytes, &nlayers,
                  &barriers_per_layer, &counter, &generation, &out,
                  &iterations};
  dim3 grid(grid_blocks), block(block_size);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  cudaError_t err = cudaLaunchCooperativeKernel(
      (void *)lower_bound_probe, grid, block, args, 0, nullptr);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaFree(counter);
  cudaFree(generation);
  cudaFree(out);
  if (err != cudaSuccess) return 5;
  *ms_per_step_out = (double)ms / (double)iterations;
  return 0;
}
"""


def _weight_layout(model_path: str, dims: ModelDims):  # pragma: no cover - GPU only
    """Return ``(flat_weight_buffer, total_2d_weight_bytes)``.

    Only the 2-D projection weights are streamed (the embed table is a gather
    and the 1-D norms are negligible); every layer has the same parameter set,
    so the buffer is streamed once per layer by wrapping.
    """

    import torch
    from safetensors import safe_open

    index_path = Path(model_path) / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = sorted(set(index["weight_map"].values()))
    else:
        shards = [path.name for path in sorted(Path(model_path).glob("*.safetensors"))]

    buffers = []
    total_bytes = 0
    for shard in shards:
        with safe_open(Path(model_path) / shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if "embed_tokens" in key or "lm_head" in key:
                    continue
                tensor = handle.get_tensor(key)
                if tensor.dtype in (torch.bfloat16, torch.float16) and tensor.dim() == 2:
                    buffers.append(tensor.to("cuda").contiguous().view(-1))
                    total_bytes += tensor.numel() * tensor.element_size()
    if not buffers:
        raise RuntimeError("no 2-D weight tensors found for the probe")
    return torch.cat(buffers), total_bytes


def bench_lower_bound(  # pragma: no cover - GPU only
    model_path: str,
    dims: ModelDims,
    batches: list[int],
    grid_blocks: int,
    block_size: int,
    iterations: int,
    warmup: int,
    runs: int,
) -> dict:
    import torch
    from torch.utils.cpp_extension import load_inline

    flat, total_bytes = _weight_layout(model_path, dims)
    bytes_per_layer = (total_bytes // dims.num_layers) & ~15  # 16B aligned
    buffer_bytes = flat.numel() * flat.element_size()
    module = load_inline(
        name="ldg_lower_bound_probe",
        cpp_sources=(
            "int run_lower_bound_probe(const void*, unsigned long long, "
            "unsigned long long, int, int, int, int, int, double*);"
        ),
        cuda_sources=_PROBE_SOURCE,
        functions=["run_lower_bound_probe"],
        extra_cuda_cflags=["-O3", "-arch=sm_80"],
        verbose=False,
    )

    results: list[dict] = []
    for batch in batches:
        # Batch does not change the weight traffic (weights are shared); the
        # probe is run per batch so the report is explicit about it.
        samples = []
        for _ in range(max(0, warmup)):
            module.run_lower_bound_probe(
                flat.data_ptr(), bytes_per_layer, buffer_bytes, dims.num_layers,
                BARRIERS_PER_LAYER, grid_blocks, block_size, 1,
            )
        for _ in range(max(1, runs)):
            ms = module.run_lower_bound_probe(
                flat.data_ptr(), bytes_per_layer, buffer_bytes, dims.num_layers,
                BARRIERS_PER_LAYER, grid_blocks, block_size, iterations,
            )
            samples.append(float(ms))
        mean_ms = statistics.fmean(samples)
        results.append(
            {
                "batch_size": batch,
                "layers": dims.num_layers,
                "barriers_per_layer": BARRIERS_PER_LAYER,
                "grid_blocks": grid_blocks,
                "block_size": block_size,
                "iterations": iterations,
                "bytes_per_layer": bytes_per_layer,
                "ms_per_step_mean": mean_ms,
                "ms_per_step_min": min(samples),
                "ms_per_step_samples": samples,
                "weight_stream_gbps": total_bytes / (mean_ms / 1000.0) / 1e9,
                "verdict": (
                    "recoverable (R3 worth attempting)"
                    if mean_ms <= 1.5
                    else "borderline"
                    if mean_ms <= 3.0
                    else "NOT achievable (stop the rewrite)"
                ),
            }
        )
    del flat
    torch.cuda.empty_cache()
    return {"bytes_total": total_bytes, "rows": results}


# --------------------------------------------------------------------------
# C. host-free graph replay
# --------------------------------------------------------------------------
def bench_host_free_graph(  # pragma: no cover - GPU only
    dims: ModelDims, batches: list[int], replays: int, warmup: int, runs: int
) -> list[dict]:
    import torch

    results: list[dict] = []
    for batch in batches:
        weights = [
            torch.randn(
                dims.hidden_size, dims.hidden_size, dtype=torch.bfloat16, device="cuda"
            )
            for _ in range(4)
        ]
        activation = torch.randn(
            batch, dims.hidden_size, dtype=torch.bfloat16, device="cuda"
        )

        def decode_step() -> None:
            out = activation
            for weight in weights:
                out = (out @ weight).contiguous()
            activation.copy_(out)

        # Warm up outside capture, then capture one step.
        for _ in range(max(1, warmup)):
            decode_step()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            decode_step()
        torch.cuda.synchronize()

        # Host-inclusive: synchronize after every replay (the cudagraph status
        # quo: one host round trip per step).
        def replay_synced() -> float:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(replays):
                graph.replay()
                torch.cuda.synchronize()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / replays

        # Host-free: queue all replays back to back, then synchronize once.
        def replay_burst() -> float:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(replays):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / replays

        synced = [replay_synced() for _ in range(max(1, runs))]
        burst = [replay_burst() for _ in range(max(1, runs))]
        results.append(
            {
                "batch_size": batch,
                "replays_per_measurement": replays,
                "ms_per_step_host_inclusive": statistics.fmean(synced),
                "ms_per_step_host_free": statistics.fmean(burst),
                "host_per_step_ms": statistics.fmean(synced) - statistics.fmean(burst),
            }
        )
        del weights, activation
    torch.cuda.empty_cache()
    return results


# --------------------------------------------------------------------------
def build_report(args) -> dict:  # pragma: no cover - GPU only
    dims = ModelDims()
    sections = {part.strip() for part in args.sections.split(",") if part.strip()}
    report: dict = {"sections": sorted(sections), "dims": dims.__dict__}
    if "probe" in sections and args.model:
        report["lower_bound_probe"] = bench_lower_bound(
            args.model, dims, args.batches, args.grid_blocks, args.block_size,
            args.iterations, args.warmup, args.runs,
        )
        probe = report["lower_bound_probe"]["rows"][0]
        model = RooflineModel(dims, HardwareSpec())
        report["roofline_reference"] = {
            "weight_bytes": model.weight_bytes(),
            "spec_bandwidth_gbps": model.hardware.effective_bandwidth_gbps(),
            "spec_weight_floor_ms": model.weight_bytes()
            / (model.hardware.effective_bandwidth_gbps() * 1e9)
            * 1000.0,
            "probe_ms_per_step": probe["ms_per_step_mean"],
            "barrier_budget_ms": probe["barriers_per_layer"]
            * dims.num_layers
            * 1.63e-3,
        }
    if "graph" in sections:
        report["host_free_graph"] = bench_host_free_graph(
            dims, args.batches, args.replays, args.warmup, args.runs
        )
    return report


def parse_batches(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=None, help="local Qwen3-0.6B dir")
    parser.add_argument(
        "--sections", default="probe,graph", help="comma list of probe,graph"
    )
    parser.add_argument("--batches", type=parse_batches, default=[1, 8, 32])
    parser.add_argument("--grid-blocks", type=int, default=80)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--replays", type=int, default=50)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args(argv)

    _require_cuda()
    import torch

    if args.iterations < 1 or args.replays < 1:
        parser.error("--iterations and --replays must be >= 1")
    if "probe" in args.sections and not args.model:
        parser.error("--model is required for the probe section")

    report = build_report(args)
    print(json.dumps(report, indent=2))
    if args.json_path:
        Path(args.json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.json_path}", file=sys.stderr)
    del torch
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
