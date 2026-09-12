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
from typing import Any

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

// Structurally identical to the production AtomicGridSync in kernel.cu:
//  - acquire/release fences at GPU scope (fence.acq_rel.gpu), not __threadfence,
//    because the production barrier relies on the acquire/release pair to
//    publish cross-SM writes;
//  - the local generation is captured before arriving and compared with `<=`,
//    which tolerates a fast peer advancing it more than once (the `==` form can
//    miss an increment).
// The probe does no arithmetic and writes nothing cross-SM, but the barrier
// cost must match production or the measured floor is not the real one.
// Keep in sync with kernel.cu; tests/test_probe_fidelity.py guards it.
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

// Stream exactly `bytes` starting at `base`, 16 bytes per thread per step.
// Never runs past `base + bytes`: the loop tests the *end* of each vector
// read, and the trailing partial vector (if any) is covered separately.  This
// matters because the probe deliberately points `base` at interior offsets to
// wrap the weight buffer.
__device__ __forceinline__ float stream_exact(
    const char *__restrict__ base, unsigned long long bytes, int block_id,
    int nblocks, int block_size) {
  const unsigned long long stride = (unsigned long long)block_size * 16ull;
  const unsigned long long start =
      (unsigned long long)(block_id * block_size + threadIdx.x) * 16ull;
  const unsigned long long grid_step = (unsigned long long)nblocks * stride;
  float sink = 0.0f;
  for (unsigned long long off = start; off + 16ull <= bytes; off += grid_step) {
    const uint4 raw = *reinterpret_cast<const uint4 *>(base + off);
    sink += (float)(raw.x ^ raw.y ^ raw.z ^ raw.w);
  }
  // Cover the tail beyond the last full vector with a scalar read.
  if (bytes % 16ull != 0ull) {
    const unsigned long long tail_start = bytes - (bytes % 16ull);
    const unsigned long long tail_grid = (unsigned long long)nblocks * block_size;
    for (unsigned long long i = (unsigned long long)(block_id * block_size + threadIdx.x);
         i < bytes - tail_start; i += tail_grid) {
      sink += (float)(unsigned char)base[tail_start + i];
    }
  }
  return sink;
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
  // Largest 16B-aligned start that still lets a full bytes_per_layer read fit
  // inside the buffer.  Decode reserves a 256-token KV block, so the real
  // weight footprint is always strictly smaller than the buffer.
  const unsigned long long max_base_offset =
      buffer_bytes > bytes_per_layer ? buffer_bytes - bytes_per_layer : 0ull;
  unsigned long long base_offset = 0ull;
  float sink = 0.0f;
  unsigned int local_gen = 0u;
  const unsigned long long step_clock_start = clock64();
  for (int iter = 0; iter < iterations; ++iter) {
    for (int layer = 0; layer < nlayers; ++layer) {
      const char *base = weights + base_offset;
      sink += stream_exact(base, bytes_per_layer, block_id, nblocks, block_size);
      base_offset += bytes_per_layer;
      if (base_offset > max_base_offset) base_offset = 0ull;
      for (int b = 0; b < barriers_per_layer; ++b) {
        atomic_grid_sync(counter, sync_generation, (unsigned int)nblocks,
                         local_gen);
        local_gen += 1u;
      }
    }
  }
  const unsigned long long elapsed = clock64() - step_clock_start;
  // One global write per block: keeps `sink` alive, costs nothing measurable.
  if (threadIdx.x == 0) {
    out[block_id] = (unsigned long long)(sink * 1e-30f) + elapsed;
  }
}

// Returns the per-step time in ms on success, a negative code on failure.
// Returning a double (rather than a double* out-param) keeps the C++ shim
// trivially bindable: torch's load_inline has no converter for `double*`.
extern "C" double run_lower_bound_probe(
    const void *weights, unsigned long long bytes_per_layer,
    unsigned long long buffer_bytes, int nlayers, int barriers_per_layer,
    int grid_blocks, int block_size, int iterations) {
  if (weights == nullptr || nlayers <= 0 || grid_blocks <= 0 ||
      buffer_bytes == 0 || iterations <= 0)
    return -1.0;
  unsigned int *counter = nullptr, *generation = nullptr;
  unsigned long long *out = nullptr;
  if (cudaMalloc(&counter, sizeof(unsigned int)) != cudaSuccess) return -2.0;
  if (cudaMalloc(&generation, sizeof(unsigned int)) != cudaSuccess) {
    cudaFree(counter);
    return -3.0;
  }
  if (cudaMalloc(&out, sizeof(unsigned long long) * grid_blocks) != cudaSuccess) {
    cudaFree(counter);
    cudaFree(generation);
    return -4.0;
  }
  cudaMemset(counter, 0, sizeof(unsigned int));
  cudaMemset(generation, 0, sizeof(unsigned int));
  cudaMemset(out, 0, sizeof(unsigned long long) * grid_blocks);

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
  if (err != cudaSuccess) return -5.0;
  return (double)ms / (double)iterations;
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
    *,
    module: Any = None,
    weights: tuple[Any, int] | None = None,
) -> dict:
    """Measure the device-side lower bound.

    ``module`` and ``weights`` are injection seams so the launch loop, the
    buffer-size guard and the failure handling can be tested without a GPU.
    """

    torch = None
    if weights is None:
        import torch as _torch

        torch = _torch
        preflight = cooperative_preflight(
            block_size, grid_blocks, torch_module=torch
        )
        if not preflight["ok"]:
            raise RuntimeError(f"cooperative preflight failed: {preflight['error']}")
        flat, total_bytes = _weight_layout(model_path, dims)
    else:
        preflight = None
        flat, total_bytes = weights
    bytes_per_layer = (total_bytes // dims.num_layers) & ~15  # 16B aligned
    buffer_bytes = flat.numel() * flat.element_size()
    # Each layer must start at a distinct buffer offset, otherwise the probe
    # would re-read the same addresses 28x and report L2-hit bandwidth instead
    # of HBM bandwidth.  For Qwen3-0.6B the weight buffer is 0.88 GB against a
    # 40 MB L2, so this holds comfortably -- but fail loudly rather than
    # produce a meaningless floor if a future model breaks the assumption.
    if buffer_bytes <= bytes_per_layer:
        raise RuntimeError(
            f"probe needs buffer_bytes ({buffer_bytes}) > bytes_per_layer "
            f"({bytes_per_layer}) to stream distinct weights per layer"
        )
    if module is None:
        from torch.utils.cpp_extension import load_inline

        module = load_inline(
            name="ldg_lower_bound_probe",
            cpp_sources=(
                "double run_lower_bound_probe(const void*, unsigned long long, "
                "unsigned long long, int, int, int, int, int);"
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
            if ms < 0:
                raise RuntimeError(
                    f"lower-bound probe failed with cuda status {ms} "
                    f"(grid={grid_blocks} block={block_size})"
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
    # Real torch frees the weight buffer; the injected-weights path has no torch.
    if torch is not None:
        torch.cuda.empty_cache()
    return {"bytes_total": total_bytes, "preflight": preflight, "rows": results}


# --------------------------------------------------------------------------
# C. host-free graph replay
# --------------------------------------------------------------------------
def bench_host_free_graph(  # pragma: no cover - GPU only
    dims: ModelDims,
    batches: list[int],
    replays: int,
    warmup: int,
    runs: int,
    *,
    torch_module: Any = None,
) -> list[dict]:
    """Measure the host overlay: the same graph replayed with/without sync.

    ``torch_module`` is an injection seam for tests.  The result is the ground
    truth for the per-step host cost of the cudagraph path: the synced replay is
    the status quo (one host round trip per step) and the burst replay queues all
    replays back to back.
    """

    torch = torch_module
    if torch is None:
        import torch as _torch

        torch = _torch

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


def cooperative_preflight(
    block_size: int = 512,
    grid_blocks: int = 80,
    *,
    torch_module: Any = None,
) -> dict:
    """Check that the cooperative launch shape is actually launchable.

    ``cudaLaunchCooperativeKernel`` fails outright if the grid exceeds the
    device's cooperative limit, and the failure surfaces only after the
    extension has finished compiling.  On a shared server that is a wasted
    session, so the limit is checked up front and recorded in the report.

    The probe kernels carry ``__launch_bounds__(512, 1)`` -- one block per SM by
    construction -- so the launchable grid is the SM count.  ``cooperativeLaunch``
    is reported when the installed torch exposes it; it is absent on older
    builds, in which case the launch itself is the only check.
    """

    torch = torch_module
    if torch is None:
        import torch as _torch

        torch = _torch
    props = torch.cuda.get_device_properties(0)
    sm_count = int(props.multi_processor_count)
    cooperative = getattr(props, "cooperativeLaunch", None)
    max_grid = sm_count  # __launch_bounds__(block_size, 1)
    result = {
        "block_size": block_size,
        "requested_grid_blocks": grid_blocks,
        "sm_count": sm_count,
        "cooperative_launch_flag": cooperative,
        "blocks_per_sm": 1,
        "max_grid": max_grid,
        "ok": True,
        "error": None,
    }
    if cooperative is False:
        result["ok"] = False
        result["error"] = (
            "device reports cooperativeLaunch=False; the persistent-kernel "
            "comparison is not meaningful on this GPU"
        )
    elif grid_blocks > max_grid:
        result["ok"] = False
        result["error"] = (
            f"requested grid {grid_blocks} exceeds the cooperative limit "
            f"{max_grid} (1 block/SM x {sm_count} SMs); re-run with "
            f"--grid-blocks {max_grid}"
        )
    return result


def probe_read_offsets(
    size: int, block_size: int, nblocks: int, vector_bytes: int = 16
) -> dict[str, list[int]]:
    """Mirror :c:`stream_exact`'s access pattern for the whole grid.

    Returns ``{"vector": [...], "tail": [...]}``.  A ``vector`` offset is the
    start of a ``vector_bytes`` read and must satisfy
    ``offset + vector_bytes <= size``; a ``tail`` offset is a single byte and
    must satisfy ``offset < size``.  Block 16-byte reads are *not* issued into
    the trailing partial vector -- those bytes are covered by the scalar tail
    loop instead, exactly like the CUDA code.

    Kept in Python so a CPU test can prove the probe covers exactly ``size``
    bytes and never steps outside ``[0, size)`` -- the probe wraps the weight
    buffer, so an off-by-one here would read out of bounds on the GPU.
    """

    stride = block_size * vector_bytes
    grid_step = nblocks * stride
    vector_region = size - (size % vector_bytes)
    vector_offsets: list[int] = []
    for thread in range(block_size * nblocks):
        start = thread * vector_bytes
        if start + vector_bytes > vector_region:
            break
        vector_offsets.extend(range(start, vector_region, grid_step))
    tail_offsets: list[int] = []
    for thread in range(block_size * nblocks):
        offset = vector_region + thread
        if offset >= size:
            break
        tail_offsets.append(offset)
    return {"vector": sorted(vector_offsets), "tail": sorted(tail_offsets)}


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
