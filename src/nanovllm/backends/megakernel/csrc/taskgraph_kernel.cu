/**
 * Task-Graph Megakernel for Qwen3-0.6B on A100 (sm_80).
 *
 * One persistent cooperative launch (grid = #SMs) executes the complete decode
 * forward pass. Each SM runs one resident CTA that walks a host-built per-SM
 * instruction stream (inst_buf). Task dependencies are expressed as atomic
 * uint32 counters in global memory (bar_*): producers __threadfence + atomicAdd,
 * consumers volatile-poll until >= target. No per-layer full-grid grid.sync().
 *
 * See docs/11_task_graph重写设计.md for the full design.
 *
 * Phase 0 (this file): skeleton — wait/arrive primitives, controller/worker
 * loop, dispatch switch with stub op bodies. Phase 1 fills the op bodies.
 *
 * Primary target: NVIDIA A100 (sm_80). WMMA m16n16k16; no TMA, no wgmma.
 */

#include <cstdint>
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "tg_abi.h"

using namespace tg;  // Bring TG_INST_WIDTH, TgOp, TgLaunchDesc, etc. into scope

// =============================================================================
// Model constants (Qwen3-0.6B) — mirrored from kernel.cu:25-32.
// =============================================================================

constexpr int TG_WARP_SIZE = 32;
constexpr int TG_HIDDEN = 1024;
constexpr int TG_INTERMEDIATE = 3072;
constexpr int TG_NUM_Q_HEADS = 16;
constexpr int TG_NUM_KV_HEADS = 8;
constexpr int TG_HEAD_DIM = 128;
constexpr int TG_Q_SIZE = TG_NUM_Q_HEADS * TG_HEAD_DIM;    // 2048
constexpr int TG_KV_SIZE = TG_NUM_KV_HEADS * TG_HEAD_DIM;  // 1024
constexpr int TG_VOCAB = 151936;
constexpr int TG_NUM_LAYERS = 28;
constexpr float TG_RMS_EPS = 1e-6f;

// =============================================================================
// Build-time configuration.
//
// These mirror the spirit of kernel.cu's LDG_* flags but are independent so
// the task-graph path can be tuned without disturbing the frozen batch=1
// kernel. build.py passes them as -D. The #ifndef fallbacks MUST equal the
// build.py defaults; test_taskgraph_abi.py enforces this (same pattern as
// test_batch_launch_guards.py).
//
// The grid size and the batch are **runtime** values (``desc.num_sms`` and
// ``desc.BS``), so there is deliberately no compile-time SM-count or batch
// ceiling here. ``TG_NUM_BLOCKS`` / ``TG_MAX_BATCH`` used to be defined and
// passed by build.py while nothing read them; a build flag that looks like it
// configures the launch geometry is worse than no flag. The Python-side batch
// ceiling (scratch/bar sizing) is ``TG_MAX_BATCH`` in
// ``backends/megakernel/taskgraph.py``.
// =============================================================================

#ifndef TG_BLOCK_SIZE
#define TG_BLOCK_SIZE 256  // 8 warps/CTA; occupancy=1 block/SM via __launch_bounds__
#endif

constexpr int TG_NUM_WARPS = TG_BLOCK_SIZE / TG_WARP_SIZE;        // 8
constexpr int TG_WORKER_WARPS = TG_NUM_WARPS - 1;                 // 7 (warp 0 is controller)

// =============================================================================
// Cross-SM barrier primitives (bar_wait / bar_arrive).
//
// The bar arrays are uint32 counters in GMEM. Producers atomicAdd, consumers
// volatile-poll until >= target. This replaces per-layer grid.sync() with
// fine-grained dependency signaling.
//
// sm_80 has no TMA and therefore no fence.proxy.async. We use fence.acq_rel.gpu
// (same primitive as AtomicGridSync in kernel.cu:108-145, proven on A100) for
// both release and acquire: it guarantees GPU-scope visibility of prior global
// stores before the atomicAdd that signals a producer, and visibility of the
// producer's stores to a consumer that has observed the counter cross the
// threshold.
//
// Correctness argument (mirrors docs/11 §4.2):
//  - Release: fence.acq_rel.gpu publishes this CTA's global writes to GPU
//    scope BEFORE atomicAdd. Any CTA that later observes the incremented
//    counter is guaranteed to see those writes.
//  - Acquire: the consumer's fence.acq_rel.gpu (after observing the counter
//    >= target) ensures subsequent loads see the producer's published stores.
//  - volatile poll prevents the counter read from being cached in registers.
//  - __nanosleep(20) throttles memory-traffic from busy-wait (cohere "DONT DELETE").
// =============================================================================

__device__ __forceinline__ void tg_fence_gpu() {
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
}

// Defined in the worker-sync / thread-index sections below; forward-declared
// because tg_wait/tg_arrive use them. tg_arrive must publish all worker warps'
// stores (barrier) and tg_wait picks the single polling warp (tg_wwarp).
__device__ __forceinline__ void tg_worker_sync_inner();
__device__ __forceinline__ int tg_wwarp();

// Bounded spin for the bar polls. A dependency that never arrives (a lost
// arrival, a corrupted counter) would otherwise spin forever and wedge the GPU
// until the process is killed — miserable to diagnose on a remote box. After
// this many nanosleeps we print the offending bar and trap, which surfaces as
// a CUDA error in seconds instead. The schedule itself is proven
// deadlock-free by test_taskgraph_abi's execution simulator, so this is a
// safety net, not an expected path. Set to 0 to disable.
#ifndef TG_WAIT_TIMEOUT_SPINS
#define TG_WAIT_TIMEOUT_SPINS 200000000  // ~4-8 s (nanosleep can overshoot)
#endif

__device__ __forceinline__ void tg_bar_spin(const volatile uint32_t* p,
                                            uint32_t target, int idx) {
#if TG_WAIT_TIMEOUT_SPINS > 0
  uint64_t spins = 0;
#endif
  while (*p < target) {
    __nanosleep(20);
#if TG_WAIT_TIMEOUT_SPINS > 0
    if (++spins > (uint64_t)TG_WAIT_TIMEOUT_SPINS) {
      printf("tg_wait timeout: bar[%d] target=%u observed=%u\n", idx, target,
             (unsigned)*p);
      __trap();
    }
#endif
  }
}

// Consumer: wait for bar[idx] >= target, then acquire.
//
// The poll must be a volatile load so the compiler does not hoist it out of the
// loop (a cached register read would spin forever). Once the counter crosses
// the threshold, fence.acq_rel.gpu orders the producer's published stores
// before our subsequent loads. ``target == 0`` means "no dependency" (used by
// layer 0, whose input is the embedding, not a previous RMSNorm).
//
// Only worker warp 0 polls; the other 6 block on the worker barrier. With 108
// resident CTAs, having all 224 threads spin would put ~24k threads on a
// handful of L2 counter lines and throttle the producers' atomicAdds. The
// acquire is published to the other warps by the barrier's release/acquire
// edges (warp 0's acquire happens-before its barrier arrival).
__device__ __forceinline__ void tg_wait(const uint32_t* bar, int idx,
                                        uint32_t target) {
  if (target == 0) {
    tg_fence_gpu();
    return;
  }
  if (tg_wwarp() == 0) {
    tg_bar_spin(bar + idx, target, idx);
    tg_fence_gpu();
  }
  tg_worker_sync_inner();
}

// Producer: signal bar[idx] += inc (after this CTA's global stores are visible).
//
// Two invariants, both load-bearing:
//  1. All worker warps must have finished their global stores before the
//     counter is bumped, so consumers that observe the bump see the data.
//     The worker barrier below provides that ordering.
//  2. Exactly ONE thread per CTA may bump the counter. The schedule's wait
//     targets count one arrival per instruction (see the ABI test's
//     produce/wait accounting), so a per-thread atomicAdd would multiply every
//     arrival by TG_NUM_WORKER_THREADS (224) and release every consumer 224x
//     early. This was invisible while tg_wait was a no-op and is fatal for
//     num_sms > 1.
//
// Mirrors cohere megakernel.cuh (worker_sync, then lane-0 arrive_cross_sm) and
// kernel.cu's AtomicGridSync (barrier, then a one-thread fence + atomicAdd).
__device__ __forceinline__ void tg_arrive(uint32_t* bar, int idx,
                                          uint32_t inc = 1) {
  tg_worker_sync_inner();
  // Release: publish this CTA's global writes to GPU scope before the signal.
  // The fence is per-thread, but it runs after the worker barrier, so the
  // other threads' stores are ordered before it via the barrier's release/
  // acquire edges (same argument as kernel.cu's AtomicGridSync comment).
  tg_fence_gpu();
  if (threadIdx.x == TG_WARP_SIZE) {
    atomicAdd(bar + idx, inc);
  }
}

// Wait on two counters simultaneously (ADD_RMSNORM waits on attn + ffn paths).
// Polling both in one loop lets the consumer wake as soon as the LAST of the
// two dependencies lands, instead of serializing the two waits. Same
// warp-0-polls + worker-barrier scheme as tg_wait.
__device__ __forceinline__ void tg_wait_all(const uint32_t* bar1, int idx1,
                                            uint32_t target1,
                                            const uint32_t* bar2, int idx2,
                                            uint32_t target2) {
  const bool need1 = target1 > 0;
  const bool need2 = target2 > 0;
  if (!need1 && !need2) {
    tg_fence_gpu();
    return;
  }
  if (tg_wwarp() == 0) {
    const volatile uint32_t* p1 = bar1 + idx1;
    const volatile uint32_t* p2 = bar2 + idx2;
#if TG_WAIT_TIMEOUT_SPINS > 0
    uint64_t spins = 0;
#endif
    while ((need1 && *p1 < target1) || (need2 && *p2 < target2)) {
      __nanosleep(20);
#if TG_WAIT_TIMEOUT_SPINS > 0
      if (++spins > (uint64_t)TG_WAIT_TIMEOUT_SPINS) {
        printf("tg_wait_all timeout: bar[%d] %u/%u | bar[%d] %u/%u\n", idx1,
               (unsigned)*p1, target1, idx2, (unsigned)*p2, target2);
        __trap();
      }
#endif
    }
    tg_fence_gpu();
  }
  tg_worker_sync_inner();
}

// =============================================================================
// Instruction helpers.
// =============================================================================

// Cooperative warp load: 32 lanes each read one int32, covering 32 words.
// TG_INST_WIDTH is 16, so lanes 0..15 load the instruction; lanes 16..31 idle.
__device__ __forceinline__ void tg_coop_load_inst(int* dst, const int* src) {
  if (threadIdx.x < TG_INST_WIDTH) {
    dst[threadIdx.x] = src[threadIdx.x];
  }
}

// Opcode accessor.
__device__ __forceinline__ int tg_opcode(const int* inst) { return inst[0]; }

// =============================================================================
// Worker sync (bar.sync 1 across worker warps only; controller warp 0 excluded).
//
// CRITICAL INVARIANT (cohere megakernel.cuh:515-519): every worker warp must
// execute the SAME number of bar.sync 1 calls per instruction, including the
// NOP branch. Otherwise arrival counts misalign and the CTA deadlocks. Each
// op body calls tg_worker_sync exactly once at its end; the dispatch loop
// calls it once after dispatch for the NOP/stub paths.
// =============================================================================

__device__ __forceinline__ void tg_worker_sync() {
  asm volatile("bar.sync 1, %0;" :: "r"(TG_WORKER_WARPS * TG_WARP_SIZE)
               : "memory");
}

// Worker-only sync for use INSIDE op bodies (between compute phases). Has the
// same semantics as tg_worker_sync but is called multiple times per op. The
// invariant is that ALL worker warps must call it the same number of times;
// since op bodies have no warp divergence, this is guaranteed.
__device__ __forceinline__ void tg_worker_sync_inner() {
  asm volatile("bar.sync 1, %0;" :: "r"(TG_WORKER_WARPS * TG_WARP_SIZE)
               : "memory");
}

// =============================================================================
// Worker thread index helpers.
//
// Warp 0 (threadIdx.x 0..31) is the controller and never enters op bodies.
// Worker warps are 1..7 (threadIdx.x 32..255). These helpers give a
// worker-local view so ported code from kernel.cu (which uses threadIdx.x
// starting at 0) works without per-line rewrites.
// =============================================================================

constexpr int TG_NUM_WORKER_THREADS = TG_WORKER_WARPS * TG_WARP_SIZE;  // 224

__device__ __forceinline__ int tg_wtid() { return threadIdx.x - TG_WARP_SIZE; }
__device__ __forceinline__ int tg_wwarp() { return tg_wtid() / TG_WARP_SIZE; }
__device__ __forceinline__ int tg_wlane() { return tg_wtid() % TG_WARP_SIZE; }

// =============================================================================
// Math helpers — ported from kernel.cu:155-207 (identical logic, tg_ prefix).
// =============================================================================

__device__ __forceinline__ float tg_warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = TG_WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_xor_sync(0xffffffff, val, offset);
  }
  return val;
}

constexpr float TG_LOG2E = 1.44269504088896340736f;

__device__ __forceinline__ float tg_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float tg_rcp(float x) {
  float y;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float tg_fast_exp(float x) {
  return tg_exp2(x * TG_LOG2E);
}

__device__ __forceinline__ float tg_silu(float x) {
  return x * tg_rcp(1.0f + tg_fast_exp(-x));
}

// PyTorch's BF16 path rounds at each tensor boundary. Keep accumulators in
// FP32 but model the BF16 store/load boundary before the next op.
__device__ __forceinline__ float tg_round_bf16(float x) {
  return __bfloat162float(__float2bfloat16(x));
}

// 128-bit L1-bypassing weight load (8 bf16 elements per call).
__device__ __forceinline__ uint4 tg_load_weight_u4(const uint4 *ptr) {
  uint4 out;
  asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w)
               : "l"(ptr));
  return out;
}

// =============================================================================
// Weight accessor — returns the per-layer weight pointer for a given
// TgLayerWeightIdx. The layer_weights array is [num_layers * TG_LW_COUNT]
// uint64_t pointers.
// =============================================================================

__device__ __forceinline__ const __nv_bfloat16* tg_layer_weight(
    const void* const* layer_weights, int layer, int weight_idx) {
  return reinterpret_cast<const __nv_bfloat16*>(
      layer_weights[layer * tg::TG_LW_COUNT + weight_idx]);
}

// =============================================================================
// Device-side globals wrapper (the kernel reads TgLaunchDesc through this).
// TgGlobals holds typed pointers derived from the uint64_t fields, so op bodies
// don't need to reinterpret_cast everywhere.
// =============================================================================

struct TgGlobals {
  const int* inst_buf;            // [num_sms, max_inst, TG_INST_WIDTH]
  const int* num_inst_per_sm;     // [num_sms]
  int max_inst;
  int num_sms;

  // weights
  const void* const* layer_weights;  // [num_layers * TG_LW_COUNT] -> ptr per weight
  const __nv_bfloat16* embed_weight;
  const __nv_bfloat16* final_norm_weight;
  const __nv_bfloat16* cos_table;
  const __nv_bfloat16* sin_table;

  // KV cache
  __nv_bfloat16* k_cache;
  __nv_bfloat16* v_cache;
  const int* page_table;
  const int* cache_seqlens;
  const int* row_active;
  int num_phys_pages;
  int max_pages_per_seq;
  int page_block_size;
  int64_t kv_layer_stride;

  // scratch — intermediates are float (matching kernel.cu's g_activations etc.)
  // x_resid is the only bf16 scratch (layer boundary, matching hidden_buffer).
  float* norm_out;          // [BS, hidden]    — RMSNorm output (QKV/FFN input)
  float* q_out;             // [BS, q_size]    — Q projection output
  float* k_out;             // [BS, kv_size]   — K projection output
  float* v_out;             // [BS, kv_size]   — V projection output
  float* attn_out;          // [BS, hidden]    — attention output
  float* o_proj_out;        // [BS, hidden]    — O projection output
  float* x_attn;            // [BS, hidden]    — residual after attention
  float* post_attn_norm;    // [BS, hidden]    — post-attn RMSNorm output
  float* silu_out;          // [BS, intermediate] — SiLU(gate)*up output
  float* ffn_down_out;      // [BS, hidden]    — FFN down output
  float* up_out;            // [16, intermediate] — temp up-projection output
  float* x_ffn;             // [BS, hidden]    — residual after FFN (= x_resid next)
  __nv_bfloat16* x_resid;   // [BS, hidden]    — layer input (bf16, layer boundary)
  float* lm_logits;         // [BS, vocab]     — LM head output
  float* o_partial;         // [BS, hkv, max_splits, hidden] — split attention partials
  float* lse_partial;       // [BS, hkv, max_splits]         — split attention LSE

  // bars
  uint32_t* bar_qkv;
  uint32_t* bar_attn;
  uint32_t* bar_oproj;
  uint32_t* bar_silu;
  uint32_t* bar_ffn_down;
  uint32_t* bar_layer;
  uint32_t* bar_combine;
  uint32_t* bar_lm_head;

  // geometry
  int num_layers;
  int BS;
  int hidden;
  int intermediate;
  int q_size;
  int kv_size;
  int num_q_heads;
  int num_kv_heads;
  int head_dim;
  int vocab_size;
  int oproj_splits;
  int down_splits;
  int max_attn_splits;
};

__device__ __forceinline__ TgGlobals tg_make_globals(const tg::TgLaunchDesc& d) {
  TgGlobals g;
  g.inst_buf = reinterpret_cast<const int*>(d.inst_buf);
  g.num_inst_per_sm = reinterpret_cast<const int*>(d.num_inst_per_sm);
  g.max_inst = d.max_inst;
  g.num_sms = d.num_sms;

  g.layer_weights = reinterpret_cast<const void* const*>(d.layer_weights);
  g.embed_weight = reinterpret_cast<const __nv_bfloat16*>(d.embed_weight);
  g.final_norm_weight =
      reinterpret_cast<const __nv_bfloat16*>(d.final_norm_weight);
  g.cos_table = reinterpret_cast<const __nv_bfloat16*>(d.cos_table);
  g.sin_table = reinterpret_cast<const __nv_bfloat16*>(d.sin_table);

  g.k_cache = reinterpret_cast<__nv_bfloat16*>(d.k_cache);
  g.v_cache = reinterpret_cast<__nv_bfloat16*>(d.v_cache);
  g.page_table = reinterpret_cast<const int*>(d.page_table);
  g.cache_seqlens = reinterpret_cast<const int*>(d.cache_seqlens);
  g.row_active = reinterpret_cast<const int*>(d.row_active);
  g.num_phys_pages = d.num_phys_pages;
  g.max_pages_per_seq = d.max_pages_per_seq;
  g.page_block_size = d.page_block_size;
  g.kv_layer_stride = d.kv_layer_stride;

  g.norm_out = reinterpret_cast<float*>(d.norm_out);
  g.q_out = reinterpret_cast<float*>(d.q_out);
  g.k_out = reinterpret_cast<float*>(d.k_out);
  g.v_out = reinterpret_cast<float*>(d.v_out);
  g.attn_out = reinterpret_cast<float*>(d.attn_out);
  g.o_proj_out = reinterpret_cast<float*>(d.o_proj_out);
  g.x_attn = reinterpret_cast<float*>(d.x_attn);
  g.post_attn_norm = reinterpret_cast<float*>(d.post_attn_norm);
  g.silu_out = reinterpret_cast<float*>(d.silu_out);
  g.ffn_down_out = reinterpret_cast<float*>(d.ffn_down_out);
  g.up_out = reinterpret_cast<float*>(d.up_out);
  g.x_ffn = reinterpret_cast<float*>(d.x_ffn);
  g.x_resid = reinterpret_cast<__nv_bfloat16*>(d.x_resid);
  g.lm_logits = reinterpret_cast<float*>(d.lm_logits);
  g.o_partial = reinterpret_cast<float*>(d.o_partial);
  g.lse_partial = reinterpret_cast<float*>(d.lse_partial);

  g.bar_qkv = reinterpret_cast<uint32_t*>(d.bars.bar_qkv);
  g.bar_attn = reinterpret_cast<uint32_t*>(d.bars.bar_attn);
  g.bar_oproj = reinterpret_cast<uint32_t*>(d.bars.bar_oproj);
  g.bar_silu = reinterpret_cast<uint32_t*>(d.bars.bar_silu);
  g.bar_ffn_down = reinterpret_cast<uint32_t*>(d.bars.bar_ffn_down);
  g.bar_layer = reinterpret_cast<uint32_t*>(d.bars.bar_layer);
  g.bar_combine = reinterpret_cast<uint32_t*>(d.bars.bar_combine);
  g.bar_lm_head = reinterpret_cast<uint32_t*>(d.bars.bar_lm_head);

  g.num_layers = d.num_layers;
  g.BS = d.BS;
  g.hidden = d.hidden;
  g.intermediate = d.intermediate;
  g.q_size = d.q_size;
  g.kv_size = d.kv_size;
  g.num_q_heads = d.num_q_heads;
  g.num_kv_heads = d.num_kv_heads;
  g.head_dim = d.head_dim;
  g.vocab_size = d.vocab_size;
  g.oproj_splits = d.oproj_splits;
  g.down_splits = d.down_splits;
  g.max_attn_splits = d.max_attn_splits;
  return g;
}

// =============================================================================
// KV cache offset (paged) — adapted from kernel.cu:360-376.
// =============================================================================

__device__ __forceinline__ int64_t tg_kv_offset(int position, int kv_head,
                                                int head_dim, int kv_size,
                                                int block_size,
                                                const int* block_table) {
  int logical_block = position / block_size;
  int block_offset = position % block_size;
  int physical_block = block_table[logical_block];
  return (int64_t)physical_block * block_size * kv_size +
         (int64_t)block_offset * kv_size + (int64_t)kv_head * head_dim;
}

// =============================================================================
// RMSNorm — ported from kernel.cu:228-283, adapted for 7 worker warps.
//
// Computes: out[i] = round_bf16(round_bf16(input[i] * rstd) * weight[i])
// where rstd = rsqrt(sum(input[i]^2) / hidden + eps).
//
// `smem` must be at least `hidden` floats (stages the input + normalized output).
// `smem_reduce` must be at least TG_WORKER_WARPS floats (block reduce).
// After return, `smem` contains the normalized output (float, already BF16-rounded).
// =============================================================================

// Variant A: bf16 input (for layer input from x_resid)
__device__ __forceinline__ void tg_rmsnorm(
    float* __restrict__ smem, float* __restrict__ smem_reduce,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ norm_weight,
    int hidden) {
  int wtid = tg_wtid();
  int wwarp = tg_wwarp();
  int wlane = tg_wlane();

  float local_sum_sq = 0.0f;
  for (int i = wtid; i < hidden; i += TG_NUM_WORKER_THREADS * 2) {
    float v0 = __bfloat162float(__ldg(input + i));
    smem[i] = v0;
    local_sum_sq += v0 * v0;
    int i1 = i + TG_NUM_WORKER_THREADS;
    if (i1 < hidden) {
      float v1 = __bfloat162float(__ldg(input + i1));
      smem[i1] = v1;
      local_sum_sq += v1 * v1;
    }
  }

  local_sum_sq = tg_warp_reduce_sum(local_sum_sq);
  if (wlane == 0) smem_reduce[wwarp] = local_sum_sq;
  tg_worker_sync_inner();

  if (wwarp == 0) {
    float sum = (wlane < TG_WORKER_WARPS) ? smem_reduce[wlane] : 0.0f;
    sum = tg_warp_reduce_sum(sum);
    if (wlane == 0)
      smem_reduce[0] = rsqrtf(sum / float(hidden) + TG_RMS_EPS);
  }
  tg_worker_sync_inner();

  float rstd = smem_reduce[0];

  for (int i = wtid; i < hidden; i += TG_NUM_WORKER_THREADS * 2) {
    float w0 = __bfloat162float(__ldg(norm_weight + i));
    smem[i] = tg_round_bf16(tg_round_bf16(smem[i] * rstd) * w0);
    int i1 = i + TG_NUM_WORKER_THREADS;
    if (i1 < hidden) {
      float w1 = __bfloat162float(__ldg(norm_weight + i1));
      smem[i1] = tg_round_bf16(tg_round_bf16(smem[i1] * rstd) * w1);
    }
  }
  tg_worker_sync_inner();
}

// Variant B: float input (for intermediate activations like x_attn, x_ffn)
// Reads from `input` (float*), writes normalized output to `smem` (float*).
__device__ __forceinline__ void tg_rmsnorm_f(
    float* __restrict__ smem, float* __restrict__ smem_reduce,
    const float* __restrict__ input,
    const __nv_bfloat16* __restrict__ norm_weight,
    int hidden) {
  int wtid = tg_wtid();
  int wwarp = tg_wwarp();
  int wlane = tg_wlane();

  float local_sum_sq = 0.0f;
  for (int i = wtid; i < hidden; i += TG_NUM_WORKER_THREADS * 2) {
    float v0 = input[i];
    smem[i] = v0;
    local_sum_sq += v0 * v0;
    int i1 = i + TG_NUM_WORKER_THREADS;
    if (i1 < hidden) {
      float v1 = input[i1];
      smem[i1] = v1;
      local_sum_sq += v1 * v1;
    }
  }

  local_sum_sq = tg_warp_reduce_sum(local_sum_sq);
  if (wlane == 0) smem_reduce[wwarp] = local_sum_sq;
  tg_worker_sync_inner();

  if (wwarp == 0) {
    float sum = (wlane < TG_WORKER_WARPS) ? smem_reduce[wlane] : 0.0f;
    sum = tg_warp_reduce_sum(sum);
    if (wlane == 0)
      smem_reduce[0] = rsqrtf(sum / float(hidden) + TG_RMS_EPS);
  }
  tg_worker_sync_inner();

  float rstd = smem_reduce[0];

  for (int i = wtid; i < hidden; i += TG_NUM_WORKER_THREADS * 2) {
    float w0 = __bfloat162float(__ldg(norm_weight + i));
    smem[i] = tg_round_bf16(tg_round_bf16(smem[i] * rstd) * w0);
    int i1 = i + TG_NUM_WORKER_THREADS;
    if (i1 < hidden) {
      float w1 = __bfloat162float(__ldg(norm_weight + i1));
      smem[i1] = tg_round_bf16(tg_round_bf16(smem[i1] * rstd) * w1);
    }
  }
  tg_worker_sync_inner();
}

// =============================================================================
// cp.async weight pipeline (sm_80 has no TMA).
//
// The GEMM is weight-bandwidth bound: per step the model streams ~1.2 GB of
// weights, and the LM head alone is 311 MB. A synchronous
// wmma::load_matrix_sync(b, W + ..., K) stalls the warp for the full global
// latency every K step, so the SM can only keep a handful of loads in flight
// (~550 GB/s measured headroom). We stage the B tile through shared memory
// with cp.async and keep TG_B_STAGES copies in flight per warp.
//
// cohere does the same with TMA + `mbarrier.arrive.expect_tx`; on sm_80 the
// equivalent is `cp.async.cg` + `cp.async.commit_group` /
// `cp.async.wait_group`. `.cg` bypasses L1 (the weights have no reuse).
// =============================================================================

using namespace nvcuda;
constexpr int TG_WMMA_TILE = 16;
// In-flight weight stages per warp. TG_B_STAGES-1 real loads stay outstanding
// after each wait, so the SM keeps (STAGES-1) * active_warps * 512 B in flight.
//
// The K loop advances one stage per iteration and blocks in
// tg_cp_async_wait<STAGES-1>, so in the latency-bound steady state the loop
// rate is `latency / (STAGES-1)` per K-step: 3 stages deep gives ~200 ns per
// K-step for a 600 ns HBM latency, which is what capped the A100 measurement
// at ~4.4 GB/s per SM. 8 stages -> 7 deep -> ~86 ns, a ~2.3x deeper pipeline.
// The A100 allows at most 8 outstanding cp.async commit groups, so 8 stages
// (7 prologue groups + the one issued before each wait) is the maximum.
// Env-overridable (TG_B_STAGES=<n>) so the depth can be A/B'd on hardware
// without editing the source. build.py's default MUST equal this fallback
// (test_taskgraph_abi.py::BuildFlagSyncTests enforces it).
#ifndef TG_B_STAGES
#define TG_B_STAGES 8
#endif
// The prologue commits TG_B_STAGES-1 groups and one more is issued before each
// wait, so TG_B_STAGES groups are outstanding at the wait. 8 is the hardware
// maximum for cp.async commit groups.
static_assert(TG_B_STAGES >= 2 && TG_B_STAGES <= 8,
              "TG_B_STAGES must stay within the cp.async commit-group limit");

// Per-warp B staging. File-scope __shared__ so the op bodies do not have to
// thread another pointer through their signatures.
__shared__ __align__(16)
    __nv_bfloat16 tg_s_b[TG_WORKER_WARPS][TG_B_STAGES][TG_WMMA_TILE * TG_WMMA_TILE];

__device__ __forceinline__ void tg_cp_async_16(__nv_bfloat16* s_dst,
                                               const void* g_src) {
  uint32_t saddr = (uint32_t)__cvta_generic_to_shared(s_dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :: "r"(saddr), "l"(g_src));
}

__device__ __forceinline__ void tg_cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

// Wait until at most PENDING async groups are outstanding.
template <int PENDING>
__device__ __forceinline__ void tg_cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(PENDING) : "memory");
}

// Stage W[n0..n0+15, k..k+15] (row-major, row stride K) into s_dst as a
// row-major [16][16] tile. Each lane moves one 16-byte chunk (16 bf16 = 32 B
// per row = 2 chunks; 16 rows * 2 = 32 lanes).
__device__ __forceinline__ void tg_cp_async_b_tile(
    __nv_bfloat16* s_dst, const __nv_bfloat16* W_base, int K, int k) {
  int lane = tg_wlane();
  int r = lane >> 1;   // row 0..15
  int h = lane & 1;    // 16-byte half of the row
  tg_cp_async_16(s_dst + r * TG_WMMA_TILE + h * 8,
                 W_base + r * K + k + h * 8);
}

// =============================================================================
// WMMA GEMM — ported from kernel.cu:962-1012, adapted for 7 worker warps.
//
// Computes C[M,N] = H[M,K] @ W[N,K]^T (W is stored row-major as [N,K]).
// Each warp handles one 16x16 tile. `s_a` is a per-warp slice of shared memory
// for the A fragment (float→bf16 staging): the A data is the same for every
// warp (M≤16 so tiles_M=1), but staging it per-warp keeps the K loop free of
// CTA barriers.
//
// The B fragment is staged through tg_s_b with a TG_B_STAGES-deep cp.async
// pipeline so the weight load of K step ks+TG_B_STAGES-1 overlaps the A
// staging and mma of step ks.
//
// STRIDE_K: row stride of H (e.g. HIDDEN for QKV input).
// STRIDE_N: row stride of C (e.g. Q_SIZE for Q output).
// =============================================================================

template <int STRIDE_K, int STRIDE_N>
__device__ __forceinline__ void tg_wmma_gemm(
    const float* __restrict__ H,
    const __nv_bfloat16* __restrict__ W,
    float* __restrict__ C, int M, int N, int K,
    int global_warp, int num_warps,
    __nv_bfloat16* s_a) {
  const int tiles_M = (M + TG_WMMA_TILE - 1) / TG_WMMA_TILE;
  const int tiles_N = (N + TG_WMMA_TILE - 1) / TG_WMMA_TILE;
  const int total_tiles = tiles_M * tiles_N;
  // CRITICAL: all warps must execute the SAME number of bar.sync 1 calls.
  // Compute the max iterations any warp will do, and have all warps loop
  // that many times (idle warps skip work but still hit the syncs).
  const int my_max_tiles = (total_tiles + num_warps - 1) / num_warps;
  const int num_k = K / TG_WMMA_TILE;
  __nv_bfloat16* s_b = &tg_s_b[global_warp][0][0];
  __nv_bfloat16* my_a = s_a + global_warp * (TG_WMMA_TILE * TG_WMMA_TILE);

  for (int t = 0; t < my_max_tiles; ++t) {
    int tile = global_warp + t * num_warps;
    bool active = (tile < total_tiles);
    int tr = active ? (tile / tiles_N) : 0;
    int tc = active ? (tile % tiles_N) : 0;
    int m0 = tr * TG_WMMA_TILE;
    int n0 = tc * TG_WMMA_TILE;
    const __nv_bfloat16* w_base = W + n0 * K;

    wmma::fragment<wmma::accumulator, TG_WMMA_TILE, TG_WMMA_TILE,
                   TG_WMMA_TILE, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    // Prologue: fill the pipeline with the first TG_B_STAGES-1 K steps. An
    // empty commit_group is still a group, which keeps the group accounting
    // uniform for short K (and for idle warps).
    #pragma unroll
    for (int s = 0; s < TG_B_STAGES - 1; ++s) {
      if (active && s < num_k)
        tg_cp_async_b_tile(s_b + s * (TG_WMMA_TILE * TG_WMMA_TILE), w_base, K,
                           s * TG_WMMA_TILE);
      tg_cp_async_commit();
    }

    for (int ks = 0; ks < num_k; ++ks) {
      // Issue one stage ahead. Always commit so exactly TG_B_STAGES groups
      // are outstanding at the wait (wait<STAGES-1> then completes stage ks).
      const int ahead = ks + TG_B_STAGES - 1;
      if (active && ahead < num_k)
        tg_cp_async_b_tile(
            s_b + (ahead % TG_B_STAGES) * (TG_WMMA_TILE * TG_WMMA_TILE),
            w_base, K, ahead * TG_WMMA_TILE);
      tg_cp_async_commit();

      // Per-warp A staging: each warp converts its own A fragment into its own
      // slice of s_a. The A data is identical across warps (tiles_M=1), but a
      // shared copy needed two CTA barriers per K step, forcing all 7 warps
      // into lockstep; with K up to 3072 that overhead was comparable to the
      // weight traffic. Redundant staging is cheap (16x16 elements).
      //
      // Vectorized 4-wide: one lane moves a float4 (4 K values of one row) and
      // writes the four converted bf16 as a single 8-byte store — 2 loads + 2
      // stores per lane instead of 8 + 8. Every address the loop touches is
      // 16-byte aligned: H is a torch/static buffer, STRIDE_K and k are
      // multiples of 4 floats, and ki is in {0,4,8,12}. The guard cannot cut a
      // vector short because k + ki + 3 <= K - 1 for every ks < K/16.
      //
      // ORDER MATTERS: this runs BEFORE the wait below. The staging's loads are
      // then in flight while the warp blocks in cp_async_wait, so the two
      // latencies overlap instead of adding up. It only touches this warp's own
      // s_a slice, so it cannot race with the B pipeline.
      //
      // Both loads are issued before either is consumed, on purpose: written as
      // a two-iteration loop the compiler reuses the load registers and emits
      // LDS.128 -> F2FP -> STS.64 -> LDS.128, serialising the two latencies
      // (verified in SASS). The lane map is fixed:
      //   lane -> (row lane/4, cols k + 4*(lane%4) .. +3) and the same +8 rows,
      //   i.e. s_a[4*lane] and s_a[4*lane + 128].
      const int k = ks * TG_WMMA_TILE;
      const int lane = tg_wlane();
      const int ki = (lane % 4) * 4;      // 0, 4, 8 or 12
      const int row_lo = lane / 4;        // 0..7
      const int col = k + ki;
      float4 f_lo = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      float4 f_hi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (active && m0 + row_lo < M && col < K)
        f_lo = *reinterpret_cast<const float4*>(
            H + (m0 + row_lo) * STRIDE_K + col);
      if (active && m0 + row_lo + 8 < M && col < K)
        f_hi = *reinterpret_cast<const float4*>(
            H + (m0 + row_lo + 8) * STRIDE_K + col);
      const int idx_lo = lane * 4;        // == row_lo * 16 + ki
      const int idx_hi = idx_lo + 128;    // == (row_lo + 8) * 16 + ki
      #pragma unroll
      for (int half = 0; half < 2; ++half) {
        const float4 f = (half == 0) ? f_lo : f_hi;
        const __nv_bfloat162 lo = __floats2bfloat162_rn(f.x, f.y);
        const __nv_bfloat162 hi = __floats2bfloat162_rn(f.z, f.w);
        uint2 packed;
        packed.x = *reinterpret_cast<const uint32_t*>(&lo);
        packed.y = *reinterpret_cast<const uint32_t*>(&hi);
        *reinterpret_cast<uint2*>(my_a + (half == 0 ? idx_lo : idx_hi)) = packed;
      }

      tg_cp_async_wait<TG_B_STAGES - 1>();
      __syncwarp();  // make this warp's cp.async + staging stores visible

      if (active) {
        wmma::fragment<wmma::matrix_a, TG_WMMA_TILE, TG_WMMA_TILE,
                       TG_WMMA_TILE, __nv_bfloat16, wmma::row_major> a;
        wmma::load_matrix_sync(a, my_a, TG_WMMA_TILE);

        // The staged tile is row-major [16][16] = W[n0+r, k+c], so the B
        // fragment b(r,c) = W[n0+c, k+r] is its transpose: load col_major.
        wmma::fragment<wmma::matrix_b, TG_WMMA_TILE, TG_WMMA_TILE,
                       TG_WMMA_TILE, __nv_bfloat16, wmma::col_major> b;
        wmma::load_matrix_sync(
            b, s_b + (ks % TG_B_STAGES) * (TG_WMMA_TILE * TG_WMMA_TILE),
            TG_WMMA_TILE);

        wmma::mma_sync(acc, a, b, acc);
      }
      // No CTA barrier: my_a and the B stages are per-warp, so the next
      // iteration can overwrite them as soon as this warp's mma has read them.
      // (mma_sync is synchronous, so its shared-memory reads are done.)
    }
    // Drain the trailing (empty) commit groups so the next tile's pipeline
    // starts from a known group count.
    tg_cp_async_wait<0>();
    if (active) {
      wmma::store_matrix_sync(C + m0 * STRIDE_N + n0, acc, STRIDE_N,
                              wmma::mem_row_major);
    }
  }
}

// =============================================================================
// RoPE (rotate_half / GPT-NeoX style) — ported from kernel.cu:434-476.
//
// Rotates a head_dim-length vector in-place (or to a separate output):
//   first half (i < hd/2):  out[i] = x[i]*cos[i] - x[i+hd/2]*sin[i]
//   second half (i >= hd/2): out[i] = x[i-hd/2]*sin[i] + x[i]*cos[i]
//
// Each lane owns the elements i = lane + j*WARP_SIZE (4 of them for head_dim
// 128), i.e. strided, NOT contiguous. With that layout the mirror partner
// (i +/- head_dim/2) lives in lane (pi % 32) at index pi/32, which is exactly
// what the __shfl_sync below fetches. This must match kernel.cu's RoPE and the
// inlined copy in tg_attn_decode_op; an earlier revision used a lane*4 layout
// here and silently wrote only the 32 multiples of 4.
//
// cos/sin tables are [max_seq_len, head_dim] (NOT head_dim*2 — the
// .repeat(1,2) in model.py doubles the 64-wide freqs to 128-wide = HEAD_DIM).
//
// NOTE: currently unused — tg_attn_decode_op inlines its own copy. Kept
// correct so it is safe to call if the two are ever unified.
// =============================================================================

__device__ __forceinline__ void tg_rope(
    float* val,  // [head_dim] — rotated in-place
    const __nv_bfloat16* cos_pos, const __nv_bfloat16* sin_pos,
    int head_dim) {
  int lane = tg_wlane();
  float local[4];
  #pragma unroll
  for (int i = lane, j = 0; i < head_dim; i += TG_WARP_SIZE, j++)
    local[j] = val[i];

  #pragma unroll
  for (int i = lane, j = 0; i < head_dim; i += TG_WARP_SIZE, j++) {
    float cv = __bfloat162float(__ldg(cos_pos + i));
    float sv = __bfloat162float(__ldg(sin_pos + i));
    int po = (i < head_dim / 2) ? head_dim / 2 : -head_dim / 2;
    int pi = i + po;
    int pj = pi / TG_WARP_SIZE;
    float pv = __shfl_sync(0xffffffff, local[pj], pi % TG_WARP_SIZE);
    val[i] = (i < head_dim / 2) ? local[j] * cv - pv * sv
                                 : pv * sv + local[j] * cv;
  }
}

// =============================================================================
// Phase 1 op bodies. Each does: dependency wait → compute → dependency arrive.
// Shared memory is declared in the kernel entry and passed via TgGlobals or
// directly. The dispatch loop calls tg_worker_sync() once after each op body
// to keep the bar.sync 1 arrival count aligned.
// =============================================================================

// QKV_PROJ: RMSNorm(x_resid) → WMMA GEMM for a pack of Q/K/V N tiles.
//
// Instruction layout:
//   inst[1]=layer, inst[2]=tile_n(primary), inst[3]=gemm_idx(0=Q,1=K,2=V),
//   inst[4]=needs_rope, inst[5]=wait_bar_idx(bar_layer), inst[6]=wait_target,
//   inst[7]=produce_bar_idx(bar_qkv), inst[8]=batch_row,
//   inst[9]=num_tiles, inst[10..15]=consecutive tile_n list
//
// batch_row is always 0 (the op covers all BS rows). The RMSNorm is computed
// once per instruction (redundantly across instructions, but cheap next to the
// GEMM) and stored in shared memory for the WMMA A fragment.
//
// All packed tiles share (gemm_idx, produce_bar_idx), so they use the same
// weight and signal the same bar_qkv slot with inc = num_tiles. The attention
// consumer's target counts the total tile arrivals, so the sum is unchanged.

// Shared memory layout for QKV / GEMM ops (declared in kernel entry):
//   s_a[7][256]    — per-warp WMMA A fragment staging (3.5 KiB)
//   s_norm[hidden] — RMSNorm input/output (4 KiB)
//   s_reduce[wwarps] — block reduce (28 bytes)

__device__ __forceinline__ void tg_qkv_proj_op(
    const int* inst, TgGlobals& g, int bid,
    __nv_bfloat16* s_a, float* s_norm, float* s_reduce) {
  int layer = inst[1];
  int gemm_idx = inst[3];
  int num_tiles = inst[9];

  // Wait for previous layer's RMSNorm (bar_layer)
  tg_wait(g.bar_layer, inst[5], (uint32_t)inst[6]);

  // RMSNorm: x_resid → s_norm (normalized, float, BF16-rounded)
  // For BS>1, normalize each batch row independently into s_norm[batch*hidden].
  const __nv_bfloat16* norm_w =
      tg_layer_weight(g.layer_weights, layer, tg::TG_LW_INPUT_LAYERNORM);
  for (int batch = 0; batch < g.BS; batch++) {
    const __nv_bfloat16* input = g.x_resid + batch * g.hidden;
    tg_rmsnorm(s_norm + batch * g.hidden, s_reduce, input, norm_w, g.hidden);
  }

  // Select weight and output buffer based on gemm_idx
  int weight_idx;
  float* output;
  if (gemm_idx == 0) {
    weight_idx = tg::TG_LW_Q_PROJ;
    output = g.q_out;
  } else if (gemm_idx == 1) {
    weight_idx = tg::TG_LW_K_PROJ;
    output = g.k_out;
  } else {
    weight_idx = tg::TG_LW_V_PROJ;
    output = g.v_out;
  }
  const __nv_bfloat16* weight =
      tg_layer_weight(g.layer_weights, layer, weight_idx);

  // ONE GEMM covering all packed (consecutive) tile_n: N = num_tiles*16, so
  // tg_wmma_gemm splits the N tiles across the worker warps. M=BS, K=hidden.
  int wwarp = tg_wwarp();
  const int tile_n0 = inst[10];
  const int tile_offset = tile_n0 * TG_WMMA_TILE;
  const int n_total = num_tiles * TG_WMMA_TILE;
  if (gemm_idx == 0) {
    // Q: STRIDE_K=hidden, STRIDE_N=q_size
    tg_wmma_gemm<TG_HIDDEN, TG_Q_SIZE>(
        s_norm, weight + tile_offset * g.hidden,
        output + tile_offset,
        g.BS, n_total, g.hidden,
        wwarp, TG_WORKER_WARPS, s_a);
  } else {
    // K or V: STRIDE_K=hidden, STRIDE_N=kv_size
    tg_wmma_gemm<TG_HIDDEN, TG_KV_SIZE>(
        s_norm, weight + tile_offset * g.hidden,
        output + tile_offset,
        g.BS, n_total, g.hidden,
        wwarp, TG_WORKER_WARPS, s_a);
  }

  tg_arrive(g.bar_qkv, inst[7], (uint32_t)num_tiles);
}

// ATTN_DECODE: QK-norm + RoPE + KV cache write + scalar flash attention.
//
// Instruction layout:
//   inst[1]=layer, inst[2]=batch_idx, inst[3]=kv_head_idx,
//   inst[4]=split_idx, inst[5]=num_splits, inst[6]=window_size,
//   inst[7]=wait_bar_idx(bar_qkv), inst[8]=wait_target,
//   inst[9]=produce_bar_idx (bit 31 = combine flag)
//
// For Phase 1 with attn_splits=1, there's one split covering all positions.
// Each instruction handles one (batch, kv_head) pair. The 7 worker warps
// cooperate on the scalar flash attention over the position range.
//
// The op body:
// 1. Wait for QKV tiles (bar_qkv for this kv_head)
// 2. QK-norm + RoPE for the Q heads in this kv_head's group (2 Q heads per kv_head)
// 3. K-norm + RoPE + KV cache write
// 4. V cache write
// 5. Scalar flash attention (all Q heads for this kv_head)
// 6. Signal bar_attn (or bar_combine if split > 1)

__device__ __forceinline__ void tg_attn_decode_op(
    const int* inst, TgGlobals& g, int bid,
    float* s_max, float* s_sum, float* s_out_acc) {
  int layer = inst[1];
  int batch = inst[2];
  int kv_head = inst[3];
  int split_idx = inst[4];
  int num_splits = inst[5];
  // inst[6] = window_size (unused for Phase 1)
  // bit 31 of inst[9]: this split publishes o_partial/lse_partial for
  // ATTN_COMBINE instead of writing attn_out directly.
  const bool is_combine = ((inst[9] >> 31) & 1) != 0;

  // Wait for QKV tiles for this kv_head
  tg_wait(g.bar_qkv, inst[7], (uint32_t)inst[8]);

  int wwarp = tg_wwarp();
  int wlane = tg_wlane();
  int wtid = tg_wtid();

  // KV cache layer base
  __nv_bfloat16* k_cache_layer = g.k_cache + (int64_t)layer * g.kv_layer_stride;
  __nv_bfloat16* v_cache_layer = g.v_cache + (int64_t)layer * g.kv_layer_stride;

  // Position and block table for this batch row
  int cache_len = g.cache_seqlens[batch] + 1;  // +1 for the new token
  int position = cache_len - 1;  // current position (0-indexed)
  const int* block_table = g.page_table + batch * g.max_pages_per_seq;
  int block_size = g.page_block_size;

  // cos/sin tables for this position
  const __nv_bfloat16* cos_pos = g.cos_table + position * g.head_dim;
  const __nv_bfloat16* sin_pos = g.sin_table + position * g.head_dim;

  // QK-norm weights
  const __nv_bfloat16* q_norm_w =
      tg_layer_weight(g.layer_weights, layer, tg::TG_LW_Q_NORM);
  const __nv_bfloat16* k_norm_w =
      tg_layer_weight(g.layer_weights, layer, tg::TG_LW_K_NORM);

  // Q/K/V pointers for this batch row
  float* q_base = g.q_out + batch * g.q_size;
  float* k_base = g.k_out + batch * g.kv_size;
  float* v_base = g.v_out + batch * g.kv_size;

  // GQA: 2 Q heads per kv_head (NUM_Q_HEADS=16, NUM_KV_HEADS=8)
  int group_size = g.num_q_heads / g.num_kv_heads;  // 2
  int q_head_start = kv_head * group_size;

  // --- Phase A: K-norm + RoPE + KV cache write (one warp, warp 0 of workers) ---
  // K and V are written to the paged cache at the current position.
  //
  // Inactive rows must NOT write: a continuous-batching runtime frees the
  // pages of finished sequences and hands them to a new sequence, so a stale
  // row writing "its" old pages would corrupt the new owner. Everything else
  // the row computes is harmless (row-private scratch; its output is ignored),
  // and all bars still arrive so the wait targets stay valid.
  const bool row_on = (g.row_active[batch] != 0);
  if (wwarp == 0 && row_on) {
    float* k_head = k_base + kv_head * g.head_dim;
    float* v_head = v_base + kv_head * g.head_dim;

    // K-norm (RMSNorm over head_dim=128)
    float ss = 0.0f;
    for (int i = wlane; i < g.head_dim; i += TG_WARP_SIZE)
      ss += k_head[i] * k_head[i];
    ss = tg_warp_reduce_sum(ss);
    float sc = rsqrtf(ss / float(g.head_dim) + TG_RMS_EPS);
    sc = __shfl_sync(0xffffffff, sc, 0);

    // Apply K-norm weight + RoPE
    float kl[4];
    #pragma unroll
    for (int i = wlane, j = 0; i < g.head_dim; i += TG_WARP_SIZE, j++)
      kl[j] = tg_round_bf16(
          tg_round_bf16(k_head[i] * sc) *
          __bfloat162float(__ldg(k_norm_w + i)));

    // KV cache pointers
    int64_t kv_off = tg_kv_offset(position, kv_head, g.head_dim,
                                  g.kv_size, block_size, block_table);
    __nv_bfloat16* kc = k_cache_layer + kv_off;
    __nv_bfloat16* vc = v_cache_layer + kv_off;

    #pragma unroll
    for (int i = wlane, j = 0; i < g.head_dim; i += TG_WARP_SIZE, j++) {
      float cv = __bfloat162float(__ldg(cos_pos + i));
      float sv = __bfloat162float(__ldg(sin_pos + i));
      int po = (i < g.head_dim / 2) ? g.head_dim / 2 : -g.head_dim / 2;
      int pi = i + po, pj = pi / TG_WARP_SIZE;
      float pv = __shfl_sync(0xffffffff, kl[pj], pi % TG_WARP_SIZE);
      float kf = (i < g.head_dim / 2) ? kl[j] * cv - pv * sv
                                      : pv * sv + kl[j] * cv;
      kc[i] = __float2bfloat16(kf);
      vc[i] = __float2bfloat16(v_head[i]);
    }
  }

  // --- Phase A2: Q-norm + RoPE for Q heads in this kv_head's group ---
  // Each worker warp handles one Q head (warp 0 → q_head_start, warp 1 → q_head_start+1)
  for (int qh_offset = wwarp; qh_offset < group_size; qh_offset += TG_WORKER_WARPS) {
    int qh = q_head_start + qh_offset;
    float* qh_ptr = q_base + qh * g.head_dim;

    // Q-norm (RMSNorm over head_dim=128)
    float ss = 0.0f;
    for (int i = wlane; i < g.head_dim; i += TG_WARP_SIZE)
      ss += qh_ptr[i] * qh_ptr[i];
    ss = tg_warp_reduce_sum(ss);
    float sc = rsqrtf(ss / float(g.head_dim) + TG_RMS_EPS);
    sc = __shfl_sync(0xffffffff, sc, 0);

    float ql[4];
    #pragma unroll
    for (int i = wlane, j = 0; i < g.head_dim; i += TG_WARP_SIZE, j++)
      ql[j] = tg_round_bf16(
          tg_round_bf16(qh_ptr[i] * sc) *
          __bfloat162float(__ldg(q_norm_w + i)));

    #pragma unroll
    for (int i = wlane, j = 0; i < g.head_dim; i += TG_WARP_SIZE, j++) {
      float cv = __bfloat162float(__ldg(cos_pos + i));
      float sv = __bfloat162float(__ldg(sin_pos + i));
      int po = (i < g.head_dim / 2) ? g.head_dim / 2 : -g.head_dim / 2;
      int pi = i + po, pj = pi / TG_WARP_SIZE;
      float pv = __shfl_sync(0xffffffff, ql[pj], pi % TG_WARP_SIZE);
      qh_ptr[i] = tg_round_bf16(
          (i < g.head_dim / 2) ? ql[j] * cv - pv * sv
                               : pv * sv + ql[j] * cv);
    }
  }

  // Sync: ensure KV cache write (warp 0) and Q-norm (all warps) are done.
  // CRITICAL: bar.sync 1 only orders shared memory. Warp 0's global KV cache
  // stores need fence.acq_rel.gpu to be visible to other warps after the
  // barrier. Without this, the attention reads stale KV data non-deterministically.
  tg_fence_gpu();
  tg_worker_sync_inner();
  tg_fence_gpu();

  // --- Phase D: Scalar flash attention ---
  // Each Q head is processed by all worker warps (striding over positions).
  float attn_scale = 1.0f / sqrtf((float)g.head_dim);

  for (int qh_offset = 0; qh_offset < group_size; qh_offset++) {
    int qh = q_head_start + qh_offset;
    const float* q_head = q_base + qh * g.head_dim;
    float* out_head = g.attn_out + batch * g.q_size + qh * g.head_dim;

    // Each lane owns 4 of 128 head dims
    int q_idx = wlane * 4;
    float q_local[4];
    q_local[0] = q_head[q_idx + 0];
    q_local[1] = q_head[q_idx + 1];
    q_local[2] = q_head[q_idx + 2];
    q_local[3] = q_head[q_idx + 3];

    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float out_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Determine position range for this split
    int pos_start, pos_end;
    if (num_splits <= 1) {
      pos_start = 0;
      pos_end = cache_len;
    } else {
      int per_split = (cache_len + num_splits - 1) / num_splits;
      pos_start = split_idx * per_split;
      pos_end = min(pos_start + per_split, cache_len);
    }

    // Each warp processes a subset of positions
    for (int pos = pos_start + wwarp; pos < pos_end; pos += TG_WORKER_WARPS) {
      int64_t kv_off = tg_kv_offset(pos, kv_head, g.head_dim,
                                    g.kv_size, block_size, block_table);
      const __nv_bfloat16* k_pos = k_cache_layer + kv_off;
      const __nv_bfloat16* v_pos = v_cache_layer + kv_off;

      // Q @ K^T (scalar dot product over 128 dims, 4 per lane)
      float score = 0.0f;
      uint2 k_u2 = __ldg(reinterpret_cast<const uint2*>(k_pos + q_idx));
      __nv_bfloat16* k_ptr = reinterpret_cast<__nv_bfloat16*>(&k_u2);
      score += q_local[0] * __bfloat162float(k_ptr[0]) +
               q_local[1] * __bfloat162float(k_ptr[1]) +
               q_local[2] * __bfloat162float(k_ptr[2]) +
               q_local[3] * __bfloat162float(k_ptr[3]);
      score = tg_warp_reduce_sum(score) * attn_scale;

      // Online softmax
      float old_max = max_score;
      max_score = fmaxf(max_score, score);
      float exp_diff = tg_fast_exp(old_max - max_score);
      sum_exp = sum_exp * exp_diff + tg_fast_exp(score - max_score);

      // P @ V
      float weight = tg_fast_exp(score - max_score);
      uint2 v_u2 = __ldg(reinterpret_cast<const uint2*>(v_pos + q_idx));
      __nv_bfloat16* v_ptr = reinterpret_cast<__nv_bfloat16*>(&v_u2);
      out_acc[0] = out_acc[0] * exp_diff + weight * __bfloat162float(v_ptr[0]);
      out_acc[1] = out_acc[1] * exp_diff + weight * __bfloat162float(v_ptr[1]);
      out_acc[2] = out_acc[2] * exp_diff + weight * __bfloat162float(v_ptr[2]);
      out_acc[3] = out_acc[3] * exp_diff + weight * __bfloat162float(v_ptr[3]);
    }

    // --- Phase E: Cross-warp online-softmax combine ---
    if (wlane == 0) {
      s_max[wwarp] = max_score;
      s_sum[wwarp] = sum_exp;
    }
    int out_base = wlane * 4;
    s_out_acc[wwarp * g.head_dim + out_base + 0] = out_acc[0];
    s_out_acc[wwarp * g.head_dim + out_base + 1] = out_acc[1];
    s_out_acc[wwarp * g.head_dim + out_base + 2] = out_acc[2];
    s_out_acc[wwarp * g.head_dim + out_base + 3] = out_acc[3];
    tg_worker_sync_inner();

    if (wwarp == 0) {
      float global_max = s_max[0];
      for (int w = 1; w < TG_WORKER_WARPS; w++) {
        if (s_max[w] > -INFINITY)
          global_max = fmaxf(global_max, s_max[w]);
      }

      float total_sum_exp = 0.0f;
      float final_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};

      for (int w = 0; w < TG_WORKER_WARPS; w++) {
        if (s_max[w] > -INFINITY) {
          float scale = tg_fast_exp(s_max[w] - global_max);
          total_sum_exp += s_sum[w] * scale;
          final_out[0] += s_out_acc[w * g.head_dim + out_base + 0] * scale;
          final_out[1] += s_out_acc[w * g.head_dim + out_base + 1] * scale;
          final_out[2] += s_out_acc[w * g.head_dim + out_base + 2] * scale;
          final_out[3] += s_out_acc[w * g.head_dim + out_base + 3] * scale;
        }
      }

      if (is_combine) {
        // Publish this split's UNNORMALIZED partial and its LSE for
        // ATTN_COMBINE. o_partial is [BS, hkv, max_splits, hidden]; the q
        // heads of one kv_head occupy the first group_size*head_dim floats.
        // ATTN_COMBINE then computes
        //   out = sum_s exp(lse_s - M) * o_s / sum_s exp(lse_s - M)
        // which equals the full softmax because
        //   exp(lse_s - M) * o_s = sum_{pos in s} exp(score - M) * v.
        float* partial =
            g.o_partial +
            ((batch * g.num_kv_heads + kv_head) * g.max_attn_splits +
             split_idx) * g.hidden +
            qh_offset * g.head_dim;
        partial[out_base + 0] = final_out[0];
        partial[out_base + 1] = final_out[1];
        partial[out_base + 2] = final_out[2];
        partial[out_base + 3] = final_out[3];
        if (wlane == 0) {
          // Empty split -> global_max = -inf, total_sum_exp = 0 -> lse = -inf,
          // which ATTN_COMBINE skips.
          g.lse_partial[(batch * g.num_kv_heads + kv_head) * g.max_attn_splits +
                        split_idx] = global_max + logf(total_sum_exp);
        }
      } else {
        out_head[out_base + 0] = tg_round_bf16(final_out[0] / total_sum_exp);
        out_head[out_base + 1] = tg_round_bf16(final_out[1] / total_sum_exp);
        out_head[out_base + 2] = tg_round_bf16(final_out[2] / total_sum_exp);
        out_head[out_base + 3] = tg_round_bf16(final_out[3] / total_sum_exp);
      }
    }
    tg_worker_sync_inner();
  }

  // Signal produce bar (bar_attn or bar_combine, determined by bit 31 of inst[9])
  uint32_t produce_idx = (uint32_t)inst[9] & ~(1u << 31);
  if (is_combine) {
    tg_arrive(g.bar_combine, produce_idx);
  } else {
    tg_arrive(g.bar_attn, produce_idx);
  }
}

// ATTN_COMBINE: combine partial attention outputs from multiple splits.
// Only emitted when attn_splits > 1. Each ATTN_DECODE split published its
// unnormalized output o_s and lse_s = m_s + log Z_s; this op computes
// out = sum_s exp(lse_s - M) * o_s / sum_s exp(lse_s - M) with M = max lse_s,
// which equals the full softmax over all positions.

__device__ __forceinline__ void tg_attn_combine_op(
    const int* inst, TgGlobals& g, int bid,
    float* s_max, float* s_sum, float* s_out_acc) {
  // inst[1] (layer) is unused until the partials are actually written.
  int batch = inst[2];
  int kv_head = inst[3];
  int num_splits = inst[4];

  tg_wait(g.bar_combine, inst[5], (uint32_t)inst[6]);

  // For Phase 1 with attn_splits=1, combine is not needed. The combine
  // instruction is not emitted by the schedule when splits=1.
  // Phase 2 will implement the LSE-style combine across splits.

  int wwarp = tg_wwarp();
  int wlane = tg_wlane();

  int group_size = g.num_q_heads / g.num_kv_heads;
  int q_head_start = kv_head * group_size;

  for (int qh_offset = 0; qh_offset < group_size; qh_offset++) {
    int qh = q_head_start + qh_offset;
    float* out_head = g.attn_out + batch * g.q_size + qh * g.head_dim;

    // Combine partials from all splits using LSE rescaling
    // o_partial: [BS, hkv, max_splits, hidden]
    // lse_partial: [BS, hkv, max_splits]
    float global_max = -INFINITY;
    for (int s = 0; s < num_splits; s++) {
      float lse = g.lse_partial[batch * g.num_kv_heads * g.max_attn_splits +
                                 kv_head * g.max_attn_splits + s];
      if (lse > global_max) global_max = lse;
    }

    float total_weight = 0.0f;
    float final_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int out_base = wlane * 4;

    for (int s = 0; s < num_splits; s++) {
      float lse = g.lse_partial[batch * g.num_kv_heads * g.max_attn_splits +
                                 kv_head * g.max_attn_splits + s];
      if (lse <= -INFINITY) continue;
      float w = tg_fast_exp(lse - global_max);
      total_weight += w;
      // The q heads of this kv_head are packed at the start of the split's
      // g.hidden floats, so offset by qh_offset (GQA: group_size heads).
      float* partial = g.o_partial + batch * g.num_kv_heads * g.max_attn_splits * g.hidden +
                        kv_head * g.max_attn_splits * g.hidden + s * g.hidden +
                        qh_offset * g.head_dim;
      final_out[0] += partial[out_base + 0] * w;
      final_out[1] += partial[out_base + 1] * w;
      final_out[2] += partial[out_base + 2] * w;
      final_out[3] += partial[out_base + 3] * w;
    }

    if (wwarp == 0) {
      out_head[out_base + 0] = tg_round_bf16(final_out[0] / total_weight);
      out_head[out_base + 1] = tg_round_bf16(final_out[1] / total_weight);
      out_head[out_base + 2] = tg_round_bf16(final_out[2] / total_weight);
      out_head[out_base + 3] = tg_round_bf16(final_out[3] / total_weight);
    }
  }

  tg_arrive(g.bar_attn, inst[7]);
}

// O_PROJ: WMMA GEMM, attn_out → o_proj_out.
// Generic GEMM layout: inst[6]=wait_bar_idx, inst[7]=wait_target,
// inst[8]=produce_bar_idx, inst[9]=num_tiles, inst[10..15]=packed tile IDs.

__device__ __forceinline__ void tg_o_proj_op(
    const int* inst, TgGlobals& g, int bid,
    __nv_bfloat16* s_a) {
  int layer = inst[1];
  int num_tiles = inst[9];

  tg_wait(g.bar_attn, inst[6], (uint32_t)inst[7]);

  const __nv_bfloat16* weight =
      tg_layer_weight(g.layer_weights, layer, tg::TG_LW_O_PROJ);
  int wwarp = tg_wwarp();

  // ONE GEMM call covering all packed tiles (they are contiguous tile_n; the
  // packer enforces this). Passing N = num_tiles*16 makes tg_wmma_gemm split
  // the N tiles across all worker warps. Looping one 16-wide GEMM per tile
  // would serialize them on warp 0 and leave the other 6 warps idle.
  const int tile_n0 = inst[10] & 0xFFFF;
  tg_wmma_gemm<TG_Q_SIZE, TG_HIDDEN>(
      g.attn_out,  // [BS, q_size]
      weight + tile_n0 * TG_WMMA_TILE * TG_Q_SIZE,
      g.o_proj_out + tile_n0 * TG_WMMA_TILE,  // [BS, hidden]
      g.BS,  // M = batch size
      num_tiles * TG_WMMA_TILE,  // N = 16 per packed tile
      TG_Q_SIZE,  // K = q_size
      wwarp, TG_WORKER_WARPS, s_a);

  tg_arrive(g.bar_oproj, inst[8], (uint32_t)num_tiles);
}

// FFN_UPGATE_ACT: fused residual add + post-attn RMSNorm + gate/up GEMM + SiLU.
//
// Waits on bar_oproj (O_PROJ done). The op body:
// 1. x_attn = x_resid + o_proj_out (residual add)
// 2. post_attn_norm = RMSNorm(x_attn) → shared memory
// 3. gate = post_attn_norm @ gate_weight^T (WMMA GEMM)
// 4. up = post_attn_norm @ up_weight^T (WMMA GEMM)
// 5. silu_out = SiLU(gate) * up (elementwise)
// 6. Store x_attn to scratch (for ADD_RMSNORM to read)

__device__ __forceinline__ void tg_ffn_upgate_op(
    const int* inst, TgGlobals& g, int bid,
    __nv_bfloat16* s_a, float* s_norm, float* s_reduce) {
  int layer = inst[1];
  int num_tiles = inst[9];

  tg_wait(g.bar_oproj, inst[6], (uint32_t)inst[7]);

  int wwarp = tg_wwarp();
  int wtid = tg_wtid();

  // Phase 1: residual add + RMSNorm (per batch row, redundantly computed).
  // For BS=1 there is one row; for BS>1 every tile instruction still needs all
  // rows because M=BS in the GEMM.
  //
  // x_attn is staged in the shared s_norm slice and normalized in place, so
  // the RMSNorm never round-trips through global memory. Only the first
  // packed instruction (tile_n0 == 0) also publishes x_attn to g.x_attn for
  // ADD_RMSNORM. Previously every instruction wrote the same global addresses
  // from every CTA — same-value, so functionally safe, but a data race that
  // racecheck flags. ADD_RMSNORM still reads a complete x_attn because it
  // waits on bar_ffn_down, which is only produced after every FFN_UPGATE
  // (including the publisher) has arrived.
  const int tile_n0 = inst[10] & 0xFFFF;
  const bool publishes_x_attn = (tile_n0 == 0);
  for (int batch = 0; batch < g.BS; batch++) {
    float* xa_smem = s_norm + batch * g.hidden;  // staged x_attn (shared)
    float* o_proj = g.o_proj_out + batch * g.hidden;
    const __nv_bfloat16* x_res = g.x_resid + batch * g.hidden;

    for (int i = wtid; i < g.hidden; i += TG_NUM_WORKER_THREADS) {
      xa_smem[i] = tg_round_bf16(o_proj[i]) + __bfloat162float(x_res[i]);
    }
    tg_worker_sync_inner();

    if (publishes_x_attn) {
      float* x_attn = g.x_attn + batch * g.hidden;
      for (int i = wtid; i < g.hidden; i += TG_NUM_WORKER_THREADS) {
        x_attn[i] = xa_smem[i];
      }
      tg_worker_sync_inner();
    }

    // In-place RMSNorm: input and output are the same shared slice, which is
    // safe because the staging pass only re-reads what it just wrote.
    const __nv_bfloat16* post_norm_w =
        tg_layer_weight(g.layer_weights, layer, tg::TG_LW_POST_ATTN_LAYERNORM);
    tg_rmsnorm_f(xa_smem, s_reduce, xa_smem, post_norm_w, g.hidden);
  }

  // Phase 2: Gate + Up GEMM
  const __nv_bfloat16* gate_w =
      tg_layer_weight(g.layer_weights, layer, tg::TG_LW_GATE_PROJ);
  const __nv_bfloat16* up_w =
      tg_layer_weight(g.layer_weights, layer, tg::TG_LW_UP_PROJ);

  // One GEMM per operand covering all packed (contiguous) tiles, so the N
  // tiles are split across the worker warps instead of serialized on warp 0.
  // (tile_n0 is declared in Phase 1.)
  const int n_total = num_tiles * TG_WMMA_TILE;
  // Gate GEMM → silu_out:
  tg_wmma_gemm<TG_HIDDEN, TG_INTERMEDIATE>(
      s_norm,
      gate_w + tile_n0 * TG_WMMA_TILE * g.hidden,
      g.silu_out + tile_n0 * TG_WMMA_TILE,
      g.BS, n_total, g.hidden,
      wwarp, TG_WORKER_WARPS, s_a);
  // Up GEMM → up_out:
  tg_wmma_gemm<TG_HIDDEN, TG_INTERMEDIATE>(
      s_norm,
      up_w + tile_n0 * TG_WMMA_TILE * g.hidden,
      g.up_out + tile_n0 * TG_WMMA_TILE,
      g.BS, n_total, g.hidden,
      wwarp, TG_WORKER_WARPS, s_a);

  // CRITICAL: synchronize between GEMMs and SiLU. The gate/up GEMMs write
  // to g.silu_out and g.up_out (global memory) with different warps handling
  // different tiles. Without this fence+barrier, a fast warp could reach the
  // SiLU phase and read a tile written by a slow warp that hasn't finished
  // its wmma::store_matrix_sync yet — a classic cross-warp global memory race.
  tg_fence_gpu();
  tg_worker_sync_inner();

  // Phase 3: SiLU(gate) * up (elementwise) — ONLY for this instruction's tiles.
  // This instruction packs ``num_tiles`` CONSECUTIVE tiles starting at
  // ``tile_n0`` (inst[10]); each tile covers TG_WMMA_TILE columns of the
  // intermediate dimension. We must NOT process all TG_INTERMEDIATE positions,
  // only the ones this instruction's GEMM wrote, otherwise later instructions
  // re-apply SiLU to positions already processed by earlier ones.
  //
  // The tile index is ``tile_n0 + t``, derived from the range. This is
  // mandatory, not stylistic: a per-tile list cannot address a full pack.
  // ``inst`` here is ``inst_shared``, a ``__shared__ int[TG_INST_WIDTH]``
  // local copy (see the kernel entry — op bodies never rebase it), so only
  // inst[10..15] exist after the range start and ``_TILE_SLOTS_PER_INST`` is 7.
  //
  // It used to read the per-tile list as ``inst[10 + t]``. ``f3d8114``
  // replaced that list with a range, zeroed inst[11..15] and raised the pack
  // cap 6 -> 7, leaving this loop behind. The consequences were not subtle:
  //
  //   t == 0      inst[10]     = tile_n0  -> only the 28 pack STARTS activated
  //   1 <= t <= 5 inst[11..15] = 0        -> tile 0 re-Silu'd 3836x
  //   t == 6      inst[16]     -> ONE WORD PAST inst_shared: undefined, and in
  //                                         practice whatever __shared__ object
  //                                         the compiler laid out next (s_a
  //                                         holds the GEMM's own bf16 A fragment,
  //                                         so its bits read back as tile ids in
  //                                         the tens of thousands, e.g. bf16 1.0
  //                                         = 0x3F80 = 16256 -> ~1 MB past a
  //                                         192 KiB silu_out). The exact value
  //                                         is layout-dependent; the out-of-
  //                                         bounds read-modify-write is not.
  //
  // Measured against the emitted BS=1 schedule: 28 of the 192 intermediate
  // tiles were ever activated, 164 (85%) kept their RAW gate projection, and
  // 756 instructions per step read inst[16]. That unchecked store is a prime
  // suspect for the grid > 1 illegal memory access on sm_120.
  //
  // For BS>1, process all batch rows for each tile column.
  for (int t = 0; t < num_tiles; t++) {
    int tile_n = tile_n0 + t;
    int offset = tile_n * TG_WMMA_TILE;
    for (int i = wtid; i < TG_WMMA_TILE * g.BS; i += TG_NUM_WORKER_THREADS) {
      int col = i % TG_WMMA_TILE;
      int batch = i / TG_WMMA_TILE;
      int idx = batch * g.intermediate + offset + col;
      float gate_val = g.silu_out[idx];
      float up_val = g.up_out[idx];
      g.silu_out[idx] = tg_round_bf16(tg_round_bf16(tg_silu(gate_val)) * up_val);
    }
  }

  tg_arrive(g.bar_silu, inst[8], (uint32_t)num_tiles);
}

// FFN_DOWN: WMMA GEMM, silu_out → ffn_down_out.

__device__ __forceinline__ void tg_ffn_down_op(
    const int* inst, TgGlobals& g, int bid,
    __nv_bfloat16* s_a) {
  int layer = inst[1];
  int num_tiles = inst[9];

  tg_wait(g.bar_silu, inst[6], (uint32_t)inst[7]);

  const __nv_bfloat16* weight =
      tg_layer_weight(g.layer_weights, layer, tg::TG_LW_DOWN_PROJ);
  int wwarp = tg_wwarp();

  // One GEMM for all packed (contiguous) tiles; warps split the N tiles.
  const int tile_n0 = inst[10] & 0xFFFF;
  tg_wmma_gemm<TG_INTERMEDIATE, TG_HIDDEN>(
      g.silu_out,  // [BS, intermediate]
      weight + tile_n0 * TG_WMMA_TILE * g.intermediate,
      g.ffn_down_out + tile_n0 * TG_WMMA_TILE,  // [BS, hidden]
      g.BS, num_tiles * TG_WMMA_TILE, g.intermediate,
      wwarp, TG_WORKER_WARPS, s_a);

  tg_arrive(g.bar_ffn_down, inst[8], (uint32_t)num_tiles);
}

// ADD_RMSNORM: FFN residual + RMSNorm for next layer.
//
// Waits on bar_oproj (for x_attn, transitively ready via bar_ffn_down) and
// bar_ffn_down. The op body:
// 1. x_ffn = x_attn + ffn_down_out (FFN residual add)
// 2. norm_out = RMSNorm(x_ffn) → for next layer's QKV GEMM
// 3. x_resid = x_ffn (bf16, for next layer's input)
// 4. Signal bar_layer

__device__ __forceinline__ void tg_add_rmsnorm_op(
    const int* inst, TgGlobals& g, int bid,
    float* s_norm, float* s_reduce) {
  int layer = inst[1];
  int batch = inst[2];
  int is_final = inst[8];

  // Wait for both attention path (bar_oproj) and FFN path (bar_ffn_down).
  // bar_oproj is transitively satisfied via bar_silu → bar_ffn_down, but we
  // keep the wait for correctness (x_attn is stored by FFN_UPGATE which
  // waits on bar_oproj).
  tg_wait_all(g.bar_oproj, inst[3], (uint32_t)inst[4],
              g.bar_ffn_down, inst[5], (uint32_t)inst[6]);

  int wtid = tg_wtid();

  // FFN residual: x_ffn = x_attn + ffn_down_out
  float* x_attn = g.x_attn + batch * g.hidden;
  float* ffn_down = g.ffn_down_out + batch * g.hidden;
  float* x_ffn = g.x_ffn + batch * g.hidden;
  __nv_bfloat16* x_res = g.x_resid + batch * g.hidden;

  for (int i = wtid; i < g.hidden; i += TG_NUM_WORKER_THREADS) {
    float val = tg_round_bf16(ffn_down[i]) + x_attn[i];
    x_ffn[i] = val;
    x_res[i] = __float2bfloat16(val);
  }
  tg_worker_sync_inner();

  // RMSNorm: x_ffn → norm_out (for next layer's QKV)
  // Use the float-input RMSNorm variant, output goes to global norm_out.
  // We need to use s_norm as scratch, then copy to global norm_out.
  const __nv_bfloat16* norm_w = (is_final)
      ? g.final_norm_weight
      : tg_layer_weight(g.layer_weights, layer + 1, tg::TG_LW_INPUT_LAYERNORM);

  tg_rmsnorm_f(s_norm, s_reduce, x_ffn, norm_w, g.hidden);

  // Copy from shared to global norm_out
  float* norm_out = g.norm_out + batch * g.hidden;
  for (int i = wtid; i < g.hidden; i += TG_NUM_WORKER_THREADS) {
    norm_out[i] = s_norm[i];
  }
  tg_worker_sync_inner();

  tg_arrive(g.bar_layer, inst[7]);
}

// LM_HEAD: WMMA logits for a pack of vocab tiles. Waits on bar_layer[num_layers]
// and produces bar_lm_head[0] with inc = NUM_TILES.
//
// The argmax is NOT done here: with num_sms > 1 another SM may not have written
// its vocab tiles yet, so a per-instruction scan would read stale logits. The
// terminal TG_LM_ARGMAX instruction waits for every tile via bar_lm_head.

__device__ __forceinline__ void tg_lm_head_op(
    const int* inst, TgGlobals& g, int bid,
    __nv_bfloat16* s_a, float* s_norm) {
  int num_tiles = inst[9];

  tg_wait(g.bar_layer, inst[6], (uint32_t)inst[7]);

  int wtid = tg_wtid();
  int wwarp = tg_wwarp();

  // The last ADD_RMSNORM already stored the final RMSNorm into norm_out.
  // Copy all BS rows into shared memory for the WMMA A fragment.
  for (int i = wtid; i < g.BS * g.hidden; i += TG_NUM_WORKER_THREADS) {
    s_norm[i] = g.norm_out[i];
  }
  tg_worker_sync_inner();

  // WMMA GEMM: lm_logits = norm_out @ embed_weight^T
  // embed_weight is [vocab, hidden] row-major (= LM head weight, tied).
  // One GEMM for all packed (contiguous) tiles; warps split the N tiles. This
  // is the largest GEMM of the step (vocab x hidden), so the fan-out matters.
  const int tile_n0 = inst[10] & 0xFFFF;
  tg_wmma_gemm<TG_HIDDEN, TG_VOCAB>(
      s_norm,  // [BS, hidden] in shared memory
      g.embed_weight + tile_n0 * TG_WMMA_TILE * g.hidden,
      g.lm_logits + tile_n0 * TG_WMMA_TILE,  // [BS, vocab]
      g.BS, num_tiles * TG_WMMA_TILE, g.hidden,
      wwarp, TG_WORKER_WARPS, s_a);

  tg_arrive(g.bar_lm_head, inst[8], (uint32_t)num_tiles);
}

// LM_ARGMAX: terminal per-batch argmax over the fully-written logits.
//
// Waits on bar_lm_head[inst[6]] >= inst[7] (== total LM_HEAD tiles), so every
// SM's tiles are visible. Then each worker warp scans a slice of the vocab and
// warp 0 reduces to the token id, stored as int32 at lm_logits[batch, 0].
__device__ __forceinline__ void tg_lm_argmax_op(
    const int* inst, TgGlobals& g, int bid,
    float* s_argmax_val, int* s_argmax_idx) {
  tg_wait(g.bar_lm_head, inst[6], (uint32_t)inst[7]);
  tg_fence_gpu();
  tg_worker_sync_inner();

  int wtid = tg_wtid();
  int wwarp = tg_wwarp();
  int wlane = tg_wlane();

  for (int batch = 0; batch < g.BS; batch++) {
    float* logits = g.lm_logits + batch * g.vocab_size;
    float local_max = -INFINITY;
    int local_max_idx = 0;
    for (int i = wtid; i < g.vocab_size; i += TG_NUM_WORKER_THREADS) {
      float val = logits[i];
      if (val > local_max) {
        local_max = val;
        local_max_idx = i;
      }
    }

    // Warp-level argmax
    for (int offset = TG_WARP_SIZE / 2; offset > 0; offset /= 2) {
      float other_max = __shfl_xor_sync(0xffffffff, local_max, offset);
      int other_idx = __shfl_xor_sync(0xffffffff, local_max_idx, offset);
      if (other_max > local_max) {
        local_max = other_max;
        local_max_idx = other_idx;
      }
    }

    // Block-level argmax via shared memory (passed from kernel entry)
    if (wlane == 0) {
      s_argmax_val[wwarp] = local_max;
      s_argmax_idx[wwarp] = local_max_idx;
    }
    tg_worker_sync_inner();

    if (wwarp == 0) {
      float best_val = s_argmax_val[0];
      int best_idx = s_argmax_idx[0];
      for (int w = 1; w < TG_WORKER_WARPS; w++) {
        if (s_argmax_val[w] > best_val) {
          best_val = s_argmax_val[w];
          best_idx = s_argmax_idx[w];
        }
      }
      if (wlane == 0) {
        // Store the argmax token id as int32 at the start of this batch
        // row's logits. The runtime reads lm_logits[batch, 0] as int32.
        reinterpret_cast<int*>(logits)[0] = best_idx;
      }
    }
    tg_worker_sync_inner();
  }
}

// =============================================================================
// Shared memory layout for compute ops. Declared once in the kernel entry;
// pointers passed to each op body via this struct.
// =============================================================================

struct TgSmem {
  __nv_bfloat16* s_a;           // [7][16*16] — per-warp WMMA A staging (3.5 KiB)
  float* s_norm;                // [1024] = 4 KiB (RMSNorm input/output)
  float* s_reduce;              // [7] = 28 B (block reduce)
  float* s_max;                 // [7] = 28 B (attention warp max)
  float* s_sum;                 // [7] = 28 B (attention warp sum_exp)
  float* s_out_acc;             // [7*128] = 3.5 KiB (attention warp outputs)
  float* s_argmax_val;          // [7] = 28 B (LM head argmax val)
  int* s_argmax_idx;            // [7] = 28 B (LM head argmax idx)
};

// =============================================================================
// Instruction dispatch. Big switch on opcode. Each branch calls one op helper
// and then tg_worker_sync exactly once (the invariant).
// =============================================================================

__device__ __forceinline__ void tg_dispatch(const int* inst, TgGlobals& g,
                                             int bid, TgSmem& smem) {
  int op = tg_opcode(inst);
  switch (op) {
    case tg::TG_NOP:
      break;
    case tg::TG_QKV_PROJ:
      tg_qkv_proj_op(inst, g, bid, smem.s_a, smem.s_norm, smem.s_reduce);
      break;
    case tg::TG_ATTN_DECODE:
      tg_attn_decode_op(inst, g, bid, smem.s_max, smem.s_sum, smem.s_out_acc);
      break;
    case tg::TG_ATTN_COMBINE:
      tg_attn_combine_op(inst, g, bid, smem.s_max, smem.s_sum, smem.s_out_acc);
      break;
    case tg::TG_O_PROJ:
      tg_o_proj_op(inst, g, bid, smem.s_a);
      break;
    case tg::TG_FFN_UPGATE_ACT:
      tg_ffn_upgate_op(inst, g, bid, smem.s_a, smem.s_norm, smem.s_reduce);
      break;
    case tg::TG_FFN_DOWN:
      tg_ffn_down_op(inst, g, bid, smem.s_a);
      break;
    case tg::TG_ADD_RMSNORM:
      tg_add_rmsnorm_op(inst, g, bid, smem.s_norm, smem.s_reduce);
      break;
    case tg::TG_LM_HEAD:
      tg_lm_head_op(inst, g, bid, smem.s_a, smem.s_norm);
      break;
    case tg::TG_LM_ARGMAX:
      tg_lm_argmax_op(inst, g, bid, smem.s_argmax_val, smem.s_argmax_idx);
      break;
    default:
      break;
  }
  tg_worker_sync();
}

// =============================================================================
// Persistent kernel entry.
// =============================================================================

__global__ __launch_bounds__(TG_BLOCK_SIZE, 1)
void tg_decode_kernel(__grid_constant__ const tg::TgLaunchDesc desc) {
  const int bid = blockIdx.x;
  TgGlobals g = tg_make_globals(desc);
  const int num_inst = g.num_inst_per_sm[bid];
  const int* sm_base = g.inst_buf + (int64_t)bid * g.max_inst * TG_INST_WIDTH;

  // One shared instruction buffer. The active implementation uses a
  // correctness-first load/sync/dispatch/sync loop; an asynchronous ring can
  // be introduced later only with a dedicated correctness and performance A/B.
  __shared__ int inst_shared[TG_INST_WIDTH];

  // Shared memory for compute ops (see TgSmem).
  //
  // s_norm is in dynamic shared memory because the per-warp B staging plus the
  // A fragments already exceed the 48 KB static limit on their own. It holds
  // the RMSNorm staging row indexed as ``s_norm[batch * hidden]``, and it is the
  // WMMA A operand for GEMMs with ``M = BS``: it needs **BS rows**, not
  // ``TG_WMMA_TILE`` (16) rows. It was previously sized at
  // ``TG_WMMA_TILE * TG_HIDDEN * 4`` = 64 KB, which is 8x the batch-8
  // requirement; on sm_120 that pushed static + dynamic to 102608 B against a
  // 102400 B per-SM budget and ``cudaFuncSetAttribute`` failed with "invalid
  // argument", so the very first launch died. The size is now derived from
  // ``desc->BS`` in ``tg_launch_decode``; see the comment there.
  //
  // Per-warp A staging (each warp converts its own 16x16 fragment), so the
  // GEMM K loop needs no CTA barrier. __align__(16) is required by
  // wmma::load_matrix_sync.
  __shared__ __align__(16)
      __nv_bfloat16 s_a[TG_WORKER_WARPS * TG_WMMA_TILE * TG_WMMA_TILE];
  __shared__ float s_reduce[TG_WORKER_WARPS];
  __shared__ float s_max[TG_WORKER_WARPS];
  __shared__ float s_sum[TG_WORKER_WARPS];
  __shared__ float s_out_acc[TG_WORKER_WARPS * TG_HEAD_DIM];
  __shared__ float s_argmax_val[TG_WORKER_WARPS];
  __shared__ int s_argmax_idx[TG_WORKER_WARPS];

  // Dynamic shared memory for s_norm: one float row per batch row (see the
  // sizing comment above). The launch helper requests ``BS * hidden`` floats.
  extern __shared__ float s_norm[];

  TgSmem smem;
  smem.s_a = s_a;
  smem.s_norm = s_norm;
  smem.s_reduce = s_reduce;
  smem.s_max = s_max;
  smem.s_sum = s_sum;
  smem.s_out_acc = s_out_acc;
  smem.s_argmax_val = s_argmax_val;
  smem.s_argmax_idx = s_argmax_idx;

  const int warp_id = threadIdx.x / TG_WARP_SIZE;

  // Simple __syncthreads-based dispatch (no ring buffer).
  // Controller (warp 0) loads instruction, all threads sync, workers dispatch,
  // all threads sync.
  for (int i = 0; i < num_inst; ++i) {
    // Controller loads one instruction into shared memory.
    if (warp_id == 0) {
      tg_coop_load_inst(inst_shared, sm_base + i * TG_INST_WIDTH);
    }
    __syncthreads();
    // Workers dispatch.
    if (warp_id != 0) {
      tg_dispatch(inst_shared, g, bid, smem);
    }
    __syncthreads();
  }
}

// =============================================================================
// Host-side launch helper (called by runtime.cu). Mirrors ldg_launch_cooperative
// (kernel.cu:2259-2292): validates cooperative support, queries occupancy,
// launches with cudaLaunchCooperativeKernel so all CTAs are resident (required
// for the spinning tg_wait to make progress).
// =============================================================================

// Returns cudaSuccess or the launch error. The binding turns a non-success
// into a TORCH_CHECK: a silent failure here would leave the previous step's
// tokens in lm_logits and look like a wrong answer.
extern "C" cudaError_t tg_launch_decode(const tg::TgLaunchDesc* desc,
                                        cudaStream_t stream) {
  // s_norm must hold one float row per batch row: the RMSNorm op writes
  // s_norm[batch * hidden] for every batch, and the WMMA A operand is read as
  // [BS, hidden]. Sizing it at TG_WMMA_TILE * hidden (64 KB) rather than
  // BS * hidden (32 KB at BS=8) is not merely wasteful -- static (37072 B
  // measured) + 65536 B exceeds the 102400 B per-SM shared memory budget on
  // sm_120 (RTX 5090), so cudaFuncSetAttribute fails with "invalid argument"
  // and the very first decode launch dies. On the A100 the budget was larger
  // than the request, which is why this surfaced only on the 5090.
  if (desc->BS < 1 || desc->hidden < 1) {
    fprintf(stderr, "tg_launch_decode: invalid geometry BS=%d hidden=%d\n",
            desc->BS, desc->hidden);
    return cudaErrorInvalidValue;
  }
  const int dyn_smem_bytes =
      (int)((size_t)desc->BS * (size_t)desc->hidden * sizeof(float));

  // Raise the dynamic shared memory ceiling to the largest value any call has
  // needed. Tracking the value (rather than a "already set" flag) matters
  // because the batch size differs between steps: a flag set by a BS=1 step
  // would leave a 4 KB limit in place for a later BS=8 step.
  static int smem_attr_bytes = 0;
  if (dyn_smem_bytes > smem_attr_bytes) {
    cudaError_t attr_err = cudaFuncSetAttribute(
        (void*)tg_decode_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        dyn_smem_bytes);
    if (attr_err != cudaSuccess) {
      // Report the real budget instead of a bare "invalid argument": this
      // failure is static + dynamic exceeding the per-SM limit, which is easy
      // to misread as a driver or architecture problem.
      cudaFuncAttributes fn_attr{};
      cudaFuncGetAttributes(&fn_attr, (void*)tg_decode_kernel);
      int per_sm = 0;
      cudaDeviceGetAttribute(&per_sm,
                             cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
      fprintf(stderr,
              "tg_launch_decode: cudaFuncSetAttribute(dynamic=%d bytes, "
              "BS=%d, hidden=%d) failed: %s; static=%zu + dynamic=%d = %zu "
              "vs per-SM shared memory %d bytes. Lower the batch size.\n",
              dyn_smem_bytes, desc->BS, desc->hidden,
              cudaGetErrorString(attr_err), fn_attr.sharedSizeBytes,
              dyn_smem_bytes,
              fn_attr.sharedSizeBytes + (size_t)dyn_smem_bytes, per_sm);
      return attr_err;
    }
    smem_attr_bytes = dyn_smem_bytes;
  }
  void* args[] = {(void*)desc};
  cudaError_t err = cudaLaunchCooperativeKernel(
      (void*)tg_decode_kernel,
      dim3(desc->num_sms), dim3(TG_BLOCK_SIZE), args,
      dyn_smem_bytes, stream);
  if (err != cudaSuccess) {
    fprintf(stderr, "tg_launch_decode: cudaLaunchCooperativeKernel(grid=%d, "
            "block=%d) failed: %s\n",
            desc->num_sms, TG_BLOCK_SIZE, cudaGetErrorString(err));
  }
  return err;
}

// =============================================================================
// Per-step zeroing kernel. Clears all scratch + bar arrays in one launch.
// Runtime uploads a packed TgZeroRegion array; this kernel processes them.
// Adapted from cohere's nmc_zero_regions_kernel (runtime.cu:506-544).
// =============================================================================

// One region per blockIdx.y; blockIdx.x * blockDim.x + threadIdx.x threads
// cooperatively clear that region with a grid-stride loop starting at offset 0.
// (The previous version used the *region index* as the starting byte offset and
// a grid-wide stride, so it only cleared a sparse 1/(gridDim.x*blockDim.x)
// fraction of each region.)
__global__ void tg_zero_regions_kernel(const tg::TgZeroRegion* regions,
                                       int num_regions) {
  const int region = blockIdx.y;
  if (region >= num_regions) return;
  const tg::TgZeroRegion& r = regions[region];
  if (r.ptr == 0 || r.bytes == 0) return;
  char* p = reinterpret_cast<char*>(r.ptr);
  const int64_t bytes = (int64_t)r.bytes;
  const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t off = tid; off < bytes; off += stride) {
    p[off] = 0;
  }
}

extern "C" void tg_launch_zero_regions(const tg::TgZeroRegion* regions,
                                       int num_regions, int blocks_per_region,
                                       cudaStream_t stream) {
  if (num_regions <= 0) return;
  if (blocks_per_region < 1) blocks_per_region = 1;
  const int threads = 256;
  dim3 grid(blocks_per_region, num_regions);
  tg_zero_regions_kernel<<<grid, threads, 0, stream>>>(regions, num_regions);
}
