/**
 * Fused single-kernel decode for Qwen3-0.6B on A100.
 *
 * Everything — embedding lookup, 28 transformer layers (RMSNorm, QKV, RoPE,
 * attention, O-proj, MLP), and final norm — runs inside one cooperative kernel
 * launch.  The LM head (vocab projection + argmax) is a separate
 * non-cooperative kernel launched immediately after.
 *
 * Primary target: NVIDIA A100 (sm_80); other architectures use build-time flags.
 * Model:         Qwen/Qwen3-0.6B (bf16 weights)
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <mma.h>

// =============================================================================
// Model constants (Qwen3-0.6B)
// =============================================================================

constexpr int WARP_SIZE = 32;
constexpr int HIDDEN_SIZE = 1024;
constexpr int INTERMEDIATE_SIZE = 3072;
constexpr int NUM_Q_HEADS = 16;
constexpr int NUM_KV_HEADS = 8;
constexpr int HEAD_DIM = 128;
constexpr int Q_SIZE = NUM_Q_HEADS * HEAD_DIM;   // 2048
constexpr int KV_SIZE = NUM_KV_HEADS * HEAD_DIM; // 1024

#ifndef LDG_NUM_BLOCKS
#define LDG_NUM_BLOCKS 80
#endif
#ifndef LDG_BLOCK_SIZE
#define LDG_BLOCK_SIZE 512
#endif
#ifndef LDG_LM_NUM_BLOCKS
#define LDG_LM_NUM_BLOCKS 1184
#endif
#ifndef LDG_LM_BLOCK_SIZE
#define LDG_LM_BLOCK_SIZE 256
#endif
#ifndef LDG_LM_ROWS_PER_WARP
#define LDG_LM_ROWS_PER_WARP 2
#endif
#ifndef LDG_ATTN_BLOCKS
#define LDG_ATTN_BLOCKS NUM_Q_HEADS
#endif
#ifndef LDG_PREFETCH_QK
#define LDG_PREFETCH_QK 1
#endif
#ifndef LDG_PREFETCH_DOWN
#define LDG_PREFETCH_DOWN 1
#endif
#ifndef LDG_PREFETCH_THREAD_STRIDE
#define LDG_PREFETCH_THREAD_STRIDE 1
#endif
#ifndef LDG_PREFETCH_ELEM_STRIDE
#define LDG_PREFETCH_ELEM_STRIDE 1
#endif
#ifndef LDG_PREFETCH_BLOCK_STRIDE
#define LDG_PREFETCH_BLOCK_STRIDE 1
#endif
#ifndef LDG_PREFETCH_GATE
#define LDG_PREFETCH_GATE 1
#endif
#ifndef LDG_PREFETCH_UP
#define LDG_PREFETCH_UP 1
#endif
#ifndef LDG_MAX_BATCH
#define LDG_MAX_BATCH 32
#endif

constexpr int LDG_NUM_WARPS = LDG_BLOCK_SIZE / WARP_SIZE;
constexpr float LDG_RMS_EPS = 1e-6f;

// LM head
constexpr int LDG_VOCAB_SIZE = 151936;

struct LDGLayerWeights {
  const __nv_bfloat16 *input_layernorm_weight;
  const __nv_bfloat16 *q_proj_weight;
  const __nv_bfloat16 *k_proj_weight;
  const __nv_bfloat16 *v_proj_weight;
  const __nv_bfloat16 *q_norm_weight;
  const __nv_bfloat16 *k_norm_weight;
  const __nv_bfloat16 *o_proj_weight;
  const __nv_bfloat16 *post_attn_layernorm_weight;
  const __nv_bfloat16 *gate_proj_weight;
  const __nv_bfloat16 *up_proj_weight;
  const __nv_bfloat16 *down_proj_weight;
};

// =============================================================================
// Atomic barrier for persistent kernel (replaces cooperative grid.sync())
// =============================================================================

struct AtomicGridSync {
  unsigned int *counter;
  unsigned int *generation;
  unsigned int nblocks;
  unsigned int local_gen;

  __device__ void sync() {
    __syncthreads();
    // All threads in the block must publish their global stores (e.g. WMMA
    // store_matrix_sync by other warps) at GPU scope before thread 0 signals
    // arrival; fence.acq_rel.gpu is a per-thread fence, so only the executing
    // thread's writes are made visible. Without this, reader blocks can see
    // stale GEMM output after the barrier releases (cross-SM WMMA race).
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
    if (threadIdx.x == 0) {
      unsigned int my_gen = local_gen;
      unsigned int arrived = atomicAdd(counter, 1);
      if (arrived == nblocks - 1) {
        *counter = 0;
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        atomicAdd(generation, 1);
      } else {
        volatile unsigned int *vgen = (volatile unsigned int *)generation;
        while (*vgen <= my_gen) {
        }
        // Acquire fence: the release-side fence.acq_rel.gpu publishes writes
        // to GPU scope before the last block increments generation.  Without
        // this acquire fence, the waiting block sees the new generation but
        // has no guarantee that those published stores are visible to it.
        // On Ampere, L2 coherence masks this at some batch sizes (b2/b4) but
        // not others (b3/b5), producing deterministic early divergence.
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
      }
      local_gen = my_gen + 1;
    }
    __syncthreads();
  }
};

// =============================================================================
// Helpers
// =============================================================================

__device__ __forceinline__ float ldg_warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

constexpr float LOG2E = 1.44269504088896340736f;

__device__ __forceinline__ float ptx_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float ptx_rcp(float x) {
  float y;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float fast_exp(float x) {
#ifdef LDG_FAST_MATH
  return ptx_exp2(x * LOG2E);
#else
  return expf(x);
#endif
}

__device__ __forceinline__ float ldg_silu(float x) {
#ifdef LDG_FAST_MATH
  return x * ptx_rcp(1.0f + fast_exp(-x));
#else
  return x / (1.0f + fast_exp(-x));
#endif
}

// PyTorch's Qwen3 BF16 path rounds at each tensor boundary (RMSNorm, linear,
// RoPE, attention, and residual). Keep the kernel's accumulators in FP32, but
// explicitly model the BF16 store/load boundary before the next operation.
__device__ __forceinline__ float ldg_round_bf16(float x) {
  return __bfloat162float(__float2bfloat16(x));
}

// 128-bit L1-bypassing weight loads
__device__ __forceinline__ uint4 ldg_load_weight_u4(const uint4 *ptr) {
  uint4 out;
  asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w)
               : "l"(ptr));
  return out;
}

// =============================================================================
// Optimized matvec with __ldg and aggressive unrolling
// =============================================================================

__device__ void
ldg_matvec_qkv(auto &grid, int block_id, int num_blocks,
               const __nv_bfloat16 *__restrict__ input,
               const __nv_bfloat16 *__restrict__ norm_weight,
               const __nv_bfloat16 *__restrict__ q_weight,
               const __nv_bfloat16 *__restrict__ k_weight,
               const __nv_bfloat16 *__restrict__ v_weight,
               float *__restrict__ g_normalized, float *__restrict__ g_residual,
               float *__restrict__ q_out, float *__restrict__ k_out,
               float *__restrict__ v_out) {
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;
  __shared__ __align__(16) float s_norm[HIDDEN_SIZE];

  // ALL blocks compute RMSNorm redundantly
  {
    __shared__ float smem_reduce[LDG_NUM_WARPS];

    float local_sum_sq = 0.0f;

    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE * 2) {
      float v0 = __bfloat162float(__ldg(input + i));
      s_norm[i] = v0;
      local_sum_sq += v0 * v0;
      int i1 = i + LDG_BLOCK_SIZE;
      if (i1 < HIDDEN_SIZE) {
        float v1 = __bfloat162float(__ldg(input + i1));
        s_norm[i1] = v1;
        local_sum_sq += v1 * v1;
      }
    }

    // Block 0 saves residual for later use
    if (block_id == 0) {
      for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE * 2) {
        g_residual[i] = s_norm[i];
        int i1 = i + LDG_BLOCK_SIZE;
        if (i1 < HIDDEN_SIZE) {
          g_residual[i1] = s_norm[i1];
        }
      }
    }

    local_sum_sq = ldg_warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) {
      smem_reduce[warp_id] = local_sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
      float sum = (lane_id < LDG_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
      sum = ldg_warp_reduce_sum(sum);
      if (lane_id == 0) {
        smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + LDG_RMS_EPS);
      }
    }
    __syncthreads();

    float rstd = smem_reduce[0];

    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE * 2) {
      float w0 = __bfloat162float(__ldg(norm_weight + i));
      s_norm[i] = ldg_round_bf16(ldg_round_bf16(s_norm[i] * rstd) * w0);
      int i1 = i + LDG_BLOCK_SIZE;
      if (i1 < HIDDEN_SIZE) {
        float w1 = __bfloat162float(__ldg(norm_weight + i1));
        s_norm[i1] = ldg_round_bf16(ldg_round_bf16(s_norm[i1] * rstd) * w1);
      }
    }
    __syncthreads();
  }

  // QKV projection with vec4 and __ldg
  constexpr int TOTAL_ROWS = Q_SIZE + KV_SIZE + KV_SIZE;
  int rows_per_block = (TOTAL_ROWS + num_blocks - 1) / num_blocks;
  int row_start = block_id * rows_per_block;
  int row_end = min(row_start + rows_per_block, TOTAL_ROWS);

  for (int m_base = row_start; m_base < row_end; m_base += LDG_NUM_WARPS) {
    int m = m_base + warp_id;

    if (m < row_end) {
      const __nv_bfloat16 *weight_row;
      float *output_ptr;

      if (m < Q_SIZE) {
        weight_row = q_weight + m * HIDDEN_SIZE;
        output_ptr = q_out + m;
      } else if (m < Q_SIZE + KV_SIZE) {
        weight_row = k_weight + (m - Q_SIZE) * HIDDEN_SIZE;
        output_ptr = k_out + (m - Q_SIZE);
      } else {
        weight_row = v_weight + (m - Q_SIZE - KV_SIZE) * HIDDEN_SIZE;
        output_ptr = v_out + (m - Q_SIZE - KV_SIZE);
      }

      float sum = 0.0f;
#pragma unroll 4
      for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
        uint4 w_u4 =
            ldg_load_weight_u4(reinterpret_cast<const uint4 *>(weight_row + k));
        __nv_bfloat16 *w_ptr = reinterpret_cast<__nv_bfloat16 *>(&w_u4);
        float4 a1 = *reinterpret_cast<const float4 *>(s_norm + k);
        float4 a2 = *reinterpret_cast<const float4 *>(s_norm + k + 4);

        sum += __bfloat162float(w_ptr[0]) * a1.x +
               __bfloat162float(w_ptr[1]) * a1.y +
               __bfloat162float(w_ptr[2]) * a1.z +
               __bfloat162float(w_ptr[3]) * a1.w +
               __bfloat162float(w_ptr[4]) * a2.x +
               __bfloat162float(w_ptr[5]) * a2.y +
               __bfloat162float(w_ptr[6]) * a2.z +
               __bfloat162float(w_ptr[7]) * a2.w;
      }

      sum = ldg_warp_reduce_sum(sum);
      if (lane_id == 0) {
        *output_ptr = ldg_round_bf16(sum);
      }
    }
  }

  grid.sync();
}

// =============================================================================
// Attention with __ldg for KV cache + block divergence for prefetching
// =============================================================================

// Prefetch weights into L2 cache using __ldg reads
__device__ void
ldg_prefetch_weights_l2(const __nv_bfloat16 *__restrict__ weights,
                        int num_elements) {
  // Bulk L2 prefetch (Blackwell) or fallback to cached loads
  // Each thread prefetches strided elements to warm L2 cache
  float dummy = 0.0f;
  for (int i = threadIdx.x; i < num_elements; i += LDG_BLOCK_SIZE) {
    // Read but don't use - compiler won't optimize out due to volatile-like
    // __ldg
    dummy += __bfloat162float(__ldg(weights + i));
  }
  // Prevent optimization (result stored to shared but never used)
  __shared__ float s_dummy;
  if (threadIdx.x == 0)
    s_dummy = dummy;
}

__device__ inline __nv_bfloat16 *ldg_kv_ptr(
    __nv_bfloat16 *__restrict__ cache, int kv_head, int position,
    int64_t max_seq_len, const int *__restrict__ block_table,
    int64_t block_size) {
  if (block_table == nullptr) {
    int64_t offset = static_cast<int64_t>(kv_head) * max_seq_len * HEAD_DIM +
                     static_cast<int64_t>(position) * HEAD_DIM;
    return cache + offset;
  }
  int64_t logical_block = static_cast<int64_t>(position) / block_size;
  int64_t block_offset = static_cast<int64_t>(position) % block_size;
  int physical_block = block_table[logical_block];
  int64_t offset = static_cast<int64_t>(physical_block) * block_size * KV_SIZE +
                   block_offset * KV_SIZE +
                   static_cast<int64_t>(kv_head) * HEAD_DIM;
  return cache + offset;
}

__device__ void ldg_attention(
    auto &grid, int block_id, int num_blocks, float *__restrict__ q,
    float *__restrict__ k,
    const float *__restrict__ v, __nv_bfloat16 *__restrict__ k_cache,
    __nv_bfloat16 *__restrict__ v_cache, float *__restrict__ attn_out,
    int cache_len, int64_t max_seq_len, float attn_scale,
    // QK norm parameters (fused to eliminate a grid.sync)
    const __nv_bfloat16 *__restrict__ q_norm_weight,
    const __nv_bfloat16 *__restrict__ k_norm_weight,
    const __nv_bfloat16 *__restrict__ cos_table,
    const __nv_bfloat16 *__restrict__ sin_table, int position,
    const int *__restrict__ block_table, int64_t block_size,
    // Weights to prefetch during attention (for blocks not doing attention)
    const __nv_bfloat16 *__restrict__ o_weight,
    const __nv_bfloat16 *__restrict__ gate_weight,
    const __nv_bfloat16 *__restrict__ up_weight,
    const __nv_bfloat16 *__restrict__ down_weight,
    // Lightweight flag syncs (independent of grid barrier). null = use
    // grid.sync().
    unsigned int *__restrict__ kv_flag =
        nullptr, // block 0 signals KV cache ready
    unsigned int *__restrict__ attn_flag =
        nullptr, // attention blocks signal completion
    int layer_idx = 0) {
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;

  const int ATTN_BLOCKS = LDG_ATTN_BLOCKS;
  const __nv_bfloat16 *cos_pos = cos_table + position * HEAD_DIM;
  const __nv_bfloat16 *sin_pos = sin_table + position * HEAD_DIM;

  // -- Fused QK norm: block 0 handles all K heads, attention blocks handle Q --
  // Block 0: K norm + RoPE + KV cache write (8 heads × 128 dim — trivial)
  if (block_id == 0) {
    for (int h = warp_id; h < NUM_KV_HEADS; h += LDG_NUM_WARPS) {
      float *k_head = k + h * HEAD_DIM;
      const float *v_head = v + h * HEAD_DIM;
      __nv_bfloat16 *kc = ldg_kv_ptr(
          k_cache, h, position, max_seq_len, block_table, block_size);
      __nv_bfloat16 *vc = ldg_kv_ptr(
          v_cache, h, position, max_seq_len, block_table, block_size);

      float ss = 0.0f;
      for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE)
        ss += k_head[i] * k_head[i];
      ss = ldg_warp_reduce_sum(ss);
      float sc = rsqrtf(ss / float(HEAD_DIM) + LDG_RMS_EPS);
      sc = __shfl_sync(0xffffffff, sc, 0);

      float kl[HEAD_DIM / WARP_SIZE];
#pragma unroll
      for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++)
        kl[j] = ldg_round_bf16(
            ldg_round_bf16(k_head[i] * sc) *
            __bfloat162float(__ldg(k_norm_weight + i)));
#pragma unroll
      for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
        float cv = __bfloat162float(__ldg(cos_pos + i));
        float sv = __bfloat162float(__ldg(sin_pos + i));
        int po = (i < HEAD_DIM / 2) ? HEAD_DIM / 2 : -HEAD_DIM / 2;
        int pi = i + po, pj = pi / WARP_SIZE;
        float pv = __shfl_sync(0xffffffff, kl[pj], pi % WARP_SIZE);
        float kf =
            (i < HEAD_DIM / 2) ? kl[j] * cv - pv * sv : pv * sv + kl[j] * cv;
        kc[i] = __float2bfloat16(kf);
        vc[i] = __float2bfloat16(v_head[i]);
      }
    }
  }
  // Attention blocks: Q norm + RoPE for own head (warp 0 only — 128 elements)
  if (block_id < ATTN_BLOCKS && warp_id == 0) {
    int heads_per_block = (NUM_Q_HEADS + ATTN_BLOCKS - 1) / ATTN_BLOCKS;
    int head_start = block_id * heads_per_block;
    int head_end = min(head_start + heads_per_block, NUM_Q_HEADS);
    for (int qh_idx = head_start; qh_idx < head_end; qh_idx++) {
      float *qh = q + qh_idx * HEAD_DIM;
      float ss = 0.0f;
      for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE)
        ss += qh[i] * qh[i];
      ss = ldg_warp_reduce_sum(ss);
      float sc = rsqrtf(ss / float(HEAD_DIM) + LDG_RMS_EPS);
      sc = __shfl_sync(0xffffffff, sc, 0);
      float ql[HEAD_DIM / WARP_SIZE];
#pragma unroll
      for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++)
        ql[j] = ldg_round_bf16(
            ldg_round_bf16(qh[i] * sc) *
            __bfloat162float(__ldg(q_norm_weight + i)));
#pragma unroll
      for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
        float cv = __bfloat162float(__ldg(cos_pos + i));
        float sv = __bfloat162float(__ldg(sin_pos + i));
        int po = (i < HEAD_DIM / 2) ? HEAD_DIM / 2 : -HEAD_DIM / 2;
        int pi = i + po, pj = pi / WARP_SIZE;
        float pv = __shfl_sync(0xffffffff, ql[pj], pi % WARP_SIZE);
        qh[i] = ldg_round_bf16(
            (i < HEAD_DIM / 2) ? ql[j] * cv - pv * sv
                               : pv * sv + ql[j] * cv);
      }
    }
  }

  // Non-attention blocks: prefetch while QK norm runs above (overlapped work)
  if (LDG_PREFETCH_QK && block_id >= ATTN_BLOCKS) {
    int prefetch_block_id = block_id - ATTN_BLOCKS;
    int num_prefetch_blocks = num_blocks - ATTN_BLOCKS;
    int o_blocks = num_prefetch_blocks * 2 / 11;
    int gate_blocks = num_prefetch_blocks * 3 / 11;
    int up_blocks = num_prefetch_blocks * 3 / 11;
    if (o_blocks < 1)
      o_blocks = 1;

    if (prefetch_block_id < o_blocks) {
      int total = Q_SIZE * HIDDEN_SIZE;
      int epb = (total + o_blocks - 1) / o_blocks;
      int start = prefetch_block_id * epb;
      int count = min(epb, total - start);
      if (count > 0)
        ldg_prefetch_weights_l2(o_weight + start, count);
    } else if (prefetch_block_id < o_blocks + gate_blocks) {
      int adj = prefetch_block_id - o_blocks;
      int total = HIDDEN_SIZE * INTERMEDIATE_SIZE;
      int epb = (total + gate_blocks - 1) / gate_blocks;
      int start = adj * epb;
      int count = min(epb, total - start);
      if (count > 0)
        ldg_prefetch_weights_l2(gate_weight + start, count);
    } else if (prefetch_block_id < o_blocks + gate_blocks + up_blocks) {
      int adj = prefetch_block_id - o_blocks - gate_blocks;
      int total = HIDDEN_SIZE * INTERMEDIATE_SIZE;
      int epb = (total + up_blocks - 1) / up_blocks;
      int start = adj * epb;
      int count = min(epb, total - start);
      if (count > 0)
        ldg_prefetch_weights_l2(up_weight + start, count);
    } else {
      int adj = prefetch_block_id - o_blocks - gate_blocks - up_blocks;
      int db = num_prefetch_blocks - o_blocks - gate_blocks - up_blocks;
      int total = INTERMEDIATE_SIZE * HIDDEN_SIZE;
      int epb = (total + db - 1) / db;
      int start = adj * epb;
      int count = min(epb, total - start);
      if (count > 0)
        ldg_prefetch_weights_l2(down_weight + start, count);
    }
  }

  // --- Barrier #2: KV cache must be ready before attention ---
  if (kv_flag) {
    // All attention blocks need __syncthreads() to finish QK norm writes
    if (block_id < ATTN_BLOCKS) {
      __syncthreads();
    }
    // Block 0: signal KV cache is written (after intra-block sync). All
    // threads of block 0 fence their KV writes at GPU scope (fence is
    // per-thread) before thread 0 raises the flag, so attention blocks on
    // other SMs are guaranteed to see the freshly written K/V.
    if (block_id == 0) {
      asm volatile("fence.acq_rel.gpu;" ::: "memory");
      if (threadIdx.x == 0)
        atomicExch(kv_flag, (unsigned int)(layer_idx + 1));
    }
    // Blocks 1-15: wait for KV cache
    if (block_id > 0 && block_id < ATTN_BLOCKS) {
      if (threadIdx.x == 0) {
        volatile unsigned int *vf = (volatile unsigned int *)kv_flag;
        while (*vf < (unsigned int)(layer_idx + 1)) {
        }
        // Acquire fence: see the K/V writes published by block 0's
        // fence.acq_rel.gpu before the kv_flag signal.
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
      }
      __syncthreads();
    }
    // Blocks 16-127: skip (they don't do attention)
  } else {
    grid.sync();
  }

  // Shared memory for cross-warp reduction of online softmax
  __shared__ float s_max_score[LDG_NUM_WARPS];
  __shared__ float s_sum_exp[LDG_NUM_WARPS];
  __shared__ float s_out_acc[LDG_NUM_WARPS][HEAD_DIM];

  // Each of the 16 attention blocks handles one Q head
  int heads_per_block = (NUM_Q_HEADS + ATTN_BLOCKS - 1) / ATTN_BLOCKS;
  int head_start = block_id * heads_per_block;
  int head_end = min(head_start + heads_per_block, NUM_Q_HEADS);

  for (int qh = head_start; qh < head_end; qh++) {
    int kv_head = qh / (NUM_Q_HEADS / NUM_KV_HEADS);
    const float *q_head = q + qh * HEAD_DIM;
    float *out_head = attn_out + qh * HEAD_DIM;

    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float out_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    int q_idx = lane_id * 4;
    float q_local[4];
    q_local[0] = q_head[q_idx + 0];
    q_local[1] = q_head[q_idx + 1];
    q_local[2] = q_head[q_idx + 2];
    q_local[3] = q_head[q_idx + 3];

    // Each warp processes a subset of cache positions
    for (int pos = warp_id; pos < cache_len; pos += LDG_NUM_WARPS) {
      const __nv_bfloat16 *k_pos = ldg_kv_ptr(
          k_cache, kv_head, pos, max_seq_len, block_table, block_size);
      const __nv_bfloat16 *v_pos = ldg_kv_ptr(
          v_cache, kv_head, pos, max_seq_len, block_table, block_size);

      // Q @ K with __ldg
      float score = 0.0f;
      uint2 k_u2 = __ldg(reinterpret_cast<const uint2 *>(k_pos + q_idx));
      __nv_bfloat16 *k_ptr = reinterpret_cast<__nv_bfloat16 *>(&k_u2);
      score += q_local[0] * __bfloat162float(k_ptr[0]) +
               q_local[1] * __bfloat162float(k_ptr[1]) +
               q_local[2] * __bfloat162float(k_ptr[2]) +
               q_local[3] * __bfloat162float(k_ptr[3]);
      score = ldg_warp_reduce_sum(score) * attn_scale;
      score = __shfl_sync(0xffffffff, score, 0);

      float old_max = max_score;
      max_score = fmaxf(max_score, score);
      float exp_diff = fast_exp(old_max - max_score);
      sum_exp = sum_exp * exp_diff + fast_exp(score - max_score);

      float weight = fast_exp(score - max_score);
      uint2 v_u2 = __ldg(reinterpret_cast<const uint2 *>(v_pos + q_idx));
      __nv_bfloat16 *v_ptr = reinterpret_cast<__nv_bfloat16 *>(&v_u2);
      out_acc[0] = out_acc[0] * exp_diff + weight * __bfloat162float(v_ptr[0]);
      out_acc[1] = out_acc[1] * exp_diff + weight * __bfloat162float(v_ptr[1]);
      out_acc[2] = out_acc[2] * exp_diff + weight * __bfloat162float(v_ptr[2]);
      out_acc[3] = out_acc[3] * exp_diff + weight * __bfloat162float(v_ptr[3]);
    }

    // Store each warp's partial results to shared memory
    if (lane_id == 0) {
      s_max_score[warp_id] = max_score;
      s_sum_exp[warp_id] = sum_exp;
    }
    int out_base = lane_id * 4;
    s_out_acc[warp_id][out_base + 0] = out_acc[0];
    s_out_acc[warp_id][out_base + 1] = out_acc[1];
    s_out_acc[warp_id][out_base + 2] = out_acc[2];
    s_out_acc[warp_id][out_base + 3] = out_acc[3];
    __syncthreads();

    // Warp 0 combines results from all warps
    if (warp_id == 0) {
      // Find global max across all warps
      float global_max = s_max_score[0];
      for (int w = 1; w < LDG_NUM_WARPS; w++) {
        if (s_max_score[w] >
            -INFINITY) { // Only consider warps that processed positions
          global_max = fmaxf(global_max, s_max_score[w]);
        }
      }

      // Rescale and sum the partial results
      float total_sum_exp = 0.0f;
      float final_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};

      for (int w = 0; w < LDG_NUM_WARPS; w++) {
        if (s_max_score[w] >
            -INFINITY) { // Only consider warps that processed positions
          float scale = fast_exp(s_max_score[w] - global_max);
          total_sum_exp += s_sum_exp[w] * scale;

          int base = lane_id * 4;
          final_out[0] += s_out_acc[w][base + 0] * scale;
          final_out[1] += s_out_acc[w][base + 1] * scale;
          final_out[2] += s_out_acc[w][base + 2] * scale;
          final_out[3] += s_out_acc[w][base + 3] * scale;
        }
      }

      // Write final normalized output
      int base = lane_id * 4;
      out_head[base + 0] = ldg_round_bf16(final_out[0] / total_sum_exp);
      out_head[base + 1] = ldg_round_bf16(final_out[1] / total_sum_exp);
      out_head[base + 2] = ldg_round_bf16(final_out[2] / total_sum_exp);
      out_head[base + 3] = ldg_round_bf16(final_out[3] / total_sum_exp);
    }
    __syncthreads();
  }

  // --- Barrier #3: attention output must be complete before O proj ---
  if (attn_flag) {
    // Attention blocks (0-15): signal completion
    if (block_id < ATTN_BLOCKS) {
      asm volatile("fence.acq_rel.gpu;" ::: "memory");
      if (threadIdx.x == 0)
        atomicAdd(attn_flag, 1);
    }
    // Non-attention blocks: prefetch O+gate+up weights while waiting
    // (refreshes L2 so O proj + MLP reads hit L2 instead of DRAM)
    if (block_id >= ATTN_BLOCKS && threadIdx.x != 0 &&
        (threadIdx.x % LDG_PREFETCH_THREAD_STRIDE == 0)) {
      int prefetch_id = block_id - ATTN_BLOCKS;
      if (prefetch_id % LDG_PREFETCH_BLOCK_STRIDE != 0) {
        // Skip this block's prefetch work.
      } else {
        int npb = num_blocks - ATTN_BLOCKS; // 112 blocks
        // Split: ~20% O proj, ~40% gate, ~40% up (proportional to size)
        int o_total = Q_SIZE * HIDDEN_SIZE; // ~2M
        int g_total = LDG_PREFETCH_GATE ? (HIDDEN_SIZE * INTERMEDIATE_SIZE) : 0;
        int u_total = LDG_PREFETCH_UP ? (HIDDEN_SIZE * INTERMEDIATE_SIZE) : 0;
        int d_total = LDG_PREFETCH_DOWN ? (INTERMEDIATE_SIZE * HIDDEN_SIZE) : 0;
        int all_total = o_total + g_total + u_total + d_total; // ~11M
        int per = (all_total + npb - 1) / npb;
        int s = prefetch_id * per, e = min(s + per, all_total);
        int step = (LDG_BLOCK_SIZE - 1) * LDG_PREFETCH_ELEM_STRIDE;
        for (int i = s + (threadIdx.x - 1); i < e; i += step) {
          const __nv_bfloat16 *ptr;
          if (i < o_total)
            ptr = o_weight + i;
          else if (i < o_total + g_total)
            ptr = gate_weight + (i - o_total);
          else if (i < o_total + g_total + u_total)
            ptr = up_weight + (i - o_total - g_total);
          else
            ptr = down_weight + (i - o_total - g_total - u_total);
          asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr));
        }
      }
    }
    // Thread 0: wait for all 16 attention heads to finish
    if (threadIdx.x == 0) {
      unsigned int target = (unsigned int)(ATTN_BLOCKS * (layer_idx + 1));
      volatile unsigned int *vf = (volatile unsigned int *)attn_flag;
      while (*vf < target) {
      }
      // Acquire fence: see the g_attn_out writes published by the attention
      // blocks' fence.acq_rel.gpu before they atomicAdd'd attn_flag.
      asm volatile("fence.acq_rel.gpu;" ::: "memory");
    }
    __syncthreads();
  } else {
    grid.sync();
  }
}

// =============================================================================
// O Projection + Residual + PostNorm + MLP (all with __ldg)
// =============================================================================

__device__ void ldg_o_proj_postnorm_mlp(
    auto &grid, int block_id, int num_blocks,
    const __nv_bfloat16 *__restrict__ o_weight,
    const __nv_bfloat16 *__restrict__ post_norm_weight,
    const __nv_bfloat16 *__restrict__ gate_weight,
    const __nv_bfloat16 *__restrict__ up_weight,
    const __nv_bfloat16 *__restrict__ down_weight,
    const float *__restrict__ attn_out, float *__restrict__ g_residual,
    float *__restrict__ g_activations, float *__restrict__ g_mlp_intermediate,
    __nv_bfloat16 *__restrict__ hidden_out) {
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;
  __shared__ __align__(16) float s_attn[Q_SIZE];
  __shared__ __align__(16) float s_act[HIDDEN_SIZE];
  __shared__ __align__(16) float s_mlp[INTERMEDIATE_SIZE];

  // Cache attention output for reuse across rows in this block.
  for (int i = threadIdx.x; i < Q_SIZE; i += LDG_BLOCK_SIZE) {
    s_attn[i] = attn_out[i];
  }
  __syncthreads();

  // O Projection + Residual
  int hid_per_block = (HIDDEN_SIZE + num_blocks - 1) / num_blocks;
  int hid_start = block_id * hid_per_block;
  int hid_end = min(hid_start + hid_per_block, HIDDEN_SIZE);

  for (int m_base = hid_start; m_base < hid_end; m_base += LDG_NUM_WARPS) {
    int m = m_base + warp_id;

    if (m < hid_end) {
      const __nv_bfloat16 *o_row = o_weight + m * Q_SIZE;

      float sum = 0.0f;
#pragma unroll 4
      for (int k = lane_id * 8; k < Q_SIZE; k += WARP_SIZE * 8) {
        uint4 w_u4 =
            ldg_load_weight_u4(reinterpret_cast<const uint4 *>(o_row + k));
        __nv_bfloat16 *w_ptr = reinterpret_cast<__nv_bfloat16 *>(&w_u4);
        float4 a1 = *reinterpret_cast<const float4 *>(s_attn + k);
        float4 a2 = *reinterpret_cast<const float4 *>(s_attn + k + 4);

        sum += __bfloat162float(w_ptr[0]) * a1.x +
               __bfloat162float(w_ptr[1]) * a1.y +
               __bfloat162float(w_ptr[2]) * a1.z +
               __bfloat162float(w_ptr[3]) * a1.w +
               __bfloat162float(w_ptr[4]) * a2.x +
               __bfloat162float(w_ptr[5]) * a2.y +
               __bfloat162float(w_ptr[6]) * a2.z +
               __bfloat162float(w_ptr[7]) * a2.w;
      }

      sum = ldg_warp_reduce_sum(sum);
      if (lane_id == 0) {
        g_activations[m] = ldg_round_bf16(sum) + g_residual[m];
      }
    }
  }

  grid.sync();

  // ALL blocks compute post-attention RMSNorm redundantly (eliminates
  // grid.sync)
  {
    __shared__ float smem_reduce[LDG_NUM_WARPS];

    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE * 2) {
      float v0 = g_activations[i];
      s_act[i] = v0;
      local_sum_sq += v0 * v0;
      int i1 = i + LDG_BLOCK_SIZE;
      if (i1 < HIDDEN_SIZE) {
        float v1 = g_activations[i1];
        s_act[i1] = v1;
        local_sum_sq += v1 * v1;
      }
    }

    // Block 0 saves residual for later use
    if (block_id == 0) {
      for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE * 2) {
        g_residual[i] = ldg_round_bf16(s_act[i]);
        int i1 = i + LDG_BLOCK_SIZE;
        if (i1 < HIDDEN_SIZE) {
          g_residual[i1] = ldg_round_bf16(s_act[i1]);
        }
      }
    }

    local_sum_sq = ldg_warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) {
      smem_reduce[warp_id] = local_sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
      float sum = (lane_id < LDG_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
      sum = ldg_warp_reduce_sum(sum);
      if (lane_id == 0) {
        smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + LDG_RMS_EPS);
      }
    }
    __syncthreads();

    float rstd = smem_reduce[0];

    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE * 2) {
      float w0 = __bfloat162float(__ldg(post_norm_weight + i));
      s_act[i] = ldg_round_bf16(ldg_round_bf16(s_act[i] * rstd) * w0);
      int i1 = i + LDG_BLOCK_SIZE;
      if (i1 < HIDDEN_SIZE) {
        float w1 = __bfloat162float(__ldg(post_norm_weight + i1));
        s_act[i1] = ldg_round_bf16(ldg_round_bf16(s_act[i1] * rstd) * w1);
      }
    }
    __syncthreads();
  }

  // Gate + Up + SiLU
  int int_per_block = (INTERMEDIATE_SIZE + num_blocks - 1) / num_blocks;
  int int_start = block_id * int_per_block;
  int int_end = min(int_start + int_per_block, INTERMEDIATE_SIZE);

  for (int m_base = int_start; m_base < int_end; m_base += LDG_NUM_WARPS) {
    int m = m_base + warp_id;

    if (m < int_end) {
      const __nv_bfloat16 *gate_row = gate_weight + m * HIDDEN_SIZE;
      const __nv_bfloat16 *up_row = up_weight + m * HIDDEN_SIZE;

      float gate_sum = 0.0f, up_sum = 0.0f;
#pragma unroll 4
      for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
        uint4 g_u4 =
            ldg_load_weight_u4(reinterpret_cast<const uint4 *>(gate_row + k));
        uint4 u_u4 =
            ldg_load_weight_u4(reinterpret_cast<const uint4 *>(up_row + k));
        __nv_bfloat16 *g_ptr = reinterpret_cast<__nv_bfloat16 *>(&g_u4);
        __nv_bfloat16 *u_ptr = reinterpret_cast<__nv_bfloat16 *>(&u_u4);
        float4 a1 = *reinterpret_cast<const float4 *>(s_act + k);
        float4 a2 = *reinterpret_cast<const float4 *>(s_act + k + 4);

        gate_sum += __bfloat162float(g_ptr[0]) * a1.x +
                    __bfloat162float(g_ptr[1]) * a1.y +
                    __bfloat162float(g_ptr[2]) * a1.z +
                    __bfloat162float(g_ptr[3]) * a1.w +
                    __bfloat162float(g_ptr[4]) * a2.x +
                    __bfloat162float(g_ptr[5]) * a2.y +
                    __bfloat162float(g_ptr[6]) * a2.z +
                    __bfloat162float(g_ptr[7]) * a2.w;

        up_sum += __bfloat162float(u_ptr[0]) * a1.x +
                  __bfloat162float(u_ptr[1]) * a1.y +
                  __bfloat162float(u_ptr[2]) * a1.z +
                  __bfloat162float(u_ptr[3]) * a1.w +
                  __bfloat162float(u_ptr[4]) * a2.x +
                  __bfloat162float(u_ptr[5]) * a2.y +
                  __bfloat162float(u_ptr[6]) * a2.z +
                  __bfloat162float(u_ptr[7]) * a2.w;
      }

      gate_sum = ldg_warp_reduce_sum(gate_sum);
      up_sum = ldg_warp_reduce_sum(up_sum);

      if (lane_id == 0) {
        float gate_bf16 = ldg_round_bf16(gate_sum);
        float up_bf16 = ldg_round_bf16(up_sum);
        float silu_bf16 = ldg_round_bf16(ldg_silu(gate_bf16));
        g_mlp_intermediate[m] = ldg_round_bf16(silu_bf16 * up_bf16);
      }
    }
  }

  grid.sync();

  for (int i = threadIdx.x; i < INTERMEDIATE_SIZE; i += LDG_BLOCK_SIZE) {
    s_mlp[i] = g_mlp_intermediate[i];
  }
  __syncthreads();

  // Down projection + residual
  const float *mlp_in = g_mlp_intermediate;
  mlp_in = s_mlp;
  for (int m_base = hid_start; m_base < hid_end; m_base += LDG_NUM_WARPS) {
    int m = m_base + warp_id;

    if (m < hid_end) {
      const __nv_bfloat16 *down_row = down_weight + m * INTERMEDIATE_SIZE;

      float sum = 0.0f;
#pragma unroll 4
      for (int k = lane_id * 8; k < INTERMEDIATE_SIZE; k += WARP_SIZE * 8) {
        uint4 d_u4 =
            ldg_load_weight_u4(reinterpret_cast<const uint4 *>(down_row + k));
        __nv_bfloat16 *d_ptr = reinterpret_cast<__nv_bfloat16 *>(&d_u4);
        float4 a1 = *reinterpret_cast<const float4 *>(g_mlp_intermediate + k);
        float4 a2 =
            *reinterpret_cast<const float4 *>(g_mlp_intermediate + k + 4);

        sum += __bfloat162float(d_ptr[0]) * a1.x +
               __bfloat162float(d_ptr[1]) * a1.y +
               __bfloat162float(d_ptr[2]) * a1.z +
               __bfloat162float(d_ptr[3]) * a1.w +
               __bfloat162float(d_ptr[4]) * a2.x +
               __bfloat162float(d_ptr[5]) * a2.y +
               __bfloat162float(d_ptr[6]) * a2.z +
               __bfloat162float(d_ptr[7]) * a2.w;
      }

      sum = ldg_warp_reduce_sum(sum);
      if (lane_id == 0) {
        hidden_out[m] = __float2bfloat16(
            ldg_round_bf16(sum) + g_residual[m]);
      }
    }
  }

  grid.sync();
}


// =============================================================================
// Batched WMMA (tensor-core) GEMM utilities
// =============================================================================
using namespace nvcuda;

constexpr int WMMA_TILE = 16;

template <int STRIDE_K, int STRIDE_N>
__device__ void wmma_gemm(const float *__restrict__ H,
                          const __nv_bfloat16 *__restrict__ W,
                          float *__restrict__ C, int M, int N, int K,
                          int global_warp, int num_warps) {
  __shared__ __nv_bfloat16 s_a[WMMA_TILE * WMMA_TILE];
  const int tiles_M = (M + WMMA_TILE - 1) / WMMA_TILE;
  const int tiles_N = (N + WMMA_TILE - 1) / WMMA_TILE;
  const int total_tiles = tiles_M * tiles_N;

  for (int tile = global_warp; tile < total_tiles; tile += num_warps) {
    int tr = tile / tiles_N, tc = tile % tiles_N;
    int m0 = tr * WMMA_TILE, m_lim = min(m0 + WMMA_TILE, M);
    int n0 = tc * WMMA_TILE;

    wmma::fragment<wmma::accumulator, WMMA_TILE, WMMA_TILE, WMMA_TILE, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    for (int k = 0; k < K; k += WMMA_TILE) {
      for (int idx = threadIdx.x; idx < WMMA_TILE * WMMA_TILE; idx += LDG_BLOCK_SIZE) {
        int mi = idx / WMMA_TILE, ki = idx % WMMA_TILE;
        float v = 0.0f;
        if (m0 + mi < M && k + ki < K)
          v = H[(m0 + mi) * STRIDE_K + k + ki];
        s_a[idx] = __float2bfloat16(v);
      }
      __syncthreads();

      wmma::fragment<wmma::matrix_a, WMMA_TILE, WMMA_TILE, WMMA_TILE,
                     __nv_bfloat16, wmma::row_major> a;
      wmma::load_matrix_sync(a, s_a, WMMA_TILE);

      wmma::fragment<wmma::matrix_b, WMMA_TILE, WMMA_TILE, WMMA_TILE,
                     __nv_bfloat16, wmma::col_major> b;
      wmma::load_matrix_sync(b, W + n0 * K + k, K);

      wmma::mma_sync(acc, a, b, acc);
      // Ensure every warp has finished reading s_a before the next k
      // iteration's fill overwrites it.  Without this, a fast warp can
      // advance to the next fill and clobber s_a while a slow warp in
      // the same block is still loading it (deterministic shared-memory
      // race that only manifests at certain batch sizes / SM schedules).
      __syncthreads();
    }
    // The batch output buffers are padded to a multiple of 16 rows, so the
    // full-16-row store_matrix_sync (whose padded rows for M<16 are garbage)
    // stays within each buffer's padding and never overruns into neighbours.
    wmma::store_matrix_sync(C + m0 * STRIDE_N + n0, acc, STRIDE_N,
                            wmma::mem_row_major);
  }
}

// Batched WMMA O-proj + post-attn RMSNorm + Gate/Up + SiLU + Down + residual.
// Replaces the per-seq scalar ldg_o_proj_postnorm_mlp in the WMMA batch kernel.
// All GEMMs use the full grid (all blocks cooperate on the batched GEMM,
// M=batch_size), while RMSNorm / SiLU / residual-add are elementwise.
//
// Buffers (batch rows are contiguous, row-major):
//   g_attn_out:  [batch * Q_SIZE]  float  (input, consumed by O proj)
//   g_residual:  [batch * H]       float  (residual storage)
//   g_activations:[batch * H]      float  (O out → norm in → Down out)
//   g_mlp_inter: [batch * M]       float  (Gate output, then SiLU*Up result)
//   g_mlp_up:    [batch * M]       float  (Up output, temp)
//   hidden_out:  [batch * H]       bf16   (final hidden output)
__device__ void ldg_o_proj_mlp_wmma(
    auto &grid, int block_id, int grid_size, int batch_size,
    const __nv_bfloat16 *__restrict__ o_weight,
    const __nv_bfloat16 *__restrict__ post_norm_weight,
    const __nv_bfloat16 *__restrict__ gate_weight,
    const __nv_bfloat16 *__restrict__ up_weight,
    const __nv_bfloat16 *__restrict__ down_weight,
    float *__restrict__ g_attn_out,    // [batch * Q]
    float *__restrict__ g_residual,    // [batch * H]
    float *__restrict__ g_activations, // [batch * H]
    float *__restrict__ g_mlp_inter,   // [batch * M]
    float *__restrict__ g_mlp_up,      // [batch * M]
    __nv_bfloat16 *__restrict__ hidden_out) { // [batch * H]
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;
  int nw = grid_size * LDG_NUM_WARPS;
  int gw = block_id * LDG_NUM_WARPS + warp_id;

  constexpr int H = HIDDEN_SIZE, Q = Q_SIZE, M = INTERMEDIATE_SIZE;

  // --- 1: Batched O projection ---
  // g_activations[s*H + h] = sum_k g_attn_out[s*Q + k] * o_weight[h*Q + k]
  wmma_gemm<Q, H>(g_attn_out, o_weight, g_activations, batch_size, H, Q, gw, nw);
  grid.sync();

  // --- 2: Residual add (elementwise, all blocks) ---
  {
    int total = batch_size * H;
    for (int i = threadIdx.x + block_id * LDG_BLOCK_SIZE; i < total;
         i += grid_size * LDG_BLOCK_SIZE) {
      g_activations[i] = g_activations[i] + g_residual[i];
    }
  }
  grid.sync();

  // --- 3: Post-attention RMSNorm (per-seq, block 0 per seq) ---
  // RMSNorm is cheap relative to GEMM; let block 0 of each seq do it.
  // With the full grid cooperating, we assign seq s to block s (if s < grid_size).
  // For batch ≤ 5 and grid_size = 80, each seq gets its own block.
  {
    __shared__ float smem_reduce[LDG_NUM_WARPS];
    for (int s = block_id; s < batch_size; s += grid_size) {
      float *act = g_activations + s * H;
      float *res = g_residual + s * H;

      float ss = 0.0f;
      for (int i = threadIdx.x; i < H; i += LDG_BLOCK_SIZE * 2) {
        float v0 = act[i]; ss += v0 * v0;
        int i1 = i + LDG_BLOCK_SIZE;
        if (i1 < H) { float v1 = act[i1]; ss += v1 * v1; }
      }
      ss = ldg_warp_reduce_sum(ss);
      if (lane_id == 0) smem_reduce[warp_id] = ss;
      __syncthreads();
      if (warp_id == 0) {
        float sum = (lane_id < LDG_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
        sum = ldg_warp_reduce_sum(sum);
        if (lane_id == 0) smem_reduce[0] = rsqrtf(sum / float(H) + LDG_RMS_EPS);
      }
      __syncthreads();
      float rstd = smem_reduce[0];

      for (int i = threadIdx.x; i < H; i += LDG_BLOCK_SIZE * 2) {
        res[i] = ldg_round_bf16(act[i]);
        float w0 = __bfloat162float(__ldg(post_norm_weight + i));
        act[i] = ldg_round_bf16(ldg_round_bf16(act[i] * rstd) * w0);
        int i1 = i + LDG_BLOCK_SIZE;
        if (i1 < H) {
          res[i1] = ldg_round_bf16(act[i1]);
          float w1 = __bfloat162float(__ldg(post_norm_weight + i1));
          act[i1] = ldg_round_bf16(ldg_round_bf16(act[i1] * rstd) * w1);
        }
      }
      __syncthreads();
    }
  }
  grid.sync();

  // --- 4: Batched Gate projection ---
  // g_mlp_inter[s*M + m] = sum_k g_activations[s*H + k] * gate_weight[m*H + k]
  wmma_gemm<H, M>(g_activations, gate_weight, g_mlp_inter, batch_size, M, H, gw, nw);
  grid.sync();

  // --- 5: Batched Up projection ---
  // g_mlp_up[s*M + m] = sum_k g_activations[s*H + k] * up_weight[m*H + k]
  wmma_gemm<H, M>(g_activations, up_weight, g_mlp_up, batch_size, M, H, gw, nw);
  grid.sync();

  // --- 6: SiLU(gate) * up (elementwise, all blocks) ---
  {
    int total = batch_size * M;
    for (int i = threadIdx.x + block_id * LDG_BLOCK_SIZE; i < total;
         i += grid_size * LDG_BLOCK_SIZE) {
      float g = g_mlp_inter[i];
      float u = g_mlp_up[i];
      g_mlp_inter[i] = ldg_round_bf16(ldg_round_bf16(ldg_silu(g)) * u);
    }
  }
  grid.sync();

  // --- 7: Batched Down projection → g_activations (float) ---
  // g_activations[s*H + h] = sum_k g_mlp_inter[s*M + k] * down_weight[h*M + k]
  wmma_gemm<M, H>(g_mlp_inter, down_weight, g_activations, batch_size, H, M, gw, nw);
  grid.sync();

  // --- 8: Residual add + bf16 cast (elementwise, all blocks) ---
  {
    int total = batch_size * H;
    for (int i = threadIdx.x + block_id * LDG_BLOCK_SIZE; i < total;
         i += grid_size * LDG_BLOCK_SIZE) {
      hidden_out[i] = __float2bfloat16(
          ldg_round_bf16(g_activations[i]) + g_residual[i]);
    }
  }
  grid.sync();
}

// =============================================================================
// WMMA-batched decode kernel: syncs all blocks globally, uses WMMA tensor-core
// for all GEMMs (QKV, O, Gate, Up, Down), per-seq attention.
// =============================================================================

__global__ void __launch_bounds__(LDG_BLOCK_SIZE, 1)
ldg_decode_kernel_batch_wmma(
    const __nv_bfloat16 *__restrict__ embed_weight,
    const LDGLayerWeights *__restrict__ layer_weights,
    const __nv_bfloat16 *__restrict__ final_norm_weight,
    const __nv_bfloat16 *__restrict__ cos_table,
    const __nv_bfloat16 *__restrict__ sin_table,
    __nv_bfloat16 *__restrict__ k_cache, __nv_bfloat16 *__restrict__ v_cache,
    __nv_bfloat16 *__restrict__ hidden_buffer,
    float *__restrict__ g_activations, float *__restrict__ g_residual,
    float *__restrict__ g_q, float *__restrict__ g_k, float *__restrict__ g_v,
    float *__restrict__ g_attn_out, float *__restrict__ g_mlp_intermediate,
    float *__restrict__ g_mlp_up,
    float *__restrict__ g_normalized,
    unsigned int *__restrict__ barrier_counter,
    unsigned int *__restrict__ barrier_sense,
    unsigned int *__restrict__ kv_flag, unsigned int *__restrict__ attn_flag,
    int num_layers, const int64_t *__restrict__ input_token_ids,
    const int64_t *__restrict__ positions,
    const int *__restrict__ block_tables, int batch_size,
    int block_table_stride, int block_size, int num_cache_blocks,
    int blocks_per_seq, float attn_scale) {

  if (batch_size > LDG_MAX_BATCH) return;

  // Read the current generation from barrier_sense (like ldg_decode_one does)
  // so we don't need ldg_reset_decode_state between no-sync steps.
  unsigned int initial_generation = *barrier_sense;
  AtomicGridSync grid{barrier_counter, barrier_sense,
                      (unsigned int)(blocks_per_seq * batch_size),
                      initial_generation};

  int block_id = blockIdx.x;
  int seq_id = block_id / blocks_per_seq;
  if (seq_id >= batch_size) return;

  // Reset per-seq flags (like ldg_decode_one does) so we don't need
  // ldg_reset_decode_state between no-sync steps.
  if (block_id % blocks_per_seq == 0 && threadIdx.x == 0) {
    atomicExch(kv_flag + seq_id, 0u);
    atomicExch(attn_flag + seq_id, 0u);
  }
  grid.sync();

  constexpr int H = HIDDEN_SIZE, Q = Q_SIZE, K = KV_SIZE, M = INTERMEDIATE_SIZE;
  int64_t kvls = block_tables
      ? static_cast<int64_t>(num_cache_blocks) * block_size * K
      : static_cast<int64_t>(NUM_KV_HEADS) * block_size * num_cache_blocks * HEAD_DIM;

  for (int layer = 0; layer < num_layers; layer++) {
    const LDGLayerWeights &w = layer_weights[layer];
    __nv_bfloat16 *lk = k_cache + layer * kvls;
    __nv_bfloat16 *lv = v_cache + layer * kvls;

    // --- 1: Per-seq RMSNorm ---
    {
      const __nv_bfloat16 *inp = (layer == 0)
          ? (embed_weight + input_token_ids[seq_id] * HIDDEN_SIZE)
          : (hidden_buffer + seq_id * H);
      float *on = g_activations + seq_id * H;
      float *or_ = g_residual + seq_id * H;
      int wid = threadIdx.x / WARP_SIZE, lid = threadIdx.x % WARP_SIZE;
      __shared__ float sr[LDG_NUM_WARPS];

      float ss = 0.0f;
      for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE * 2) {
        float v0 = __bfloat162float(__ldg(inp + i)); on[i] = v0; ss += v0 * v0;
        int i1 = i + LDG_BLOCK_SIZE;
        if (i1 < HIDDEN_SIZE) { float v1 = __bfloat162float(__ldg(inp + i1)); on[i1] = v1; ss += v1 * v1; }
      }
      if (block_id % blocks_per_seq == 0)
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE * 2) {
          or_[i] = on[i]; int i1 = i + LDG_BLOCK_SIZE; if (i1 < HIDDEN_SIZE) or_[i1] = on[i1];
        }
      ss = ldg_warp_reduce_sum(ss); if (lid == 0) sr[wid] = ss; __syncthreads();
      if (wid == 0) { float sum=(lid<LDG_NUM_WARPS)?sr[lid]:0.0f; sum=ldg_warp_reduce_sum(sum); if(lid==0)sr[0]=rsqrtf(sum/HIDDEN_SIZE+LDG_RMS_EPS); }
      __syncthreads(); float rs = sr[0];
      for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE * 2) {
        float w0=__bfloat162float(__ldg(w.input_layernorm_weight+i)); on[i]=ldg_round_bf16(ldg_round_bf16(on[i]*rs)*w0);
        int i1=i+LDG_BLOCK_SIZE; if(i1<HIDDEN_SIZE){float w1=__bfloat162float(__ldg(w.input_layernorm_weight+i1)); on[i1]=ldg_round_bf16(ldg_round_bf16(on[i1]*rs)*w1);}
      }
    }
    grid.sync();

    // --- 2: Batched QKV WMMA ---
    {
      int nw = gridDim.x * LDG_NUM_WARPS, gw = block_id * LDG_NUM_WARPS + (threadIdx.x/WARP_SIZE);
      wmma_gemm<H, Q>(g_activations, w.q_proj_weight, g_q, batch_size, Q_SIZE, HIDDEN_SIZE, gw, nw);
      wmma_gemm<H, K>(g_activations, w.k_proj_weight, g_k, batch_size, KV_SIZE, HIDDEN_SIZE, gw, nw);
      wmma_gemm<H, K>(g_activations, w.v_proj_weight, g_v, batch_size, KV_SIZE, HIDDEN_SIZE, gw, nw);
    }
    grid.sync();

    // --- 3: ldg_attention ---
    {
      int pos2 = static_cast<int>(positions[seq_id]);
      const int *btbl = block_tables ? (block_tables + seq_id * block_table_stride) : nullptr;
      ldg_attention(grid, block_id % blocks_per_seq, blocks_per_seq,
                    g_q + seq_id * Q, g_k + seq_id * K, g_v + seq_id * K,
                    lk, lv, g_attn_out + seq_id * Q,
                    pos2 + 1, block_size * num_cache_blocks, attn_scale,
                    w.q_norm_weight, w.k_norm_weight, cos_table, sin_table,
                    pos2, btbl, block_size, w.o_proj_weight,
                    w.gate_proj_weight, w.up_proj_weight, w.down_proj_weight,
                    kv_flag + seq_id, attn_flag + seq_id, layer);
    }

    // Full grid sync: ldg_attention uses per-seq flag sync, so different
    // sequences may finish attention at different times. ldg_o_proj_mlp_wmma
    // uses ALL blocks and reads ALL seqs' g_attn_out, so every seq's attention
    // must be complete before the batched GEMM starts.
    grid.sync();

    // --- 4-9: Batched WMMA O proj + RMSNorm + Gate/Up/SiLU/Down ---
    // All blocks cooperate on the batched GEMM (M=batch_size), unlike the
    // old per-seq scalar path which only used blocks_per_seq blocks.
    ldg_o_proj_mlp_wmma(
        grid, block_id, gridDim.x, batch_size, w.o_proj_weight,
        w.post_attn_layernorm_weight, w.gate_proj_weight,
        w.up_proj_weight, w.down_proj_weight,
        g_attn_out, g_residual, g_activations, g_mlp_intermediate,
        g_mlp_up, hidden_buffer);
    grid.sync();
  }

  // Final RMSNorm (one block per seq)
  if (block_id % blocks_per_seq == 0) {
    int wid=threadIdx.x/WARP_SIZE,lid=threadIdx.x%WARP_SIZE; __shared__ float srf[LDG_NUM_WARPS];
    float *no=g_normalized+seq_id*H,*ac=g_activations+seq_id*H;
    float ss5=0;for(int i=threadIdx.x;i<HIDDEN_SIZE;i+=LDG_BLOCK_SIZE*2){float v=__bfloat162float(hidden_buffer[seq_id*H+i]);ac[i]=v;ss5+=v*v;int i1=i+LDG_BLOCK_SIZE;if(i1<HIDDEN_SIZE){float v1=__bfloat162float(hidden_buffer[seq_id*H+i1]);ac[i1]=v1;ss5+=v1*v1;}}
    ss5=ldg_warp_reduce_sum(ss5);if(lid==0)srf[wid]=ss5;__syncthreads();
    if(wid==0){float sum=(lid<LDG_NUM_WARPS)?srf[lid]:0.0f;sum=ldg_warp_reduce_sum(sum);if(lid==0)srf[0]=rsqrtf(sum/HIDDEN_SIZE+LDG_RMS_EPS);}
    __syncthreads();float rs5=srf[0];
    for(int i=threadIdx.x;i<HIDDEN_SIZE;i+=LDG_BLOCK_SIZE*2){float w0=__bfloat162float(__ldg(final_norm_weight+i));no[i]=ldg_round_bf16(ldg_round_bf16(ac[i]*rs5)*w0);int i1=i+LDG_BLOCK_SIZE;if(i1<HIDDEN_SIZE){float w1=__bfloat162float(__ldg(final_norm_weight+i1));no[i1]=ldg_round_bf16(ldg_round_bf16(ac[i1]*rs5)*w1);}}
  }
}

__global__ void ldg_lm_head_phase1(const float *__restrict__ hidden,
                                   const __nv_bfloat16 *__restrict__ weight,
                                   float *__restrict__ block_max_vals,
                                   int *__restrict__ block_max_idxs) {
  __shared__ __align__(16) float s_hidden[HIDDEN_SIZE];

  for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_LM_BLOCK_SIZE) {
    s_hidden[i] = hidden[i];
  }
  __syncthreads();

  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;

  int rows_per_block = (LDG_VOCAB_SIZE + gridDim.x - 1) / gridDim.x;
  int row_start = blockIdx.x * rows_per_block;
  int row_end = min(row_start + rows_per_block, LDG_VOCAB_SIZE);

  float local_max = -INFINITY;
  int local_max_idx = -1;

  int warp_stride = LDG_LM_BLOCK_SIZE / WARP_SIZE;
  int base = row_start + warp_id * LDG_LM_ROWS_PER_WARP;

  for (int m_base = base; m_base < row_end;
       m_base += warp_stride * LDG_LM_ROWS_PER_WARP) {
    int rows[LDG_LM_ROWS_PER_WARP];
    bool valid[LDG_LM_ROWS_PER_WARP];
#pragma unroll
    for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
      rows[r] = m_base + r;
      valid[r] = rows[r] < row_end;
    }

    float sum[LDG_LM_ROWS_PER_WARP];
#pragma unroll
    for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
      sum[r] = 0.0f;
    }

#pragma unroll 4
    for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
      float4 a1 = *reinterpret_cast<const float4 *>(s_hidden + k);
      float4 a2 = *reinterpret_cast<const float4 *>(s_hidden + k + 4);

#pragma unroll
      for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
        if (!valid[r]) {
          continue;
        }
        const __nv_bfloat16 *w_row = weight + rows[r] * HIDDEN_SIZE;
        uint4 w_u4 =
            ldg_load_weight_u4(reinterpret_cast<const uint4 *>(w_row + k));
        __nv_bfloat16 *w_ptr = reinterpret_cast<__nv_bfloat16 *>(&w_u4);

        sum[r] += __bfloat162float(w_ptr[0]) * a1.x +
                  __bfloat162float(w_ptr[1]) * a1.y +
                  __bfloat162float(w_ptr[2]) * a1.z +
                  __bfloat162float(w_ptr[3]) * a1.w +
                  __bfloat162float(w_ptr[4]) * a2.x +
                  __bfloat162float(w_ptr[5]) * a2.y +
                  __bfloat162float(w_ptr[6]) * a2.z +
                  __bfloat162float(w_ptr[7]) * a2.w;
      }
    }

#pragma unroll
    for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
      if (!valid[r]) {
        continue;
      }
      float reduced = ldg_round_bf16(ldg_warp_reduce_sum(sum[r]));
      if (lane_id == 0 && reduced > local_max) {
        local_max = reduced;
        local_max_idx = rows[r];
      }
    }
  }

  local_max = __shfl_sync(0xffffffff, local_max, 0);
  local_max_idx = __shfl_sync(0xffffffff, local_max_idx, 0);

  __shared__ float warp_max[LDG_LM_BLOCK_SIZE / WARP_SIZE];
  __shared__ int warp_idx[LDG_LM_BLOCK_SIZE / WARP_SIZE];

  if (lane_id == 0) {
    warp_max[warp_id] = local_max;
    warp_idx[warp_id] = local_max_idx;
  }
  __syncthreads();

  if (warp_id == 0) {
    float max_val = (lane_id < LDG_LM_BLOCK_SIZE / WARP_SIZE)
                        ? warp_max[lane_id]
                        : -INFINITY;
    int max_idx =
        (lane_id < LDG_LM_BLOCK_SIZE / WARP_SIZE) ? warp_idx[lane_id] : -1;

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      float other_val = __shfl_down_sync(0xffffffff, max_val, offset);
      int other_idx = __shfl_down_sync(0xffffffff, max_idx, offset);
      if (other_val > max_val) {
        max_val = other_val;
        max_idx = other_idx;
      }
    }

    if (lane_id == 0) {
      block_max_vals[blockIdx.x] = max_val;
      block_max_idxs[blockIdx.x] = max_idx;
    }
  }
}

__global__ void ldg_lm_head_phase2(const float *__restrict__ block_max_vals,
                                   const int *__restrict__ block_max_idxs,
                                   int *__restrict__ output_token,
                                   int num_blocks) {
  __shared__ float s_max_vals[1024];
  __shared__ int s_max_idxs[1024];

  int tid = threadIdx.x;

  float local_max = -INFINITY;
  int local_idx = -1;

  for (int i = tid; i < num_blocks; i += blockDim.x) {
    float val = block_max_vals[i];
    if (val > local_max) {
      local_max = val;
      local_idx = block_max_idxs[i];
    }
  }

  s_max_vals[tid] = local_max;
  s_max_idxs[tid] = local_idx;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (s_max_vals[tid + s] > s_max_vals[tid]) {
        s_max_vals[tid] = s_max_vals[tid + s];
        s_max_idxs[tid] = s_max_idxs[tid + s];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    *output_token = s_max_idxs[0];
  }
}

__global__ void ldg_lm_head_fused(const float *__restrict__ hidden,
                                  const __nv_bfloat16 *__restrict__ weight,
                                  float *__restrict__ block_max_vals,
                                  int *__restrict__ block_max_idxs,
                                  int *__restrict__ output_token,
                                  unsigned int *__restrict__ counter,
                                  int num_blocks) {
  __shared__ __align__(16) float s_hidden[HIDDEN_SIZE];

  for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_LM_BLOCK_SIZE) {
    s_hidden[i] = hidden[i];
  }
  __syncthreads();

  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;

  int rows_per_block = (LDG_VOCAB_SIZE + gridDim.x - 1) / gridDim.x;
  int row_start = blockIdx.x * rows_per_block;
  int row_end = min(row_start + rows_per_block, LDG_VOCAB_SIZE);

  float local_max = -INFINITY;
  int local_max_idx = -1;

  int warp_stride = LDG_LM_BLOCK_SIZE / WARP_SIZE;
  int base = row_start + warp_id * LDG_LM_ROWS_PER_WARP;

  for (int m_base = base; m_base < row_end;
       m_base += warp_stride * LDG_LM_ROWS_PER_WARP) {
    int rows[LDG_LM_ROWS_PER_WARP];
    bool valid[LDG_LM_ROWS_PER_WARP];
#pragma unroll
    for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
      rows[r] = m_base + r;
      valid[r] = rows[r] < row_end;
    }

    float sum[LDG_LM_ROWS_PER_WARP];
#pragma unroll
    for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
      sum[r] = 0.0f;
    }

#pragma unroll 4
    for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
      float4 a1 = *reinterpret_cast<const float4 *>(s_hidden + k);
      float4 a2 = *reinterpret_cast<const float4 *>(s_hidden + k + 4);

#pragma unroll
      for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
        if (!valid[r]) {
          continue;
        }
        const __nv_bfloat16 *w_row = weight + rows[r] * HIDDEN_SIZE;
        uint4 w_u4 =
            ldg_load_weight_u4(reinterpret_cast<const uint4 *>(w_row + k));
        __nv_bfloat16 *w_ptr = reinterpret_cast<__nv_bfloat16 *>(&w_u4);

        sum[r] += __bfloat162float(w_ptr[0]) * a1.x +
                  __bfloat162float(w_ptr[1]) * a1.y +
                  __bfloat162float(w_ptr[2]) * a1.z +
                  __bfloat162float(w_ptr[3]) * a1.w +
                  __bfloat162float(w_ptr[4]) * a2.x +
                  __bfloat162float(w_ptr[5]) * a2.y +
                  __bfloat162float(w_ptr[6]) * a2.z +
                  __bfloat162float(w_ptr[7]) * a2.w;
      }
    }

#pragma unroll
    for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
      if (!valid[r]) {
        continue;
      }
      float reduced = ldg_round_bf16(ldg_warp_reduce_sum(sum[r]));
      if (lane_id == 0 && reduced > local_max) {
        local_max = reduced;
        local_max_idx = rows[r];
      }
    }
  }

  local_max = __shfl_sync(0xffffffff, local_max, 0);
  local_max_idx = __shfl_sync(0xffffffff, local_max_idx, 0);

  __shared__ float warp_max[LDG_LM_BLOCK_SIZE / WARP_SIZE];
  __shared__ int warp_idx[LDG_LM_BLOCK_SIZE / WARP_SIZE];

  if (lane_id == 0) {
    warp_max[warp_id] = local_max;
    warp_idx[warp_id] = local_max_idx;
  }
  __syncthreads();

  if (warp_id == 0) {
    float max_val = (lane_id < LDG_LM_BLOCK_SIZE / WARP_SIZE)
                        ? warp_max[lane_id]
                        : -INFINITY;
    int max_idx =
        (lane_id < LDG_LM_BLOCK_SIZE / WARP_SIZE) ? warp_idx[lane_id] : -1;

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      float other_val = __shfl_down_sync(0xffffffff, max_val, offset);
      int other_idx = __shfl_down_sync(0xffffffff, max_idx, offset);
      if (other_val > max_val) {
        max_val = other_val;
        max_idx = other_idx;
      }
    }

    if (lane_id == 0) {
      block_max_vals[blockIdx.x] = max_val;
      block_max_idxs[blockIdx.x] = max_idx;
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    atomicAdd(counter, 1);
  }

  if (blockIdx.x == 0) {
    if (threadIdx.x == 0) {
      volatile unsigned int *vc = (volatile unsigned int *)counter;
      while (*vc < (unsigned int)num_blocks) {
      }
      __threadfence();
    }
    __syncthreads();

    __shared__ float s_max_vals[1024];
    __shared__ int s_max_idxs[1024];

    int tid = threadIdx.x;
    float local_max2 = -INFINITY;
    int local_idx2 = -1;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
      float val = block_max_vals[i];
      if (val > local_max2) {
        local_max2 = val;
        local_idx2 = block_max_idxs[i];
      }
    }

    s_max_vals[tid] = local_max2;
    s_max_idxs[tid] = local_idx2;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s) {
        if (s_max_vals[tid + s] > s_max_vals[tid]) {
          s_max_vals[tid] = s_max_vals[tid + s];
          s_max_idxs[tid] = s_max_idxs[tid + s];
        }
      }
      __syncthreads();
    }

    if (tid == 0) {
      *output_token = s_max_idxs[0];
    }
  }
}

// =============================================================================
__global__ void ldg_lm_head_fused_batch(
    const float *__restrict__ hidden,
    const __nv_bfloat16 *__restrict__ weight,
    float *__restrict__ block_max_vals,
    int *__restrict__ block_max_idxs,
    int *__restrict__ output_token,
    unsigned int *__restrict__ counter, int num_blocks,
    int batch_size, int hidden_stride, int block_max_stride) {
  int global_block = blockIdx.x;
  int seq_id = global_block / num_blocks;
  int local_block = global_block % num_blocks;
  if (seq_id >= batch_size)
    return;

  hidden += seq_id * hidden_stride;
  block_max_vals += seq_id * block_max_stride;
  block_max_idxs += seq_id * block_max_stride;
  output_token += seq_id;
  counter += seq_id;

  __shared__ __align__(16) float s_hidden[HIDDEN_SIZE];

  for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_LM_BLOCK_SIZE) {
    s_hidden[i] = hidden[i];
  }
  __syncthreads();

  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;

  int rows_per_block = (LDG_VOCAB_SIZE + num_blocks - 1) / num_blocks;
  int row_start = local_block * rows_per_block;
  int row_end = min(row_start + rows_per_block, LDG_VOCAB_SIZE);

  float local_max = -INFINITY;
  int local_max_idx = -1;

  int warp_stride = LDG_LM_BLOCK_SIZE / WARP_SIZE;
  int base = row_start + warp_id * LDG_LM_ROWS_PER_WARP;

  for (int m_base = base; m_base < row_end;
       m_base += warp_stride * LDG_LM_ROWS_PER_WARP) {
    int rows[LDG_LM_ROWS_PER_WARP];
    bool valid[LDG_LM_ROWS_PER_WARP];
#pragma unroll
    for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
      rows[r] = m_base + r;
      valid[r] = rows[r] < row_end;
    }

    float sum[LDG_LM_ROWS_PER_WARP];
#pragma unroll
    for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
      sum[r] = 0.0f;
    }

#pragma unroll 4
    for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
      float4 a1 = *reinterpret_cast<const float4 *>(s_hidden + k);
      float4 a2 = *reinterpret_cast<const float4 *>(s_hidden + k + 4);

#pragma unroll
      for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
        if (!valid[r]) {
          continue;
        }
        const __nv_bfloat16 *w_row = weight + rows[r] * HIDDEN_SIZE;
        uint4 w_u4 =
            ldg_load_weight_u4(reinterpret_cast<const uint4 *>(w_row + k));
        __nv_bfloat16 *w_ptr = reinterpret_cast<__nv_bfloat16 *>(&w_u4);

        sum[r] += __bfloat162float(w_ptr[0]) * a1.x +
                  __bfloat162float(w_ptr[1]) * a1.y +
                  __bfloat162float(w_ptr[2]) * a1.z +
                  __bfloat162float(w_ptr[3]) * a1.w +
                  __bfloat162float(w_ptr[4]) * a2.x +
                  __bfloat162float(w_ptr[5]) * a2.y +
                  __bfloat162float(w_ptr[6]) * a2.z +
                  __bfloat162float(w_ptr[7]) * a2.w;
      }
    }

#pragma unroll
    for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
      if (!valid[r]) {
        continue;
      }
      float reduced = ldg_round_bf16(ldg_warp_reduce_sum(sum[r]));
      if (lane_id == 0 && reduced > local_max) {
        local_max = reduced;
        local_max_idx = rows[r];
      }
    }
  }

  local_max = __shfl_sync(0xffffffff, local_max, 0);
  local_max_idx = __shfl_sync(0xffffffff, local_max_idx, 0);

  __shared__ float warp_max[LDG_LM_BLOCK_SIZE / WARP_SIZE];
  __shared__ int warp_idx[LDG_LM_BLOCK_SIZE / WARP_SIZE];

  if (lane_id == 0) {
    warp_max[warp_id] = local_max;
    warp_idx[warp_id] = local_max_idx;
  }
  __syncthreads();

  if (warp_id == 0) {
    float max_val = (lane_id < LDG_LM_BLOCK_SIZE / WARP_SIZE)
                        ? warp_max[lane_id]
                        : -INFINITY;
    int max_idx =
        (lane_id < LDG_LM_BLOCK_SIZE / WARP_SIZE) ? warp_idx[lane_id] : -1;

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      float other_val = __shfl_down_sync(0xffffffff, max_val, offset);
      int other_idx = __shfl_down_sync(0xffffffff, max_idx, offset);
      if (other_val > max_val) {
        max_val = other_val;
        max_idx = other_idx;
      }
    }

    if (lane_id == 0) {
      block_max_vals[local_block] = max_val;
      block_max_idxs[local_block] = max_idx;
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    atomicAdd(counter, 1);
  }

  if (local_block == 0) {
    if (threadIdx.x == 0) {
      volatile unsigned int *vc = (volatile unsigned int *)counter;
      while (*vc < (unsigned int)num_blocks) {
      }
      __threadfence();
    }
    __syncthreads();

    __shared__ float s_max_vals[1024];
    __shared__ int s_max_idxs[1024];

    int tid = threadIdx.x;
    float local_max2 = -INFINITY;
    int local_idx2 = -1;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
      float val = block_max_vals[i];
      if (val > local_max2) {
        local_max2 = val;
        local_idx2 = block_max_idxs[i];
      }
    }

    s_max_vals[tid] = local_max2;
    s_max_idxs[tid] = local_idx2;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s) {
        if (s_max_vals[tid + s] > s_max_vals[tid]) {
          s_max_vals[tid] = s_max_vals[tid + s];
          s_max_idxs[tid] = s_max_idxs[tid + s];
        }
      }
      __syncthreads();
    }

    if (tid == 0) {
      *output_token = s_max_idxs[0];
    }
  }

}

// Persistent (non-cooperative) decode kernel
// =============================================================================

__global__ void
__launch_bounds__(LDG_BLOCK_SIZE, 1) ldg_decode_kernel_persistent(
    const __nv_bfloat16 *__restrict__ embed_weight,
    const LDGLayerWeights *__restrict__ layer_weights,
    const __nv_bfloat16 *__restrict__ final_norm_weight,
    const __nv_bfloat16 *__restrict__ cos_table,
    const __nv_bfloat16 *__restrict__ sin_table,
    __nv_bfloat16 *__restrict__ k_cache, __nv_bfloat16 *__restrict__ v_cache,
    __nv_bfloat16 *__restrict__ hidden_buffer,
    float *__restrict__ g_activations, float *__restrict__ g_residual,
    float *__restrict__ g_q, float *__restrict__ g_k, float *__restrict__ g_v,
    float *__restrict__ g_attn_out, float *__restrict__ g_mlp_intermediate,
    float *__restrict__ g_normalized,
    unsigned int *__restrict__ barrier_counter,
    unsigned int *__restrict__ barrier_sense,
    unsigned int *__restrict__ kv_flag, unsigned int *__restrict__ attn_flag,
    int num_layers, const int *__restrict__ d_position,
    const int *__restrict__ d_token_id, int max_seq_len, float attn_scale) {
  // Read mutable params from device memory (allows CUDA graph replay)
  int position = *d_position;
  int input_token_id = *d_token_id;
  int cache_len = position + 1;
  int block_id = blockIdx.x;

  // Host-side stream reset precedes this kernel launch.
  AtomicGridSync grid{barrier_counter, barrier_sense, (unsigned int)gridDim.x,
                      0};

  // First layer reads embed row directly (no separate embed + barrier needed)
  const __nv_bfloat16 *embed_row = embed_weight + input_token_id * HIDDEN_SIZE;

  int kv_cache_layer_stride = NUM_KV_HEADS * max_seq_len * HEAD_DIM;

  for (int layer = 0; layer < num_layers; layer++) {
    const LDGLayerWeights &w = layer_weights[layer];
    __nv_bfloat16 *layer_k_cache =
        k_cache + static_cast<int64_t>(layer) * kv_cache_layer_stride;
    __nv_bfloat16 *layer_v_cache =
        v_cache + static_cast<int64_t>(layer) * kv_cache_layer_stride;

    // Layer 0: read directly from embedding table. Other layers: from
    // hidden_buffer.
    const __nv_bfloat16 *layer_input = (layer == 0) ? embed_row : hidden_buffer;

    ldg_matvec_qkv(grid, block_id, gridDim.x, layer_input,
                   w.input_layernorm_weight, w.q_proj_weight, w.k_proj_weight,
                   w.v_proj_weight, g_activations, g_residual, g_q, g_k, g_v);

    ldg_attention(grid, block_id, gridDim.x, g_q, g_k, g_v, layer_k_cache,
                  layer_v_cache, g_attn_out, cache_len, max_seq_len, attn_scale,
                  w.q_norm_weight, w.k_norm_weight, cos_table, sin_table,
                  position, nullptr, max_seq_len, w.o_proj_weight,
                  w.gate_proj_weight, w.up_proj_weight, w.down_proj_weight,
                  kv_flag, attn_flag, layer);

    ldg_o_proj_postnorm_mlp(grid, block_id, gridDim.x, w.o_proj_weight,
                            w.post_attn_layernorm_weight, w.gate_proj_weight,
                            w.up_proj_weight, w.down_proj_weight, g_attn_out,
                            g_residual, g_activations, g_mlp_intermediate,
                            hidden_buffer);
  }

  // Final RMSNorm
  if (block_id == 0) {
    __shared__ float smem_reduce[LDG_NUM_WARPS];
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE * 2) {
      float v0 = __bfloat162float(hidden_buffer[i]);
      g_activations[i] = v0;
      local_sum_sq += v0 * v0;
      int i1 = i + LDG_BLOCK_SIZE;
      if (i1 < HIDDEN_SIZE) {
        float v1 = __bfloat162float(hidden_buffer[i1]);
        g_activations[i1] = v1;
        local_sum_sq += v1 * v1;
      }
    }
    local_sum_sq = ldg_warp_reduce_sum(local_sum_sq);
    if (lane_id == 0)
      smem_reduce[warp_id] = local_sum_sq;
    __syncthreads();
    if (warp_id == 0) {
      float sum = (lane_id < LDG_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
      sum = ldg_warp_reduce_sum(sum);
      if (lane_id == 0)
        smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + LDG_RMS_EPS);
    }
    __syncthreads();
    float rstd = smem_reduce[0];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE * 2) {
      float wt0 = __bfloat162float(__ldg(final_norm_weight + i));
      g_normalized[i] = ldg_round_bf16(
          ldg_round_bf16(g_activations[i] * rstd) * wt0);
      int i1 = i + LDG_BLOCK_SIZE;
      if (i1 < HIDDEN_SIZE) {
        float wt1 = __bfloat162float(__ldg(final_norm_weight + i1));
        g_normalized[i1] = ldg_round_bf16(
            ldg_round_bf16(g_activations[i1] * rstd) * wt1);
      }
    }
  }
}

__device__ void ldg_decode_one(
    const __nv_bfloat16 *__restrict__ embed_weight,
    const LDGLayerWeights *__restrict__ layer_weights,
    const __nv_bfloat16 *__restrict__ final_norm_weight,
    const __nv_bfloat16 *__restrict__ cos_table,
    const __nv_bfloat16 *__restrict__ sin_table,
    __nv_bfloat16 *__restrict__ k_cache, __nv_bfloat16 *__restrict__ v_cache,
    __nv_bfloat16 *__restrict__ hidden_buffer,
    float *__restrict__ g_activations, float *__restrict__ g_residual,
    float *__restrict__ g_q, float *__restrict__ g_k, float *__restrict__ g_v,
    float *__restrict__ g_attn_out, float *__restrict__ g_mlp_intermediate,
    float *__restrict__ g_normalized,
    unsigned int *__restrict__ barrier_counter,
    unsigned int *__restrict__ barrier_sense,
    unsigned int *__restrict__ kv_flag, unsigned int *__restrict__ attn_flag,
    int num_layers, int position, int input_token_id, int max_seq_len,
    const int *__restrict__ block_table, int block_size, int num_cache_blocks,
    float attn_scale, int block_id, int num_blocks) {

  int cache_len = position + 1;

  // The host resets the counter before the launch. For batch decode, this
  // function is called once per sequence, so seed the local generation from
  // the current global generation rather than restarting at zero.
  unsigned int initial_generation = *barrier_sense;
  AtomicGridSync grid{barrier_counter, barrier_sense, (unsigned int)num_blocks,
                      initial_generation};

  // Flags are per-sequence state. Reset them only after the previous sequence
  // has completed its final grid barrier; this block-0 write is followed by a
  // grid barrier and never races with a counter reset.
  if (block_id == 0 && threadIdx.x == 0) {
    atomicExch(kv_flag, 0u);
    atomicExch(attn_flag, 0u);
  }
  grid.sync();

  const __nv_bfloat16 *embed_row = embed_weight + input_token_id * HIDDEN_SIZE;

  // The paged KV layer stride can exceed INT32_MAX on an 80 GB A100
  // (2516 blocks * 256 tokens * 1024 elements ~= 660M elements per layer).
  // Keep all cache offsets 64-bit; an int overflow here points later layers
  // several GiB before the allocation and causes illegal global writes.
  int64_t kv_cache_layer_stride =
      block_table
          ? static_cast<int64_t>(num_cache_blocks) * block_size * KV_SIZE
          : static_cast<int64_t>(NUM_KV_HEADS) * max_seq_len * HEAD_DIM;

  for (int layer = 0; layer < num_layers; layer++) {
    const LDGLayerWeights &w = layer_weights[layer];
    __nv_bfloat16 *layer_k_cache = k_cache + layer * kv_cache_layer_stride;
    __nv_bfloat16 *layer_v_cache = v_cache + layer * kv_cache_layer_stride;

    const __nv_bfloat16 *layer_input = (layer == 0) ? embed_row : hidden_buffer;

    ldg_matvec_qkv(grid, block_id, num_blocks, layer_input,
                   w.input_layernorm_weight, w.q_proj_weight, w.k_proj_weight,
                   w.v_proj_weight, g_activations, g_residual, g_q, g_k, g_v);

    ldg_attention(grid, block_id, num_blocks, g_q, g_k, g_v, layer_k_cache,
                  layer_v_cache, g_attn_out, cache_len, max_seq_len, attn_scale,
                  w.q_norm_weight, w.k_norm_weight, cos_table, sin_table,
                  position, block_table, block_size, w.o_proj_weight,
                  w.gate_proj_weight, w.up_proj_weight, w.down_proj_weight,
                  kv_flag, attn_flag, layer);

    ldg_o_proj_postnorm_mlp(grid, block_id, num_blocks, w.o_proj_weight,
                            w.post_attn_layernorm_weight, w.gate_proj_weight,
                            w.up_proj_weight, w.down_proj_weight, g_attn_out,
                            g_residual, g_activations, g_mlp_intermediate,
                            hidden_buffer);
  }

  if (block_id == 0) {
    __shared__ float smem_reduce[LDG_NUM_WARPS];
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE * 2) {
      float v0 = __bfloat162float(hidden_buffer[i]);
      g_activations[i] = v0;
      local_sum_sq += v0 * v0;
      int i1 = i + LDG_BLOCK_SIZE;
      if (i1 < HIDDEN_SIZE) {
        float v1 = __bfloat162float(hidden_buffer[i1]);
        g_activations[i1] = v1;
        local_sum_sq += v1 * v1;
      }
    }
    local_sum_sq = ldg_warp_reduce_sum(local_sum_sq);
    if (lane_id == 0)
      smem_reduce[warp_id] = local_sum_sq;
    __syncthreads();
    if (warp_id == 0) {
      float sum = (lane_id < LDG_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
      sum = ldg_warp_reduce_sum(sum);
      if (lane_id == 0)
        smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + LDG_RMS_EPS);
    }
    __syncthreads();
    float rstd = smem_reduce[0];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE * 2) {
      float wt0 = __bfloat162float(__ldg(final_norm_weight + i));
      g_normalized[i] = ldg_round_bf16(
          ldg_round_bf16(g_activations[i] * rstd) * wt0);
      int i1 = i + LDG_BLOCK_SIZE;
      if (i1 < HIDDEN_SIZE) {
        float wt1 = __bfloat162float(__ldg(final_norm_weight + i1));
        g_normalized[i1] = ldg_round_bf16(
            ldg_round_bf16(g_activations[i1] * rstd) * wt1);
      }
    }
  }
  grid.sync();
}

__global__ void __launch_bounds__(LDG_BLOCK_SIZE, 1) ldg_decode_kernel_direct(
    const __nv_bfloat16 *__restrict__ embed_weight,
    const LDGLayerWeights *__restrict__ layer_weights,
    const __nv_bfloat16 *__restrict__ final_norm_weight,
    const __nv_bfloat16 *__restrict__ cos_table,
    const __nv_bfloat16 *__restrict__ sin_table,
    __nv_bfloat16 *__restrict__ k_cache, __nv_bfloat16 *__restrict__ v_cache,
    __nv_bfloat16 *__restrict__ hidden_buffer,
    float *__restrict__ g_activations, float *__restrict__ g_residual,
    float *__restrict__ g_q, float *__restrict__ g_k, float *__restrict__ g_v,
    float *__restrict__ g_attn_out, float *__restrict__ g_mlp_intermediate,
    float *__restrict__ g_normalized,
    unsigned int *__restrict__ barrier_counter,
    unsigned int *__restrict__ barrier_sense,
    unsigned int *__restrict__ kv_flag, unsigned int *__restrict__ attn_flag,
    int num_layers, int position, int input_token_id, int max_seq_len,
    const int *__restrict__ block_table, int block_size, int num_cache_blocks,
    float attn_scale) {
  ldg_decode_one(
      embed_weight, layer_weights, final_norm_weight, cos_table, sin_table,
      k_cache, v_cache, hidden_buffer, g_activations, g_residual, g_q, g_k,
      g_v, g_attn_out, g_mlp_intermediate, g_normalized, barrier_counter,
      barrier_sense, kv_flag, attn_flag, num_layers, position, input_token_id,
      max_seq_len, block_table, block_size, num_cache_blocks, attn_scale,
      blockIdx.x, gridDim.x);
}

__global__ void __launch_bounds__(LDG_BLOCK_SIZE, 1) ldg_decode_kernel_batch(
    const __nv_bfloat16 *__restrict__ embed_weight,
    const LDGLayerWeights *__restrict__ layer_weights,
    const __nv_bfloat16 *__restrict__ final_norm_weight,
    const __nv_bfloat16 *__restrict__ cos_table,
    const __nv_bfloat16 *__restrict__ sin_table,
    __nv_bfloat16 *__restrict__ k_cache, __nv_bfloat16 *__restrict__ v_cache,
    __nv_bfloat16 *__restrict__ hidden_buffer,
    float *__restrict__ g_activations, float *__restrict__ g_residual,
    float *__restrict__ g_q, float *__restrict__ g_k, float *__restrict__ g_v,
    float *__restrict__ g_attn_out, float *__restrict__ g_mlp_intermediate,
    float *__restrict__ g_normalized,
    unsigned int *__restrict__ barrier_counter,
    unsigned int *__restrict__ barrier_sense,
    unsigned int *__restrict__ kv_flag, unsigned int *__restrict__ attn_flag,
    int num_layers, const int64_t *__restrict__ input_token_ids,
    const int64_t *__restrict__ positions,
    const int *__restrict__ block_tables, int batch_size,
    int block_table_stride, int block_size, int num_cache_blocks,
    int blocks_per_seq, float attn_scale) {
  constexpr int HIDDEN_STRIDE = HIDDEN_SIZE;
  constexpr int Q_STRIDE = Q_SIZE;
  constexpr int KV_STRIDE = KV_SIZE;
  constexpr int MLP_STRIDE = INTERMEDIATE_SIZE;

  // Parallel batch: partition the cooperative grid into disjoint per-sequence
  // block groups. Each block belongs to exactly one sequence (seq_id), which
  // syncs only within its own `blocks_per_seq` group using per-sequence
  // barrier/flags. This replaces the previous serial `for (seq_id...)` loop
  // with genuine sequence-level concurrency.
  int seq_id = blockIdx.x / blocks_per_seq;
  int local_block = blockIdx.x % blocks_per_seq;
  if (seq_id >= batch_size)
    return;

  ldg_decode_one(
      embed_weight, layer_weights, final_norm_weight, cos_table, sin_table,
      k_cache, v_cache, hidden_buffer + seq_id * HIDDEN_STRIDE,
      g_activations + seq_id * HIDDEN_STRIDE,
      g_residual + seq_id * HIDDEN_STRIDE, g_q + seq_id * Q_STRIDE,
      g_k + seq_id * KV_STRIDE, g_v + seq_id * KV_STRIDE,
      g_attn_out + seq_id * Q_STRIDE, g_mlp_intermediate + seq_id * MLP_STRIDE,
      g_normalized + seq_id * HIDDEN_STRIDE, barrier_counter + seq_id,
      barrier_sense + seq_id, kv_flag + seq_id, attn_flag + seq_id, num_layers,
      static_cast<int>(positions[seq_id]),
      static_cast<int>(input_token_ids[seq_id]), block_size * num_cache_blocks,
      block_tables + seq_id * block_table_stride, block_size, num_cache_blocks,
      attn_scale, local_block, blocks_per_seq);
}

// Persistent paged decode: reads the mutable position/token from device memory
// so N steps can be queued back-to-back with no host sync (the no-sync path).
// Paged KV write/read uses the (fixed across the span) device block table.
__global__ void __launch_bounds__(LDG_BLOCK_SIZE, 1) ldg_decode_kernel_persistent_paged(
    const __nv_bfloat16 *__restrict__ embed_weight,
    const LDGLayerWeights *__restrict__ layer_weights,
    const __nv_bfloat16 *__restrict__ final_norm_weight,
    const __nv_bfloat16 *__restrict__ cos_table,
    const __nv_bfloat16 *__restrict__ sin_table,
    __nv_bfloat16 *__restrict__ k_cache, __nv_bfloat16 *__restrict__ v_cache,
    __nv_bfloat16 *__restrict__ hidden_buffer,
    float *__restrict__ g_activations, float *__restrict__ g_residual,
    float *__restrict__ g_q, float *__restrict__ g_k, float *__restrict__ g_v,
    float *__restrict__ g_attn_out, float *__restrict__ g_mlp_intermediate,
    float *__restrict__ g_normalized,
    unsigned int *__restrict__ barrier_counter,
    unsigned int *__restrict__ barrier_sense,
    unsigned int *__restrict__ kv_flag, unsigned int *__restrict__ attn_flag,
    int num_layers, const int *__restrict__ d_position,
    const int *__restrict__ d_token_id, const int *__restrict__ block_table,
    int block_size, int num_cache_blocks, float attn_scale) {
  int position = *d_position;
  int input_token_id = *d_token_id;
  ldg_decode_one(
      embed_weight, layer_weights, final_norm_weight, cos_table, sin_table,
      k_cache, v_cache, hidden_buffer, g_activations, g_residual, g_q, g_k,
      g_v, g_attn_out, g_mlp_intermediate, g_normalized, barrier_counter,
      barrier_sense, kv_flag, attn_flag, num_layers, position, input_token_id,
      block_size * num_cache_blocks, block_table, block_size, num_cache_blocks,
      attn_scale, blockIdx.x, gridDim.x);
}

// Device-side step update: copies LM head output -> next token input,
// increments position, logs output token. Uses device-side step counter
// so the kernel arg is fixed (graph-compatible). When ``eos_token >= 0`` and
// the generated token equals it, the sequence is deactivated (position/token
// frozen); subsequent steps write -1 sentinel to output_log so the host can
// truncate. Mirrors ldg_update_step_batch's per-seq EOS semantics.
__global__ void ldg_update_step(const int *__restrict__ lm_output,
                                int *__restrict__ d_token_id,
                                int *__restrict__ d_position,
                                int *__restrict__ output_log,
                                int *__restrict__ d_step_counter,
                                int *__restrict__ d_active,
                                int eos_token) {
  int step = *d_step_counter;
  if (!*d_active) {
    output_log[step] = -1;
    *d_step_counter = step + 1;
    return;
  }
  int tok = *lm_output;
  output_log[step] = tok;
  if (eos_token >= 0 && tok == eos_token) {
    *d_active = 0;
  } else {
    *d_token_id = tok;
    *d_position = *d_position + 1;
  }
  *d_step_counter = step + 1;
}

// Batch version of ldg_update_step: per-sequence token/position advance with
// device-side EOS. Inactive sequences freeze (don't advance), so a completed
// sequence re-decodes a frozen position in later burst steps (wasted compute,
// but the cooperative WMMA grid can't shrink mid-burst without deadlock).
// output_log is [batch_size * max_steps] int32 (row-major by seq).
__global__ void ldg_update_step_batch(
    const int *__restrict__ lm_output,   // [batch_size] int32
    int64_t *__restrict__ d_positions,   // [batch_size]
    int64_t *__restrict__ d_tokens,      // [batch_size]
    int *__restrict__ output_log,        // [batch_size * max_steps]
    int *__restrict__ d_active,          // [batch_size]
    int eos_token, int batch_size, int max_steps,
    int *__restrict__ d_step_counter) {
  int step = *d_step_counter;
  if (threadIdx.x == 0) {
    for (int s = 0; s < batch_size; s++) {
      if (!d_active[s])
        continue;
      int tok = lm_output[s];
      output_log[s * max_steps + step] = tok;
      // eos_token < 0 means "never deactivate" (ignore_eos), matching
      // ldg_update_step above.  This is what lets the burst path serve
      // ignore_eos workloads, which is what makes the megakernel-vs-cudagraph
      // A/B fair: the controlled benchmark suites run with ignore_eos=True, so
      // before this the burst could never be exercised by them.
      if (eos_token >= 0 && tok == eos_token) {
        d_active[s] = 0;
      } else {
        d_tokens[s] = tok;
        d_positions[s] = d_positions[s] + 1;
      }
    }
    *d_step_counter = step + 1;
  }
}

// =============================================================================
// Launch functions
// =============================================================================

static unsigned int *d_barrier_counter = nullptr;
static unsigned int *d_barrier_sense = nullptr;
static unsigned int *d_kv_flag = nullptr;
static unsigned int *d_attn_flag = nullptr;
static unsigned int *d_lm_head_counter = nullptr;
static int *d_mutable_position = nullptr;
static int *d_mutable_token_id = nullptr;
int *h_pinned_position = nullptr;
int *h_pinned_token_id = nullptr;

// barrier/flags are per-sequence so parallel batch (one cooperative grid
// partitioned into disjoint per-sequence block groups) can sync each sequence
// independently using its own counter/sense generation and flag cells.
static void ensure_barrier_alloc() {
  if (!d_barrier_counter) {
    cudaMalloc(&d_barrier_counter, sizeof(unsigned int) * LDG_MAX_BATCH);
    cudaMalloc(&d_barrier_sense, sizeof(unsigned int) * LDG_MAX_BATCH);
    cudaMalloc(&d_kv_flag, sizeof(unsigned int) * LDG_MAX_BATCH);
    cudaMalloc(&d_attn_flag, sizeof(unsigned int) * LDG_MAX_BATCH);
    cudaMalloc(&d_lm_head_counter, sizeof(unsigned int) * LDG_MAX_BATCH);
    cudaMalloc(&d_mutable_position, sizeof(int));
    cudaMalloc(&d_mutable_token_id, sizeof(int));
    cudaHostAlloc(&h_pinned_position, sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&h_pinned_token_id, sizeof(int), cudaHostAllocDefault);
    cudaMemset(d_barrier_counter, 0, sizeof(unsigned int) * LDG_MAX_BATCH);
    cudaMemset(d_barrier_sense, 0, sizeof(unsigned int) * LDG_MAX_BATCH);
    cudaMemset(d_kv_flag, 0, sizeof(unsigned int) * LDG_MAX_BATCH);
    cudaMemset(d_attn_flag, 0, sizeof(unsigned int) * LDG_MAX_BATCH);
    cudaMemset(d_lm_head_counter, 0, sizeof(unsigned int) * LDG_MAX_BATCH);
  }
}

static cudaError_t ldg_reset_decode_state(cudaStream_t stream) {
  cudaError_t error = cudaMemsetAsync(d_barrier_counter, 0,
                                      sizeof(unsigned int) * LDG_MAX_BATCH, stream);
  if (error != cudaSuccess)
    return error;
  error = cudaMemsetAsync(d_barrier_sense, 0,
                          sizeof(unsigned int) * LDG_MAX_BATCH, stream);
  if (error != cudaSuccess)
    return error;
  error = cudaMemsetAsync(d_kv_flag, 0,
                          sizeof(unsigned int) * LDG_MAX_BATCH, stream);
  if (error != cudaSuccess)
    return error;
  error = cudaMemsetAsync(d_attn_flag, 0,
                          sizeof(unsigned int) * LDG_MAX_BATCH, stream);
  if (error != cudaSuccess)
    return error;
  return cudaMemsetAsync(d_lm_head_counter, 0,
                         sizeof(unsigned int) * LDG_MAX_BATCH, stream);
}

static bool ldg_check_phase(const char *phase, cudaStream_t stream) {
  cudaError_t error = cudaPeekAtLastError();
  if (error != cudaSuccess) {
    fprintf(stderr, "Megakernel %s launch failed: %s\n", phase,
            cudaGetErrorString(error));
    return false;
  }
  if (std::getenv("NANOVLLM_CUDA_DEBUG") != nullptr) {
    error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) {
      fprintf(stderr, "Megakernel %s failed: %s\n", phase,
              cudaGetErrorString(error));
      return false;
    }
  }
  return true;
}

#define LDG_ARG(value) reinterpret_cast<void *>(&(value))

template <typename Kernel>
static bool ldg_launch_cooperative(Kernel kernel, int grid_blocks,
                                   void **args, cudaStream_t stream) {
  // Cooperative-launch support is a device property; query it once and cache
  // instead of issuing cudaDeviceGetAttribute on every step of a no-sync
  // generation loop (host-side launch latency dominates the per-token cost).
  static int supported = -1;
  if (supported < 0) {
    int value = 0;
    cudaError_t error = cudaDeviceGetAttribute(
        &value, cudaDevAttrCooperativeLaunch, 0);
    if (error != cudaSuccess) {
      fprintf(stderr, "Megakernel cooperative-launch query failed: %s\n",
              cudaGetErrorString(error));
      return false;
    }
    supported = value;
  }
  if (!supported) {
    fprintf(stderr,
            "Megakernel requires cooperative launch for cross-block sync, "
            "but the active CUDA device does not support it\n");
    return false;
  }
  cudaError_t error = cudaLaunchCooperativeKernel(
      reinterpret_cast<const void *>(kernel), dim3(grid_blocks),
      dim3(LDG_BLOCK_SIZE), args, 0, stream);
  if (error != cudaSuccess) {
    fprintf(stderr, "Megakernel cooperative launch failed: %s\n",
            cudaGetErrorString(error));
    return false;
  }
  return true;
}

template <typename Kernel>
static int ldg_resident_grid_blocks(Kernel kernel, int requested) {
  int device = 0;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  int active_blocks = 0;
  cudaError_t error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, kernel, LDG_BLOCK_SIZE, 0);
  if (error != cudaSuccess || active_blocks <= 0) {
    return 0;
  }
  int resident_blocks = active_blocks * prop.multiProcessorCount;
  return min(requested, resident_blocks);
}

// Largest batch the WMMA batch kernel can actually launch.  The kernel splits
// the resident grid across sequences and needs LDG_ATTN_BLOCKS blocks per
// sequence for attention, so batch is bounded by grid_blocks / LDG_ATTN_BLOCKS
// -- typically 80/8 = 10, far below LDG_MAX_BATCH=32.  Exposed so the Python
// backend and the torch binding can reject unsupported batches up front instead
// of the launch printing to stderr and returning with stale output tokens.
extern "C" int ldg_wmma_max_supported_batch(int requested_blocks) {
  int grid = ldg_resident_grid_blocks(ldg_decode_kernel_batch_wmma,
                                      requested_blocks > 0 ? requested_blocks
                                                           : LDG_NUM_BLOCKS);
  if (grid < LDG_ATTN_BLOCKS) {
    return 0;
  }
  return grid / LDG_ATTN_BLOCKS;
}

// Check the batch against the real cooperative limit before consuming the
// decode state.  The launch helpers return void, so a guard that only prints to
// stderr would let the Python caller read the previous step's tokens out of the
// output buffer and treat them as fresh output -- a silent wrong answer.  The
// Python backend and the torch binding both know this limit up front (see
// ldg_wmma_max_supported_batch), so reaching this check means a caller bypassed
// them; keep it as a backstop and make the limit explicit in the message.
static bool ldg_batch_is_launchable(int batch_size, int grid_blocks) {
  const char *reason = nullptr;
  char message[256];
  if (batch_size <= 0) {
    reason = "batch_size must be positive";
  } else if (grid_blocks < LDG_ATTN_BLOCKS) {
    snprintf(message, sizeof(message),
             "resident grid %d is below LDG_ATTN_BLOCKS=%d, so the cooperative "
             "launch cannot run",
             grid_blocks, LDG_ATTN_BLOCKS);
    reason = message;
  } else {
    int blocks_per_seq = grid_blocks / batch_size;
    if (blocks_per_seq < LDG_ATTN_BLOCKS) {
      snprintf(message, sizeof(message),
               "batch=%d needs >= %d blocks/seq but the grid only provides %d, "
               "so the maximum batch for this grid is %d; use torch-cudagraph "
               "for larger batches",
               batch_size, LDG_ATTN_BLOCKS, blocks_per_seq,
               grid_blocks / LDG_ATTN_BLOCKS);
      reason = message;
    }
  }
  if (reason == nullptr) {
    return true;
  }
  fprintf(stderr, "Megakernel WMMA batch: %s\n", reason);
  // This helper returns void, so stderr alone would leave the caller reading the
  // previous step's tokens as fresh output.  The observable rejection is done
  // before the launch by the torch binding (check_wmma_batch_supported) and by
  // MegakernelBackend.compatibility(); this backstop only covers a caller that
  // bypassed both, and it deliberately does not fake a CUDA error.
  (void)cudaGetLastError();
  return false;
}

static inline void ldg_configure_kernel_attributes(); // forward decl

// Decode state is reset by the host on the same stream immediately before each
// launch. It must not be reset by block 0 inside the kernel: CUDA provides no
// ordering guarantee that block 0 starts before the other resident blocks.
static cudaError_t ldg_reset_decode_state(cudaStream_t stream);

extern "C" void launch_ldg_decode_direct(
    int input_token_id, int *output_token_id, const void *embed_weight,
    const LDGLayerWeights *layer_weights, const void *final_norm_weight,
    const void *lm_head_weight, const void *cos_table, const void *sin_table,
    void *k_cache, void *v_cache, void *hidden_buffer, void *g_activations,
    void *g_residual, void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_normalized, void *block_max_vals,
    void *block_max_idxs, int num_layers, int position, int max_seq_len,
    const int *block_table, int block_size, int num_cache_blocks,
    float attn_scale, cudaStream_t stream) {
  ldg_configure_kernel_attributes();
  ensure_barrier_alloc();
  cudaError_t reset_error = ldg_reset_decode_state(stream);
  if (reset_error != cudaSuccess) {
    fprintf(stderr, "Megakernel decode-state reset failed: %s\n",
            cudaGetErrorString(reset_error));
    return;
  }
  int grid_blocks = ldg_resident_grid_blocks(
      ldg_decode_kernel_direct, LDG_NUM_BLOCKS);
  if (grid_blocks < LDG_ATTN_BLOCKS) {
    fprintf(stderr,
            "Megakernel direct launch requires at least %d resident blocks; "
            "occupancy allows %d\n",
            LDG_ATTN_BLOCKS, grid_blocks);
    return;
  }

  void *kernel_args[] = {
      LDG_ARG(embed_weight),       LDG_ARG(layer_weights),
      LDG_ARG(final_norm_weight),   LDG_ARG(cos_table),
      LDG_ARG(sin_table),           LDG_ARG(k_cache),
      LDG_ARG(v_cache),             LDG_ARG(hidden_buffer),
      LDG_ARG(g_activations),       LDG_ARG(g_residual),
      LDG_ARG(g_q),                 LDG_ARG(g_k),
      LDG_ARG(g_v),                 LDG_ARG(g_attn_out),
      LDG_ARG(g_mlp_intermediate),  LDG_ARG(g_normalized),
      LDG_ARG(d_barrier_counter),   LDG_ARG(d_barrier_sense),
      LDG_ARG(d_kv_flag),           LDG_ARG(d_attn_flag),
      LDG_ARG(num_layers),          LDG_ARG(position),
      LDG_ARG(input_token_id),      LDG_ARG(max_seq_len),
      LDG_ARG(block_table),         LDG_ARG(block_size),
      LDG_ARG(num_cache_blocks),    LDG_ARG(attn_scale),
  };
  if (!ldg_launch_cooperative(ldg_decode_kernel_direct, grid_blocks,
                              kernel_args, stream))
    return;
  if (!ldg_check_phase("transformer decode", stream))
    return;

  cudaMemsetAsync(d_lm_head_counter, 0, sizeof(unsigned int), stream);
  ldg_lm_head_fused<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
      (const float *)g_normalized, (const __nv_bfloat16 *)lm_head_weight,
      (float *)block_max_vals, (int *)block_max_idxs, output_token_id,
      d_lm_head_counter, LDG_LM_NUM_BLOCKS);
  if (!ldg_check_phase("LM head", stream))
    return;
}

extern "C" void launch_ldg_decode_paged_batch(
    const void *input_token_ids, const void *positions,
    const void *block_tables, int *output_token_ids,
    const void *embed_weight, const LDGLayerWeights *layer_weights,
    const void *final_norm_weight, const void *lm_head_weight,
    const void *cos_table, const void *sin_table, void *k_cache,
    void *v_cache, void *hidden_buffer, void *g_activations,
    void *g_residual, void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_normalized, void *block_max_vals,
    void *block_max_idxs, int num_layers, int batch_size,
    int block_table_stride, int block_size, int num_cache_blocks,
    float attn_scale, cudaStream_t stream) {
  ldg_configure_kernel_attributes();
  ensure_barrier_alloc();
  cudaError_t reset_error = ldg_reset_decode_state(stream);
  if (reset_error != cudaSuccess) {
    fprintf(stderr, "Megakernel decode-state reset failed: %s\n",
            cudaGetErrorString(reset_error));
    return;
  }
  int grid_blocks = ldg_resident_grid_blocks(
      ldg_decode_kernel_batch, LDG_NUM_BLOCKS);
  // Partition the cooperative grid into per-sequence groups. Each sequence
  // needs at least LDG_ATTN_BLOCKS blocks for its attention; we must also use
  // a multiple of batch_size so every sequence gets an equal group.
  int blocks_per_seq = grid_blocks / batch_size;
  if (blocks_per_seq < LDG_ATTN_BLOCKS) {
    fprintf(stderr,
            "Megakernel parallel batch needs >= %d blocks per sequence "
            "(occupancy allows %d total for %d seqs)\n",
            LDG_ATTN_BLOCKS, grid_blocks, batch_size);
    return;
  }
  grid_blocks = blocks_per_seq * batch_size;

  void *kernel_args[] = {
      LDG_ARG(embed_weight),       LDG_ARG(layer_weights),
      LDG_ARG(final_norm_weight),   LDG_ARG(cos_table),
      LDG_ARG(sin_table),           LDG_ARG(k_cache),
      LDG_ARG(v_cache),             LDG_ARG(hidden_buffer),
      LDG_ARG(g_activations),       LDG_ARG(g_residual),
      LDG_ARG(g_q),                 LDG_ARG(g_k),
      LDG_ARG(g_v),                 LDG_ARG(g_attn_out),
      LDG_ARG(g_mlp_intermediate),  LDG_ARG(g_normalized),
      LDG_ARG(d_barrier_counter),   LDG_ARG(d_barrier_sense),
      LDG_ARG(d_kv_flag),           LDG_ARG(d_attn_flag),
      LDG_ARG(num_layers),          LDG_ARG(input_token_ids),
      LDG_ARG(positions),            LDG_ARG(block_tables),
      LDG_ARG(batch_size),           LDG_ARG(block_table_stride),
      LDG_ARG(block_size),            LDG_ARG(num_cache_blocks),
      LDG_ARG(blocks_per_seq),        LDG_ARG(attn_scale),
  };
  if (!ldg_launch_cooperative(ldg_decode_kernel_batch, grid_blocks,
                              kernel_args, stream))
    return;
  if (!ldg_check_phase("transformer decode", stream))
    return;

  cudaMemsetAsync(d_lm_head_counter, 0,
                  sizeof(unsigned int) * batch_size, stream);
  ldg_lm_head_fused_batch<<<LDG_LM_NUM_BLOCKS * batch_size,
                            LDG_LM_BLOCK_SIZE, 0, stream>>>(
      (const float *)g_normalized, (const __nv_bfloat16 *)lm_head_weight,
      (float *)block_max_vals, (int *)block_max_idxs, output_token_ids,
      d_lm_head_counter, LDG_LM_NUM_BLOCKS, batch_size, HIDDEN_SIZE,
      LDG_LM_NUM_BLOCKS);
  if (!ldg_check_phase("LM head", stream))
    return;
}

extern "C" void launch_ldg_decode_batch_wmma(
    const void *input_token_ids, const void *positions,
    const void *block_tables, int *output_token_ids,
    const void *embed_weight, const LDGLayerWeights *layer_weights,
    const void *final_norm_weight, const void *lm_head_weight,
    const void *cos_table, const void *sin_table, void *k_cache,
    void *v_cache, void *hidden_buffer, void *g_activations,
    void *g_residual, void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_mlp_up, void *g_normalized,
    void *block_max_vals, void *block_max_idxs, int num_layers,
    int batch_size, int block_table_stride, int block_size,
    int num_cache_blocks, float attn_scale, cudaStream_t stream) {
  ldg_configure_kernel_attributes();
  ensure_barrier_alloc();
  if (ldg_reset_decode_state(stream) != cudaSuccess) {
    fprintf(stderr, "Megakernel WMMA decode-state reset failed\n");
    return;
  }
  int grid_blocks = ldg_resident_grid_blocks(
      ldg_decode_kernel_batch_wmma, LDG_NUM_BLOCKS);
  if (!ldg_batch_is_launchable(batch_size, grid_blocks)) {
    return;
  }
  int blocks_per_seq = grid_blocks / batch_size;
  grid_blocks = blocks_per_seq * batch_size;
  void *kernel_args[] = {
      LDG_ARG(embed_weight),       LDG_ARG(layer_weights),
      LDG_ARG(final_norm_weight),   LDG_ARG(cos_table),
      LDG_ARG(sin_table),           LDG_ARG(k_cache),
      LDG_ARG(v_cache),             LDG_ARG(hidden_buffer),
      LDG_ARG(g_activations),       LDG_ARG(g_residual),
      LDG_ARG(g_q),                 LDG_ARG(g_k),
      LDG_ARG(g_v),                 LDG_ARG(g_attn_out),
      LDG_ARG(g_mlp_intermediate),  LDG_ARG(g_mlp_up),
      LDG_ARG(g_normalized),
      LDG_ARG(d_barrier_counter),   LDG_ARG(d_barrier_sense),
      LDG_ARG(d_kv_flag),           LDG_ARG(d_attn_flag),
      LDG_ARG(num_layers),          LDG_ARG(input_token_ids),
      LDG_ARG(positions),            LDG_ARG(block_tables),
      LDG_ARG(batch_size),           LDG_ARG(block_table_stride),
      LDG_ARG(block_size),            LDG_ARG(num_cache_blocks),
      LDG_ARG(blocks_per_seq),        LDG_ARG(attn_scale),
  };
  if (!ldg_launch_cooperative(ldg_decode_kernel_batch_wmma, grid_blocks,
                              kernel_args, stream))
    return;
  if (!ldg_check_phase("transformer decode (WMMA)", stream))
    return;
  cudaMemsetAsync(d_lm_head_counter, 0,
                  sizeof(unsigned int) * batch_size, stream);
  ldg_lm_head_fused_batch<<<LDG_LM_NUM_BLOCKS * batch_size,
                            LDG_LM_BLOCK_SIZE, 0, stream>>>(
      (const float *)g_normalized, (const __nv_bfloat16 *)lm_head_weight,
      (float *)block_max_vals, (int *)block_max_idxs, output_token_ids,
      d_lm_head_counter, LDG_LM_NUM_BLOCKS, batch_size, HIDDEN_SIZE,
      LDG_LM_NUM_BLOCKS);
}

extern "C" void launch_ldg_decode_persistent(
    int input_token_id, int *output_token_id, const void *embed_weight,
    const LDGLayerWeights *layer_weights, const void *final_norm_weight,
    const void *lm_head_weight, const void *cos_table, const void *sin_table,
    void *k_cache, void *v_cache, void *hidden_buffer, void *g_activations,
    void *g_residual, void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_normalized, void *block_max_vals,
    void *block_max_idxs, int num_layers, int position, int cache_len,
    int max_seq_len, float attn_scale, cudaStream_t stream) {
  ldg_configure_kernel_attributes();
  ensure_barrier_alloc();
  cudaError_t reset_error = ldg_reset_decode_state(stream);
  if (reset_error != cudaSuccess) {
    fprintf(stderr, "Megakernel decode-state reset failed: %s\n",
            cudaGetErrorString(reset_error));
    return;
  }
  int grid_blocks = ldg_resident_grid_blocks(
      ldg_decode_kernel_persistent, LDG_NUM_BLOCKS);
  if (grid_blocks < LDG_ATTN_BLOCKS) {
    fprintf(stderr,
            "Megakernel persistent launch requires at least %d resident blocks; "
            "occupancy allows %d\n",
            LDG_ATTN_BLOCKS, grid_blocks);
    return;
  }

  // Write mutable params via pinned host memory (barriers reset on-device)
  *h_pinned_position = position;
  *h_pinned_token_id = input_token_id;
  cudaMemcpyAsync(d_mutable_position, h_pinned_position, sizeof(int),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_mutable_token_id, h_pinned_token_id, sizeof(int),
                  cudaMemcpyHostToDevice, stream);

  void *kernel_args[] = {
      LDG_ARG(embed_weight),       LDG_ARG(layer_weights),
      LDG_ARG(final_norm_weight),   LDG_ARG(cos_table),
      LDG_ARG(sin_table),           LDG_ARG(k_cache),
      LDG_ARG(v_cache),             LDG_ARG(hidden_buffer),
      LDG_ARG(g_activations),       LDG_ARG(g_residual),
      LDG_ARG(g_q),                 LDG_ARG(g_k),
      LDG_ARG(g_v),                 LDG_ARG(g_attn_out),
      LDG_ARG(g_mlp_intermediate),  LDG_ARG(g_normalized),
      LDG_ARG(d_barrier_counter),   LDG_ARG(d_barrier_sense),
      LDG_ARG(d_kv_flag),           LDG_ARG(d_attn_flag),
      LDG_ARG(num_layers),          LDG_ARG(d_mutable_position),
      LDG_ARG(d_mutable_token_id),  LDG_ARG(max_seq_len),
      LDG_ARG(attn_scale),
  };
  if (!ldg_launch_cooperative(ldg_decode_kernel_persistent, grid_blocks,
                              kernel_args, stream))
    return;

  cudaMemsetAsync(d_lm_head_counter, 0, sizeof(unsigned int), stream);
  ldg_lm_head_fused<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
      (const float *)g_normalized, (const __nv_bfloat16 *)lm_head_weight,
      (float *)block_max_vals, (int *)block_max_idxs, output_token_id,
      d_lm_head_counter, LDG_LM_NUM_BLOCKS);
}

// N-step generate with NO per-step CPU sync. All steps queued back-to-back.
// The update kernel feeds output_token -> d_token_id on device between steps.
extern "C" void launch_ldg_generate_nosync(
    int first_token_id, int num_steps, const void *embed_weight,
    const LDGLayerWeights *layer_weights, const void *final_norm_weight,
    const void *lm_head_weight, const void *cos_table, const void *sin_table,
    void *k_cache, void *v_cache, void *hidden_buffer, void *g_activations,
    void *g_residual, void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_normalized, void *block_max_vals,
    void *block_max_idxs,
    int *output_log, // device int[num_steps]: all generated tokens
    int num_layers, int start_position, int max_seq_len, float attn_scale,
    cudaStream_t stream) {

  ldg_configure_kernel_attributes();
  ensure_barrier_alloc();
  cudaError_t reset_error = ldg_reset_decode_state(stream);
  if (reset_error != cudaSuccess) {
    fprintf(stderr, "Megakernel decode-state reset failed: %s\n",
            cudaGetErrorString(reset_error));
    return;
  }
  int grid_blocks = ldg_resident_grid_blocks(
      ldg_decode_kernel_persistent, LDG_NUM_BLOCKS);
  if (grid_blocks < LDG_ATTN_BLOCKS) {
    fprintf(stderr,
            "Megakernel generate launch requires at least %d resident blocks; "
            "occupancy allows %d\n",
            LDG_ATTN_BLOCKS, grid_blocks);
    return;
  }

  // Allocate device-side step counter
  static int *d_step_counter = nullptr;
  static int *d_output_token = nullptr;
  static int *d_active = nullptr;
  if (!d_step_counter) {
    cudaMalloc(&d_step_counter, sizeof(int));
    cudaMalloc(&d_output_token, sizeof(int));
    cudaMalloc(&d_active, sizeof(int));
  }
  cudaMemsetAsync(d_step_counter, 0, sizeof(int), stream);
  cudaMemsetAsync(d_active, 1, sizeof(int), stream);

  // Set initial position and token_id on device
  *h_pinned_position = start_position;
  *h_pinned_token_id = first_token_id;
  cudaMemcpyAsync(d_mutable_position, h_pinned_position, sizeof(int),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_mutable_token_id, h_pinned_token_id, sizeof(int),
                  cudaMemcpyHostToDevice, stream);

  // Launch all N steps back-to-back with NO CPU sync between them.
  // Each step: persistent kernel -> LM head phase 1 -> LM head phase 2 ->
  // update
  for (int step = 0; step < num_steps; step++) {
    void *kernel_args[] = {
        LDG_ARG(embed_weight),       LDG_ARG(layer_weights),
        LDG_ARG(final_norm_weight),   LDG_ARG(cos_table),
        LDG_ARG(sin_table),           LDG_ARG(k_cache),
        LDG_ARG(v_cache),             LDG_ARG(hidden_buffer),
        LDG_ARG(g_activations),       LDG_ARG(g_residual),
        LDG_ARG(g_q),                 LDG_ARG(g_k),
        LDG_ARG(g_v),                 LDG_ARG(g_attn_out),
        LDG_ARG(g_mlp_intermediate),  LDG_ARG(g_normalized),
        LDG_ARG(d_barrier_counter),   LDG_ARG(d_barrier_sense),
        LDG_ARG(d_kv_flag),           LDG_ARG(d_attn_flag),
        LDG_ARG(num_layers),          LDG_ARG(d_mutable_position),
        LDG_ARG(d_mutable_token_id),  LDG_ARG(max_seq_len),
        LDG_ARG(attn_scale),
    };
    if (!ldg_launch_cooperative(ldg_decode_kernel_persistent, grid_blocks,
                                kernel_args, stream))
      return;

    cudaMemsetAsync(d_lm_head_counter, 0, sizeof(unsigned int), stream);
    ldg_lm_head_fused<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float *)g_normalized, (const __nv_bfloat16 *)lm_head_weight,
        (float *)block_max_vals, (int *)block_max_idxs, d_output_token,
        d_lm_head_counter, LDG_LM_NUM_BLOCKS);

    // Update step: feed output token back, increment position, log result
    // eos_token=-1 disables device-side EOS (host checks between steps).
    ldg_update_step<<<1, 1, 0, stream>>>(d_output_token, d_mutable_token_id,
                                         d_mutable_position, output_log,
                                         d_step_counter, d_active, -1);
  }
}

// N-step paged generate with NO per-step CPU sync, mirroring the contiguous
// generate_nosync but writing/reading the nano-vLLM paged KV cache via the
// (fixed over the span) device block table. Requires all blocks for the
// generation span to be pre-allocated in the block table.
extern "C" void launch_ldg_generate_nosync_paged(
    int first_token_id, int num_steps, const void *embed_weight,
    const LDGLayerWeights *layer_weights, const void *final_norm_weight,
    const void *lm_head_weight, const void *cos_table, const void *sin_table,
    void *k_cache, void *v_cache, void *hidden_buffer, void *g_activations,
    void *g_residual, void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_normalized, void *block_max_vals,
    void *block_max_idxs, const int *block_table,
    int *output_log, // device int[num_steps]: all generated tokens
    int num_layers, int start_position, int block_size, int num_cache_blocks,
    int eos_token, float attn_scale, cudaStream_t stream) {

  ldg_configure_kernel_attributes();
  ensure_barrier_alloc();
  cudaError_t reset_error = ldg_reset_decode_state(stream);
  if (reset_error != cudaSuccess) {
    fprintf(stderr, "Megakernel paged decode-state reset failed: %s\n",
            cudaGetErrorString(reset_error));
    return;
  }
  int grid_blocks = ldg_resident_grid_blocks(
      ldg_decode_kernel_persistent_paged, LDG_NUM_BLOCKS);
  if (grid_blocks < LDG_ATTN_BLOCKS) {
    fprintf(stderr,
            "Megakernel paged generate launch requires at least %d resident "
            "blocks; occupancy allows %d\n",
            LDG_ATTN_BLOCKS, grid_blocks);
    return;
  }

  static int *d_step_counter = nullptr;
  static int *d_output_token = nullptr;
  static int *d_active = nullptr;
  if (!d_step_counter) {
    cudaMalloc(&d_step_counter, sizeof(int));
    cudaMalloc(&d_output_token, sizeof(int));
    cudaMalloc(&d_active, sizeof(int));
  }
  cudaMemsetAsync(d_step_counter, 0, sizeof(int), stream);
  cudaMemsetAsync(d_active, 1, sizeof(int), stream);

  // Set initial position and token_id on device
  *h_pinned_position = start_position;
  *h_pinned_token_id = first_token_id;
  cudaMemcpyAsync(d_mutable_position, h_pinned_position, sizeof(int),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_mutable_token_id, h_pinned_token_id, sizeof(int),
                  cudaMemcpyHostToDevice, stream);

  // Launch all N steps back-to-back with NO CPU sync between them.
  for (int step = 0; step < num_steps; step++) {
    void *kernel_args[] = {
        LDG_ARG(embed_weight),       LDG_ARG(layer_weights),
        LDG_ARG(final_norm_weight),   LDG_ARG(cos_table),
        LDG_ARG(sin_table),           LDG_ARG(k_cache),
        LDG_ARG(v_cache),             LDG_ARG(hidden_buffer),
        LDG_ARG(g_activations),       LDG_ARG(g_residual),
        LDG_ARG(g_q),                 LDG_ARG(g_k),
        LDG_ARG(g_v),                 LDG_ARG(g_attn_out),
        LDG_ARG(g_mlp_intermediate),  LDG_ARG(g_normalized),
        LDG_ARG(d_barrier_counter),   LDG_ARG(d_barrier_sense),
        LDG_ARG(d_kv_flag),           LDG_ARG(d_attn_flag),
        LDG_ARG(num_layers),          LDG_ARG(d_mutable_position),
        LDG_ARG(d_mutable_token_id),  LDG_ARG(block_table),
        LDG_ARG(block_size),          LDG_ARG(num_cache_blocks),
        LDG_ARG(attn_scale),
    };
    if (!ldg_launch_cooperative(ldg_decode_kernel_persistent_paged,
                                grid_blocks, kernel_args, stream))
      return;

    cudaMemsetAsync(d_lm_head_counter, 0, sizeof(unsigned int), stream);
    ldg_lm_head_fused<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float *)g_normalized, (const __nv_bfloat16 *)lm_head_weight,
        (float *)block_max_vals, (int *)block_max_idxs, d_output_token,
        d_lm_head_counter, LDG_LM_NUM_BLOCKS);

    ldg_update_step<<<1, 1, 0, stream>>>(d_output_token, d_mutable_token_id,
                                         d_mutable_position, output_log,
                                         d_step_counter, d_active, eos_token);
  }
}

// N-step BATCH paged generate with NO per-step CPU sync. Loops over
// launch_ldg_decode_batch_wmma's WMMA batch kernel + batch LM head + batch
// device-side update. Positions/tokens are int64 device arrays mutated by
// ldg_update_step_batch; block tables are fixed across the burst (R4). Each
// seq's output is row s of output_log[batch_size * max_steps]; EOS stops a seq.
extern "C" void launch_ldg_generate_nosync_paged_batch(
    const void *input_token_ids, const void *positions,
    const void *block_tables, const void *embed_weight,
    const LDGLayerWeights *layer_weights, const void *final_norm_weight,
    const void *lm_head_weight, const void *cos_table, const void *sin_table,
    void *k_cache, void *v_cache, void *hidden_buffer, void *g_activations,
    void *g_residual, void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_mlp_up, void *g_normalized,
    void *block_max_vals, void *block_max_idxs,
    int *output_log, // device int[batch*max_steps]
    int num_layers, int batch_size, int block_table_stride, int block_size,
    int num_cache_blocks, int max_steps, int eos_token, float attn_scale,
    cudaStream_t stream) {
  ldg_configure_kernel_attributes();
  ensure_barrier_alloc();
  if (ldg_reset_decode_state(stream) != cudaSuccess) {
    fprintf(stderr, "Megakernel WMMA batch no-sync reset failed\n");
    return;
  }
  int grid_blocks = ldg_resident_grid_blocks(
      ldg_decode_kernel_batch_wmma, LDG_NUM_BLOCKS);
  if (!ldg_batch_is_launchable(batch_size, grid_blocks)) {
    return;
  }
  int blocks_per_seq = grid_blocks / batch_size;
  grid_blocks = blocks_per_seq * batch_size;

  static int *d_step_counter = nullptr;
  static int *d_lm_output = nullptr;
  static int *d_active = nullptr;
  if (!d_step_counter) {
    cudaMalloc(&d_step_counter, sizeof(int));
    cudaMalloc(&d_lm_output, sizeof(int) * LDG_MAX_BATCH);
    cudaMalloc(&d_active, sizeof(int) * LDG_MAX_BATCH);
  }
  cudaMemsetAsync(d_step_counter, 0, sizeof(int), stream);
  cudaMemsetAsync(d_active, 1, sizeof(int) * batch_size, stream);

  for (int step = 0; step < max_steps; step++) {
    void *kernel_args[] = {
        LDG_ARG(embed_weight),       LDG_ARG(layer_weights),
        LDG_ARG(final_norm_weight),   LDG_ARG(cos_table),
        LDG_ARG(sin_table),           LDG_ARG(k_cache),
        LDG_ARG(v_cache),             LDG_ARG(hidden_buffer),
        LDG_ARG(g_activations),       LDG_ARG(g_residual),
        LDG_ARG(g_q),                 LDG_ARG(g_k),
        LDG_ARG(g_v),                 LDG_ARG(g_attn_out),
        LDG_ARG(g_mlp_intermediate),  LDG_ARG(g_mlp_up),
        LDG_ARG(g_normalized),
        LDG_ARG(d_barrier_counter),   LDG_ARG(d_barrier_sense),
        LDG_ARG(d_kv_flag),           LDG_ARG(d_attn_flag),
        LDG_ARG(num_layers),          LDG_ARG(input_token_ids),
        LDG_ARG(positions),            LDG_ARG(block_tables),
        LDG_ARG(batch_size),           LDG_ARG(block_table_stride),
        LDG_ARG(block_size),            LDG_ARG(num_cache_blocks),
        LDG_ARG(blocks_per_seq),        LDG_ARG(attn_scale),
    };
    if (!ldg_launch_cooperative(ldg_decode_kernel_batch_wmma, grid_blocks,
                                kernel_args, stream))
      return;

    cudaMemsetAsync(d_lm_head_counter, 0,
                    sizeof(unsigned int) * batch_size, stream);
    ldg_lm_head_fused_batch<<<LDG_LM_NUM_BLOCKS * batch_size,
                              LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float *)g_normalized, (const __nv_bfloat16 *)lm_head_weight,
        (float *)block_max_vals, (int *)block_max_idxs, d_lm_output,
        d_lm_head_counter, LDG_LM_NUM_BLOCKS, batch_size, HIDDEN_SIZE,
        LDG_LM_NUM_BLOCKS);

    ldg_update_step_batch<<<1, 1, 0, stream>>>(
        d_lm_output, (int64_t *)positions, (int64_t *)input_token_ids,
        output_log, d_active, eos_token, batch_size, max_steps,
        d_step_counter);
  }
}

// Cross-block synchronization in the decode kernels is valid only when the
// whole grid is launched cooperatively and all blocks are resident together.
static inline void ldg_configure_kernel_attributes() {
  static bool configured = false;
  if (configured) {
    return;
  }
  configured = true;
  cudaFuncSetAttribute(ldg_decode_kernel_persistent,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  cudaFuncSetAttribute(ldg_lm_head_phase1,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxL1);
  cudaFuncSetAttribute(ldg_lm_head_phase2,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxL1);
}

#undef LDG_ARG
