#pragma once

// =============================================================================
// Task-Graph Megakernel ABI (layout-locked descriptors).
//
// Shared by:
//   - csrc/taskgraph_kernel.cu      (device kernel, NVCC)
//   - csrc/torch_bindings.cpp       (pybind bindings + launch helpers)
//   - schedule.py                   (Python schedule builder, via bindings)
//
// MUST COMPILE WITHOUT NVCC / CUDA headers: taskgraph_bindings.cpp includes
// this from a plain C++ TU. Device pointers are carried as uint64_t to stay
// free of any CUDA or torch type.
//
// LAYOUT-LOCKED: TgLaunchDesc crosses no dlsym boundary in v1 (no JIT), but
// it is uploaded to the device as a __grid_constant__ and read by the kernel.
// Appending fields is safe; inserting/reordering is not. Every field change
// must be mirrored in:
//   - schedule.py._build_launch_desc (writer)
//   - taskgraph_bindings.cpp (RW bindings + structural tests)
//   - taskgraph_kernel.cu TgGlobals<Cfg> (device reader)
// The first omission fails test_taskgraph_abi.py, which parses this header
// and compares it against the bindings.
// =============================================================================

#include <cstdint>

namespace tg {

// ---------------------------------------------------------------------------
// Instruction encoding. 16 int32 per task. See docs/11 §3 for the layout.
// ---------------------------------------------------------------------------
constexpr int TG_INST_WIDTH = 16;

// Opcode enumeration. Dense-only (no MoE). See docs/11 §3.2.
enum TgOp : int32_t {
  TG_NOP              = 0,
  TG_QKV_PROJ         = 1,
  TG_ATTN_DECODE      = 2,
  TG_ATTN_COMBINE     = 3,
  TG_O_PROJ           = 4,
  TG_FFN_UPGATE_ACT   = 5,
  TG_FFN_DOWN         = 6,
  TG_ADD_RMSNORM      = 7,
  TG_LM_HEAD          = 8,
  TG_LM_ARGMAX        = 9,  // terminal per-batch argmax over the LM head logits
};

// Instruction as a flat int array. Field offsets are opcode-specific; the
// named accessors below document the layout but the on-wire form is `data[]`.
struct TgInstruction {
  int32_t data[TG_INST_WIDTH];
};

// ---------------------------------------------------------------------------
// Barrier arrays. 7 arrays for the dense model. See docs/11 §4.1.
// Each is a device uint32 array of the given length, zeroed per step.
// The struct holds device pointers (uint64_t) so it can cross the C++/CUDA
// boundary without torch types; the kernel dereferences them as uint32_t*.
// ---------------------------------------------------------------------------
struct TgBars {
  uint64_t bar_qkv;       // [layers * hkv]
  uint64_t bar_attn;      // [layers * oproj_splits]
  uint64_t bar_oproj;     // [layers]
  uint64_t bar_silu;      // [layers * down_splits]
  uint64_t bar_ffn_down;  // [layers]
  uint64_t bar_layer;     // [layers + 1]
  uint64_t bar_combine;   // [layers * BS * hkv]   (split>1 only; 0 otherwise)
  uint64_t bar_lm_head;   // [1]  LM_HEAD tiles -> LM_ARGMAX (cross-SM)
};

// ---------------------------------------------------------------------------
// Per-layer weight pointers (11 tensors/layer, matching LDGLayerWeights).
// Carried as a flat device array of uint64: [num_layers * 11].
// Field order MUST match LDGLayerWeights in kernel.cu:90 and
// weights_from_nano_model in model.py:156.
// ---------------------------------------------------------------------------
enum TgLayerWeightIdx : int {
  TG_LW_INPUT_LAYERNORM = 0,
  TG_LW_Q_PROJ,
  TG_LW_K_PROJ,
  TG_LW_V_PROJ,
  TG_LW_Q_NORM,
  TG_LW_K_NORM,
  TG_LW_O_PROJ,
  TG_LW_POST_ATTN_LAYERNORM,
  TG_LW_GATE_PROJ,
  TG_LW_UP_PROJ,
  TG_LW_DOWN_PROJ,
  TG_LW_COUNT,  // 11
};

// ---------------------------------------------------------------------------
// Launch descriptor. Uploaded as __grid_constant__ to the persistent kernel.
// See docs/11 §5.1. Append-only after v1 freeze.
// ---------------------------------------------------------------------------
struct TgLaunchDesc {
  // --- grid / instruction stream -------------------------------------------
  int32_t num_sms;            // grid size (== #SMs used, e.g. 108 on A100)
  uint64_t inst_buf;          // device int32 [num_sms, max_inst, TG_INST_WIDTH]
  uint64_t num_inst_per_sm;   // device int32 [num_sms]
  int32_t max_inst;           // instruction capacity per SM (inst_buf stride)

  // --- weights -------------------------------------------------------------
  uint64_t layer_weights;     // device uint64 [num_layers * TG_LW_COUNT]
  uint64_t embed_weight;      // device bf16 [vocab, hidden]  (tied LM head)
  uint64_t final_norm_weight; // device bf16 [hidden]
  uint64_t cos_table;         // device bf16 [max_seq_len, head_dim*2]
  uint64_t sin_table;         // device bf16 [max_seq_len, head_dim*2]

  // --- KV cache (paged, nano-vLLM layout) ----------------------------------
  uint64_t k_cache;           // device bf16 [num_layers, num_blocks, block_size, kv_heads, head_dim]
  uint64_t v_cache;           // same shape
  uint64_t page_table;        // device int32 [BS, max_pages_per_seq]
  uint64_t cache_seqlens;     // device int32 [BS]  (current length per row, pre-decode)
  uint64_t row_active;        // device int32 [BS]  (1=decode this row, 0=skip)
  int32_t num_phys_pages;
  int32_t max_pages_per_seq;
  int32_t page_block_size;    // tokens per page (block_size)
  int64_t kv_layer_stride;    // elements per layer in k/v cache (int64 for 80GB safety; matches kernel.cu)

  // --- scratch (device, zeroed per step) -----------------------------------
  //
  // WMMA_PAD CONTRACT: every buffer written by a WMMA GEMM must have at least
  // TG_WMMA_TILE (16) rows. wmma::store_matrix_sync always stores a full 16x16
  // tile, so with BS < 16 it writes rows BS..15 as zeros. The row stride must
  // equal the column count documented here (the kernel hardcodes it as a
  // template parameter). torch_bindings.cpp enforces this in tg_decode_step.
  //
  // Values are float unless marked bf16; the kernel rounds at the same tensor
  // boundaries kernel.cu does.
  uint64_t norm_out;          // float [BS, hidden]         (RMSNorm output)
  uint64_t q_out;             // float [16, q_size]         (WMMA out)
  uint64_t k_out;             // float [16, kv_size]        (WMMA out, pre-RoPE/KV-write)
  uint64_t v_out;             // float [16, kv_size]        (WMMA out)
  uint64_t attn_out;          // float [BS, q_size]         (post attention, pre O_PROJ)
  uint64_t o_proj_out;        // float [16, hidden]         (WMMA out, pre residual)
  uint64_t x_attn;            // float [BS, hidden]         (residual after attention)
  uint64_t post_attn_norm;    // float [BS, hidden]         (reserved; FFN folds the norm)
  uint64_t silu_out;          // float [16, intermediate]   (WMMA out + SiLU)
  uint64_t ffn_down_out;      // float [16, hidden]         (WMMA out)
  uint64_t up_out;            // float [16, intermediate]   (WMMA out, temp up-proj)
  uint64_t x_ffn;             // float [BS, hidden]         (residual after FFN)
  uint64_t x_resid;           // bf16  [BS, hidden]         (layer input / next layer input)
  uint64_t lm_logits;         // float [16, vocab]          (WMMA out, pre-argmax)
  // split-K attention partials (when attn split > 1; NOT implemented yet —
  // tg_attn_decode_op does not write them, so keep attn_splits == 1)
  uint64_t o_partial;         // float [BS, hkv, max_splits, hidden]
  uint64_t lse_partial;       // float [BS, hkv, max_splits]

  // --- barriers ------------------------------------------------------------
  TgBars bars;

  // --- model geometry ------------------------------------------------------
  int32_t num_layers;         // 28 for Qwen3-0.6B
  int32_t BS;                 // batch size (rows in the fixed geometry)
  int32_t hidden;             // 1024
  int32_t intermediate;       // 3072
  int32_t q_size;             // num_q_heads * head_dim = 2048
  int32_t kv_size;            // num_kv_heads * head_dim = 1024
  int32_t num_q_heads;        // 16
  int32_t num_kv_heads;       // 8
  int32_t head_dim;           // 128
  int32_t vocab_size;         // 151936
  int32_t oproj_splits;       // how O_PROJ splits its N dimension across bars
  int32_t down_splits;        // how FFN_DOWN splits its K dimension across bars
  int32_t max_attn_splits;    // capacity of o_partial/lse_partial (>= any bucket's splits)

  // --- sampling / step control ---------------------------------------------
  int32_t timing;             // 1 = record per-step kernel ms into timing_ms
  // Appended fields go below this line; existing offsets stay stable.
};

// ---------------------------------------------------------------------------
// One device region the runtime zeroes before each decode step (scratch + bars).
// Kept POD: the runtime uploads a packed array of these to a small kernel.
// ---------------------------------------------------------------------------
struct TgZeroRegion {
  uint64_t ptr;    // device address; may be 0 only when bytes is 0
  uint64_t bytes;
};

// ---------------------------------------------------------------------------
// Token callback. Called from the decode thread (no GIL) after each step's
// D2H + watchdog sync. The binding trampoline acquires the GIL before invoking
// the Python callable. See docs/11 §7.4.
//   context        : opaque host pointer passed through
//   sampled_tokens : host int64 [BS], the token sampled this step per row
//   emitted_rows   : host int32 [BS], 1 = this row produced a token this step
//   batch_size     : current geometry BS
// Returns non-zero to signal a fatal error (terminates the decode loop).
// ---------------------------------------------------------------------------
using TgTokenCallback = int (*)(
    void* context,
    const long long* sampled_tokens,
    const int32_t* emitted_rows,
    int32_t batch_size);

}  // namespace tg
