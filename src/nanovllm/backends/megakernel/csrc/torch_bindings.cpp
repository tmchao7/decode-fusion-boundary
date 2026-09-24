#include <Python.h>
#include <algorithm>
#include <cstdlib>
#include <vector>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/library.h>

#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)
#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

#define REGISTER_EXTENSION(NAME)                                               \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                     \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT,                 \
                                        STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                           \
  }

// Task-graph launch ABI and device entrypoints.
#include "tg_abi.h"

extern "C" cudaError_t tg_launch_decode(const tg::TgLaunchDesc* desc,
                                        cudaStream_t stream);
extern "C" void tg_launch_zero_regions(const tg::TgZeroRegion* regions,
                                        int num_regions, int blocks_per_region,
                                        cudaStream_t stream);

static uint64_t tensor_ptr(const torch::Tensor& t) {
  return reinterpret_cast<uint64_t>(t.data_ptr());
}

// ---------------------------------------------------------------------------
// Scratch-buffer layout guards.
//
// The kernel hardcodes each GEMM's row stride as a template parameter and
// wmma::store_matrix_sync always stores a full 16x16 tile, so every WMMA
// output buffer must have at least 16 rows even when BS < 16. A buffer that
// matches the old "[BS, cols]" documentation would be silently overrun.
// ---------------------------------------------------------------------------
constexpr int64_t TG_WMMA_PAD_ROWS = 16;

static void tg_require_wmma_buffer(const torch::Tensor& t, int64_t rows,
                                   int64_t cols, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.dim() == 2 && t.size(0) >= rows && t.size(1) == cols,
              name, " must be [>= ", rows, ", ", cols,
              "] because wmma::store_matrix_sync writes a full 16-row tile; got ",
              t.sizes());
}

static void tg_require_scratch(const torch::Tensor& t, int64_t rows,
                               int64_t cols, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.dim() == 2 && t.size(0) >= rows && t.size(1) == cols,
              name, " must be [>= ", rows, ", ", cols, "]; got ", t.sizes());
}

void tg_decode_step(
    // instruction stream
    torch::Tensor inst_buf, torch::Tensor num_inst_per_sm,
    int64_t max_inst, int64_t num_sms,
    // weights
    torch::Tensor layer_weights_packed, torch::Tensor embed_weight,
    torch::Tensor final_norm_weight, torch::Tensor cos_table,
    torch::Tensor sin_table,
    // KV cache + paging
    torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor page_table,
    torch::Tensor cache_seqlens, torch::Tensor row_active,
    int64_t num_phys_pages, int64_t max_pages_per_seq,
    int64_t page_block_size, int64_t kv_layer_stride,
    // scratch buffers
    torch::Tensor norm_out, torch::Tensor q_out, torch::Tensor k_out,
    torch::Tensor v_out, torch::Tensor attn_out, torch::Tensor o_proj_out,
    torch::Tensor x_attn, torch::Tensor post_attn_norm, torch::Tensor silu_out,
    torch::Tensor ffn_down_out, torch::Tensor up_out, torch::Tensor x_ffn, torch::Tensor x_resid,
    torch::Tensor lm_logits, torch::Tensor o_partial, torch::Tensor lse_partial,
    // bars
    torch::Tensor bar_qkv, torch::Tensor bar_attn, torch::Tensor bar_oproj,
    torch::Tensor bar_silu, torch::Tensor bar_ffn_down, torch::Tensor bar_layer,
    torch::Tensor bar_combine, torch::Tensor bar_lm_head,
    // geometry
    int64_t num_layers, int64_t BS, int64_t hidden, int64_t intermediate,
    int64_t q_size, int64_t kv_size, int64_t num_q_heads,
    int64_t num_kv_heads, int64_t head_dim, int64_t vocab_size,
    int64_t oproj_splits, int64_t down_splits, int64_t max_attn_splits) {
  // Layout guards: WMMA outputs need 16 rows; elementwise scratch needs BS.
  tg_require_wmma_buffer(q_out, TG_WMMA_PAD_ROWS, q_size, "q_out");
  tg_require_wmma_buffer(k_out, TG_WMMA_PAD_ROWS, kv_size, "k_out");
  tg_require_wmma_buffer(v_out, TG_WMMA_PAD_ROWS, kv_size, "v_out");
  tg_require_wmma_buffer(o_proj_out, TG_WMMA_PAD_ROWS, hidden, "o_proj_out");
  tg_require_wmma_buffer(silu_out, TG_WMMA_PAD_ROWS, intermediate, "silu_out");
  tg_require_wmma_buffer(ffn_down_out, TG_WMMA_PAD_ROWS, hidden, "ffn_down_out");
  tg_require_wmma_buffer(up_out, TG_WMMA_PAD_ROWS, intermediate, "up_out");
  tg_require_wmma_buffer(lm_logits, TG_WMMA_PAD_ROWS, vocab_size, "lm_logits");
  tg_require_scratch(norm_out, BS, hidden, "norm_out");
  tg_require_scratch(attn_out, BS, q_size, "attn_out");
  tg_require_scratch(x_attn, BS, hidden, "x_attn");
  tg_require_scratch(x_ffn, BS, hidden, "x_ffn");
  tg_require_scratch(x_resid, BS, hidden, "x_resid");
  tg_require_scratch(post_attn_norm, BS, hidden, "post_attn_norm");
  TORCH_CHECK(bar_lm_head.is_cuda() && bar_lm_head.numel() >= 1,
              "bar_lm_head must be a CUDA tensor with at least one element");

  // Build TgLaunchDesc on the stack.
  tg::TgLaunchDesc desc{};

  desc.num_sms = (int32_t)num_sms;
  desc.inst_buf = tensor_ptr(inst_buf);
  desc.num_inst_per_sm = tensor_ptr(num_inst_per_sm);
  desc.max_inst = (int32_t)max_inst;

  desc.layer_weights = tensor_ptr(layer_weights_packed);
  desc.embed_weight = tensor_ptr(embed_weight);
  desc.final_norm_weight = tensor_ptr(final_norm_weight);
  desc.cos_table = tensor_ptr(cos_table);
  desc.sin_table = tensor_ptr(sin_table);

  desc.k_cache = tensor_ptr(k_cache);
  desc.v_cache = tensor_ptr(v_cache);
  desc.page_table = tensor_ptr(page_table);
  desc.cache_seqlens = tensor_ptr(cache_seqlens);
  desc.row_active = tensor_ptr(row_active);
  desc.num_phys_pages = (int32_t)num_phys_pages;
  desc.max_pages_per_seq = (int32_t)max_pages_per_seq;
  desc.page_block_size = (int32_t)page_block_size;
  desc.kv_layer_stride = (int64_t)kv_layer_stride;

  desc.norm_out = tensor_ptr(norm_out);
  desc.q_out = tensor_ptr(q_out);
  desc.k_out = tensor_ptr(k_out);
  desc.v_out = tensor_ptr(v_out);
  desc.attn_out = tensor_ptr(attn_out);
  desc.o_proj_out = tensor_ptr(o_proj_out);
  desc.x_attn = tensor_ptr(x_attn);
  desc.post_attn_norm = tensor_ptr(post_attn_norm);
  desc.silu_out = tensor_ptr(silu_out);
  desc.ffn_down_out = tensor_ptr(ffn_down_out);
  desc.up_out = tensor_ptr(up_out);
  desc.x_ffn = tensor_ptr(x_ffn);
  desc.x_resid = tensor_ptr(x_resid);
  desc.lm_logits = tensor_ptr(lm_logits);
  desc.o_partial = tensor_ptr(o_partial);
  desc.lse_partial = tensor_ptr(lse_partial);

  desc.bars.bar_qkv = tensor_ptr(bar_qkv);
  desc.bars.bar_attn = tensor_ptr(bar_attn);
  desc.bars.bar_oproj = tensor_ptr(bar_oproj);
  desc.bars.bar_silu = tensor_ptr(bar_silu);
  desc.bars.bar_ffn_down = tensor_ptr(bar_ffn_down);
  desc.bars.bar_layer = tensor_ptr(bar_layer);
  desc.bars.bar_combine = tensor_ptr(bar_combine);
  desc.bars.bar_lm_head = tensor_ptr(bar_lm_head);

  desc.num_layers = (int32_t)num_layers;
  desc.BS = (int32_t)BS;
  desc.hidden = (int32_t)hidden;
  desc.intermediate = (int32_t)intermediate;
  desc.q_size = (int32_t)q_size;
  desc.kv_size = (int32_t)kv_size;
  desc.num_q_heads = (int32_t)num_q_heads;
  desc.num_kv_heads = (int32_t)num_kv_heads;
  desc.head_dim = (int32_t)head_dim;
  desc.vocab_size = (int32_t)vocab_size;
  desc.oproj_splits = (int32_t)oproj_splits;
  desc.down_splits = (int32_t)down_splits;
  desc.max_attn_splits = (int32_t)max_attn_splits;
  desc.timing = 0;

  // Launch on the current CUDA stream. A failed cooperative launch (e.g. the
  // grid does not fit co-resident) must raise, not silently leave the caller
  // reading the previous step's lm_logits.
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  cudaError_t err = tg_launch_decode(&desc, stream.stream());
  TORCH_CHECK(err == cudaSuccess,
              "tg_launch_decode failed (num_sms=", num_sms, "): ",
              cudaGetErrorString(err));
}

void tg_zero_regions_binding(torch::Tensor regions_packed,
                              int64_t num_regions) {
  if (num_regions <= 0) return;
  // regions_packed is [num_regions, 2] int64: each row is (ptr, bytes).
  // It is typically a CUDA tensor (the tests build it with device="cuda"), so
  // read it through a CPU copy: dereferencing data_ptr() directly on the host
  // segfaults on a device address.
  auto host = regions_packed.detach().to(torch::kCPU).contiguous();
  TORCH_CHECK(host.scalar_type() == torch::kInt64 && host.numel() >= num_regions * 2,
              "regions_packed must be an int64 tensor with >= 2*num_regions elements");
  const int64_t* raw = host.data_ptr<int64_t>();
  std::vector<tg::TgZeroRegion> regions(num_regions);
  uint64_t max_bytes = 0;
  for (int64_t i = 0; i < num_regions; i++) {
    regions[i].ptr = (uint64_t)raw[i * 2 + 0];
    regions[i].bytes = (uint64_t)raw[i * 2 + 1];
    max_bytes = std::max(max_bytes, regions[i].bytes);
  }
  // Intra-region parallelism: grid = (blocks_per_region, num_regions).
  const int64_t threads = 256;
  int64_t blocks_per_region = (int64_t)((max_bytes + threads - 1) / threads);
  if (blocks_per_region > 256) blocks_per_region = 256;
  if (blocks_per_region < 1) blocks_per_region = 1;

  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  // Cache the device staging buffer: cudaMalloc/cudaFree per step would
  // implicitly synchronize the device and dominate a sub-millisecond step.
  static tg::TgZeroRegion* d_regions = nullptr;
  static int64_t d_regions_cap = 0;
  if (num_regions > d_regions_cap) {
    if (d_regions != nullptr) cudaFree(d_regions);
    cudaMalloc(&d_regions, num_regions * sizeof(tg::TgZeroRegion));
    d_regions_cap = num_regions;
  }
  cudaMemcpyAsync(d_regions, regions.data(),
                   num_regions * sizeof(tg::TgZeroRegion),
                   cudaMemcpyHostToDevice, stream.stream());
  tg_launch_zero_regions(d_regions, (int)num_regions, (int)blocks_per_region,
                          stream.stream());
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def(
      "tg_decode_step("
      // instruction stream
      "Tensor inst_buf, Tensor num_inst_per_sm, int max_inst, int num_sms, "
      // weights
      "Tensor layer_weights_packed, Tensor embed_weight, "
      "Tensor final_norm_weight, Tensor cos_table, Tensor sin_table, "
      // KV cache + paging
      "Tensor k_cache, Tensor v_cache, Tensor page_table, "
      "Tensor cache_seqlens, Tensor row_active, "
      "int num_phys_pages, int max_pages_per_seq, int page_block_size, "
      "int kv_layer_stride, "
      // scratch buffers (float intermediates + bf16 x_resid)
      "Tensor norm_out, Tensor q_out, Tensor k_out, Tensor v_out, "
      "Tensor attn_out, Tensor o_proj_out, Tensor x_attn, "
       "Tensor post_attn_norm, Tensor silu_out, Tensor ffn_down_out, "
       "Tensor up_out, "
       "Tensor x_ffn, Tensor x_resid, Tensor lm_logits, "
      "Tensor o_partial, Tensor lse_partial, "
      // bars
      "Tensor bar_qkv, Tensor bar_attn, Tensor bar_oproj, "
      "Tensor bar_silu, Tensor bar_ffn_down, Tensor bar_layer, "
      "Tensor bar_combine, Tensor bar_lm_head, "
      // geometry
      "int num_layers, int BS, int hidden, int intermediate, "
      "int q_size, int kv_size, int num_q_heads, int num_kv_heads, "
      "int head_dim, int vocab_size, int oproj_splits, int down_splits, "
      "int max_attn_splits) -> ()");
  ops.impl("tg_decode_step", torch::kCUDA, &tg_decode_step);

  ops.def(
      "tg_zero_regions(Tensor regions_packed, int num_regions) -> ()");
  // The binding internally copies to CPU and then to a cached device buffer,
  // so it accepts both CPU and CUDA input tensors.
  ops.impl("tg_zero_regions", torch::kCUDA, &tg_zero_regions_binding);
  ops.impl("tg_zero_regions", torch::kCPU, &tg_zero_regions_binding);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
