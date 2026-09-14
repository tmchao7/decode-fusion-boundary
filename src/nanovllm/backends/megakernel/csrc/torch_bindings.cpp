#include <Python.h>
#include <cstdlib>
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

static void maybe_sync_debug(const char *phase);

// Largest batch the WMMA batch kernel can actually run, derived from the real
// cooperative occupancy rather than from LDG_MAX_BATCH.  The kernel partitions
// the resident grid across sequences (blocks_per_seq = grid_blocks / batch) and
// drops the launch when blocks_per_seq < LDG_ATTN_BLOCKS, so with the default
// 80-block grid and LDG_ATTN_BLOCKS=8 only batch <= 10 can run even though
// LDG_MAX_BATCH and this file's TORCH_CHECK allow 32.  Defined in kernel.cu.
extern "C" int ldg_wmma_max_supported_batch(int requested_blocks);

/// Reject a batch the batch kernel cannot launch.
///
/// Both the scalar and WMMA batch kernels partition the resident cooperative
/// grid across sequences (blocks_per_seq = grid_blocks / batch_size) and need
/// at least LDG_ATTN_BLOCKS blocks per sequence.  Without this check the C++
/// launch helper prints to stderr and returns, the void binding reports
/// success, and the Python caller reads the *previous* step's tokens out of
/// ``output_tokens`` as if they were fresh output.
static void check_batch_supported(int64_t batch_size) {
  const int max_batch = ldg_wmma_max_supported_batch(0);
  if (max_batch > 0 && batch_size > max_batch) {
    TORCH_CHECK(false,
                "megakernel batch decode supports at most ", max_batch,
                " sequences (got ", batch_size,
                "); the cooperative grid is partitioned per sequence and needs "
                "at least LDG_ATTN_BLOCKS=8 blocks per sequence. Use "
                "torch-cudagraph for larger batches.");
  }
}

static void validate_paged_metadata(const torch::Tensor &block_table,
                                    const torch::Tensor &k_cache,
                                    int64_t position, int64_t block_size,
                                    int64_t num_cache_blocks) {
  if (std::getenv("NANOVLLM_CUDA_DEBUG") == nullptr)
    return;

  auto host_table = block_table.to(torch::kCPU).contiguous();
  const int *values = host_table.data_ptr<int>();
  const int64_t required = position / block_size + 1;
  TORCH_CHECK(required <= host_table.numel(),
              "paged block table is too short: required=", required,
              " entries, actual=", host_table.numel());
  std::fprintf(stderr,
               "Megakernel paged metadata: position=%ld block_size=%ld "
               "num_cache_blocks=%ld table_numel=%ld k_cache_ptr=%p\n",
               static_cast<long>(position), static_cast<long>(block_size),
               static_cast<long>(num_cache_blocks),
               static_cast<long>(host_table.numel()), k_cache.data_ptr());
  for (int64_t i = 0; i < required; ++i) {
    TORCH_CHECK(values[i] >= 0 && values[i] < num_cache_blocks,
                "invalid paged physical block: index=", i,
                " value=", values[i], " num_cache_blocks=", num_cache_blocks);
    std::fprintf(stderr, "  block_table[%ld]=%d\n", static_cast<long>(i),
                 values[i]);
  }
}

struct LDGLayerWeights {
  const void *input_layernorm_weight;
  const void *q_proj_weight;
  const void *k_proj_weight;
  const void *v_proj_weight;
  const void *q_norm_weight;
  const void *k_norm_weight;
  const void *o_proj_weight;
  const void *post_attn_layernorm_weight;
  const void *gate_proj_weight;
  const void *up_proj_weight;
  const void *down_proj_weight;
};

extern "C" void launch_ldg_decode_persistent(
    int input_token_id, int *output_token_id, const void *embed_weight,
    const LDGLayerWeights *layer_weights, const void *final_norm_weight,
    const void *lm_head_weight, const void *cos_table, const void *sin_table,
    void *k_cache, void *v_cache, void *hidden_buffer, void *g_activations,
    void *g_residual, void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_normalized, void *block_max_vals,
    void *block_max_idxs, int num_layers, int position, int cache_len,
    int max_seq_len, float attn_scale, cudaStream_t stream);

extern "C" void launch_ldg_decode_direct(
    int input_token_id, int *output_token_id, const void *embed_weight,
    const LDGLayerWeights *layer_weights, const void *final_norm_weight,
    const void *lm_head_weight, const void *cos_table, const void *sin_table,
    void *k_cache, void *v_cache, void *hidden_buffer, void *g_activations,
    void *g_residual, void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_normalized, void *block_max_vals,
    void *block_max_idxs, int num_layers, int position, int max_seq_len,
    const int *block_table, int block_size, int num_cache_blocks,
    float attn_scale, cudaStream_t stream);

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
    float attn_scale, cudaStream_t stream);

extern "C" void launch_ldg_decode_batch_wmma(
    const void *input_token_ids, const void *positions,
    const void *block_tables, int *output_token_ids,
    const void *embed_weight, const LDGLayerWeights *layer_weights,
    const void *final_norm_weight, const void *lm_head_weight,
    const void *cos_table, const void *sin_table, void *k_cache,
    void *v_cache, void *hidden_buffer, void *g_activations,
    void *g_residual, void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_mlp_up, void *g_normalized,
    void *block_max_vals, void *block_max_idxs, int num_layers, int batch_size,
    int block_table_stride, int block_size, int num_cache_blocks,
    float attn_scale, cudaStream_t stream);

void decode(torch::Tensor output_token, int64_t input_token_id,
            torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
            torch::Tensor final_norm_weight, torch::Tensor lm_head_weight,
            torch::Tensor cos_table, torch::Tensor sin_table,
            torch::Tensor k_cache, torch::Tensor v_cache,
            torch::Tensor hidden_buffer, torch::Tensor activations,
            torch::Tensor residual, torch::Tensor q, torch::Tensor k,
            torch::Tensor v, torch::Tensor attn_out,
            torch::Tensor mlp_intermediate, torch::Tensor normalized,
            torch::Tensor block_max_vals, torch::Tensor block_max_idxs,
            int64_t num_layers, int64_t position, int64_t max_seq_len,
            double attn_scale) {
  launch_ldg_decode_direct(
      static_cast<int>(input_token_id),
      static_cast<int *>(output_token.data_ptr()), embed_weight.data_ptr(),
      reinterpret_cast<const LDGLayerWeights *>(
          layer_weights_packed.data_ptr()),
      final_norm_weight.data_ptr(), lm_head_weight.data_ptr(),
      cos_table.data_ptr(), sin_table.data_ptr(), k_cache.data_ptr(),
      v_cache.data_ptr(), hidden_buffer.data_ptr(), activations.data_ptr(),
      residual.data_ptr(), q.data_ptr(), k.data_ptr(), v.data_ptr(),
      attn_out.data_ptr(), mlp_intermediate.data_ptr(), normalized.data_ptr(),
      block_max_vals.data_ptr(), block_max_idxs.data_ptr(),
      static_cast<int>(num_layers), static_cast<int>(position),
      static_cast<int>(max_seq_len), nullptr, static_cast<int>(max_seq_len), 1,
      static_cast<float>(attn_scale),
      c10::cuda::getCurrentCUDAStream().stream());
  maybe_sync_debug("decode");
}

static void maybe_sync_debug(const char *phase) {
  if (std::getenv("NANOVLLM_CUDA_DEBUG") != nullptr) {
    auto error = cudaDeviceSynchronize();
    TORCH_CHECK(error == cudaSuccess, "Megakernel ", phase,
                " failed: ", cudaGetErrorString(error));
  }
}

void decode_paged(
    torch::Tensor output_token, int64_t input_token_id,
    torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight, torch::Tensor lm_head_weight,
    torch::Tensor cos_table, torch::Tensor sin_table,
    torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor hidden_buffer, torch::Tensor activations,
    torch::Tensor residual, torch::Tensor q, torch::Tensor k,
    torch::Tensor v, torch::Tensor attn_out, torch::Tensor mlp_intermediate,
    torch::Tensor normalized, torch::Tensor block_max_vals,
    torch::Tensor block_max_idxs, torch::Tensor block_table,
    int64_t num_layers, int64_t position, int64_t block_size,
    int64_t num_cache_blocks, double attn_scale) {
  TORCH_CHECK(block_table.is_cuda(), "block_table must be a CUDA tensor");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32,
              "block_table must have int32 dtype");
  TORCH_CHECK(block_table.dim() == 1 && block_table.is_contiguous(),
              "block_table must be a contiguous 1D tensor");
  TORCH_CHECK(k_cache.is_cuda() && v_cache.is_cuda(),
              "paged KV cache tensors must be CUDA tensors");
  TORCH_CHECK(k_cache.scalar_type() == torch::kBFloat16 &&
                  v_cache.scalar_type() == torch::kBFloat16,
              "paged KV cache tensors must be BF16 tensors");
  TORCH_CHECK(block_size > 0 && num_cache_blocks > 0,
              "block_size and num_cache_blocks must be positive");
  TORCH_CHECK(position >= 0, "position must be non-negative");
  TORCH_CHECK(output_token.is_cuda() && output_token.scalar_type() == torch::kInt32 &&
                  output_token.numel() >= 1,
              "output_token must be a CUDA int32 tensor with one element");
  TORCH_CHECK(k_cache.dim() == 5 && v_cache.dim() == 5 &&
                  k_cache.is_contiguous() && v_cache.is_contiguous() &&
                  k_cache.sizes() == v_cache.sizes(),
              "paged KV cache views must be matching contiguous 5D tensors");
  TORCH_CHECK(k_cache.size(0) == num_layers && k_cache.size(1) == num_cache_blocks &&
                  k_cache.size(2) == block_size && k_cache.size(3) == 8 &&
                  k_cache.size(4) == 128,
              "paged KV cache shape does not match Qwen3-0.6B metadata");
  TORCH_CHECK(block_table.numel() >= (position / block_size) + 1,
              "block_table is too short for the requested position");
  validate_paged_metadata(block_table, k_cache, position, block_size,
                          num_cache_blocks);

  launch_ldg_decode_direct(
      static_cast<int>(input_token_id),
      static_cast<int *>(output_token.data_ptr()), embed_weight.data_ptr(),
      reinterpret_cast<const LDGLayerWeights *>(layer_weights_packed.data_ptr()),
      final_norm_weight.data_ptr(), lm_head_weight.data_ptr(),
      cos_table.data_ptr(), sin_table.data_ptr(), k_cache.data_ptr(),
      v_cache.data_ptr(), hidden_buffer.data_ptr(), activations.data_ptr(),
      residual.data_ptr(), q.data_ptr(), k.data_ptr(), v.data_ptr(),
      attn_out.data_ptr(), mlp_intermediate.data_ptr(), normalized.data_ptr(),
      block_max_vals.data_ptr(), block_max_idxs.data_ptr(),
      static_cast<int>(num_layers), static_cast<int>(position),
      static_cast<int>(block_size * num_cache_blocks), block_table.data_ptr<int>(),
      static_cast<int>(block_size), static_cast<int>(num_cache_blocks),
      static_cast<float>(attn_scale), c10::cuda::getCurrentCUDAStream().stream());
}

void decode_paged_batch(
    torch::Tensor output_tokens, torch::Tensor input_token_ids,
    torch::Tensor positions, torch::Tensor block_tables,
    torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight, torch::Tensor lm_head_weight,
    torch::Tensor cos_table, torch::Tensor sin_table,
    torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor hidden_buffer, torch::Tensor activations,
    torch::Tensor residual, torch::Tensor q, torch::Tensor k,
    torch::Tensor v, torch::Tensor attn_out, torch::Tensor mlp_intermediate,
    torch::Tensor normalized, torch::Tensor block_max_vals,
    torch::Tensor block_max_idxs, int64_t num_layers, int64_t block_size,
    int64_t num_cache_blocks, double attn_scale) {
  TORCH_CHECK(output_tokens.is_cuda() && output_tokens.scalar_type() == torch::kInt32,
              "output_tokens must be a CUDA int32 tensor");
  TORCH_CHECK(input_token_ids.is_cuda() &&
                  input_token_ids.scalar_type() == torch::kInt64,
              "input_token_ids must be a CUDA int64 tensor");
  TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == torch::kInt64,
              "positions must be a CUDA int64 tensor");
  TORCH_CHECK(block_tables.is_cuda() &&
                  block_tables.scalar_type() == torch::kInt32,
              "block_tables must be a CUDA int32 tensor");
  TORCH_CHECK(output_tokens.dim() == 1 && input_token_ids.dim() == 1 &&
                  positions.dim() == 1,
              "batch metadata tensors must be 1D");
  TORCH_CHECK(block_tables.dim() == 2 && block_tables.is_contiguous() &&
                  block_tables.size(1) > 0,
              "block_tables must be a non-empty contiguous 2D tensor");
  TORCH_CHECK(input_token_ids.numel() == positions.numel() &&
                  output_tokens.numel() == input_token_ids.numel() &&
                  block_tables.size(0) == input_token_ids.numel(),
              "batch metadata sizes must match");
  TORCH_CHECK(input_token_ids.numel() > 0 && input_token_ids.numel() <= 32,
              "decode_paged_batch supports batch sizes from 1 to 32");
  TORCH_CHECK(output_tokens.is_contiguous() && input_token_ids.is_contiguous() &&
                  positions.is_contiguous(),
              "batch metadata tensors must be contiguous");
  TORCH_CHECK(k_cache.is_cuda() && v_cache.is_cuda() && k_cache.dim() == 5 &&
                  k_cache.is_contiguous() && v_cache.is_contiguous() &&
                  v_cache.sizes() == k_cache.sizes(),
              "paged KV cache views must be matching contiguous CUDA 5D tensors");
  TORCH_CHECK(k_cache.size(0) == num_layers && k_cache.size(1) == num_cache_blocks &&
                  k_cache.size(2) == block_size && k_cache.size(3) == 8 &&
                  k_cache.size(4) == 128,
              "paged KV cache shape does not match Qwen3-0.6B metadata");
  TORCH_CHECK(block_size > 0 && num_cache_blocks > 0,
              "block_size and num_cache_blocks must be positive");
  if (std::getenv("NANOVLLM_CUDA_DEBUG") != nullptr) {
    auto host_tables = block_tables.to(torch::kCPU).contiguous();
    auto host_positions = positions.to(torch::kCPU).contiguous();
    const int *tables = host_tables.data_ptr<int>();
    const int64_t *pos = host_positions.data_ptr<int64_t>();
    for (int64_t row = 0; row < block_tables.size(0); ++row) {
      int64_t required = pos[row] / block_size + 1;
      TORCH_CHECK(required <= block_tables.size(1),
                  "paged batch block table is too short for row=", row);
      for (int64_t col = 0; col < required; ++col) {
        int value = tables[row * block_tables.size(1) + col];
        TORCH_CHECK(value >= 0 && value < num_cache_blocks,
                    "invalid paged batch physical block: row=", row,
                    " col=", col, " value=", value,
                    " num_cache_blocks=", num_cache_blocks);
      }
    }
  }

  check_batch_supported(input_token_ids.numel());
  // Defence-in-depth: sentinel the output so that if the launch aborts after
  // this point (e.g. a caller bypassed the check via a different path), the
  // Python caller reads -1 instead of the previous step's tokens.
  output_tokens.fill_(-1);

  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  launch_ldg_decode_paged_batch(
      input_token_ids.data_ptr(), positions.data_ptr(), block_tables.data_ptr(),
      output_tokens.data_ptr<int>(), embed_weight.data_ptr(),
      reinterpret_cast<const LDGLayerWeights *>(layer_weights_packed.data_ptr()),
      final_norm_weight.data_ptr(), lm_head_weight.data_ptr(),
      cos_table.data_ptr(), sin_table.data_ptr(), k_cache.data_ptr(),
      v_cache.data_ptr(), hidden_buffer.data_ptr(), activations.data_ptr(),
      residual.data_ptr(), q.data_ptr(), k.data_ptr(), v.data_ptr(),
      attn_out.data_ptr(), mlp_intermediate.data_ptr(), normalized.data_ptr(),
      block_max_vals.data_ptr(), block_max_idxs.data_ptr(),
      static_cast<int>(num_layers), static_cast<int>(input_token_ids.numel()),
      static_cast<int>(block_tables.size(1)), static_cast<int>(block_size),
      static_cast<int>(num_cache_blocks), static_cast<float>(attn_scale), stream);
}

void decode_paged_batch_wmma(
    torch::Tensor output_tokens, torch::Tensor input_token_ids,
    torch::Tensor positions, torch::Tensor block_tables,
    torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight, torch::Tensor lm_head_weight,
    torch::Tensor cos_table, torch::Tensor sin_table, torch::Tensor k_cache,
    torch::Tensor v_cache, torch::Tensor hidden_buffer,
    torch::Tensor activations, torch::Tensor residual, torch::Tensor q,
    torch::Tensor k, torch::Tensor v, torch::Tensor attn_out,
    torch::Tensor mlp_intermediate, torch::Tensor mlp_up,
    torch::Tensor normalized,
    torch::Tensor block_max_vals, torch::Tensor block_max_idxs,
    int64_t num_layers, int64_t block_size, int64_t num_cache_blocks,
    double attn_scale) {
  TORCH_CHECK(input_token_ids.numel() > 0 && input_token_ids.numel() <= 32,
              "decode_paged_batch_wmma supports 1-32");
  check_batch_supported(input_token_ids.numel());
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  launch_ldg_decode_batch_wmma(
      input_token_ids.data_ptr(), positions.data_ptr(),
      block_tables.data_ptr(), output_tokens.data_ptr<int>(),
      embed_weight.data_ptr(),
      reinterpret_cast<const LDGLayerWeights *>(layer_weights_packed.data_ptr()),
      final_norm_weight.data_ptr(), lm_head_weight.data_ptr(),
      cos_table.data_ptr(), sin_table.data_ptr(), k_cache.data_ptr(),
      v_cache.data_ptr(), hidden_buffer.data_ptr(), activations.data_ptr(),
      residual.data_ptr(), q.data_ptr(), k.data_ptr(), v.data_ptr(),
      attn_out.data_ptr(), mlp_intermediate.data_ptr(), mlp_up.data_ptr(),
      normalized.data_ptr(), block_max_vals.data_ptr(),
      block_max_idxs.data_ptr(), static_cast<int>(num_layers),
      static_cast<int>(input_token_ids.numel()),
      static_cast<int>(block_tables.size(1)),
      static_cast<int>(block_size), static_cast<int>(num_cache_blocks),
      static_cast<float>(attn_scale), stream);
}


extern "C" void launch_ldg_generate_nosync(
    int first_token_id, int num_steps, const void *embed_weight,
    const LDGLayerWeights *layer_weights, const void *final_norm_weight,
    const void *lm_head_weight, const void *cos_table, const void *sin_table,
    void *k_cache, void *v_cache, void *hidden_buffer, void *g_activations,
    void *g_residual, void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_normalized, void *block_max_vals,
    void *block_max_idxs, int *output_log, int num_layers, int start_position,
    int max_seq_len, float attn_scale, cudaStream_t stream);

torch::Tensor generate_nosync(
    int64_t first_token_id, int64_t num_steps, torch::Tensor embed_weight,
    torch::Tensor layer_weights_packed, torch::Tensor final_norm_weight,
    torch::Tensor lm_head_weight, torch::Tensor cos_table,
    torch::Tensor sin_table, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor hidden_buffer, torch::Tensor activations,
    torch::Tensor residual, torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor attn_out, torch::Tensor mlp_intermediate,
    torch::Tensor normalized, torch::Tensor block_max_vals,
    torch::Tensor block_max_idxs, int64_t num_layers, int64_t start_position,
    int64_t max_seq_len, double attn_scale) {

  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  auto output_log = torch::empty(
      {num_steps}, torch::dtype(torch::kInt32).device(torch::kCUDA));

  launch_ldg_generate_nosync(
      static_cast<int>(first_token_id), static_cast<int>(num_steps),
      embed_weight.data_ptr(),
      reinterpret_cast<const LDGLayerWeights *>(
          layer_weights_packed.data_ptr()),
      final_norm_weight.data_ptr(), lm_head_weight.data_ptr(),
      cos_table.data_ptr(), sin_table.data_ptr(), k_cache.data_ptr(),
      v_cache.data_ptr(), hidden_buffer.data_ptr(), activations.data_ptr(),
      residual.data_ptr(), q.data_ptr(), k.data_ptr(), v.data_ptr(),
      attn_out.data_ptr(), mlp_intermediate.data_ptr(), normalized.data_ptr(),
      block_max_vals.data_ptr(), block_max_idxs.data_ptr(),
      static_cast<int *>(output_log.data_ptr()), static_cast<int>(num_layers),
      static_cast<int>(start_position), static_cast<int>(max_seq_len),
      static_cast<float>(attn_scale), stream);

  // Single sync at end -- all N steps already queued
  cudaStreamSynchronize(stream);
  return output_log;
}


extern "C" void launch_ldg_generate_nosync_paged(
    int first_token_id, int num_steps, const void *embed_weight,
    const LDGLayerWeights *layer_weights, const void *final_norm_weight,
    const void *lm_head_weight, const void *cos_table, const void *sin_table,
    void *k_cache, void *v_cache, void *hidden_buffer, void *g_activations,
    void *g_residual, void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_normalized, void *block_max_vals,
    void *block_max_idxs, const int *block_table, int *output_log,
    int num_layers, int start_position, int block_size, int num_cache_blocks,
    int eos_token, float attn_scale, cudaStream_t stream);

torch::Tensor generate_nosync_paged(
    int64_t first_token_id, int64_t num_steps, torch::Tensor embed_weight,
    torch::Tensor layer_weights_packed, torch::Tensor final_norm_weight,
    torch::Tensor lm_head_weight, torch::Tensor cos_table,
    torch::Tensor sin_table, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor hidden_buffer, torch::Tensor activations,
    torch::Tensor residual, torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor attn_out, torch::Tensor mlp_intermediate,
    torch::Tensor normalized, torch::Tensor block_max_vals,
    torch::Tensor block_max_idxs, torch::Tensor block_table,
    int64_t num_layers, int64_t start_position, int64_t block_size,
    int64_t num_cache_blocks, int64_t eos_token, double attn_scale) {

  TORCH_CHECK(block_table.is_cuda() && block_table.scalar_type() == torch::kInt32 &&
                  block_table.is_contiguous() && block_table.dim() == 1,
              "block_table must be a contiguous 1D int32 CUDA tensor");
  TORCH_CHECK(num_steps > 0, "num_steps must be positive");

  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  // Initialise to -1 so that entries past EOS (where the device-side update
  // kernel skips writing) are sentinel-valued rather than garbage.
  auto output_log = torch::full(
      {num_steps}, -1,
      torch::dtype(torch::kInt32).device(torch::kCUDA));

  launch_ldg_generate_nosync_paged(
      static_cast<int>(first_token_id), static_cast<int>(num_steps),
      embed_weight.data_ptr(),
      reinterpret_cast<const LDGLayerWeights *>(
          layer_weights_packed.data_ptr()),
      final_norm_weight.data_ptr(), lm_head_weight.data_ptr(),
      cos_table.data_ptr(), sin_table.data_ptr(), k_cache.data_ptr(),
      v_cache.data_ptr(), hidden_buffer.data_ptr(), activations.data_ptr(),
      residual.data_ptr(), q.data_ptr(), k.data_ptr(), v.data_ptr(),
      attn_out.data_ptr(), mlp_intermediate.data_ptr(), normalized.data_ptr(),
      block_max_vals.data_ptr(), block_max_idxs.data_ptr(),
      block_table.data_ptr<int>(), static_cast<int *>(output_log.data_ptr()),
      static_cast<int>(num_layers), static_cast<int>(start_position),
      static_cast<int>(block_size), static_cast<int>(num_cache_blocks),
      static_cast<int>(eos_token), static_cast<float>(attn_scale), stream);

  // Single sync at end -- all N steps already queued
  cudaStreamSynchronize(stream);
  return output_log;
}

extern "C" void launch_ldg_generate_nosync_paged_batch(
    const void *input_token_ids, const void *positions,
    const void *block_tables, const void *embed_weight,
    const LDGLayerWeights *layer_weights, const void *final_norm_weight,
    const void *lm_head_weight, const void *cos_table, const void *sin_table,
    void *k_cache, void *v_cache, void *hidden_buffer, void *g_activations,
    void *g_residual, void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_mlp_up, void *g_normalized,
    void *block_max_vals, void *block_max_idxs, int *output_log,
    int num_layers, int batch_size, int block_table_stride, int block_size,
    int num_cache_blocks, int max_steps, int eos_token, float attn_scale,
    cudaStream_t stream);

torch::Tensor generate_nosync_paged_batch(
    torch::Tensor input_token_ids, torch::Tensor positions,
    torch::Tensor block_tables, torch::Tensor embed_weight,
    torch::Tensor layer_weights_packed, torch::Tensor final_norm_weight,
    torch::Tensor lm_head_weight, torch::Tensor cos_table,
    torch::Tensor sin_table, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor hidden_buffer, torch::Tensor activations,
    torch::Tensor residual, torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor attn_out, torch::Tensor mlp_intermediate,
    torch::Tensor mlp_up, torch::Tensor normalized,
    torch::Tensor block_max_vals, torch::Tensor block_max_idxs,
    int64_t num_layers, int64_t max_steps, int64_t eos_token,
    int64_t block_size, int64_t num_cache_blocks, double attn_scale) {
  TORCH_CHECK(input_token_ids.is_cuda() && input_token_ids.is_contiguous() &&
                  input_token_ids.scalar_type() == torch::kInt64 &&
                  input_token_ids.dim() == 1 && input_token_ids.numel() > 0 &&
                  input_token_ids.numel() <= 32,
              "input_token_ids must be a contiguous 1D int64 CUDA tensor 1-32");
  TORCH_CHECK(positions.is_cuda() && positions.is_contiguous() &&
                  positions.scalar_type() == torch::kInt64 &&
                  input_token_ids.numel() == positions.numel(),
              "positions must match input_token_ids");
  TORCH_CHECK(block_tables.is_cuda() && block_tables.is_contiguous() &&
                  block_tables.dim() == 2 &&
                  block_tables.size(0) == input_token_ids.numel(),
              "block_tables must be a contiguous int32 2D CUDA tensor");
  TORCH_CHECK(max_steps > 0, "max_steps must be positive");
  check_batch_supported(input_token_ids.numel());

  int batch_size = static_cast<int>(input_token_ids.numel());
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  // Initialise to -1 so that entries past a per-seq EOS (where the device-side
  // update kernel skips writing) are sentinel-valued rather than garbage.
  auto output_log = torch::full(
      {batch_size, max_steps}, -1,
      torch::dtype(torch::kInt32).device(torch::kCUDA));

  // Reset positions/tokens in place (the launcher mutates them via the batch
  // update kernel), then run the whole burst without host sync.
  // (The caller owns these buffers; the launcher updates them on device.)
  launch_ldg_generate_nosync_paged_batch(
      input_token_ids.data_ptr(), positions.data_ptr(),
      block_tables.data_ptr(), embed_weight.data_ptr(),
      reinterpret_cast<const LDGLayerWeights *>(layer_weights_packed.data_ptr()),
      final_norm_weight.data_ptr(), lm_head_weight.data_ptr(),
      cos_table.data_ptr(), sin_table.data_ptr(), k_cache.data_ptr(),
      v_cache.data_ptr(), hidden_buffer.data_ptr(), activations.data_ptr(),
      residual.data_ptr(), q.data_ptr(), k.data_ptr(), v.data_ptr(),
      attn_out.data_ptr(), mlp_intermediate.data_ptr(), mlp_up.data_ptr(),
      normalized.data_ptr(), block_max_vals.data_ptr(), block_max_idxs.data_ptr(),
      static_cast<int *>(output_log.data_ptr()),
      static_cast<int>(num_layers), batch_size,
      static_cast<int>(block_tables.size(1)), static_cast<int>(block_size),
      static_cast<int>(num_cache_blocks), static_cast<int>(max_steps),
      static_cast<int>(eos_token), static_cast<float>(attn_scale), stream);

  cudaStreamSynchronize(stream);
  return output_log;
}

/// Largest batch the WMMA batch decode path can launch on the active device.
///
/// The backend queries this to reject unsupported batches up front with a clear
/// message instead of letting a void launch helper fail silently.
int64_t wmma_max_supported_batch() {
  return static_cast<int64_t>(ldg_wmma_max_supported_batch(0));
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("decode(Tensor output_token, int input_token_id, "
          "Tensor embed_weight, Tensor layer_weights_packed, "
          "Tensor final_norm_weight, Tensor lm_head_weight, "
          "Tensor cos_table, Tensor sin_table, "
          "Tensor k_cache, Tensor v_cache, "
          "Tensor hidden_buffer, Tensor activations, Tensor residual, "
          "Tensor q, Tensor k, Tensor v, Tensor attn_out, "
          "Tensor mlp_intermediate, Tensor normalized, "
          "Tensor block_max_vals, Tensor block_max_idxs, "
          "int num_layers, int position, int max_seq_len, "
          "float attn_scale) -> ()");
  ops.impl("decode", torch::kCUDA, &decode);

  ops.def("decode_paged(Tensor output_token, int input_token_id, "
          "Tensor embed_weight, Tensor layer_weights_packed, "
          "Tensor final_norm_weight, Tensor lm_head_weight, "
          "Tensor cos_table, Tensor sin_table, Tensor k_cache, Tensor v_cache, "
          "Tensor hidden_buffer, Tensor activations, Tensor residual, "
          "Tensor q, Tensor k, Tensor v, Tensor attn_out, "
          "Tensor mlp_intermediate, Tensor normalized, "
          "Tensor block_max_vals, Tensor block_max_idxs, Tensor block_table, "
          "int num_layers, int position, int block_size, "
          "int num_cache_blocks, float attn_scale) -> ()");
  ops.impl("decode_paged", torch::kCUDA, &decode_paged);

  ops.def("decode_paged_batch(Tensor output_tokens, Tensor input_token_ids, "
          "Tensor positions, Tensor block_tables, Tensor embed_weight, "
          "Tensor layer_weights_packed, Tensor final_norm_weight, "
          "Tensor lm_head_weight, Tensor cos_table, Tensor sin_table, "
          "Tensor k_cache, Tensor v_cache, Tensor hidden_buffer, "
          "Tensor activations, Tensor residual, Tensor q, Tensor k, Tensor v, "
          "Tensor attn_out, Tensor mlp_intermediate, Tensor normalized, "
          "Tensor block_max_vals, Tensor block_max_idxs, int num_layers, "
          "int block_size, int num_cache_blocks, float attn_scale) -> ()");
  ops.impl("decode_paged_batch", torch::kCUDA, &decode_paged_batch);

  ops.def("wmma_max_supported_batch() -> int");
  ops.impl("wmma_max_supported_batch", torch::kCUDA, &wmma_max_supported_batch);

  ops.def("decode_paged_batch_wmma(Tensor output_tokens, Tensor input_token_ids, "
          "Tensor positions, Tensor block_tables, Tensor embed_weight, "
          "Tensor layer_weights_packed, Tensor final_norm_weight, "
          "Tensor lm_head_weight, Tensor cos_table, Tensor sin_table, "
          "Tensor k_cache, Tensor v_cache, Tensor hidden_buffer, "
          "Tensor activations, Tensor residual, Tensor q, Tensor k, Tensor v, "
          "Tensor attn_out, Tensor mlp_intermediate, Tensor mlp_up, "
          "Tensor normalized, "
          "Tensor block_max_vals, Tensor block_max_idxs, int num_layers, "
          "int block_size, int num_cache_blocks, float attn_scale) -> ()");
  ops.impl("decode_paged_batch_wmma", torch::kCUDA, &decode_paged_batch_wmma);

  ops.def("generate_nosync(int first_token_id, int num_steps, "
          "Tensor embed_weight, Tensor layer_weights_packed, "
          "Tensor final_norm_weight, Tensor lm_head_weight, "
          "Tensor cos_table, Tensor sin_table, "
          "Tensor k_cache, Tensor v_cache, "
          "Tensor hidden_buffer, Tensor activations, Tensor residual, "
          "Tensor q, Tensor k, Tensor v, Tensor attn_out, "
          "Tensor mlp_intermediate, Tensor normalized, "
          "Tensor block_max_vals, Tensor block_max_idxs, "
          "int num_layers, int start_position, int max_seq_len, "
          "float attn_scale) -> Tensor");
  ops.impl("generate_nosync", torch::kCUDA, &generate_nosync);

  ops.def("generate_nosync_paged(int first_token_id, int num_steps, "
          "Tensor embed_weight, Tensor layer_weights_packed, "
          "Tensor final_norm_weight, Tensor lm_head_weight, "
          "Tensor cos_table, Tensor sin_table, "
          "Tensor k_cache, Tensor v_cache, "
          "Tensor hidden_buffer, Tensor activations, Tensor residual, "
          "Tensor q, Tensor k, Tensor v, Tensor attn_out, "
          "Tensor mlp_intermediate, Tensor normalized, "
          "Tensor block_max_vals, Tensor block_max_idxs, Tensor block_table, "
          "int num_layers, int start_position, int block_size, "
          "int num_cache_blocks, int eos_token, float attn_scale) -> Tensor");
  ops.impl("generate_nosync_paged", torch::kCUDA, &generate_nosync_paged);

  ops.def("generate_nosync_paged_batch(Tensor input_token_ids, Tensor positions, "
          "Tensor block_tables, Tensor embed_weight, "
          "Tensor layer_weights_packed, Tensor final_norm_weight, "
          "Tensor lm_head_weight, Tensor cos_table, Tensor sin_table, "
          "Tensor k_cache, Tensor v_cache, "
          "Tensor hidden_buffer, Tensor activations, Tensor residual, "
          "Tensor q, Tensor k, Tensor v, Tensor attn_out, "
          "Tensor mlp_intermediate, Tensor mlp_up, Tensor normalized, "
          "Tensor block_max_vals, Tensor block_max_idxs, "
          "int num_layers, int max_steps, int eos_token, "
          "int block_size, int num_cache_blocks, float attn_scale) -> Tensor");
  ops.impl("generate_nosync_paged_batch", torch::kCUDA, &generate_nosync_paged_batch);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
