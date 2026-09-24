import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "src/nanovllm/backends/megakernel/csrc/kernel.cu"
BINDINGS = ROOT / "src/nanovllm/backends/megakernel/csrc/torch_bindings.cpp"


class PagedKernelSourceTests(unittest.TestCase):
    def test_kernel_contains_paged_address_mapping(self):
        source = KERNEL.read_text(encoding="utf-8")
        self.assertIn("ldg_kv_ptr", source)
        self.assertIn("logical_block = static_cast<int64_t>(position) / block_size", source)
        self.assertIn("static_cast<int64_t>(physical_block) * block_size * KV_SIZE", source)
        self.assertIn("const int *__restrict__ block_table", source)
        self.assertIn("ldg_decode_kernel_batch", source)
        self.assertIn("ldg_lm_head_fused_batch", source)
        self.assertIn("ldg_resident_grid_blocks", source)
        self.assertIn("cudaOccupancyMaxActiveBlocksPerMultiprocessor", source)
        self.assertIn("ldg_reset_decode_state", source)
        self.assertIn("AtomicGridSync grid", source)
        self.assertIn("(unsigned int)gridDim.x,\n                      0", source)
        self.assertNotIn(
            """if (block_id == 0 && threadIdx.x == 0) {
    *barrier_counter = 0;
    *barrier_sense = 0;""",
            source,
        )

    def test_paged_operator_is_registered(self):
        source = BINDINGS.read_text(encoding="utf-8")
        kernel_source = KERNEL.read_text(encoding="utf-8")
        self.assertIn('ops.def("decode_paged', source)
        self.assertIn('ops.impl("decode_paged"', source)
        self.assertIn("block_table.data_ptr<int>()", source)
        self.assertIn('ops.def("decode_paged_batch', source)
        self.assertIn('ops.impl("decode_paged_batch"', source)
        self.assertIn('ldg_check_phase("transformer decode"', kernel_source)
        self.assertIn('ldg_check_phase("LM head"', kernel_source)
        runner_source = (ROOT / "src/nanovllm/engine/model_runner.py").read_text(encoding="utf-8")
        # CUDA-graph capture must stay off for EVERY megakernel backend. The
        # check used to be an inline string comparison against "megakernel";
        # it now goes through the shared predicate so adding a backend (e.g.
        # "megakernel-tg") cannot silently leave graph capture enabled.
        self.assertIn("is_megakernel_backend(config.decode_backend)", runner_source)
        self.assertNotIn("config.decode_backend != \"megakernel\"", runner_source)
        self.assertIn("not self.use_cudagraph", runner_source)
        self.assertIn("cudaPeekAtLastError", kernel_source)
        self.assertIn("cudaLaunchCooperativeKernel", kernel_source)
        self.assertIn("cudaDevAttrCooperativeLaunch", kernel_source)
        self.assertIn("validate_paged_metadata", source)
        self.assertIn("invalid paged physical block", source)
        self.assertIn("int64_t kv_cache_layer_stride", kernel_source)
        self.assertIn("static_cast<int64_t>(layer) * kv_cache_layer_stride", kernel_source)
        self.assertIn("static_cast<int64_t>(physical_block)", kernel_source)
        self.assertIn("#ifdef LDG_FAST_MATH", kernel_source)
        self.assertIn("return expf(x);", kernel_source)
        self.assertIn("ldg_round_bf16", kernel_source)
        self.assertIn("g_residual[i] = ldg_round_bf16", kernel_source)
        self.assertIn("g_mlp_intermediate[m] = ldg_round_bf16", kernel_source)
        self.assertIn("float reduced = ldg_round_bf16(ldg_warp_reduce_sum(sum[r]))", kernel_source)
        self.assertIn("initial_generation = *barrier_sense", kernel_source)
        self.assertIn("atomicExch(kv_flag, 0u)", kernel_source)
        self.assertIn("atomicExch(attn_flag, 0u)", kernel_source)
        self.assertNotIn("maybe_sync_debug();", source)
        self.assertIn('static void maybe_sync_debug(const char *phase);', source)


if __name__ == "__main__":
    unittest.main()
