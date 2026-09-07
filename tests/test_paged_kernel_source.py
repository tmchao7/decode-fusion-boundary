import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "src/nanovllm/backends/megakernel/csrc/kernel.cu"
BINDINGS = ROOT / "src/nanovllm/backends/megakernel/csrc/torch_bindings.cpp"


class PagedKernelSourceTests(unittest.TestCase):
    def test_kernel_contains_paged_address_mapping(self):
        source = KERNEL.read_text(encoding="utf-8")
        self.assertIn("ldg_kv_ptr", source)
        self.assertIn("position / block_size", source)
        self.assertIn("physical_block * block_size * KV_SIZE", source)
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
        self.assertIn("cudaPeekAtLastError", kernel_source)
        self.assertIn("cudaLaunchCooperativeKernel", kernel_source)
        self.assertIn("cudaDevAttrCooperativeLaunch", kernel_source)
        self.assertNotIn("maybe_sync_debug();", source)
        self.assertIn('static void maybe_sync_debug(const char *phase);', source)


if __name__ == "__main__":
    unittest.main()
