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

    def test_paged_operator_is_registered(self):
        source = BINDINGS.read_text(encoding="utf-8")
        self.assertIn('ops.def("decode_paged', source)
        self.assertIn('ops.impl("decode_paged"', source)
        self.assertIn("block_table.data_ptr<int>()", source)


if __name__ == "__main__":
    unittest.main()
