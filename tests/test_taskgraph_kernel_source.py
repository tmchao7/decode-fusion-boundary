import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "src/nanovllm/backends/megakernel/csrc/taskgraph_kernel.cu"
BINDINGS = ROOT / "src/nanovllm/backends/megakernel/csrc/torch_bindings.cpp"


class TaskGraphKernelSourceTests(unittest.TestCase):
    def test_taskgraph_kernel_and_binding_are_present(self):
        kernel = KERNEL.read_text(encoding="utf-8")
        bindings = BINDINGS.read_text(encoding="utf-8")
        self.assertIn("tg_decode_kernel", kernel)
        self.assertIn("cudaLaunchCooperativeKernel", kernel)
        self.assertIn("TG_MAX_BATCH", kernel)
        self.assertIn('ops.def("tg_decode_step"', bindings)
        self.assertIn('ops.impl("tg_decode_step"', bindings)

    def test_active_backend_has_no_legacy_kernel_source(self):
        self.assertFalse(
            (ROOT / "src/nanovllm/backends/megakernel/csrc/kernel.cu").exists()
        )
        self.assertFalse(
            (ROOT / "src/nanovllm/backends/megakernel/model.py").exists()
        )


if __name__ == "__main__":
    unittest.main()
