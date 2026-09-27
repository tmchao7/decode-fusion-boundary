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
        # The launch geometry comes from the runtime descriptor (no compile-time
        # SM/batch ceiling); this is what TG_MAX_BATCH used to be mistaken for.
        self.assertIn("desc->num_sms", kernel)
        self.assertIn('"tg_decode_step(', bindings)
        self.assertIn('ops.impl("tg_decode_step"', bindings)

    def test_active_backend_has_no_legacy_kernel_source(self):
        self.assertFalse(
            (ROOT / "src/nanovllm/backends/megakernel/csrc/kernel.cu").exists()
        )
        self.assertTrue(
            (ROOT / "src/nanovllm/backends/megakernel/weights.py").exists()
        )

    def test_active_kernel_has_no_unwired_ring_scaffold(self):
        kernel = KERNEL.read_text(encoding="utf-8")
        for symbol in (
            "tg_controller_loop",
            "tg_worker_loop",
            "tg_ring_init",
            "tg_ring_wait",
        ):
            self.assertNotIn(symbol, kernel)
        self.assertIn("inst_shared", kernel)



if __name__ == "__main__":
    unittest.main()
