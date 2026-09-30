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

    def test_dynamic_smem_is_sized_from_the_batch_not_the_wmma_tile(self):
        """The s_norm request must follow ``desc->BS``, not a fixed 64 KB.

        Sizing it at ``TG_WMMA_TILE * TG_HIDDEN * 4`` made static (37072 B
        measured) + dynamic exceed the 102400 B per-SM shared memory budget on
        sm_120, so ``cudaFuncSetAttribute`` failed with "invalid argument" and
        the first decode launch died with a message that looked like a driver
        or architecture problem. Guard both halves: the size must be derived
        from the descriptor, and the stale constant must not come back.
        """

        kernel = KERNEL.read_text(encoding="utf-8")
        self.assertIn("desc->BS * (size_t)desc->hidden * sizeof(float)", kernel)
        self.assertNotIn("TG_WMMA_TILE * TG_HIDDEN * sizeof(float)", kernel)
        # A one-shot "already set" flag would leave a small limit in place when a
        # later step uses a larger batch.
        self.assertIn("if (dyn_smem_bytes > smem_attr_bytes)", kernel)
        self.assertNotIn("smem_attr_set", kernel)



if __name__ == "__main__":
    unittest.main()
