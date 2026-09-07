import os
import unittest
from unittest.mock import patch

from nanovllm.backends.megakernel import cuda_arch_for_capability

try:
    from nanovllm.backends.megakernel.build import cuda_flags
except ModuleNotFoundError:
    cuda_flags = None


class MegakernelBuildTests(unittest.TestCase):
    def test_a100_uses_sm80(self):
        self.assertEqual(cuda_arch_for_capability((8, 0)), "sm_80")

    def test_v100_uses_sm70(self):
        self.assertEqual(cuda_arch_for_capability((7, 0)), "sm_70")

    def test_unknown_architecture_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported CUDA capability"):
            cuda_arch_for_capability((9, 1))

    @unittest.skipUnless(cuda_flags is not None, "torch is required for build flag tests")
    def test_sanitizer_build_emits_line_info(self):
        with patch.dict(os.environ, {"NANOVLLM_CUDA_ARCH": "sm_80", "NANOVLLM_CUDA_SANITIZE": "1"}):
            self.assertIn("-lineinfo", cuda_flags())

    @unittest.skipUnless(cuda_flags is not None, "torch is required for build flag tests")
    def test_normal_build_does_not_add_sanitizer_flags(self):
        with patch.dict(os.environ, {"NANOVLLM_CUDA_ARCH": "sm_80"}, clear=False):
            os.environ.pop("NANOVLLM_CUDA_SANITIZE", None)
            self.assertNotIn("-lineinfo", cuda_flags())


if __name__ == "__main__":
    unittest.main()
