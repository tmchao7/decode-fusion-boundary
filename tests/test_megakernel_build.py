import unittest

from nanovllm.backends.megakernel import cuda_arch_for_capability


class MegakernelBuildTests(unittest.TestCase):
    def test_a100_uses_sm80(self):
        self.assertEqual(cuda_arch_for_capability((8, 0)), "sm_80")

    def test_v100_uses_sm70(self):
        self.assertEqual(cuda_arch_for_capability((7, 0)), "sm_70")

    def test_unknown_architecture_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported CUDA capability"):
            cuda_arch_for_capability((9, 1))


if __name__ == "__main__":
    unittest.main()
