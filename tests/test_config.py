import unittest

from nanovllm.config import VALID_DECODE_BACKENDS, validate_decode_backend
from nanovllm.sampling_params import SamplingParams


class ConfigTests(unittest.TestCase):
    def test_supported_decode_backends_are_explicit(self):
        self.assertEqual(VALID_DECODE_BACKENDS, {"torch", "megakernel"})
        self.assertEqual(validate_decode_backend("torch"), "torch")
        self.assertEqual(validate_decode_backend("megakernel"), "megakernel")

    def test_unknown_decode_backend_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "decode_backend"):
            validate_decode_backend("cuda_graph")


class SamplingParamsTests(unittest.TestCase):
    def test_zero_temperature_selects_greedy_mode(self):
        params = SamplingParams(temperature=0.0)
        self.assertTrue(params.is_greedy)

    def test_positive_temperature_remains_sampling_mode(self):
        params = SamplingParams(temperature=0.6)
        self.assertFalse(params.is_greedy)

    def test_negative_temperature_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "temperature"):
            SamplingParams(temperature=-0.1)

    def test_non_positive_max_tokens_is_rejected(self):
        """max_tokens <= 0 would decode forever instead of failing fast."""

        for max_tokens in (0, -1):
            with self.subTest(max_tokens=max_tokens):
                with self.assertRaisesRegex(ValueError, "max_tokens"):
                    SamplingParams(max_tokens=max_tokens)


if __name__ == "__main__":
    unittest.main()
