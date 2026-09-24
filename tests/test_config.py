import unittest

from nanovllm.config import VALID_DECODE_BACKENDS, validate_decode_backend
from nanovllm.sampling_params import SamplingParams


class ConfigTests(unittest.TestCase):
    def test_supported_decode_backends_are_explicit(self):
        # ``megakernel-tg`` is the task-graph rewrite (docs/11-13); it shares the
        # greedy/decode-only contract with ``megakernel`` but has its own
        # whole-batch kernel and its own admission policy.
        self.assertEqual(
            VALID_DECODE_BACKENDS, {"torch", "megakernel", "megakernel-tg"}
        )
        self.assertEqual(validate_decode_backend("torch"), "torch")
        self.assertEqual(validate_decode_backend("megakernel"), "megakernel")
        self.assertEqual(validate_decode_backend("megakernel-tg"), "megakernel-tg")

    def test_unknown_decode_backend_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "decode_backend"):
            validate_decode_backend("cuda_graph")

    def test_taskgraph_attn_splits_must_be_positive(self):
        """``megakernel_tg_attn_splits`` is baked into the schedule; 0 would
        emit an attention wave with no instructions and deadlock on bar_attn."""

        from nanovllm.config import Config

        # __post_init__ runs before the model directory / transformers checks,
        # so a fake model path is enough to reach the validation.
        with self.assertRaisesRegex(ValueError, "megakernel_tg_attn_splits"):
            Config(model="/nonexistent", megakernel_tg_attn_splits=0)


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
