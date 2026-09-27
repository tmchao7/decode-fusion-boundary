"""Production configuration validation.

``Config.__post_init__`` validates the runtime knobs that used to be
``assert``-only (stripped under ``python -O``) or missing entirely. Two
properties matter and are locked here:

1. every invalid knob fails with a ``ValueError`` that *names the field*, before
   any GPU allocation or model download is attempted; and
2. the optional ``transformers`` import stays lazy, so a bad parameter is
   rejected without the server stack installed.
"""

import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine_fixtures import make_config  # noqa: E402
from nanovllm.config import Config  # noqa: E402


class ConfigRejectionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    @property
    def model_dir(self):
        return self._tmp.name

    def test_rejects_invalid_runtime_config_without_assertions(self):
        for kwargs, field in (
            ({"max_num_batched_tokens": 0}, "max_num_batched_tokens"),
            ({"max_num_seqs": 0}, "max_num_seqs"),
            ({"max_model_len": 0}, "max_model_len"),
            ({"gpu_memory_utilization": 0.0}, "gpu_memory_utilization"),
            ({"gpu_memory_utilization": 1.5}, "gpu_memory_utilization"),
            ({"kvcache_block_size": 128}, "kvcache_block_size"),
            ({"kvcache_block_size": 300}, "kvcache_block_size"),
            ({"tensor_parallel_size": 0}, "tensor_parallel_size"),
            ({"tensor_parallel_size": 9}, "tensor_parallel_size"),
            ({"max_prefill_tokens_per_step": -1}, "max_prefill_tokens_per_step"),
            ({"fusion_groups": 0}, "fusion_groups"),
            ({"prefill_decode_ratio": 0}, "prefill_decode_ratio"),
            ({"megakernel_attn_splits": 0}, "megakernel_attn_splits"),
            ({"decode_backend": "cuda_graph"}, "decode_backend"),
        ):
            with self.subTest(field=field, kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, field):
                    Config(model=self.model_dir, **kwargs)

    def test_rejects_a_missing_model_directory(self):
        with self.assertRaisesRegex(ValueError, "model directory"):
            Config(model="/nonexistent-model-dir")

    def test_prefill_budget_accepts_zero_and_positive_values(self):
        """``0`` means "no cap"; a positive value caps one prefill iteration."""

        self.assertEqual(
            make_config(self.model_dir, max_prefill_tokens_per_step=0)
            .max_prefill_tokens_per_step,
            0,
        )
        self.assertEqual(
            make_config(self.model_dir, max_prefill_tokens_per_step=1024)
            .max_prefill_tokens_per_step,
            1024,
        )


class LazyImportTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_bad_parameters_are_rejected_before_transformers_is_used(self):
        poison = types.ModuleType("transformers")

        def _must_not_be_reached(*args, **kwargs):
            raise AssertionError("AutoConfig must not load for invalid parameters")

        poison.AutoConfig = types.SimpleNamespace(  # type: ignore[attr-defined]
            from_pretrained=_must_not_be_reached
        )
        with patch.dict(sys.modules, {"transformers": poison}):
            with self.assertRaisesRegex(ValueError, "max_num_seqs"):
                Config(model=self._tmp.name, max_num_seqs=0)
            # The valid path does load it -- otherwise this test would pass even
            # if the import were dead code.
            with self.assertRaisesRegex(AssertionError, "invalid parameters"):
                Config(model=self._tmp.name)


class NormalizationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_max_model_len_is_clamped_to_the_model_limit(self):
        config = make_config(self._tmp.name, max_model_len=16384)
        self.assertEqual(config.max_model_len, 4096)

    def test_valid_kvcache_block_size_is_accepted(self):
        config = make_config(self._tmp.name, kvcache_block_size=512)
        self.assertEqual(config.kvcache_block_size, 512)


if __name__ == "__main__":
    unittest.main()
