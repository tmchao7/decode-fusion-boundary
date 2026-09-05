import unittest
from types import SimpleNamespace

from nanovllm.backends.megakernel import MegakernelBackend


class MegakernelBackendTests(unittest.TestCase):
    def make_config(self, **overrides):
        hf_config = SimpleNamespace(
            model_type="qwen3",
            hidden_size=1024,
            intermediate_size=3072,
            num_hidden_layers=28,
            num_attention_heads=16,
            num_key_value_heads=8,
            vocab_size=151936,
            head_dim=128,
        )
        values = dict(
            decode_backend="megakernel",
            tensor_parallel_size=1,
            max_num_seqs=1,
            hf_config=hf_config,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_accepts_single_sequence_greedy_qwen_config(self):
        config = self.make_config()
        seq = SimpleNamespace(temperature=0.0)
        decision = MegakernelBackend.compatibility(config, [seq], is_prefill=False)
        self.assertTrue(decision.supported, decision.reason)

    def test_rejects_prefill_and_sampling(self):
        config = self.make_config()
        seq = SimpleNamespace(temperature=0.6)
        self.assertFalse(
            MegakernelBackend.compatibility(config, [seq], is_prefill=True).supported
        )
        self.assertFalse(
            MegakernelBackend.compatibility(config, [seq], is_prefill=False).supported
        )

    def test_rejects_batch_and_tensor_parallel(self):
        config = self.make_config(tensor_parallel_size=2)
        seq = SimpleNamespace(temperature=0.0)
        decision = MegakernelBackend.compatibility(
            config, [seq, seq], is_prefill=False
        )
        self.assertFalse(decision.supported)

    def test_rejects_default_multi_sequence_capacity_until_paged_sync(self):
        config = self.make_config(max_num_seqs=512)
        seq = SimpleNamespace(temperature=0.0)
        decision = MegakernelBackend.compatibility(config, [seq], is_prefill=False)
        self.assertFalse(decision.supported)
        self.assertIn("max_num_seqs", decision.reason)

    def test_rejects_non_qwen_shapes(self):
        config = self.make_config(hf_config=SimpleNamespace(model_type="llama"))
        seq = SimpleNamespace(temperature=0.0)
        decision = MegakernelBackend.compatibility(config, [seq], is_prefill=False)
        self.assertFalse(decision.supported)


if __name__ == "__main__":
    unittest.main()
