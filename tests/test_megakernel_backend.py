import unittest
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nanovllm.backends.megakernel import MegakernelBackend, TG_MAX_BATCH


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
            max_num_seqs=TG_MAX_BATCH,
            max_model_len=2048,
            hf_config=hf_config,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_accepts_taskgraph_batches_through_the_compile_time_ceiling(self):
        config = self.make_config()
        seq = SimpleNamespace(temperature=0.0, is_greedy=True)
        for batch in range(1, TG_MAX_BATCH + 1):
            with self.subTest(batch=batch):
                decision = MegakernelBackend.compatibility(
                    config, [seq] * batch, is_prefill=False
                )
                self.assertTrue(decision.supported, decision.reason)

    def test_rejects_prefill_and_sampling(self):
        config = self.make_config()
        seq = SimpleNamespace(temperature=0.6, is_greedy=False)
        self.assertFalse(
            MegakernelBackend.compatibility(config, [seq], is_prefill=True).supported
        )
        decision = MegakernelBackend.compatibility(config, [seq], is_prefill=False)
        self.assertFalse(decision.supported)
        self.assertIn("greedy", decision.reason)

    def test_rejects_batch_above_taskgraph_ceiling(self):
        config = self.make_config(max_num_seqs=64)
        seq = SimpleNamespace(temperature=0.0, is_greedy=True)
        decision = MegakernelBackend.compatibility(
            config, [seq] * (TG_MAX_BATCH + 1), is_prefill=False
        )
        self.assertFalse(decision.supported)
        self.assertIn(str(TG_MAX_BATCH), decision.reason)

    def test_rejects_tensor_parallel(self):
        config = self.make_config(tensor_parallel_size=2)
        seq = SimpleNamespace(temperature=0.0, is_greedy=True)
        decision = MegakernelBackend.compatibility(
            config, [seq], is_prefill=False
        )
        self.assertFalse(decision.supported)
        self.assertIn("tensor_parallel_size=1", decision.reason)

    def test_rejects_non_qwen_shapes(self):
        config = self.make_config(hf_config=SimpleNamespace(model_type="llama"))
        seq = SimpleNamespace(temperature=0.0, is_greedy=True)
        decision = MegakernelBackend.compatibility(config, [seq], is_prefill=False)
        self.assertFalse(decision.supported)
        self.assertIn("unsupported Qwen config", decision.reason)

    def test_decode_batch_delegates_to_taskgraph_rows(self):
        backend = MegakernelBackend(self.make_config(), model=None)
        calls = []

        class FakeDecoder:
            def decode_rows(self, input_ids, positions, block_tables):
                calls.append((input_ids, positions, block_tables))
                return [17, 23]

        backend.decoder = FakeDecoder()
        self.assertEqual(
            backend.decode_batch("input_ids", "positions", "block_tables"),
            [17, 23],
        )
        self.assertEqual(calls, [("input_ids", "positions", "block_tables")])

    def test_decode_uses_taskgraph_single_row_adapter(self):
        backend = MegakernelBackend(self.make_config(), model=None)
        calls = []

        class FakeDecoder:
            def step(self, token_id, position, block_table):
                calls.append((token_id, position, block_table))
                return 17

        backend.decoder = FakeDecoder()

        class SequenceStub:
            last_token = 99

            def __len__(self):
                return 1

        self.assertEqual(backend.decode(SequenceStub(), "device-table"), [17])
        self.assertEqual(calls, [(99, 0, "device-table")])

    def test_backend_has_no_legacy_burst_entrypoint(self):
        backend = MegakernelBackend(self.make_config(), model=None)
        self.assertFalse(hasattr(backend, "generate_paged_batch"))


if __name__ == "__main__":
    unittest.main()
