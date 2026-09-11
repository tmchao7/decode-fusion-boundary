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

    def test_rejects_batch_by_default_falls_back_to_cudagraph(self):
        # Hybrid architecture: batch>1 defaults to cudagraph unless
        # megakernel_batch_decode explicitly forces the megakernel batch path.
        config = self.make_config(max_num_seqs=8)
        seq = SimpleNamespace(temperature=0.0)
        decision = MegakernelBackend.compatibility(
            config, [seq, seq], is_prefill=False
        )
        self.assertFalse(decision.supported)
        self.assertIn("cudagraph", decision.reason)

    def test_accepts_small_greedy_batch_when_explicitly_enabled(self):
        config = self.make_config(max_num_seqs=8, megakernel_batch_decode=True)
        seq = SimpleNamespace(temperature=0.0)
        decision = MegakernelBackend.compatibility(
            config, [seq, seq], is_prefill=False
        )
        self.assertTrue(decision.supported, decision.reason)

    def test_rejects_batch_with_mixed_sampling_modes(self):
        config = self.make_config(max_num_seqs=8)
        seqs = [
            SimpleNamespace(temperature=0.0),
            SimpleNamespace(temperature=0.6),
        ]
        decision = MegakernelBackend.compatibility(config, seqs, is_prefill=False)
        self.assertFalse(decision.supported)
        self.assertIn("greedy", decision.reason)

    def test_rejects_batch_above_initial_limit(self):
        config = self.make_config(max_num_seqs=64)
        seqs = [SimpleNamespace(temperature=0.0) for _ in range(33)]
        decision = MegakernelBackend.compatibility(config, seqs, is_prefill=False)
        self.assertFalse(decision.supported)
        self.assertIn("32", decision.reason)

    def test_rejects_batch_and_tensor_parallel(self):
        config = self.make_config(tensor_parallel_size=2)
        seq = SimpleNamespace(temperature=0.0)
        decision = MegakernelBackend.compatibility(
            config, [seq, seq], is_prefill=False
        )
        self.assertFalse(decision.supported)

    def test_accepts_single_sequence_with_default_capacity(self):
        config = self.make_config(max_num_seqs=512)
        seq = SimpleNamespace(temperature=0.0)
        decision = MegakernelBackend.compatibility(config, [seq], is_prefill=False)
        self.assertTrue(decision.supported, decision.reason)

    def test_rejects_non_qwen_shapes(self):
        config = self.make_config(hf_config=SimpleNamespace(model_type="llama"))
        seq = SimpleNamespace(temperature=0.0)
        decision = MegakernelBackend.compatibility(config, [seq], is_prefill=False)
        self.assertFalse(decision.supported)

    def test_decode_batch_delegates_device_metadata(self):
        backend = MegakernelBackend(self.make_config(), model=None)
        calls = []

        class FakeDecoder:
            def step_batch(self, input_ids, positions, block_tables):
                calls.append((input_ids, positions, block_tables))
                return [17, 23]

        backend.decoder = FakeDecoder()
        self.assertEqual(
            backend.decode_batch("input_ids", "positions", "block_tables"),
            [17, 23],
        )
        self.assertEqual(calls, [("input_ids", "positions", "block_tables")])

    def test_decode_uses_attached_cache_and_does_not_replay_prompt(self):
        backend = MegakernelBackend(self.make_config(), model=None)
        backend.set_kv_cache(SimpleNamespace(dim=lambda: 6, shape=(2, 28, 4, 256, 8, 128)))

        calls = []

        class FakeDecoder:
            def step(self, token_id, position, block_table):
                calls.append((token_id, position, block_table))
                return 17

        backend.decoder = FakeDecoder()
        class SequenceStub:
            seq_id = 1
            last_token = 99
            prompt_token_ids = [1, 2, 3]

            def __len__(self):
                return 1

        self.assertEqual(backend.decode(SequenceStub(), block_table="device-table"), [17])
        self.assertEqual(calls, [(99, 0, "device-table")])

    def test_generate_paged_batch_dispatches_batch1_to_persistent(self):
        backend = MegakernelBackend(self.make_config(), model=None)
        calls = []

        class FakeDecoder:
            def generate_paged(self, first_token_id, start_position,
                               block_table, num_tokens, eos_token):
                calls.append((first_token_id, start_position, block_table,
                              num_tokens, eos_token))
                return [10, 11, 12]

        backend.decoder = FakeDecoder()

        class SequenceStub:
            def __init__(self, last_token, length, block_table):
                self.last_token = last_token
                self._len = length
                self.block_table = block_table

            def __len__(self):
                return self._len

        seq = SequenceStub(last_token=42, length=5, block_table=[10, 20])
        result = backend.generate_paged_batch(
            [seq], [[10, 20, 30]], max_steps=3, eos_token=999
        )
        self.assertEqual(result, [[10, 11, 12]])
        self.assertEqual(len(calls), 1)
        saved_tok, saved_pos, saved_bt, saved_steps, saved_eos = calls[0]
        self.assertEqual(saved_tok, 42)
        self.assertEqual(saved_pos, 4)
        self.assertEqual(saved_bt, [10, 20])
        self.assertEqual(saved_steps, 3)
        self.assertEqual(saved_eos, 999)

    def test_generate_paged_batch_builds_input_metadata_from_seqs(self):
        backend = MegakernelBackend(self.make_config(), model=None)
        calls = []

        class FakeDecoder:
            def generate_paged_batch(self, input_ids, positions,
                                     block_tables, max_steps, eos_token):
                calls.append((input_ids, positions, block_tables,
                              max_steps, eos_token))
                return [[10, 11, 12], [20, 21, 22]]

        backend.decoder = FakeDecoder()

        class SequenceStub:
            def __init__(self, last_token, length):
                self.last_token = last_token
                self._len = length

            def __len__(self):
                return self._len

        seq_a = SequenceStub(last_token=42, length=5)
        seq_b = SequenceStub(last_token=7, length=3)
        result = backend.generate_paged_batch(
            [seq_a, seq_b], "bt", max_steps=3, eos_token=999
        )
        self.assertEqual(result, [[10, 11, 12], [20, 21, 22]])
        self.assertEqual(len(calls), 1)
        saved_ids, saved_pos, saved_bt, saved_steps, saved_eos = calls[0]
        self.assertEqual(saved_bt, "bt")
        self.assertEqual(saved_steps, 3)
        self.assertEqual(saved_eos, 999)


if __name__ == "__main__":
    unittest.main()
