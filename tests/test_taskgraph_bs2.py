"""BS=2 smoke test for the task-graph megakernel.

Launches the kernel with two batch rows (different tokens) and verifies
both produce correct output tokens matching the PyTorch reference.

Usage:
    CUDA_VISIBLE_DEVICES=0 \
    LD_PRELOAD=.nccl_shim.so \
    HF_HUB_OFFLINE=1 \
    python -m pytest tests/test_taskgraph_bs2.py -v -s
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import sys
import glob
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nanovllm.backends.megakernel.build import get_extension
get_extension()

from nanovllm.backends.megakernel.weights import (
    load_weights, _pack_layer_weights, _rope_tables,
    NUM_LAYERS, HIDDEN_SIZE, INTERMEDIATE_SIZE,
    NUM_KV_HEADS, HEAD_DIM, Q_SIZE, KV_SIZE, MAX_SEQ_LEN,
)
from nanovllm.backends.megakernel import schedule as tg_schedule

NUM_Q_HEADS = Q_SIZE // HEAD_DIM  # 16
VOCAB = 151936
_tg_ops = torch.ops.nanovllm_megakernel_C


def _resolve_model_path() -> str:
    env = os.environ.get("QWEN3_MODEL_PATH")
    if env and os.path.isfile(os.path.join(env, "config.json")):
        return env
    if os.path.isfile("/models/Qwen3-0.6B/config.json"):
        return "/models/Qwen3-0.6B"
    for pattern in (
        "/home/*/edge_llm/*/models/Qwen3-0.6B",
        "/home/*/models/Qwen3-0.6B",
        "/models/Qwen3-0.6B",
        "/root/models/Qwen3-0.6B",
    ):
        for cand in glob.glob(pattern):
            if os.path.isfile(os.path.join(cand, "model.safetensors")) and \
               os.path.isfile(os.path.join(cand, "config.json")):
                return cand
    return "Qwen/Qwen3-0.6B"


class TaskGraphBS2Tests(unittest.TestCase):
    """Launch the task-graph kernel with BS=2 and verify both outputs."""

    @classmethod
    def setUpClass(cls):
        model_path = _resolve_model_path()
        cls.weights, cls.tokenizer = load_weights(model_path, verbose=False)
        cls.ext = get_extension()
        cls.cos_table, cls.sin_table = _rope_tables()

        cls.block_size = 16
        cls.num_cache_blocks = 256
        cls.kv_layer_stride = cls.num_cache_blocks * cls.block_size * KV_SIZE
        cls.k_cache = torch.zeros(
            NUM_LAYERS, cls.num_cache_blocks, cls.block_size,
            NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda",
        )
        cls.v_cache = torch.zeros_like(cls.k_cache)

        cls.layer_weights_packed = _pack_layer_weights(cls.weights["layer_weights"])
        cls.embed_weight = cls.weights["embed_weight"]
        cls.final_norm_weight = cls.weights["final_norm_weight"]

        # Build schedules for BS=1 and BS=2
        cls.geo = tg_schedule.default_geometry(BS=2, attn_splits=1)
        cls.sched = tg_schedule.build_schedule(cls.geo, num_sms=1)
        cls.geo1 = tg_schedule.default_geometry(BS=1, attn_splits=1)
        cls.sched1 = tg_schedule.build_schedule(cls.geo1, num_sms=1)

    def _alloc_scratch(self, BS=2, sched=None):
        sched = sched or self.sched
        f32 = dict(dtype=torch.float32, device="cuda")
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        i32 = dict(dtype=torch.int32, device="cuda")
        u32 = dict(dtype=torch.uint32, device="cuda")

        WMMA_PAD = 16
        s = {}
        s["inst_buf"] = torch.tensor(sched.flat_inst_buf, dtype=torch.uint32).cuda().view(torch.int32)
        s["num_inst_per_sm"] = torch.tensor(sched.num_inst_per_sm, dtype=torch.int32).cuda()

        s["norm_out"] = torch.empty(BS, HIDDEN_SIZE, **f32)
        s["q_out"] = torch.empty(WMMA_PAD, Q_SIZE, **f32)
        s["k_out"] = torch.empty(WMMA_PAD, KV_SIZE, **f32)
        s["v_out"] = torch.empty(WMMA_PAD, KV_SIZE, **f32)
        s["attn_out"] = torch.empty(BS, Q_SIZE, **f32)
        s["o_proj_out"] = torch.empty(WMMA_PAD, HIDDEN_SIZE, **f32)
        s["x_attn"] = torch.empty(BS, HIDDEN_SIZE, **f32)
        s["post_attn_norm"] = torch.empty(BS, HIDDEN_SIZE, **f32)
        s["silu_out"] = torch.empty(WMMA_PAD, INTERMEDIATE_SIZE, **f32)
        s["ffn_down_out"] = torch.empty(WMMA_PAD, HIDDEN_SIZE, **f32)
        s["up_out"] = torch.empty(WMMA_PAD, INTERMEDIATE_SIZE, **f32)
        s["x_ffn"] = torch.empty(BS, HIDDEN_SIZE, **f32)
        s["x_resid"] = torch.empty(BS, HIDDEN_SIZE, **bf16)
        s["lm_logits"] = torch.empty(WMMA_PAD, VOCAB, **f32)
        s["o_partial"] = torch.empty(BS, NUM_KV_HEADS, 1, HIDDEN_SIZE, **f32)
        s["lse_partial"] = torch.empty(BS, NUM_KV_HEADS, 1, **f32)

        d = sched.bar_dims
        s["bar_qkv"] = torch.zeros(d.bar_qkv, **u32)
        s["bar_attn"] = torch.zeros(d.bar_attn, **u32)
        s["bar_oproj"] = torch.zeros(d.bar_oproj, **u32)
        s["bar_silu"] = torch.zeros(d.bar_silu, **u32)
        s["bar_ffn_down"] = torch.zeros(d.bar_ffn_down, **u32)
        s["bar_layer"] = torch.zeros(d.bar_layer, **u32)
        s["bar_combine"] = torch.zeros(max(d.bar_combine, 1), **u32)
        s["bar_lm_head"] = torch.zeros(d.bar_lm_head, **u32)

        s["page_table"] = torch.zeros(BS, 64, dtype=torch.int32, device="cuda")
        s["cache_seqlens"] = torch.zeros(BS, dtype=torch.int32, device="cuda")
        s["row_active"] = torch.ones(BS, dtype=torch.int32, device="cuda")

        _DO_NOT_ZERO = {"inst_buf", "num_inst_per_sm", "page_table", "cache_seqlens", "row_active"}
        for k, v in s.items():
            if k in _DO_NOT_ZERO:
                continue
            if v.dtype in (torch.uint32, torch.float32, torch.bfloat16, torch.int32):
                v.zero_()

        return s

    def _run(self, token_ids, row_active=None):
        """Run one decode step for ``token_ids`` and return the sampled tokens."""
        BS = len(token_ids)
        sched = self.sched1 if BS == 1 else self.sched
        s = self._alloc_scratch(BS, sched=sched)
        for b, tid in enumerate(token_ids):
            s["x_resid"][b] = self.embed_weight[tid]
            s["cache_seqlens"][b] = 0  # first token, no previous KV
            s["page_table"][b, 0] = b  # each row uses a different physical page
            s["row_active"][b] = 1 if row_active is None else int(row_active[b])

        _tg_ops.tg_decode_step(
            s["inst_buf"], s["num_inst_per_sm"],
            sched.max_inst, 1,
            self.layer_weights_packed, self.embed_weight,
            self.final_norm_weight, self.cos_table, self.sin_table,
            self.k_cache, self.v_cache, s["page_table"],
            s["cache_seqlens"], s["row_active"],
            self.num_cache_blocks, 64, self.block_size,
            self.kv_layer_stride,
            s["norm_out"], s["q_out"], s["k_out"], s["v_out"],
            s["attn_out"], s["o_proj_out"], s["x_attn"],
            s["post_attn_norm"], s["silu_out"], s["ffn_down_out"],
            s["up_out"],
            s["x_ffn"], s["x_resid"], s["lm_logits"],
            s["o_partial"], s["lse_partial"],
            s["bar_qkv"], s["bar_attn"], s["bar_oproj"],
            s["bar_silu"], s["bar_ffn_down"], s["bar_layer"],
            s["bar_combine"], s["bar_lm_head"],
            NUM_LAYERS, BS, HIDDEN_SIZE, INTERMEDIATE_SIZE,
            Q_SIZE, KV_SIZE, NUM_Q_HEADS, NUM_KV_HEADS,
            HEAD_DIM, VOCAB, 1, 1, 1,
        )
        torch.cuda.synchronize()
        # The argmax token is stored as int32 at lm_logits[batch, 0].
        return [s["lm_logits"][b, 0].view(torch.int32).item() for b in range(BS)]

    def test_bs2_two_tokens(self):
        """Run BS=2 with two different tokens and verify both outputs exactly."""
        token_ids = [42, 1234]
        out_tokens = self._run(token_ids)

        print("\nBS=2 results:")
        for b, (inp, out) in enumerate(zip(token_ids, out_tokens)):
            print(f"  Batch {b}: input={inp} -> output={out}")

        for b, tok in enumerate(out_tokens):
            self.assertIsInstance(tok, int, f"Batch {b}: token is not int")
            self.assertGreaterEqual(tok, 0, f"Batch {b}: token negative")
            self.assertLess(tok, VOCAB, f"Batch {b}: token out of vocab range")

        # Exact tokens verified against the PyTorch reference (test_tg_logits_compare).
        # Override with TG_BS2_EXPECTED="a,b" if the model/checkpoint changes.
        expected = [
            int(x) for x in os.environ.get("TG_BS2_EXPECTED", "14582,38297").split(",")
        ]
        self.assertEqual(
            out_tokens, expected,
            f"BS=2 tokens {out_tokens} != expected {expected}; if the checkpoint "
            f"changed, recompute via test_tg_logits_compare and set TG_BS2_EXPECTED",
        )
        self.assertNotEqual(
            out_tokens[0], out_tokens[1],
            "Both batch rows produced the same token — rows are not independent",
        )

    def test_inactive_row_leaves_its_kv_pages_untouched(self):
        """A row with row_active == 0 must not write its KV pages.

        A continuous-batching runtime frees the pages of finished sequences and
        hands them to a new sequence, so a stale row writing "its" old pages
        would corrupt the new owner. The active row must be unaffected.
        """
        # Row 0 alone, active: the reference output.
        ref = self._run([42])[0]

        # BS=2, row 1 inactive, its physical page (1) filled with a sentinel.
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.k_cache[:, 1] = 7.0
        self.v_cache[:, 1] = 7.0
        out = self._run([42, 1234], row_active=[1, 0])

        self.assertEqual(out[0], ref,
                         "the active row changed when its neighbour went inactive")
        self.assertTrue(bool((self.k_cache[:, 1] == 7.0).all()),
                        "inactive row wrote its K page")
        self.assertTrue(bool((self.v_cache[:, 1] == 7.0).all()),
                        "inactive row wrote its V page")

    def test_bs2_matches_independent_bs1_runs(self):
        """Each BS=2 row must equal the same token decoded alone at BS=1.

        This is self-contained (no hardcoded reference): it directly detects the
        class of bugs where one row's activations leak into another's.
        """
        token_ids = [42, 1234]
        bs1 = [self._run([tid])[0] for tid in token_ids]
        bs2 = self._run(token_ids)
        print(f"\nBS=1 tokens: {bs1}  |  BS=2 tokens: {bs2}")
        self.assertEqual(
            bs2, bs1,
            "BS=2 rows disagree with independent BS=1 runs — batch rows leak",
        )

    def test_bs2_logits_nonzero(self):
        """Verify both batch rows have non-zero logits."""
        BS = 2
        s = self._alloc_scratch(BS)

        token_ids = [42, 100]
        for b, tid in enumerate(token_ids):
            s["x_resid"][b] = self.embed_weight[tid]
            s["cache_seqlens"][b] = 0
            s["page_table"][b, 0] = b
            s["row_active"][b] = 1

        _tg_ops.tg_decode_step(
            s["inst_buf"], s["num_inst_per_sm"],
            self.sched.max_inst, 1,
            self.layer_weights_packed, self.embed_weight,
            self.final_norm_weight, self.cos_table, self.sin_table,
            self.k_cache, self.v_cache, s["page_table"],
            s["cache_seqlens"], s["row_active"],
            self.num_cache_blocks, 64, self.block_size,
            self.kv_layer_stride,
            s["norm_out"], s["q_out"], s["k_out"], s["v_out"],
            s["attn_out"], s["o_proj_out"], s["x_attn"],
            s["post_attn_norm"], s["silu_out"], s["ffn_down_out"],
            s["up_out"],
            s["x_ffn"], s["x_resid"], s["lm_logits"],
            s["o_partial"], s["lse_partial"],
            s["bar_qkv"], s["bar_attn"], s["bar_oproj"],
            s["bar_silu"], s["bar_ffn_down"], s["bar_layer"],
            s["bar_combine"], s["bar_lm_head"],
            NUM_LAYERS, BS, HIDDEN_SIZE, INTERMEDIATE_SIZE,
            Q_SIZE, KV_SIZE, NUM_Q_HEADS, NUM_KV_HEADS,
            HEAD_DIM, VOCAB, 1, 1, 1,
        )

        torch.cuda.synchronize()

        for b in range(BS):
            logits = s["lm_logits"][b]
            # Skip index 0 (overwritten by argmax token id)
            max_logit = logits[1:].max().item()
            self.assertNotEqual(max_logit, 0.0,
                                f"Batch {b}: logits are all zero")
            print(f"\nBatch {b}: max logit (excl. [0]) = {max_logit:.4f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
