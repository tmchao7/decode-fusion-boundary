"""Multi-SM correctness for the task-graph megakernel.

This is the test that exercises the actual point of the task graph: instructions
spread round-robin across ``num_sms`` CTAs with cross-SM ``tg_wait`` on the bar
counters. The other functional tests all use ``num_sms=1``, where a single CTA
runs the whole stream sequentially and the cross-SM waits are trivially
satisfied.

The output token must equal the BS=1 single-SM result (token 14582 for input 42,
verified against the PyTorch reference in test_tg_logits_compare.py).

Usage:
    CUDA_VISIBLE_DEVICES=0 \
    LD_PRELOAD=.nccl_shim.so \
    HF_HUB_OFFLINE=1 \
    python -m pytest tests/test_taskgraph_multi_sm.py -v -s

    TG_MULTI_SM=8 overrides the grid size (default 8; use 108 for the full A100).
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
    NUM_KV_HEADS, HEAD_DIM, Q_SIZE, KV_SIZE,
)
from nanovllm.backends.megakernel import schedule as tg_schedule

NUM_Q_HEADS = Q_SIZE // HEAD_DIM  # 16
VOCAB = 151936
TG_MULTI_SM = int(os.environ.get("TG_MULTI_SM", "8"))
TG_REFERENCE_TOKEN = int(os.environ.get("TG_REFERENCE_TOKEN", "14582"))
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


class TaskGraphMultiSMTests(unittest.TestCase):
    """num_sms > 1: instructions really are spread across CTAs and must sync."""

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

        cls.geo = tg_schedule.default_geometry(BS=1, attn_splits=1)
        cls.sched_multi = tg_schedule.build_schedule(cls.geo, num_sms=TG_MULTI_SM)
        cls.sched_single = tg_schedule.build_schedule(cls.geo, num_sms=1)

    def _alloc_scratch(self, sched, BS=1):
        f32 = dict(dtype=torch.float32, device="cuda")
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        u32 = dict(dtype=torch.uint32, device="cuda")

        WMMA_PAD = 16
        s = {}
        s["inst_buf"] = torch.tensor(sched.flat_inst_buf, dtype=torch.int32).cuda()
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

        _DO_NOT_ZERO = {"inst_buf", "num_inst_per_sm", "page_table",
                        "cache_seqlens", "row_active"}
        for k, v in s.items():
            if k in _DO_NOT_ZERO:
                continue
            if v.dtype in (torch.uint32, torch.float32, torch.bfloat16, torch.int32):
                v.zero_()

        return s

    def _run(self, sched, num_sms, token_id):
        BS = 1
        s = self._alloc_scratch(sched, BS)
        s["x_resid"][0] = self.embed_weight[token_id]
        s["cache_seqlens"][0] = 0
        s["page_table"][0, 0] = 0
        s["row_active"][0] = 1

        _tg_ops.tg_decode_step(
            s["inst_buf"], s["num_inst_per_sm"],
            sched.max_inst, num_sms,
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
        return s["lm_logits"][0, 0].view(torch.int32).item()

    def test_multi_sm_completes_and_matches_reference(self):
        """num_sms > 1 must not deadlock and must produce the reference token."""
        tok = self._run(self.sched_multi, TG_MULTI_SM, token_id=42)
        print(f"\nnum_sms={TG_MULTI_SM} token: {tok} (reference {TG_REFERENCE_TOKEN})")
        self.assertEqual(tok, TG_REFERENCE_TOKEN,
                         f"multi-SM token {tok} != reference {TG_REFERENCE_TOKEN}")

    def test_multi_sm_matches_single_sm(self):
        """The multi-SM grid must agree with the sequential single-SM run."""
        single = self._run(self.sched_single, 1, token_id=42)
        multi = self._run(self.sched_multi, TG_MULTI_SM, token_id=42)
        print(f"\nsingle-SM token: {single}  |  num_sms={TG_MULTI_SM} token: {multi}")
        self.assertEqual(multi, single,
                         "multi-SM result disagrees with the single-SM result — "
                         "a cross-SM bar dependency is missing or mis-targeted")


if __name__ == "__main__":
    unittest.main(verbosity=2)
