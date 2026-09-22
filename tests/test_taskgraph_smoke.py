"""Smoke test for the task-graph megakernel.

Verifies that the kernel launches, executes the full instruction stream, and
produces an output token without crashing or hanging. Uses the actual
Qwen3-0.6B model weights for a meaningful forward pass.

Phase 1 goal: the task-graph kernel produces the same next-token logits as
the frozen batch=1 kernel.cu path (or close to it, allowing for minor
numerical differences from the different compute ordering).
"""

import sys
import os
import math
import unittest
import glob

# Force HF offline mode BEFORE importing transformers/huggingface_hub — the
# Qwen3-0.6B weights are already cached locally and the proxy returns 504
# on HEAD requests.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _resolve_model_path() -> str:
    """Find a local Qwen3-0.6B model directory, avoiding HF Hub network access.

    Priority:
      1. ``QWEN3_MODEL_PATH`` env var if set.
      2. ``/models/Qwen3-0.6B`` (container mount used by other tests).
      3. Any ``.../Qwen3-0.6B`` directory under common home dirs that
         contains ``model.safetensors`` and ``config.json``.
      4. Fall back to ``Qwen/Qwen3-0.6B`` (online HF Hub identifier).
    """
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


# Load the extension FIRST — model.py accesses torch.ops at import time
from nanovllm.backends.megakernel.build import get_extension
get_extension()

from nanovllm.backends.megakernel.model import (
    load_weights, _pack_layer_weights, _rope_tables,
    NUM_LAYERS, HIDDEN_SIZE, INTERMEDIATE_SIZE,
    NUM_KV_HEADS, HEAD_DIM, Q_SIZE, KV_SIZE,
    MAX_SEQ_LEN,
)
NUM_Q_HEADS = Q_SIZE // HEAD_DIM  # 16
from nanovllm.backends.megakernel import schedule as tg_schedule

# Torch ops are registered via TORCH_LIBRARY, so they live under
# ``torch.ops.<ext_name>.*`` rather than as attributes of the loaded module.
_tg_ops = torch.ops.nanovllm_megakernel_C


class TaskGraphSmokeTests(unittest.TestCase):
    """Launch the task-graph kernel with real model weights and verify it
    completes without hanging or crashing."""

    @classmethod
    def setUpClass(cls):
        model_path = _resolve_model_path()
        cls.weights, cls.tokenizer = load_weights(model_path, verbose=False)
        cls.ext = get_extension()
        cls.cos_table, cls.sin_table = _rope_tables()

        # KV cache (paged layout: [num_layers, num_blocks, block_size, kv_heads, head_dim])
        cls.block_size = 16
        cls.num_cache_blocks = 256  # 256 * 16 = 4096 max tokens
        cls.kv_layer_stride = cls.num_cache_blocks * cls.block_size * KV_SIZE  # in elements
        cls.k_cache = torch.zeros(
            NUM_LAYERS, cls.num_cache_blocks, cls.block_size,
            NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda",
        )
        cls.v_cache = torch.zeros_like(cls.k_cache)

        # Pack layer weights
        cls.layer_weights_packed = _pack_layer_weights(cls.weights["layer_weights"])
        cls.embed_weight = cls.weights["embed_weight"]
        cls.final_norm_weight = cls.weights["final_norm_weight"]

        # Build schedule for BS=1
        cls.geo = tg_schedule.default_geometry(BS=1, attn_splits=1)
        cls.sched = tg_schedule.build_schedule(cls.geo, num_sms=1)

    def _alloc_scratch(self, BS=1):
        """Allocate all scratch buffers with correct sizes."""
        f32 = dict(dtype=torch.float32, device="cuda")
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        i32 = dict(dtype=torch.int32, device="cuda")
        u32 = dict(dtype=torch.uint32, device="cuda")

        s = {}
        # Instruction stream
        s["inst_buf"] = torch.tensor(self.sched.flat_inst_buf, dtype=torch.int32).cuda()
        s["num_inst_per_sm"] = torch.tensor(self.sched.num_inst_per_sm, dtype=torch.int32).cuda()

        # Scratch (float intermediates matching kernel.cu)
        # GEMM output buffers are padded to 16 rows (TG_WMMA_TILE) because
        # wmma::store_matrix_sync always writes a full 16x16 tile even when
        # M=BS<16.  Non-GEMM buffers keep BS rows.
        WMMA_PAD = 16
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
        s["lm_logits"] = torch.empty(WMMA_PAD, 151936, **f32)
        s["o_partial"] = torch.empty(BS, NUM_KV_HEADS, 1, HIDDEN_SIZE, **f32)
        s["lse_partial"] = torch.empty(BS, NUM_KV_HEADS, 1, **f32)

        # Bars
        d = self.sched.bar_dims
        s["bar_qkv"] = torch.zeros(d.bar_qkv, **u32)
        s["bar_attn"] = torch.zeros(d.bar_attn, **u32)
        s["bar_oproj"] = torch.zeros(d.bar_oproj, **u32)
        s["bar_silu"] = torch.zeros(d.bar_silu, **u32)
        s["bar_ffn_down"] = torch.zeros(d.bar_ffn_down, **u32)
        s["bar_layer"] = torch.zeros(d.bar_layer, **u32)
        s["bar_combine"] = torch.zeros(max(d.bar_combine, 1), **u32)
        s["bar_lm_head"] = torch.zeros(d.bar_lm_head, **u32)  # min size 1

        # Paging
        s["page_table"] = torch.zeros(BS, 64, dtype=torch.int32, device="cuda")
        s["cache_seqlens"] = torch.zeros(BS, dtype=torch.int32, device="cuda")
        s["row_active"] = torch.ones(BS, dtype=torch.int32, device="cuda")

        # Zero scratch + bars (but NOT inst_buf / num_inst_per_sm / page_table /
        # cache_seqlens / row_active — those are inputs, not scratch).
        _DO_NOT_ZERO = {
            "inst_buf", "num_inst_per_sm",
            "page_table", "cache_seqlens", "row_active",
        }
        for k, v in s.items():
            if k in _DO_NOT_ZERO:
                continue
            if v.dtype in (torch.uint32, torch.float32, torch.bfloat16, torch.int32):
                v.zero_()

        return s

    def _zero_regions(self, s):
        """Build the zero-regions packed tensor for bars + scratch."""
        d = self.sched.bar_dims
        regions = []
        for key in ["bar_qkv", "bar_attn", "bar_oproj", "bar_silu",
                     "bar_ffn_down", "bar_layer", "bar_combine", "bar_lm_head",
                     "norm_out", "q_out", "k_out", "v_out", "attn_out",
                     "o_proj_out", "x_attn", "post_attn_norm", "silu_out",
                     "ffn_down_out", "up_out", "x_ffn", "lm_logits", "o_partial",
                     "lse_partial"]:
            t = s[key]
            regions.append((t.data_ptr(), t.numel() * t.element_size()))
        # Pack as [num_regions, 2] int64
        packed = torch.tensor(regions, dtype=torch.int64, device="cuda")
        return packed

    def test_kernel_completes_without_hang(self):
        """Launch the task-graph kernel for a single decode step and verify
        it completes within a reasonable time (no deadlock)."""
        BS = 1
        s = self._alloc_scratch(BS)

        # Set up input: embedding for token 42 (arbitrary test token)
        token_id = 42
        s["x_resid"][0] = self.embed_weight[token_id]  # [hidden] bf16
        s["cache_seqlens"][0] = 0  # first token, no previous KV
        s["page_table"][0, 0] = 0  # physical page 0
        s["row_active"][0] = 1

        # Zero bars + scratch (use .zero_() directly to avoid the zero-regions
        # kernel; we'll switch to tg_zero_regions once the kernel is validated).
        # Do NOT zero inst_buf / num_inst_per_sm / page_table / cache_seqlens /
        # row_active — those are inputs.
        _DO_NOT_ZERO = {
            "inst_buf", "num_inst_per_sm",
            "page_table", "cache_seqlens", "row_active",
        }
        for k, v in s.items():
            if k in _DO_NOT_ZERO:
                continue
            if v.dtype in (torch.uint32, torch.float32, torch.bfloat16, torch.int32):
                v.zero_()
        # regions = self._zero_regions(s)
        # _tg_ops.tg_zero_regions(regions, regions.shape[0])

        # Re-set x_resid after zeroing
        s["x_resid"][0] = self.embed_weight[token_id]
        s["cache_seqlens"][0] = 0
        s["page_table"][0, 0] = 0
        s["row_active"][0] = 1

        # Launch the kernel
        _tg_ops.tg_decode_step(
            # instruction stream
            s["inst_buf"], s["num_inst_per_sm"],
            self.sched.max_inst, 1,
            # weights
            self.layer_weights_packed, self.embed_weight,
            self.final_norm_weight, self.cos_table, self.sin_table,
            # KV cache + paging
            self.k_cache, self.v_cache, s["page_table"],
            s["cache_seqlens"], s["row_active"],
            self.num_cache_blocks, 64, self.block_size,
            self.kv_layer_stride,
            # scratch
            s["norm_out"], s["q_out"], s["k_out"], s["v_out"],
            s["attn_out"], s["o_proj_out"], s["x_attn"],
            s["post_attn_norm"], s["silu_out"], s["ffn_down_out"],
            s["up_out"],
            s["x_ffn"], s["x_resid"], s["lm_logits"],
            s["o_partial"], s["lse_partial"],
            # bars
            s["bar_qkv"], s["bar_attn"], s["bar_oproj"],
            s["bar_silu"], s["bar_ffn_down"], s["bar_layer"],
            s["bar_combine"], s["bar_lm_head"],
            # geometry
            NUM_LAYERS, BS, HIDDEN_SIZE, INTERMEDIATE_SIZE,
            Q_SIZE, KV_SIZE, NUM_Q_HEADS, NUM_KV_HEADS,
            HEAD_DIM, 151936, 1, 1, 1,
        )

        # Sync and check output
        torch.cuda.synchronize()

        # The argmax token id is stored as int32 at lm_logits[0, 0].
        output_token = s["lm_logits"][0, 0].view(torch.int32).item()

        self.assertIsInstance(output_token, int)
        self.assertGreaterEqual(output_token, 0)
        self.assertLess(output_token, 151936)
        print(f"\nTask-graph kernel completed. Output token: {output_token}")

    def test_kernel_produces_nonzero_logits(self):
        """Verify that lm_logits contains non-zero values (the kernel actually
        computed something, not just passed through zeros)."""
        BS = 1
        s = self._alloc_scratch(BS)

        token_id = 42
        s["x_resid"][0] = self.embed_weight[token_id]
        s["cache_seqlens"][0] = 0
        s["page_table"][0, 0] = 0
        s["row_active"][0] = 1

        _DO_NOT_ZERO = {
            "inst_buf", "num_inst_per_sm",
            "page_table", "cache_seqlens", "row_active",
        }
        for k, v in s.items():
            if k in _DO_NOT_ZERO:
                continue
            if v.dtype in (torch.uint32, torch.float32, torch.bfloat16, torch.int32):
                v.zero_()
        # regions = self._zero_regions(s)
        # _tg_ops.tg_zero_regions(regions, regions.shape[0])

        s["x_resid"][0] = self.embed_weight[token_id]
        s["cache_seqlens"][0] = 0
        s["page_table"][0, 0] = 0
        s["row_active"][0] = 1

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
            HEAD_DIM, 151936, 1, 1, 1,
        )

        torch.cuda.synchronize()

        # Check that logits are not all zero
        logits = s["lm_logits"][0]
        max_logit = logits.max().item()
        self.assertNotEqual(max_logit, 0.0, "lm_logits are all zero — kernel did no compute")
        print(f"\nMax logit: {max_logit:.4f}, argmax: {logits.argmax().item()}")

    def test_zero_regions_clears_scratch_and_bars(self):
        """tg_zero_regions must zero every region, from a CUDA-packed input.

        Regression test for two bugs: the binding dereferenced a device
        data_ptr on the host (segfault), and the kernel used the region index
        as the starting byte offset (only a sparse fraction was cleared).
        """
        BS = 1
        s = self._alloc_scratch(BS)
        skip = {"inst_buf", "num_inst_per_sm", "page_table",
                "cache_seqlens", "row_active"}
        for k, v in s.items():
            if k in skip:
                continue
            v.fill_(1)

        regions = self._zero_regions(s)
        _tg_ops.tg_zero_regions(regions, regions.shape[0])
        torch.cuda.synchronize()

        for k, v in s.items():
            if k in skip:
                continue
            self.assertEqual(int(v.max().item()), 0,
                             f"{k} was not fully zeroed by tg_zero_regions")


if __name__ == "__main__":
    unittest.main(verbosity=2)
