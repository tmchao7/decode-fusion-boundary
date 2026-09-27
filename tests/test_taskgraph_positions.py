"""Multi-position decode correctness for the task-graph megakernel.

Every other TG test decodes at position 0, where RoPE is the identity (cos=1,
sin=0), attention has a single position (softmax trivially 1, out=V) and only
page 0 / block 0 is touched. That hides whole classes of bugs:

  * RoPE at a non-zero position (partner indexing, cos/sin table row)
  * the online softmax across several positions and the cross-warp combine
  * the paged KV cache with a non-zero cache length
  * the KV write path feeding the next step's attention

This test teacher-forces a short prompt through the kernel one token at a time
(so each step's KV is written by the kernel itself) and compares the sampled
token and the logits against a causal PyTorch reference that mirrors the
kernel's bf16 rounding at each tensor boundary.

Usage:
    CUDA_VISIBLE_DEVICES=0 \
    LD_PRELOAD=.nccl_shim.so \
    HF_HUB_OFFLINE=1 \
    python -m pytest tests/test_taskgraph_positions.py -v -s
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import sys
import glob
import math
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import torch
    from nanovllm.backends.megakernel.build import get_extension
    get_extension()
    from nanovllm.backends.megakernel.weights import (
        load_weights, _pack_layer_weights, _rope_tables,
        NUM_LAYERS, HIDDEN_SIZE, INTERMEDIATE_SIZE,
        NUM_KV_HEADS, HEAD_DIM, Q_SIZE, KV_SIZE,
    )
    from nanovllm.backends.megakernel import schedule as tg_schedule
except ModuleNotFoundError as exc:  # torch / the CUDA stack is server-only
    # Skip the module instead of erroring, so a CPU-only checkout still gets a
    # green discovery run. Only ModuleNotFoundError is caught: on the server a
    # genuine extension build failure must stay loud, not become a skip.
    raise unittest.SkipTest(f"task-graph CUDA tests require torch: {exc}")

NUM_Q_HEADS = Q_SIZE // HEAD_DIM  # 16
GROUP_SIZE = NUM_Q_HEADS // NUM_KV_HEADS  # 2
VOCAB = 151936
_tg_ops = torch.ops.nanovllm_megakernel_C

# Teacher-forced prompt. Length must stay <= num_cache_blocks * block_size.
# It deliberately crosses a block boundary (block_size=16) so the paged KV
# logical->physical mapping is exercised, not just block 0.
PROMPT = [
    42, 1234, 100, 7, 999, 555, 31337, 2, 888, 17,
    4096, 63, 123, 77, 2024, 5, 1, 9999, 314, 271,
]
# Greedy tokens generated after the prompt (must fit in the KV cache).
N_GENERATE = 8
# Logit agreement with the reference. The kernel uses ex2.approx softmax and
# rounds at different boundaries, so allow absolute slack; the position-0
# comparison (test_tg_logits_compare) measured < 0.13, but errors can
# accumulate over the teacher-forced steps. A real RoPE/attention/paging bug
# at position > 0 moves logits by O(1)+, so this still catches it. Tighten
# after the first server run if the printed diffs are much smaller.
LOGIT_ATOL = 1.0
# A token is accepted if it is the reference argmax, or the reference logit at
# the kernel's token is within this of the reference max (near-tie / 1-ulp
# cascade, e.g. the documented 10.33 vs 10.27 flip).
TOKEN_GAP_ATOL = 0.5


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


# ---------------------------------------------------------------------------
# Causal PyTorch reference (mirrors the kernel's numerics)
# ---------------------------------------------------------------------------

def pytorch_causal_reference(layer_weights, embed_weight, final_norm_weight,
                             cos_table, sin_table, token_ids):
    """Full causal forward over ``token_ids``; returns logits [T, vocab].

    Matches the kernel's computation order and bf16 rounding: RMSNorm rounds
    twice (scaled value, then weight product), RoPE rounds each rotated element,
    and every tensor boundary is rounded (residual add, SiLU*up, etc.).
    """
    bf16 = torch.bfloat16
    f32 = torch.float32

    def rb(x):
        return x.to(bf16).to(f32)

    def rmsnorm(x, w):
        rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        return rb(rb(x * rstd) * w)

    def rope(x):
        # x: [T, H, D]. Tables use the .repeat(1,2) trick so both halves share
        # the same angle vector; partner of i is i +/- D/2.
        T = x.shape[0]
        c = cos_table[:T].to(f32).unsqueeze(1)   # [T, 1, D]
        s = sin_table[:T].to(f32).unsqueeze(1)
        half = HEAD_DIM // 2
        x1, x2 = x[..., :half], x[..., half:]
        c1, s1 = c[..., :half], s[..., :half]
        c2, s2 = c[..., half:], s[..., half:]
        return torch.cat([rb(x1 * c1 - x2 * s1), rb(x1 * s2 + x2 * c2)], dim=-1)

    T = len(token_ids)
    tok = torch.tensor(token_ids, device="cuda", dtype=torch.long)
    x_resid = embed_weight[tok].to(f32)  # [T, hidden]

    for layer in range(NUM_LAYERS):
        lw = layer_weights[layer * 11:(layer + 1) * 11]
        ln_w, q_w, k_w, v_w, qn_w, kn_w, o_w, post_w, gate_w, up_w, down_w = (
            t.to(f32) for t in lw
        )

        x_normed = rmsnorm(x_resid, ln_w)          # [T, hidden]
        q = x_normed @ q_w.T                       # [T, q_size]
        k = x_normed @ k_w.T                       # [T, kv_size]
        v = x_normed @ v_w.T                       # [T, kv_size]

        q = q.view(T, NUM_Q_HEADS, HEAD_DIM)
        k = k.view(T, NUM_KV_HEADS, HEAD_DIM)
        v = v.view(T, NUM_KV_HEADS, HEAD_DIM)

        q = rmsnorm(q, qn_w)                       # QK-norm per (row, head)
        k = rmsnorm(k, kn_w)

        q = rope(q)
        k = rope(k)
        # The kernel writes V to the bf16 KV cache without any norm/rope, so
        # round it here too (it also rounds K, which rope() already did).
        v = rb(v)

        k_exp = k.repeat_interleave(GROUP_SIZE, dim=1)  # [T, q_heads, D]
        v_exp = v.repeat_interleave(GROUP_SIZE, dim=1)

        # Causal attention (the kernel does online softmax per position; the
        # result is mathematically identical).
        qh = q.permute(1, 0, 2)                    # [H, T, D]
        kh = k_exp.permute(1, 0, 2)
        vh = v_exp.permute(1, 0, 2)
        scores = (qh @ kh.transpose(-1, -2)) / math.sqrt(HEAD_DIM)  # [H,T,T]
        mask = torch.triu(
            torch.ones(T, T, device="cuda", dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn_out = (attn @ vh).permute(1, 0, 2).reshape(T, Q_SIZE)
        # The kernel rounds the attention output to bf16 before O_PROJ.
        attn_out = rb(attn_out)

        x_attn = rb(attn_out @ o_w.T) + x_resid
        x_normed2 = rmsnorm(x_attn, post_w)

        gate = x_normed2 @ gate_w.T
        up = x_normed2 @ up_w.T
        silu_out = rb(rb(torch.sigmoid(gate) * gate) * up)
        # The kernel stores BOTH the unrounded sum (x_ffn, used for the final
        # norm) and its bf16 rounding (x_resid, the next layer's input).
        x_ffn = rb(silu_out @ down_w.T) + x_attn
        x_resid = rb(x_ffn)

    x_final = rmsnorm(x_ffn, final_norm_weight.to(f32))
    return x_final @ embed_weight.to(f32).T        # [T, vocab]


class TaskGraphPositionTests(unittest.TestCase):
    """Teacher-forced multi-step decode vs the causal reference."""

    @classmethod
    def setUpClass(cls):
        model_path = _resolve_model_path()
        cls.weights, cls.tokenizer = load_weights(model_path, verbose=False)
        cls.ext = get_extension()
        cls.cos_table, cls.sin_table = _rope_tables()

        cls.block_size = 16
        cls.num_cache_blocks = 4
        cls.max_pages = 4
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
        cls.sched = tg_schedule.build_schedule(cls.geo, num_sms=1)

    def _alloc_scratch(self, sched=None, BS=1, max_attn_splits=1):
        sched = sched or self.sched
        f32 = dict(dtype=torch.float32, device="cuda")
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
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
        s["o_partial"] = torch.empty(BS, NUM_KV_HEADS, max_attn_splits,
                                      HIDDEN_SIZE, **f32)
        s["lse_partial"] = torch.empty(BS, NUM_KV_HEADS, max_attn_splits, **f32)

        d = sched.bar_dims
        s["bar_qkv"] = torch.zeros(d.bar_qkv, **u32)
        s["bar_attn"] = torch.zeros(d.bar_attn, **u32)
        s["bar_oproj"] = torch.zeros(d.bar_oproj, **u32)
        s["bar_silu"] = torch.zeros(d.bar_silu, **u32)
        s["bar_ffn_down"] = torch.zeros(d.bar_ffn_down, **u32)
        s["bar_layer"] = torch.zeros(d.bar_layer, **u32)
        s["bar_combine"] = torch.zeros(max(d.bar_combine, 1), **u32)
        s["bar_lm_head"] = torch.zeros(d.bar_lm_head, **u32)

        s["page_table"] = torch.zeros(BS, self.max_pages, dtype=torch.int32, device="cuda")
        s["cache_seqlens"] = torch.zeros(BS, dtype=torch.int32, device="cuda")
        s["row_active"] = torch.ones(BS, dtype=torch.int32, device="cuda")
        return s

    def _run_step(self, s, token_id, position, sched=None, num_sms=1,
                  max_attn_splits=1):
        """One teacher-forced decode step; KV cache persists across calls."""
        sched = sched or self.sched
        skip = {"inst_buf", "num_inst_per_sm", "page_table",
                "cache_seqlens", "row_active"}
        for k, v in s.items():
            if k in skip:
                continue
            v.zero_()

        s["x_resid"][0] = self.embed_weight[token_id]
        s["cache_seqlens"][0] = position
        # Logical block 0 -> physical 0, logical block 1 -> physical 1 (the
        # prompt crosses block_size=16).
        s["page_table"][0, 0] = 0
        s["page_table"][0, 1] = 1
        s["row_active"][0] = 1

        _tg_ops.tg_decode_step(
            s["inst_buf"], s["num_inst_per_sm"],
            sched.max_inst, num_sms,
            self.layer_weights_packed, self.embed_weight,
            self.final_norm_weight, self.cos_table, self.sin_table,
            self.k_cache, self.v_cache, s["page_table"],
            s["cache_seqlens"], s["row_active"],
            self.num_cache_blocks, self.max_pages, self.block_size,
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
            NUM_LAYERS, 1, HIDDEN_SIZE, INTERMEDIATE_SIZE,
            Q_SIZE, KV_SIZE, NUM_Q_HEADS, NUM_KV_HEADS,
            HEAD_DIM, VOCAB, 1, 1, max_attn_splits,
        )
        torch.cuda.synchronize()
        return s

    def _check_sequence(self, sched, num_sms, max_attn_splits=1):
        ref_logits = pytorch_causal_reference(
            self.weights["layer_weights"], self.embed_weight,
            self.final_norm_weight, self.cos_table, self.sin_table, PROMPT,
        )
        ref_tokens = ref_logits.argmax(-1).tolist()

        # Fresh KV cache: each step writes its own position.
        self.k_cache.zero_()
        self.v_cache.zero_()
        s = self._alloc_scratch(sched, 1, max_attn_splits)

        got_tokens = []
        print(f"\nPrompt: {PROMPT}  (num_sms={num_sms})")
        for position, token_id in enumerate(PROMPT):
            s = self._run_step(s, token_id, position, sched, num_sms,
                               max_attn_splits)
            tok = s["lm_logits"][0, 0].view(torch.int32).item()
            got_tokens.append(tok)
            # Compare logits at this position (index 0 is clobbered by the
            # in-kernel argmax, so skip it).
            tg_logits = s["lm_logits"][0, 1:].clone().to(torch.float32)
            ref = ref_logits[position, 1:].to(torch.float32)
            max_diff = (tg_logits - ref).abs().max().item()
            gap = (ref_logits[position, ref_tokens[position]]
                   - ref_logits[position, tok]).item()
            print(f"  pos {position}: kernel={tok} ref={ref_tokens[position]} "
                  f"logit_maxdiff={max_diff:.4f} ref_gap={gap:.4f}")
            self.assertLess(
                max_diff, LOGIT_ATOL,
                f"pos {position}: logits differ from the causal reference by "
                f"{max_diff:.4f} (atol {LOGIT_ATOL})",
            )
            self.assertLess(
                gap, TOKEN_GAP_ATOL,
                f"pos {position}: kernel token {tok} is not the reference "
                f"argmax {ref_tokens[position]} and its reference logit is "
                f"{gap:.4f} below the max (atol {TOKEN_GAP_ATOL})",
            )

        print(f"kernel tokens: {got_tokens}")
        print(f"ref    tokens: {ref_tokens}")
        return got_tokens

    def test_multistep_decode_matches_causal_reference(self):
        """Single-SM: isolates the RoPE/attention/paging path from cross-SM."""
        self._check_sequence(self.sched, 1)

    def test_multistep_decode_matches_causal_reference_multi_sm(self):
        """Same sequence across several CTAs: multi-position + cross-SM bars."""
        num_sms = int(os.environ.get("TG_MULTI_SM", "8"))
        sched = tg_schedule.build_schedule(self.geo, num_sms=num_sms)
        self._check_sequence(sched, num_sms)

    def test_attn_splits_matches_causal_reference(self):
        """attn_splits=2 must still match the reference.

        Splitting only repartitions the positions across instructions; the
        LSE combine is mathematically identical, so the per-position tokens
        and logits must match (up to fp reduction order). This is the only
        test of the o_partial/lse_partial path.
        """
        geo2 = tg_schedule.default_geometry(BS=1, attn_splits=2)
        sched2 = tg_schedule.build_schedule(geo2, num_sms=1)
        self._check_sequence(sched2, 1, max_attn_splits=4)

    def test_greedy_generation_matches_reference(self):
        """Autoregressive greedy generation: feed the kernel's own output back.

        Every other test is teacher-forced. This is the only one that exercises
        the feedback loop real inference uses (sampled token -> next step's
        x_resid), including a KV cache that keeps growing past the prompt.
        """
        # Reference greedy decode: the model is causal, so recomputing the
        # forward on the growing sequence and taking the last position's
        # argmax is exact.
        ref_seq = list(PROMPT)
        ref_gen = []
        for _ in range(N_GENERATE):
            logits = pytorch_causal_reference(
                self.weights["layer_weights"], self.embed_weight,
                self.final_norm_weight, self.cos_table, self.sin_table, ref_seq,
            )
            tok = logits[-1].argmax().item()
            ref_gen.append(tok)
            ref_seq.append(tok)

        self.k_cache.zero_()
        self.v_cache.zero_()
        s = self._alloc_scratch(self.sched, 1)
        # Teacher-force the prompt to fill the KV cache.
        for position, token_id in enumerate(PROMPT):
            s = self._run_step(s, token_id, position, self.sched, 1)

        # Then feed each sampled token back as the next step's input.
        ker_gen = []
        pos = len(PROMPT)
        for i in range(N_GENERATE):
            tok = s["lm_logits"][0, 0].view(torch.int32).item()
            ker_gen.append(tok)
            if i + 1 < N_GENERATE:
                s = self._run_step(s, tok, pos, self.sched, 1)
                pos += 1

        print(f"\nprompt:      {PROMPT}")
        print(f"greedy ref:  {ref_gen}")
        print(f"greedy ker:  {ker_gen}")
        self.assertEqual(
            ker_gen, ref_gen,
            "autoregressive greedy generation diverged from the reference "
            "(teacher-forced steps can pass while the feedback loop is wrong)",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
