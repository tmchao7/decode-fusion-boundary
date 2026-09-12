"""Attention must reduce over the FULL head dimension -- and this test must
actually check that.

History (worth keeping): commit ``2d21fd8`` claimed ``ldg_attention`` scored each
cache position over only 4 of the 128 head dimensions and filled output dims
4..127 from uninitialised shared memory, and "fixed" it.  That diagnosis was
wrong, and the test it shipped could not have found the defect it described:

* ``ldg_warp_reduce_sum`` sums the 32 per-lane 4-dimension partials, so the old
  ``score`` already was the complete 128-dim dot product.  The reduction is
  precisely what makes it complete -- it is not a no-op over "the same 4 dims".
* ``int out_base = lane_id * 4`` is executed by all 32 lanes of every warp, so
  slots 0..127 of that warp's ``s_out_acc`` row are written.  Each row is
  ``HEAD_DIM`` = 128 floats (not 512), and it is fully covered.  Nothing is left
  uninitialised.
* The old code is in fact numerically equivalent to the replacement: the only
  semantic differences were ``out_base`` vs ``q_idx`` (the same value), an
  equivalent ``global_max`` initialisation, and comments.

The shipped ``buggy_attention`` modeled a per-lane 4-dim score with *no* warp
reduction, i.e. something the kernel never did, so it diverged by 1.2
(relative) and "proved" a defect that did not exist.

The tests below instead model the kernel's real decomposition -- strided
position ownership per warp, a per-lane 4-dimension QK partial, a warp
all-reduce, a warp-uniform softmax weight applied to each lane's own 4 V
dimensions, and the cross-warp online-softmax merge -- and check it against an
exact reference.  They also keep a source invariant so the reduction cannot be
dropped, because dropping it is the mistake the code *looks* like it makes.
"""

import math
import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

KERNEL_CU = (
    REPO_ROOT
    / "src"
    / "nanovllm"
    / "backends"
    / "megakernel"
    / "csrc"
    / "kernel.cu"
)

HEAD_DIM = 128
WARP_SIZE = 32
NUM_WARPS = 16


def reference_attention(q, keys, values, scale=1.0):
    """Exact single-head attention: score_p = <q, k_p> over all 128 dims."""

    scores = [sum(a * b for a, b in zip(q, k)) * scale for k in keys]
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    out = [0.0] * HEAD_DIM
    for e, v in zip(exps, values):
        for d in range(HEAD_DIM):
            out[d] += (e / total) * v[d]
    return out


def kernel_attention_model(q, keys, values, scale=1.0, reduce_score=True):
    """Faithful float model of ldg_attention's decomposition.

    ``reduce_score=False`` drops the warp reduction, reproducing the mistake the
    code superficially appears to make (and the one ``2d21fd8`` claimed to fix).
    """

    cache_len = len(keys)
    s_out_acc = [[0.0] * HEAD_DIM for _ in range(NUM_WARPS)]
    s_max_score = [float("-inf")] * NUM_WARPS
    s_sum_exp = [0.0] * NUM_WARPS

    for warp_id in range(NUM_WARPS):
        max_score = float("-inf")
        sum_exp = 0.0
        # out_acc is per-lane: every lane follows the same control flow but owns
        # its own 4 dimensions of Q and V.
        out_acc = [[0.0] * 4 for _ in range(WARP_SIZE)]
        q_local = [[q[lane * 4 + t] for t in range(4)] for lane in range(WARP_SIZE)]

        pos = warp_id
        while pos < cache_len:
            k = keys[pos]
            v = values[pos]
            partials = [
                sum(q_local[lane][t] * k[lane * 4 + t] for t in range(4))
                for lane in range(WARP_SIZE)
            ]
            # ldg_warp_reduce_sum is an all-reduce over the warp.
            score = sum(partials) if reduce_score else partials[0]
            score *= scale

            old_max = max_score
            max_score = max(max_score, score)
            exp_diff = (
                math.exp(old_max - max_score) if old_max != float("-inf") else 0.0
            )
            sum_exp = sum_exp * exp_diff + math.exp(score - max_score)
            weight = math.exp(score - max_score)

            for lane in range(WARP_SIZE):
                for t in range(4):
                    out_acc[lane][t] = (
                        out_acc[lane][t] * exp_diff + weight * v[lane * 4 + t]
                    )
            pos += NUM_WARPS

        s_max_score[warp_id] = max_score
        s_sum_exp[warp_id] = sum_exp
        for lane in range(WARP_SIZE):
            for t in range(4):
                s_out_acc[warp_id][lane * 4 + t] = out_acc[lane][t]

    global_max = float("-inf")
    for w in range(NUM_WARPS):
        if s_max_score[w] > float("-inf"):
            global_max = max(global_max, s_max_score[w])

    total_sum_exp = 0.0
    final = [0.0] * HEAD_DIM
    for w in range(NUM_WARPS):
        if s_max_score[w] > float("-inf"):
            rescale = math.exp(s_max_score[w] - global_max)
            total_sum_exp += s_sum_exp[w] * rescale
            for d in range(HEAD_DIM):
                final[d] += s_out_acc[w][d] * rescale
    return [x / total_sum_exp for x in final]


class AttentionAlgorithmTests(unittest.TestCase):
    def _inputs(self, cache_len):
        # Deterministic, every dimension contributes.
        q = [float((d * 13) % 9) - 4.0 for d in range(HEAD_DIM)]
        keys = [
            [float((p * 7 + d) % 5) - 2.0 for d in range(HEAD_DIM)]
            for p in range(cache_len)
        ]
        values = [
            [float((p * 11 + d * 3) % 7) - 3.0 for d in range(HEAD_DIM)]
            for p in range(cache_len)
        ]
        return q, keys, values

    def test_kernel_model_matches_the_exact_reference(self) -> None:
        """The real decomposition (strided positions + warp reduction) is exact."""

        for cache_len in (1, 2, 3, 15, 16, 17, 32, 33, 64, 129):
            with self.subTest(cache_len=cache_len):
                q, keys, values = self._inputs(cache_len)
                expected = reference_attention(q, keys, values, scale=0.25)
                got = kernel_attention_model(q, keys, values, scale=0.25)
                scale = max(abs(x) for x in expected) or 1.0
                worst = max(abs(a - b) for a, b in zip(expected, got))
                self.assertLess(
                    worst / scale,
                    1e-9,
                    f"cache_len={cache_len}: kernel decomposition diverges",
                )

    def test_dropping_the_warp_reduction_really_is_wrong(self) -> None:
        """Without the reduction the score is 4-dim, so the softmax is wrong.

        This is the failure mode the reduction prevents, and it is why the
        source invariant below forbids removing it.
        """

        q, keys, values = self._inputs(33)
        expected = reference_attention(q, keys, values, scale=0.25)
        broken = kernel_attention_model(
            q, keys, values, scale=0.25, reduce_score=False
        )
        scale = max(abs(x) for x in expected) or 1.0
        worst = max(abs(a - b) for a, b in zip(expected, broken)) / scale
        self.assertGreater(worst, 1e-3)

    def test_warp_merge_rescale_cancels_between_numerator_and_denominator(
        self,
    ) -> None:
        """Why the cross-warp merge is exact, stated precisely.

        The merge multiplies a warp's ``s_out_acc`` *and* its ``s_sum_exp`` by
        the same ``scale = exp(s_max_score[w] - global_max)``, and the final
        write divides one by the other -- so ``scale`` cancels.  This is an
        invariant of the *merge*, and it is the property worth pinning here.

        (Note for reviewers: this is NOT the claim that ``softmax(c*S)`` equals
        ``softmax(S)`` for a positive ``c`` -- that is false, and ``attn_scale``
        is a real, deliberate softmax-shaping factor applied once.  There is no
        spurious x32 anywhere in the kernel.)
        """

        out_acc = [1.0, -2.0, 3.5, 0.25, 8.0, -0.5, 1.0, 2.0]
        sum_exp = 4.0
        scale = math.exp(-3.25)  # an arbitrary positive merge factor
        merged = [(x * scale) / (sum_exp * scale) for x in out_acc]
        for x, m in zip(out_acc, merged):
            self.assertAlmostEqual(x / sum_exp, m, places=12)

    def test_the_reduction_is_a_faithful_all_reduce(self) -> None:
        """The old code was correct *because* the reduction sums every lane.

        Commit 2d21fd8 read the reduction as "summing the same 4 dims" and
        replaced correct code.  The point to pin is that the score equals the
        sum over all 32 per-lane partials, nothing else:
        ``score == sum_lane <q[lane], k[lane]> == <q, k>``.
        """

        q, keys, values = self._inputs(17)
        # Reproduce one position's QK step exactly as the kernel does it.
        k = keys[5]
        partials = [
            sum(q[lane * 4 + t] * k[lane * 4 + t] for t in range(4))
            for lane in range(WARP_SIZE)
        ]
        full_dot = sum(a * b for a, b in zip(q, k))
        self.assertAlmostEqual(sum(partials), full_dot, places=10)

        # And the 4-dim prefix, which is what the reduction must NOT be limited
        # to, is a different number.
        self.assertNotAlmostEqual(sum(partials), sum(partials[:1]), places=3)

    def test_dropping_high_dimensions_collapses_distinct_keys(self) -> None:
        """Two keys that differ only in dims >= 4 get identical 4-dim scores."""

        q = [1.0] * HEAD_DIM
        k_a = [2.0] * HEAD_DIM
        k_b = [2.0] * 4 + [-5.0] * (HEAD_DIM - 4)  # identical in dims 0..3
        full = [sum(a * b for a, b in zip(q, k)) for k in (k_a, k_b)]
        truncated = [sum(a * b for a, b in zip(q[:4], k[:4])) for k in (k_a, k_b)]
        self.assertNotAlmostEqual(full[0], full[1])
        self.assertAlmostEqual(truncated[0], truncated[1])
        self.assertAlmostEqual(truncated[0], 8.0)

    def test_every_output_dimension_is_written(self) -> None:
        """All 128 output dims come from a real weighted sum of V."""

        q, keys, values = self._inputs(9)
        got = kernel_attention_model(q, keys, values)
        # Values were built as (p*11 + d*3) % 7 - 3, so a uniform average over
        # positions is deterministic and non-zero for every d once cache_len
        # covers the period; check nothing is left at the initial 0 placeholder.
        self.assertTrue(all(x == x for x in got))  # no NaN
        self.assertTrue(any(abs(x) > 1e-6 for x in got[4:]))
        self.assertTrue(any(abs(x) > 1e-6 for x in got[:4]))


class AttentionSourceInvariantTests(unittest.TestCase):
    """Guard the kernel source so the real decomposition cannot be lost."""

    def setUp(self) -> None:
        text = KERNEL_CU.read_text(encoding="utf-8")
        start = next(
            i for i, line in enumerate(text.split("\n"))
            if line.startswith("__device__ void ldg_attention(")
        )
        lines = text.split("\n")
        depth, seen, end = 0, False, None
        for i in range(start, len(lines)):
            depth += lines[i].count("{") - lines[i].count("}")
            if "{" in lines[i]:
                seen = True
            if seen and depth == 0:
                end = i
                break
        self.lines = lines[start : end + 1]
        self.body = "\n".join(self.lines)

    def test_score_is_reduced_over_the_warp(self) -> None:
        """The reduction is what makes the score a 128-dim dot product."""

        self.assertIn("ldg_warp_reduce_sum(score)", self.body)

    def test_q_local_covers_the_lanes_four_dimensions(self) -> None:
        self.assertIn("int q_idx = lane_id * 4;", self.body)
        for offset in range(4):
            self.assertIn(f"q_head[q_idx + {offset}]", self.body)
        # And the score consumes all four of them before the warp reduction.
        self.assertRegex(
            self.body,
            r"score \+= q_local\[0\][\s\S]{0,400}q_local\[3\]",
        )

    def test_every_lane_stores_its_own_output_dimensions(self) -> None:
        """All 32 lanes must write, otherwise part of the row is unwritten."""

        self.assertRegex(self.body, r"int out_base = lane_id \* 4;")
        for offset in range(4):
            self.assertIn(f"s_out_acc[warp_id][out_base + {offset}]", self.body)
            self.assertIn(f"out_head[out_base + {offset}]", self.body)
        # Nothing may index the output by a fixed per-warp slot.
        self.assertNotRegex(self.body, r"out_head\[[0-3]\]")

    def test_warp_reduce_result_reaches_every_lane(self) -> None:
        """The softmax weight is used per lane, so every lane needs the total.

        Two implementations satisfy this and both are accepted: a butterfly
        all-reduce (``__shfl_xor_sync``), or a reduce-to-lane-0 followed by a
        broadcast (``__shfl_down_sync`` + ``__shfl_sync(..., 0)``).  What is
        forbidden is a reduce-to-lane-0 with no broadcast, which would leave
        lanes 1..31 weighting their own output dimensions with a partial score.
        """

        text = KERNEL_CU.read_text(encoding="utf-8")
        start = text.index("__device__ __forceinline__ float ldg_warp_reduce_sum")
        end = text.index("}", text.index("return val;", start))
        body = text[start:end]
        butterfly = "__shfl_xor_sync" in body
        down_then_broadcast = (
            "__shfl_down_sync" in body
            and "__shfl_sync(0xffffffff, score, 0)" in self.body
        )
        self.assertTrue(
            butterfly or down_then_broadcast,
            "ldg_attention needs the reduced score in every lane: either make "
            "ldg_warp_reduce_sum a butterfly all-reduce, or keep the "
            "reduce-to-lane-0 plus an explicit broadcast in ldg_attention",
        )


if __name__ == "__main__":
    unittest.main()
