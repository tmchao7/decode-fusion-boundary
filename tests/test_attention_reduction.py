"""Attention must reduce over the FULL head dimension.

The ``ldg_attention`` kernel (kernel.cu) partitions HEAD_DIM=128 across the 32
lanes of a warp: lane *j* loads ``q[4j..4j+3]`` and computes a 4-element partial
dot product against the corresponding key slice.  A ``ldg_warp_reduce_sum`` then
sums all 32 per-lane partials, producing the complete 128-dim dot product, and
``__shfl_sync`` broadcasts it to every lane so the online-softmax update is
uniform.  The PV accumulation is partitioned the same way: lane *j* owns output
dimensions ``4j..4j+3``, and 32 lanes × 4 dims = 128 covers every output
dimension.  Per-warp partials are merged in shared memory with the standard
max/rescale online-softmax combination.

These tests verify that the warp-reduced decomposition is mathematically
equivalent to the reference full-dim attention, and a source invariant keeps
the kernel honest (the warp reduction and ``q_idx``-indexed output must remain).

A note on history: a prior review (commit 2d21fd8) incorrectly concluded that
the 4-dim per-lane partial was the *entire* score (i.e. that the warp reduction
was a no-op).  That is wrong -- the 32 per-lane partials are different slices
of the same dot product, and their sum is the full HEAD_DIM inner product.
The "fix" in that commit was a cosmetic rename (``out_base`` -> ``q_idx``,
both equal to ``lane_id * 4``) with no semantic change.  These tests now
document the correct decomposition so the misunderstanding does not recur.
"""

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


def reference_attention(q, keys, values):
    """Exact single-head attention: score_p = <q, k_p> over all 128 dims."""

    import math

    scores = [sum(a * b for a, b in zip(q, k)) for k in keys]
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    weights = [e / total for e in exps]
    out = [0.0] * HEAD_DIM
    for w, v in zip(weights, values):
        for d in range(HEAD_DIM):
            out[d] += w * v[d]
    return out


def warp_reduced_attention(q, keys, values):
    """Mirror the kernel's decomposition: per-lane 4-dim partials summed via
    a warp reduction give the full 128-dim score; PV is partitioned by lane.

    This is what ``ldg_attention`` actually computes (one warp, for simplicity
    -- the cross-warp merge is tested separately below).
    """

    import math

    # QK: each lane j computes sum_{d=0..3} q[4j+d]*k[4j+d], then the warp
    # reduction sums all 32 lanes = full 128-dim dot product.
    scores = []
    for k in keys:
        per_lane = []
        for lane in range(WARP_SIZE):
            j = lane * 4
            partial = sum(q[j + t] * k[j + t] for t in range(4))
            per_lane.append(partial)
        full_score = sum(per_lane)  # ldg_warp_reduce_sum
        scores.append(full_score)

    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    weights = [e / total for e in exps]

    # PV: lane j owns output dims 4j..4j+3; 32 lanes cover all 128 dims.
    out = [0.0] * HEAD_DIM
    for lane in range(WARP_SIZE):
        j = lane * 4
        for d in range(4):
            out[j + d] = sum(w * v[j + d] for w, v in zip(weights, values))
    return out


def four_dim_only_score(q, keys):
    """What the score would be if the warp reduction were truly a no-op
    (i.e. only lane 0's 4-dim partial were used).  This is NOT what the kernel
    does -- it documents why the warp reduction is essential.
    """

    return [sum(q[t] * k[t] for t in range(4)) for k in keys]


class WarpReductionEquivalenceTests(unittest.TestCase):
    """The kernel's per-lane decomposition + warp reduction == reference."""

    def _inputs(self):
        keys = [[float((p * 7 + d) % 5) for d in range(HEAD_DIM)] for p in range(6)]
        values = [[float((p * 11 + d * 3) % 7) for d in range(HEAD_DIM)] for p in range(6)]
        q = [float((d * 13) % 9) for d in range(HEAD_DIM)]
        return q, keys, values

    def test_warp_reduced_matches_reference(self) -> None:
        """The kernel's decomposition produces the same result as full-dim
        attention.  This is the core correctness property."""

        q, keys, values = self._inputs()
        expected = reference_attention(q, keys, values)
        got = warp_reduced_attention(q, keys, values)
        worst = max(abs(a - b) for a, b in zip(expected, got))
        self.assertLess(worst, 1e-9)

    def test_warp_reduction_produces_full_dot_product(self) -> None:
        """Per-lane 4-dim partials summed = full 128-dim dot product."""

        q, keys, _ = self._inputs()
        for k in keys:
            full = sum(a * b for a, b in zip(q, k))
            per_lane_sum = sum(
                sum(q[j + t] * k[j + t] for t in range(4))
                for j in range(0, HEAD_DIM, 4)
            )
            self.assertAlmostEqual(full, per_lane_sum, places=5)

    def test_four_dim_only_score_diverges_from_full(self) -> None:
        """A 4-dim-only score (what the kernel does NOT do) would be wrong.
        This documents why the warp reduction is essential, not optional."""

        q, keys, _ = self._inputs()
        full_scores = [sum(a * b for a, b in zip(q, k)) for k in keys]
        truncated_scores = four_dim_only_score(q, keys)
        worst = max(abs(a - b) for a, b in zip(full_scores, truncated_scores))
        self.assertGreater(worst, 1e-3)

    def test_dropping_high_dimensions_collapses_distinct_keys(self) -> None:
        """Two keys that differ only in dims >= 4 get identical 4-dim scores
        but different full scores.  The warp reduction is what separates them."""

        q = [1.0] * HEAD_DIM
        k_a = [2.0] * HEAD_DIM
        k_b = [2.0] * 4 + [-5.0] * (HEAD_DIM - 4)
        full = [sum(a * b for a, b in zip(q, k)) for k in (k_a, k_b)]
        truncated = [sum(a * b for a, b in zip(q[:4], k[:4])) for k in (k_a, k_b)]
        self.assertNotAlmostEqual(full[0], full[1])
        self.assertAlmostEqual(truncated[0], truncated[1])


class AttentionSourceInvariantTests(unittest.TestCase):
    """Guard the kernel source so the warp-reduced form cannot regress to a
    4-dim-only score."""

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
        self.assertIn("ldg_warp_reduce_sum(score)", self.body)

    def test_output_accumulator_is_indexed_by_q_idx(self) -> None:
        self.assertIn("s_out_acc[warp_id][q_idx + 0]", self.body)
        self.assertIn("out_head[q_idx + 0]", self.body)

    def test_no_four_element_only_offsets_remain(self) -> None:
        """``lane_id * 4`` may define q_idx but must not index outputs directly."""

        for match in re.finditer(r"(?:s_out_acc\[w\]|out_head)\[([^\]]+)\]", self.body):
            index = match.group(1)
            self.assertNotIn(
                "lane_id * 4",
                index,
                f"output indexed by a per-lane 4-element offset: {index}",
            )

    def test_q_local_covers_the_lanes_four_dimensions(self) -> None:
        self.assertIn("const int q_idx = lane_id * 4;", self.body)
        for offset in range(4):
            self.assertIn(f"q_head[q_idx + {offset}]", self.body)
        self.assertIn("ldg_warp_reduce_sum(score)", self.body)
        self.assertRegex(
            self.body,
            r"score \+= q_local\[0\][\s\S]{0,400}q_local\[3\]",
        )

    def test_score_is_broadcast_to_all_lanes(self) -> None:
        """After the warp reduction, the full score must be broadcast to every
        lane so the online-softmax update is uniform."""

        self.assertIn("__shfl_sync(0xffffffff, score, 0)", self.body)


if __name__ == "__main__":
    unittest.main()
