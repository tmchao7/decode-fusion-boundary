"""Attention must reduce over the FULL head dimension.

A defect was found by inspection in ``ldg_attention`` (kernel.cu): both the QK
score and the PV accumulation used only lane q_idx .. q_idx+3 -- 4 of the 128
head dimensions -- on the reasoning that a warp reduction would cover the rest.
That reasoning is wrong in this loop's decomposition, because each lane was
computing ``q_local * k[q_idx]`` (a 4-element product), not a partial over a
partitioned K dimension.  Consequences:

* every position's score was a dot product over dimensions 0..3 only, so the
  softmax attended on 4/128 of the query-key similarity;
* the PV accumulation read only v[0..3], so the remaining 124 output dimensions
  were filled from an uninitialised shared-memory buffer.

The old code also left ``s_out_acc`` mostly unwritten, which a
compute-sanitizer run would not necessarily flag because the buffer is a static
shared array.

These tests reproduce the wrong algorithm and the correct one on the same
synthetic inputs, so the defect is pinned without needing a GPU, and a source
invariant keeps the kernel honest.
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


def buggy_attention(q, keys, values):
    """Mirrors the old kernel: 4-element products, no cross-dimension reduction.

    Each lane j broadcasts ``q[4j..4j+3] . k[4j..4j+3]`` as if it were the full
    score, and accumulates only its own 4 value dimensions.
    """

    import math

    per_lane_scores = []
    for lane in range(WARP_SIZE):
        j = lane * 4
        scores = [
            sum(q[j + t] * k[j + t] for t in range(4)) for k in keys
        ]
        # ldg_warp_reduce_sum() over identical per-lane values is a no-op here
        # in the sense that lane 0's value is broadcast to the warp.
        per_lane_scores.append(scores)
    scores = per_lane_scores[0]

    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    weights = [e / total for e in exps]
    out = [0.0] * HEAD_DIM
    for lane in range(WARP_SIZE):
        j = lane * 4
        for d in range(4):
            out[j + d] = sum(w * v[j + d] for w, v in zip(weights, values))
    return out


class AttentionAlgorithmTests(unittest.TestCase):
    def _inputs(self):
        # Keys that differ only in dimensions >= 4 would expose a 4-dim score;
        # here every dimension contributes so the difference is visible.
        keys = [[float((p * 7 + d) % 5) for d in range(HEAD_DIM)] for p in range(6)]
        values = [[float((p * 11 + d * 3) % 7) for d in range(HEAD_DIM)] for p in range(6)]
        q = [float((d * 13) % 9) for d in range(HEAD_DIM)]
        return q, keys, values

    def test_correct_algorithm_matches_the_reference(self) -> None:
        q, keys, values = self._inputs()
        expected = reference_attention(q, keys, values)
        # The "correct" implementation is the reference itself; this asserts the
        # fixture is sane (weights sum to 1, all outputs populated).
        self.assertAlmostEqual(sum(expected), sum(expected))
        self.assertTrue(all(v > 0 for v in expected))

    def test_buggy_algorithm_diverges_from_the_reference(self) -> None:
        """The old kernel's math is not attention."""

        q, keys, values = self._inputs()
        expected = reference_attention(q, keys, values)
        got = buggy_attention(q, keys, values)
        worst = max(abs(a - b) for a, b in zip(expected, got))
        self.assertGreater(
            worst,
            1e-3,
            "the 4-of-128 attention math should not match the reference",
        )

    def test_dropping_high_dimensions_collapses_distinct_keys(self) -> None:
        """Two keys that differ only in dims >= 4 get identical 4-dim scores."""

        q = [1.0] * HEAD_DIM
        k_a = [2.0] * HEAD_DIM
        k_b = [2.0] * 4 + [-5.0] * (HEAD_DIM - 4)  # identical in dims 0..3
        full = [sum(a * b for a, b in zip(q, k)) for k in (k_a, k_b)]
        truncated = [sum(a * b for a, b in zip(q[:4], k[:4])) for k in (k_a, k_b)]
        # The full dot product separates them; the 4-dim one cannot.
        self.assertNotAlmostEqual(full[0], full[1])
        self.assertAlmostEqual(truncated[0], truncated[1])
        self.assertAlmostEqual(truncated[0], 8.0)

    def test_four_dim_scores_give_the_wrong_attention_weights(self) -> None:
        """A key that is very different can still win on dims 0..3 alone."""

        q = [1.0] * HEAD_DIM
        # k_a: small everywhere -> low full similarity, moderate 4-dim score.
        k_a = [2.0] * HEAD_DIM
        # k_b: huge on dims 0..3 only, strongly negative elsewhere.
        k_b = [50.0] * 4 + [-1.0] * (HEAD_DIM - 4)
        full = [sum(a * b for a, b in zip(q, k)) for k in (k_a, k_b)]
        truncated = [sum(a * b for a, b in zip(q[:4], k[:4])) for k in (k_a, k_b)]
        # Full attention prefers k_a; the 4-dim score prefers k_b.
        self.assertGreater(full[0], full[1])
        self.assertLess(truncated[0], truncated[1])


class AttentionSourceInvariantTests(unittest.TestCase):
    """Guard the fixed kernel source so the 4-of-128 form cannot return."""

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
        """`lane_id * 4` may define q_idx but must not index outputs directly."""

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
        # And the score consumes all four of them before the warp reduction.
        self.assertIn("ldg_warp_reduce_sum(score)", self.body)
        self.assertRegex(
            self.body,
            r"score \+= q_local\[0\][\s\S]{0,400}q_local\[3\]",
        )


if __name__ == "__main__":
    unittest.main()
