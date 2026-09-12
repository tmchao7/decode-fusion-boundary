"""Fairness audit of the megakernel-vs-cudagraph comparison.

The two paths must be compared in comparable configurations, and the result an
audit must be reproducible.  Two concrete asymmetries are pinned here:

1. **The benchmark suites could never exercise the burst path.**  ``_should_burst``
   refused any ``ignore_eos`` sequence, and the controlled suites run with
   ``ignore_eos=True`` to force a fixed token count.  So the headline "megakernel
   vs cudagraph" table measured the megakernel's step-wise path (per-step host
   round trip) against a CUDA Graph, while the "--burst" tool measured the fast
   path under a different harness.  The kernel now treats ``eos_token < 0`` as
   "never deactivate", so an all-ignore_eos batch may burst.

2. **CUDA-graph batch padding.**  The torch path captures graphs at
   ``[1,2,4,8] + range(16, max_bs+1, 16)`` and replays the first bucket ``>=``
   the live batch, so a non-bucket batch pays padding.  That is legitimate (it
   is what production vLLM does) but it must be accounted for when comparing
   against the exact-batch-size megakernel.
"""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nanovllm.engine.burst_policy import (  # noqa: E402
    burst_batch_eligibility,
    burst_eos_token,
)


def make_config(**overrides):
    values = dict(
        megakernel_burst=True,
        decode_backend="megakernel",
        megakernel_batch_kernel="wmma",
        megakernel_batch_decode=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def make_engine(*, seqs, config=None):
    engine = LLMEngine.__new__(LLMEngine)
    engine.config = config or make_config()
    engine.model_runner = SimpleNamespace(
        megakernel=SimpleNamespace(disabled_reason=None),
        _cuda_poisoned=False,
    )
    engine.scheduler = SimpleNamespace(waiting=[], running=list(seqs))
    return engine


def make_seq(*, greedy=True, ignore_eos=False):
    return SimpleNamespace(is_greedy=greedy, ignore_eos=ignore_eos)


class BurstGateTests(unittest.TestCase):
    def test_ignore_eos_batch_may_now_burst(self) -> None:
        """The controlled suites set ignore_eos=True; the burst must be usable."""

        eligible, reason = burst_batch_eligibility(
            make_config(), [make_seq(ignore_eos=True), make_seq(ignore_eos=True)]
        )
        self.assertTrue(eligible, reason)

    def test_mixed_eos_batch_still_falls_back(self) -> None:
        """A single batch-wide device flag cannot express per-seq ignore_eos."""

        eligible, reason = burst_batch_eligibility(
            make_config(), [make_seq(ignore_eos=True), make_seq(ignore_eos=False)]
        )
        self.assertFalse(eligible)
        self.assertIn("mixed ignore_eos", reason)

    def test_non_ignore_eos_batch_still_bursts(self) -> None:
        eligible, reason = burst_batch_eligibility(
            make_config(), [make_seq(ignore_eos=False)]
        )
        self.assertTrue(eligible, reason)

    def test_sampling_sequences_are_still_rejected(self) -> None:
        eligible, reason = burst_batch_eligibility(
            make_config(), [make_seq(greedy=False)]
        )
        self.assertFalse(eligible)
        self.assertIn("greedy", reason)

    def test_batch_gt_one_requires_the_explicit_switch(self) -> None:
        eligible, _ = burst_batch_eligibility(
            make_config(megakernel_batch_decode=False), [make_seq(), make_seq()]
        )
        self.assertFalse(eligible)

    def test_disabled_or_wrong_backend_is_rejected(self) -> None:
        for overrides in (
            {"megakernel_burst": False},
            {"decode_backend": "torch"},
            {"megakernel_batch_kernel": "scalar"},
        ):
            with self.subTest(**overrides):
                eligible, _ = burst_batch_eligibility(
                    make_config(**overrides), [make_seq()]
                )
                self.assertFalse(eligible)

    def test_empty_batch_is_rejected(self) -> None:
        eligible, reason = burst_batch_eligibility(make_config(), [])
        self.assertFalse(eligible)
        self.assertIn("no running sequences", reason)


class EosSentinelTests(unittest.TestCase):
    """``run_burst`` must send -1 when the whole batch ignores EOS."""

    def test_all_ignore_eos_passes_the_never_stop_sentinel(self) -> None:
        seqs = [make_seq(ignore_eos=True) for _ in range(2)]
        self.assertEqual(burst_eos_token(make_config(eos=151643), seqs), -1)

    def test_normal_batch_passes_the_real_eos(self) -> None:
        self.assertEqual(
            burst_eos_token(make_config(eos=151643), [make_seq(ignore_eos=False)]),
            151643,
        )

    def test_empty_batch_falls_back_to_configured_eos(self) -> None:
        self.assertEqual(burst_eos_token(make_config(eos=151643), []), 151643)

    def test_run_burst_uses_the_helper(self) -> None:
        """The engine must not re-implement the sentinel decision."""

        source = (
            REPO_ROOT / "src" / "nanovllm" / "engine" / "model_runner.py"
        ).read_text(encoding="utf-8")
        self.assertIn("eos_token = burst_eos_token(self.config, seqs)", source)


class CudagraphPaddingDocumentationTests(unittest.TestCase):
    """The buckets are a legitimate but uneven cost the report must disclose."""

    def test_graph_buckets_match_the_documented_schedule(self) -> None:
        source = (
            REPO_ROOT / "src" / "nanovllm" / "engine" / "model_runner.py"
        ).read_text(encoding="utf-8")
        self.assertIn("[1, 2, 4, 8] + list(range(16, max_bs + 1, 16))", source)
        self.assertIn("next(x for x in self.graph_bs if x >= bs)", source)

    def test_padding_is_documented_in_the_fairness_doc(self) -> None:
        """docs/05 must state the bucket padding so the table is not read as clean."""

        fairness = (
            REPO_ROOT / "docs" / "05_对比基线与公平性.md"
        ).read_text(encoding="utf-8")
        self.assertIn("bucket（桶）", fairness)
        self.assertIn("graph_bs", fairness)
        self.assertIn("+60%", fairness)

    def test_asymmetric_acceleration_is_documented(self) -> None:
        fairness = (
            REPO_ROOT / "docs" / "05_对比基线与公平性.md"
        ).read_text(encoding="utf-8")
        self.assertIn("FlashAttention", fairness)
        self.assertIn("scalar", fairness)

    def test_architecture_doc_still_records_the_fairness_audit(self) -> None:
        arch = (
            REPO_ROOT / "docs" / "07_架构结论_megakernel为何打不过cudagraph.md"
        ).read_text(encoding="utf-8")
        self.assertIn("对比公平性", arch)
        self.assertIn("graph_bs", arch)


if __name__ == "__main__":
    unittest.main()
