"""Guard that the probe barrier stays faithful to the production one.

The device lower-bound probe and the barrier microbenchmark are only meaningful
if their grid-wide sync costs the same as ``AtomicGridSync`` in ``kernel.cu``.
Both are hand-copied CUDA strings, so they can drift silently -- and neither can
be exercised locally (no GPU).  These tests diff the barrier text instead.

If this test fails, either the production barrier changed (update the probes'
strings to match) or a probe was edited (restore the production semantics).
"""

import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from tools.bench_burst_lower_bound import (  # noqa: E402
    _PROBE_SOURCE as LOWER_BOUND_SOURCE,
)
from tools.bench_roofline_a100 import _BARRIER_SOURCE  # noqa: E402

KERNEL_CU = (
    REPO_ROOT
    / "src"
    / "nanovllm"
    / "backends"
    / "megakernel"
    / "csrc"
    / "kernel.cu"
)

PROBE_SOURCES = {
    "bench_roofline_a100._BARRIER_SOURCE": _BARRIER_SOURCE,
    "bench_burst_lower_bound._PROBE_SOURCE": LOWER_BOUND_SOURCE,
}


def production_barrier() -> str:
    text = KERNEL_CU.read_text(encoding="utf-8")
    match = re.search(
        r"struct AtomicGridSync \{(?P<body>.*?)\n\};", text, re.DOTALL
    )
    if match is None:
        raise AssertionError("could not locate AtomicGridSync in kernel.cu")
    return match.group(0)


def probe_barrier(source: str) -> str:
    match = re.search(
        r"__device__ __forceinline__ void atomic_grid_sync\((?P<body>.*?)\n\}",
        source,
        re.DOTALL,
    )
    if match is None:
        raise AssertionError("could not locate atomic_grid_sync in probe source")
    return match.group(0)


class LaunchBoundsFidelityTests(unittest.TestCase):
    """Occupancy contract: both probes must be 1 block/SM like production."""

    def test_kernel_declares_one_block_per_sm(self) -> None:
        text = KERNEL_CU.read_text(encoding="utf-8")
        self.assertIn("__launch_bounds__(LDG_BLOCK_SIZE, 1)", text)

    def test_barrier_probe_declares_one_block_per_sm(self) -> None:
        self.assertIn("__launch_bounds__(512, 1) barrier_probe", _BARRIER_SOURCE)

    def test_lower_bound_probe_declares_one_block_per_sm(self) -> None:
        self.assertIn(
            "__launch_bounds__(512, 1) lower_bound_probe", LOWER_BOUND_SOURCE
        )


class BarrierFidelityTests(unittest.TestCase):
    def test_production_barrier_uses_gpu_scope_fence(self) -> None:
        """The reference semantics we are copying."""

        body = production_barrier()
        self.assertIn("fence.acq_rel.gpu", body)
        self.assertIn("<= my_gen", body)

    def test_every_probe_uses_gpu_scope_fence(self) -> None:
        for name, source in PROBE_SOURCES.items():
            body = probe_barrier(source)
            self.assertIn("fence.acq_rel.gpu", body, name)

    def test_every_probe_waits_with_less_or_equal(self) -> None:
        """`==` can miss an increment when a fast peer advances twice."""

        for name, source in PROBE_SOURCES.items():
            body = probe_barrier(source)
            self.assertIn("<= local_gen", body, name)
            self.assertNotIn("__nanosleep", body, name)

    def test_no_threadfence_in_probe_barrier(self) -> None:
        """__threadfence only orders the executing thread, unlike the pair."""

        for name, source in PROBE_SOURCES.items():
            body = probe_barrier(source)
            self.assertNotIn("__threadfence", body, name)

    def test_probes_capture_generation_before_arriving(self) -> None:
        """The local generation must be passed in, not read after arrival."""

        for name, source in PROBE_SOURCES.items():
            body = probe_barrier(source)
            self.assertIn("local_gen", body, name)
            # Must not re-read the global generation into a local variable.
            self.assertNotIn("= *generation", body, name)

    def test_callers_increment_the_local_generation(self) -> None:
        for name, source in PROBE_SOURCES.items():
            self.assertIn("local_gen += 1u", source, name)

    def test_both_probes_share_the_same_barrier_text(self) -> None:
        """Normalised comparison, so whitespace/comment edits are allowed."""

        def normalise(body: str) -> str:
            code = "\n".join(
                line.split("//")[0] for line in body.splitlines()
            )
            return re.sub(r"\s+", " ", code).strip()

        bodies = {
            name: normalise(probe_barrier(source))
            for name, source in PROBE_SOURCES.items()
        }
        first = next(iter(bodies.values()))
        for name, body in bodies.items():
            self.assertEqual(body, first, f"{name} differs from its peer")


if __name__ == "__main__":
    unittest.main()
