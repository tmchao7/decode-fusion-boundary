import re
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from nanovllm.backends.megakernel import schedule as tg  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "src/nanovllm/backends/megakernel/csrc/taskgraph_kernel.cu"
BINDINGS = ROOT / "src/nanovllm/backends/megakernel/csrc/torch_bindings.cpp"
SCHEDULE = ROOT / "src/nanovllm/backends/megakernel/schedule.py"


def _kernel_function_body(source: str, name: str) -> str:
    """Extract a ``__device__ ... name(...) { ... }`` body.

    Brace-counted rather than regex-terminated: the bodies contain nested
    braces, and a non-greedy regex happily cuts a function in half.
    """

    start = source.index(f"void {name}(")
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[brace : index + 1]
    raise AssertionError(f"unterminated body for {name}")


class TaskGraphKernelSourceTests(unittest.TestCase):
    def test_taskgraph_kernel_and_binding_are_present(self):
        kernel = KERNEL.read_text(encoding="utf-8")
        bindings = BINDINGS.read_text(encoding="utf-8")
        self.assertIn("tg_decode_kernel", kernel)
        self.assertIn("cudaLaunchCooperativeKernel", kernel)
        # The launch geometry comes from the runtime descriptor (no compile-time
        # SM/batch ceiling); this is what TG_MAX_BATCH used to be mistaken for.
        self.assertIn("desc->num_sms", kernel)
        self.assertIn('"tg_decode_step(', bindings)
        self.assertIn('ops.impl("tg_decode_step"', bindings)

    def test_active_backend_has_no_legacy_kernel_source(self):
        self.assertFalse(
            (ROOT / "src/nanovllm/backends/megakernel/csrc/kernel.cu").exists()
        )
        self.assertTrue(
            (ROOT / "src/nanovllm/backends/megakernel/weights.py").exists()
        )

    def test_active_kernel_has_no_unwired_ring_scaffold(self):
        kernel = KERNEL.read_text(encoding="utf-8")
        for symbol in (
            "tg_controller_loop",
            "tg_worker_loop",
            "tg_ring_init",
            "tg_ring_wait",
        ):
            self.assertNotIn(symbol, kernel)
        self.assertIn("inst_shared", kernel)

    def test_dynamic_smem_is_sized_from_the_batch_not_the_wmma_tile(self):
        """The s_norm request must follow ``desc->BS``, not a fixed 64 KB.

        Sizing it at ``TG_WMMA_TILE * TG_HIDDEN * 4`` made static (37072 B
        measured) + dynamic exceed the 102400 B per-SM shared memory budget on
        sm_120, so ``cudaFuncSetAttribute`` failed with "invalid argument" and
        the first decode launch died with a message that looked like a driver
        or architecture problem. Guard both halves: the size must be derived
        from the descriptor, and the stale constant must not come back.
        """

        kernel = KERNEL.read_text(encoding="utf-8")
        self.assertIn("desc->BS * (size_t)desc->hidden * sizeof(float)", kernel)
        self.assertNotIn("TG_WMMA_TILE * TG_HIDDEN * sizeof(float)", kernel)
        # A one-shot "already set" flag would leave a small limit in place when a
        # later step uses a larger batch.
        self.assertIn("if (dyn_smem_bytes > smem_attr_bytes)", kernel)
        self.assertNotIn("smem_attr_set", kernel)

    def test_kernel_never_indexes_instruction_words_past_the_row(self) -> None:
        """A packed GEMM is a RANGE: ``inst[9]`` = count, ``inst[10]`` = first tile.

        ``f3d8114`` replaced the per-tile list in ``inst[11..15]`` with that
        range and left ``tg_ffn_upgate_op`` reading ``inst[10 + t]``:

          * ``t == 0``      reads ``inst[10]``   = ``tile_n0``, the pack start
          * ``1 <= t <= 5`` reads the *now-zero* ``inst[11..15]``, so tile 0 is
            re-activated once per tile in the pack
          * ``t == 6``      reads ``inst[16]``, one word PAST ``inst_shared``

        Measured against the emitted BS=1 schedule: only the 28 pack starts
        (tiles 0, 7, 14, ... 189) were ever activated out of 192, 164 were never
        SiLU'd, tile 0 was re-SiLU'd 3836 times, and 756 instructions per step
        read ``inst[16]`` — an out-of-bounds shared-memory read feeding an
        unchecked read-modify-write into ``g.silu_out``.

        Guard the general property, not the one call site: every instruction
        word read must be indexed by an integer *literal*. A regex that only
        looks for ``inst[<identifier>]`` misses ``inst[10 + t]``, which is
        exactly the shape that shipped.
        """

        kernel = KERNEL.read_text(encoding="utf-8")
        # Strip comments first: the fix's own explanation quotes the old
        # expression, and that must not count as a use.
        code = re.sub(r"//[^\n]*", "", kernel)
        indices = re.findall(r"inst\[([^\]]*)\]", code)
        self.assertTrue(indices, "no instruction reads found; did the kernel move?")
        computed = [expr for expr in indices if not expr.strip().isdigit()]
        self.assertEqual(
            computed,
            [],
            f"instruction words indexed by something other than a literal: {computed}",
        )

        # And the range must be walked as a range in the SiLU phase.
        body = _kernel_function_body(kernel, "tg_ffn_upgate_op")
        self.assertIn("int tile_n = tile_n0 + t;", body)
        self.assertIn("int offset = tile_n * TG_WMMA_TILE;", body)

    def test_no_literal_instruction_word_read_is_out_of_bounds(self) -> None:
        """``inst`` in an op body is ``inst_shared``, a ``__shared__ int[16]``.

        The kernel entry calls ``tg_dispatch(inst_shared, ...)`` and no op body
        rebases the pointer, so every ``inst[N]`` is a shared-memory read of a
        16-element array. A literal ``inst[16]`` is therefore an out-of-bounds
        read of whatever ``__shared__`` array was declared next — which is how
        the ``inst[10 + t]`` bug produced tile ids in the tens of thousands.
        """

        src = KERNEL.read_text(encoding="utf-8")
        # TG_INST_WIDTH lives in the ABI header, which is what sizes inst_shared.
        abi = (ROOT / "src/nanovllm/backends/megakernel/csrc/tg_abi.h").read_text(
            encoding="utf-8"
        )
        match = re.search(r"constexpr int TG_INST_WIDTH\s*=\s*(\d+)", abi)
        self.assertIsNotNone(match, "tg_abi.h no longer defines TG_INST_WIDTH")
        width = int(match.group(1))
        self.assertIn(
            "inst_shared[TG_INST_WIDTH]",
            src,
            "the dispatch buffer must stay sized by TG_INST_WIDTH",
        )
        used = {int(n) for n in re.findall(r"inst\[(\d+)\]", re.sub(r"//[^\n]*", "", src))}
        self.assertTrue(used, "no literal instruction reads found")
        self.assertLess(
            max(used), width, f"instruction word reads {sorted(used)} exceed [{width}]"
        )

        # A per-tile list needs one word per tile; the range encoding does not.
        # Lock the reason the loop above may not use `inst[10 + t]`.
        slots_after_range_start = width - 10
        self.assertGreater(
            tg._TILE_SLOTS_PER_INST,
            slots_after_range_start,
            "the pack cap now fits in inst[10..width-1], so the range encoding "
            "is no longer forced by the instruction width -- revisit this guard",
        )

    def test_silu_phase_covers_every_intermediate_tile_exactly_once(self) -> None:
        """Offline check of what the kernel's SiLU loop touches.

        Reproduces the kernel's own indexing against the emitted schedule and
        asserts it activates each of the 192 intermediate tiles exactly once.
        This is the test that would have caught the ``inst[10 + t]`` bug on a
        CPU-only checkout, before it cost a GPU session.
        """

        import sys
        from collections import defaultdict

        sys.path.insert(0, str(ROOT / "src"))
        from nanovllm.backends.megakernel import schedule as tg

        # The WMMA N tile is `ffn_bn` (16), and it is what the kernel's
        # `tile_n * TG_WMMA_TILE` stride must correspond to. Read it from the
        # geometry rather than restating 16, so a tiling change cannot leave
        # this test asserting yesterday's arithmetic.
        for batch_size in (1, 2, 4, 8):
            geo = tg.default_geometry(BS=batch_size, attn_splits=1)
            sched = tg.build_schedule(geo, num_sms=1)
            width = tg.TG_INST_WIDTH
            flat = sched.flat_inst_buf

            # Keyed by layer: the pool is per-layer, and FFN_DOWN for layer L
            # only reads the layer-L columns that layer-L's UPGATE wrote.
            touched: dict[int, dict[int, int]] = defaultdict(
                lambda: defaultdict(int)
            )
            for i in range(sched.num_inst_per_sm[0]):
                base = i * width
                inst = flat[base : base + width]
                if inst[0] != tg.TgOp.FFN_UPGATE_ACT:
                    continue
                layer, num_tiles = inst[1], inst[9]
                tile_n0 = inst[10] & 0xFFFF
                for t in range(num_tiles):
                    tile_n = tile_n0 + t  # the kernel's expression, mirrored
                    touched[layer][tile_n] += 1

            expected = set(range(geo.intermediate // geo.tiling.ffn_bn))
            with self.subTest(batch_size=batch_size):
                self.assertEqual(
                    sorted(touched), list(range(geo.num_layers))
                )
                for layer, hits in sorted(touched.items()):
                    self.assertEqual(
                        set(hits),
                        expected,
                        f"layer {layer} untouched intermediate tiles: "
                        f"{sorted(expected - set(hits))[:8]}",
                    )
                    self.assertEqual(
                        sorted(set(hits.values())),
                        [1],
                        f"layer {layer}: each intermediate tile must be "
                        "activated exactly once",
                    )



if __name__ == "__main__":
    unittest.main()
