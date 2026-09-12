"""Source invariants for the batch-launch guards.

The batch kernel's real batch limit is ``grid_blocks / LDG_ATTN_BLOCKS`` (the
cooperative grid is partitioned per sequence), which is far below the
compile-time ``LDG_MAX_BATCH=32`` ceiling.  Before this guard existed the launch
aborted, printed to stderr, and returned -- the void binding reported success and
the Python caller read the *previous* step's tokens as fresh output.

These tests read the C/C++ sources directly (they cannot run without a GPU) and
pin the relationship between the three places that must agree:

* ``build.py``           - the compile-time ceiling passed as ``LDG_MAX_BATCH``
* ``kernel.cu``          - ``ldg_wmma_max_supported_batch`` and the backstop
* ``torch_bindings.cpp`` - the observable rejection at the binding boundary
* ``backend.py``         - the compatibility check that falls back to cudagraph
"""

import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

MEGAKERNEL = REPO_ROOT / "src" / "nanovllm" / "backends" / "megakernel"
KERNEL_CU = MEGAKERNEL / "csrc" / "kernel.cu"
BINDINGS = MEGAKERNEL / "csrc" / "torch_bindings.cpp"
BUILD_PY = MEGAKERNEL / "build.py"
BACKEND_PY = MEGAKERNEL / "backend.py"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class CompileTimeCeilingTests(unittest.TestCase):
    def test_ldg_max_batch_matches_the_binding_check(self) -> None:
        build = read(BUILD_PY)
        match = re.search(r'"-DLDG_MAX_BATCH=(\d+)"', build)
        self.assertIsNotNone(match, "build.py no longer defines LDG_MAX_BATCH")
        assert match is not None
        ceiling = int(match.group(1))

        # The kernel defaults to the same number when the flag is absent.
        kernel = read(KERNEL_CU)
        default = re.search(r"#define LDG_MAX_BATCH (\d+)", kernel)
        self.assertIsNotNone(default)
        assert default is not None
        self.assertEqual(int(default.group(1)), ceiling)

        # Every binding TORCH_CHECK that gates a batch argument must use the
        # same ceiling, or a batch can slip past one and be rejected by another.
        for limit in re.findall(r"input_token_ids\.numel\(\) <= (\d+)", read(BINDINGS)):
            with self.subTest(limit=limit):
                self.assertEqual(int(limit), ceiling)


class KernelFlagDefaultTests(unittest.TestCase):
    """``kernel.cu``'s ``#ifndef`` fallbacks must equal ``build.py``'s defaults.

    The JIT build always passes these as ``-D`` flags, so the fallbacks are dead
    in the normal path -- which is exactly what makes them dangerous. Any other
    compile path (a manual ``nvcc`` line, a reduced reproduction, another
    frontend) silently gets *different* constants: the fallback for
    ``LDG_ATTN_BLOCKS`` was ``NUM_Q_HEADS`` (16) against a build default of 8,
    which silently halves the supported batch, and ``LDG_PREFETCH_QK`` fell back
    to 1 against a build default of 0, which silently enables the prefetch path
    that the shipped configuration turns off.
    """

    # Names that are compile-time constexprs in kernel.cu rather than literals.
    CONSTEXPRS = {"NUM_Q_HEADS": 16, "NUM_KV_HEADS": 8, "LDG_NUM_WARPS": 16}

    def _kernel_defaults(self) -> dict[str, int]:
        kernel = read(KERNEL_CU)
        defaults: dict[str, int] = {}
        for name, value in re.findall(
            r"#ifndef\s+(LDG_[A-Z0-9_]+)\s*\n#define\s+\1\s+([^\n/]+)", kernel
        ):
            value = value.strip()
            if value in self.CONSTEXPRS:
                defaults[name] = self.CONSTEXPRS[value]
            elif value.isdigit():
                defaults[name] = int(value)
        return defaults

    def _build_defaults(self) -> dict[str, int]:
        build = read(BUILD_PY)
        return {
            name: int(default)
            for name, default in re.findall(
                r"_env_int\('(LDG_[A-Z0-9_]+)',\s*(\d+)\)", build
            )
        }

    def test_every_tunable_flag_agrees(self) -> None:
        kernel_defaults = self._kernel_defaults()
        build_defaults = self._build_defaults()
        self.assertTrue(build_defaults, "build.py defines no LDG_* env defaults")
        for name, build_default in sorted(build_defaults.items()):
            with self.subTest(flag=name):
                self.assertIn(
                    name,
                    kernel_defaults,
                    f"{name} has no #ifndef fallback in kernel.cu",
                )
                self.assertEqual(
                    kernel_defaults[name],
                    build_default,
                    f"{name}: kernel.cu fallback {kernel_defaults[name]} != "
                    f"build.py default {build_default}",
                )

    def test_attn_blocks_fallback_keeps_the_documented_batch_limit(self) -> None:
        """LDG_ATTN_BLOCKS decides the batch limit (grid / LDG_ATTN_BLOCKS)."""

        attn = self._kernel_defaults().get("LDG_ATTN_BLOCKS")
        self.assertEqual(attn, 8)
        grid = self._build_defaults()["LDG_NUM_BLOCKS"]
        self.assertEqual(grid // attn, 10)


class RealLimitTests(unittest.TestCase):
    def test_kernel_derives_the_limit_from_the_grid(self) -> None:
        kernel = read(KERNEL_CU)
        helper = re.search(
            r"extern \"C\" int ldg_wmma_max_supported_batch\(int requested_blocks\) \{(.*?)\n\}",
            kernel,
            re.DOTALL,
        )
        self.assertIsNotNone(helper, "ldg_wmma_max_supported_batch is missing")
        assert helper is not None
        body = helper.group(1)
        self.assertIn("ldg_resident_grid_blocks", body)
        self.assertIn("LDG_ATTN_BLOCKS", body)
        self.assertIn("grid / LDG_ATTN_BLOCKS", body)

    def test_default_grid_supports_far_fewer_than_the_ceiling(self) -> None:
        """80-block grid with LDG_ATTN_BLOCKS=8 -> batch <= 10, not 32."""

        build = read(BUILD_PY)
        # The flags are f-strings, e.g.
        #   f"-DLDG_NUM_BLOCKS={_env_int('LDG_NUM_BLOCKS', 80)}",
        grid = re.search(r"LDG_NUM_BLOCKS',\s*(\d+)", build)
        attn = re.search(r"LDG_ATTN_BLOCKS',\s*(\d+)", build)
        self.assertIsNotNone(grid, "LDG_NUM_BLOCKS default not found in build.py")
        self.assertIsNotNone(attn, "LDG_ATTN_BLOCKS default not found in build.py")
        assert grid is not None and attn is not None
        grid_blocks = int(grid.group(1))
        attn_blocks = int(attn.group(1))
        real_limit = grid_blocks // attn_blocks
        self.assertEqual(real_limit, 10)
        # The docs and the compatibility message both quote this number.
        self.assertLess(real_limit, 32)

    def test_backstop_guards_both_batch_launch_sites(self) -> None:
        kernel = read(KERNEL_CU)
        self.assertEqual(kernel.count("ldg_batch_is_launchable(batch_size"), 2)
        # And the backstop itself must not claim a silent failure is fine.
        helper = re.search(
            r"static bool ldg_batch_is_launchable\(.*?\n\}", kernel, re.DOTALL
        )
        self.assertIsNotNone(helper)
        assert helper is not None
        self.assertIn("blocks_per_seq < LDG_ATTN_BLOCKS", helper.group(0))
        self.assertIn("torch-cudagraph", helper.group(0))


class ObservableRejectionTests(unittest.TestCase):
    def test_binding_rejects_before_launching(self) -> None:
        bindings = read(BINDINGS)
        # Both WMMA batch entry points call the check.
        self.assertEqual(bindings.count("check_wmma_batch_supported("), 3)  # def + 2 calls
        # Locate the *definitions* (void / torch::Tensor prefix), not the
        # forward declarations near the top of the file.
        definitions = {
            "decode_paged_batch_wmma": "void decode_paged_batch_wmma(",
            "generate_nosync_paged_batch": "torch::Tensor generate_nosync_paged_batch(",
        }
        for function, marker in definitions.items():
            start = bindings.index(marker)
            # The next top-level definition ends this body.
            following = [
                index
                for index in (
                    bindings.find("\nvoid ", start + 1),
                    bindings.find("\ntorch::Tensor ", start + 1),
                    bindings.find("\nTORCH_LIBRARY_EXPAND", start + 1),
                )
                if index > start
            ]
            self.assertTrue(following, f"could not delimit {function}")
            body = bindings[start : min(following)]
            self.assertIn(
                "check_wmma_batch_supported",
                body,
                f"{function} does not reject an unsupported batch",
            )

    def test_binding_exposes_the_limit_to_python(self) -> None:
        bindings = read(BINDINGS)
        self.assertIn('ops.def("wmma_max_supported_batch() -> int")', bindings)
        self.assertIn("ops.impl(\"wmma_max_supported_batch\"", bindings)

    def test_backend_uses_the_limit_and_falls_back_to_cudagraph(self) -> None:
        backend = read(BACKEND_PY)
        self.assertIn("def wmma_batch_limit(", backend)
        self.assertIn("wmma_batch_limit(max_batch_size)", backend)
        self.assertIn("wmma_max_supported_batch()", backend)
        # The rejection must be a Compatibility(False, ...) so the hybrid
        # dispatch falls back rather than raising mid-decode.
        start = backend.index('== "wmma":')
        block = backend[start : start + 900]
        self.assertIn("wmma_limit", block)
        self.assertIn("return Compatibility(", block)

    def test_backend_limit_falls_back_when_cuda_is_unavailable(self) -> None:
        from nanovllm.backends.megakernel.backend import (
            MAX_BATCH_SIZE,
            wmma_batch_limit,
        )

        # CPU-only path must stay usable (this is how the unit tests run).
        self.assertEqual(wmma_batch_limit(), MAX_BATCH_SIZE)


if __name__ == "__main__":
    unittest.main()
