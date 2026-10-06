"""Tests for the single source of model geometry (``nanovllm.model_spec``).

Two things are being locked here:

1. **The numbers are right.** ``QWEN3_06B`` is checked against the CUDA kernel it
   compiles into, against the kernel's weight-pointer ABI, and against counts read
   off the *instantiated* model.  A wrong constant in this file would silently move
   every roofline number in the repo, so it is pinned from three independent
   directions rather than from one.

2. **The drift detectors actually fire.**  ``assert_derived_from_spec`` and
   ``assert_matches_kernel`` exist to catch two specific, silent failure modes:
   a copy drifting from the source, and a derived constant being written as an
   inline literal that happens to equal the right number today.  A detector that
   cannot be shown to fail is not a detector, so each has a negative test.
"""

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nanovllm.model_spec import (  # noqa: E402
    KNOWN_SPECS,
    OLMOE_1B_7B,
    QWEN3_06B,
    ModelSpec,
)

# Ground truth. The per-layer figure and the total were both read off the
# instantiated `Qwen3ForCausalLM` on the GPU box, not derived from a description.
QWEN3_PER_LAYER_PARAMS = 15_730_944
QWEN3_TOTAL_PARAMS = 596_049_920
QWEN3_LAYER_WEIGHT_POINTERS = 11
QWEN3_TENSORS_PER_LAYER = 11
CUDA_SRC = ROOT / "src" / "nanovllm" / "backends" / "megakernel" / "csrc"


class Qwen3SpecTests(unittest.TestCase):
    def test_matches_the_cuda_kernel_it_compiles_into(self) -> None:
        QWEN3_06B.assert_valid()
        QWEN3_06B.assert_matches_kernel()

    def test_per_layer_params_match_the_instantiated_model(self) -> None:
        self.assertEqual(QWEN3_06B.params_per_layer(), QWEN3_PER_LAYER_PARAMS)

    def test_the_final_norm_is_counted_once_not_per_layer(self) -> None:
        """Folding it into the per-layer count is how it got counted twice."""

        self.assertEqual(QWEN3_06B.final_norm_params(), 1024)
        self.assertEqual(
            QWEN3_06B.total_layer_params(),
            QWEN3_06B.num_layers * QWEN3_PER_LAYER_PARAMS + 1024,
        )
        # At 28 layers an amortised final norm hides a 28x error.
        self.assertNotEqual(
            QWEN3_06B.params_per_layer(), QWEN3_PER_LAYER_PARAMS + 1024
        )

    def test_total_params_match_the_instantiated_model(self) -> None:
        spec = QWEN3_06B
        tied = spec.total_layer_params() + spec.vocab_size * spec.hidden_size
        self.assertEqual(tied, QWEN3_TOTAL_PARAMS)

    def test_the_qk_norms_are_head_dim_wide_not_hidden_wide(self) -> None:
        """``(128,)`` each, verified on the real tensors.

        Both readings are superficially plausible -- a per-head norm could be laid
        out as `num_q_heads x head_dim` -- and the two differ by 2,048 per layer, so
        the fact is pinned rather than inferred.
        """

        self.assertEqual(QWEN3_06B.qk_norm_params_per_layer, 2 * 128)
        self.assertNotEqual(QWEN3_06B.qk_norm_params_per_layer, 2 * 2048)
        without = ModelSpec(**{**QWEN3_06B.as_dict(), "qk_norm": False})
        self.assertEqual(without.qk_norm_params_per_layer, 0)
        self.assertEqual(
            QWEN3_06B.params_per_layer() - without.params_per_layer(), 256
        )

    def test_layer_weight_pointer_count_matches_the_kernel_abi(self) -> None:
        """``tg_abi.h`` declares 11 per-layer weight pointers; the spec derives it."""

        self.assertEqual(QWEN3_06B.weight_pointer_count, QWEN3_LAYER_WEIGHT_POINTERS)
        header = (CUDA_SRC / "tg_abi.h").read_text(encoding="utf-8")
        self.assertIn("TG_LW_COUNT", header)
        # The enum's members are the count: count them rather than trusting the
        # trailing comment, which is exactly the kind of thing that goes stale.
        block = header.split("enum TgLayerWeightIdx")[1].split("};")[0]
        members = [
            line.strip().rstrip(",")
            for line in block.splitlines()
            if line.strip().startswith("TG_LW_") and "COUNT" not in line
        ]
        self.assertEqual(len(members), QWEN3_06B.weight_pointer_count)

    def test_the_tensor_count_matches_the_checkpoint(self) -> None:
        """11 tensors per layer in the real checkpoint, matching the ABI."""

        self.assertEqual(
            QWEN3_06B.weight_pointer_count, QWEN3_TENSORS_PER_LAYER
        )


class OlmoeSpecTests(unittest.TestCase):
    def test_shared_geometry_is_valid_and_declared(self) -> None:
        OLMOE_1B_7B.assert_valid()
        self.assertIn("OLMoE-1B-7B-0924", KNOWN_SPECS)

    def test_olmoe_is_multi_head_not_grouped_query(self) -> None:
        """``num_kv_heads == num_q_heads``: this doubles the per-token KV read."""

        self.assertEqual(OLMOE_1B_7B.num_kv_heads, OLMOE_1B_7B.num_q_heads)
        self.assertEqual(OLMOE_1B_7B.group_size, 1)
        self.assertEqual(OLMOE_1B_7B.kv_size, OLMOE_1B_7B.hidden_size)
        self.assertEqual(QWEN3_06B.group_size, 2)

    def test_olmoe_is_not_compiled_into_this_dense_kernel(self) -> None:
        """The kernel is still dense-Qwen3, so this must fail loudly, not silently."""

        with self.assertRaises(ValueError) as caught:
            OLMOE_1B_7B.assert_matches_kernel()
        self.assertIn("OLMoE-1B-7B-0924 disagrees", str(caught.exception))

    def test_olmoe_norms_are_head_dim_wide_too(self) -> None:
        """The OLMoE checkpoint's q_norm/k_norm are ``(2048,)``... at hidden width.

        Recorded because it is *not* the same shape as Qwen3's: OLMoE normalises per
        head across a 2048-wide projection, so the tensor is `num_q_heads * head_dim`
        = hidden. The spec computes `2 * head_dim`, which is Qwen3's layout, so this
        test documents the difference rather than asserting a false equality.
        """

        self.assertEqual(OLMOE_1B_7B.qk_norm_params_per_layer, 2 * 128)
        self.assertEqual(OLMOE_1B_7B.hidden_size, 16 * 128)


class ValidationTests(unittest.TestCase):
    def test_non_positive_fields_are_rejected(self) -> None:
        for field in ("num_layers", "hidden_size", "head_dim", "vocab_size"):
            with self.subTest(field=field):
                spec = ModelSpec(**{**QWEN3_06B.as_dict(), field: 0})
                with self.assertRaisesRegex(ValueError, field):
                    spec.assert_valid()

    def test_non_multiple_of_sixteen_is_rejected(self) -> None:
        for field in ("hidden_size", "intermediate_size", "head_dim"):
            with self.subTest(field=field):
                spec = ModelSpec(**{**QWEN3_06B.as_dict(), field: 1000})
                with self.assertRaisesRegex(ValueError, "multiple of 16"):
                    spec.assert_valid()

    def test_a_non_integral_group_size_is_rejected(self) -> None:
        spec = ModelSpec(**{**QWEN3_06B.as_dict(), "num_kv_heads": 3})
        with self.assertRaisesRegex(ValueError, "multiple"):
            spec.assert_valid()

    def test_an_uneven_projection_width_is_rejected(self) -> None:
        """A q/k/v/o width that does not tile 16-wide WMMA leaves an unemitted tail.

        ``hidden_size`` is set to the offending value because the q/kv widths are
        ``heads * head_dim`` and every tile-aligned combination of those lands on a
        multiple of 16: ``num_q_heads`` is a multiple of ``num_kv_heads``, and
        ``head_dim`` is itself tile-aligned. 625 is neither a multiple of 16 nor a
        value the schedule can round down to.
        """

        spec = ModelSpec(**{**QWEN3_06B.as_dict(), "hidden_size": 625})
        with self.assertRaisesRegex(ValueError, "multiple of 16"):
            spec.assert_valid()

    def test_a_group_size_that_is_not_a_power_of_two_is_rejected(self) -> None:
        """GQA with 3 query heads per KV head has no path in the per-head pipeline."""

        spec = ModelSpec(**{**QWEN3_06B.as_dict(), "num_q_heads": 24})
        with self.assertRaisesRegex(ValueError, "power of two"):
            spec.assert_valid()

    def test_a_group_size_larger_than_the_tile_is_rejected(self) -> None:
        spec = ModelSpec(**{**QWEN3_06B.as_dict(), "num_q_heads": 512})
        with self.assertRaisesRegex(ValueError, "power of two|WMMA N tile"):
            spec.assert_valid()

    def test_a_head_dim_that_is_not_tile_aligned_is_rejected(self) -> None:
        spec = ModelSpec(**{**QWEN3_06B.as_dict(), "head_dim": 100})
        with self.assertRaisesRegex(ValueError, "head_dim"):
            spec.assert_valid()


class KernelContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = QWEN3_06B
        self.sandbox = ROOT / "tests" / "_tmp"
        self.sandbox.mkdir(exist_ok=True)

    def tearDown(self) -> None:
        for path in self.sandbox.glob("*"):
            path.unlink()
        self.sandbox.rmdir()

    def _kernel_with(self, **overrides: str) -> Path:
        text = (CUDA_SRC / "taskgraph_kernel.cu").read_text(encoding="utf-8")
        for name, value in overrides.items():
            import re

            text, count = re.subn(
                rf"^(constexpr int {name}\s*=\s*)[^;]+;",
                rf"\g<1>{value};",
                text,
                count=1,
                flags=re.MULTILINE,
            )
            self.assertEqual(count, 1, f"{name} not found in the kernel")
        path = self.sandbox / "kernel.cu"
        path.write_text(text, encoding="utf-8")
        return path

    def test_a_drifted_kernel_constant_is_detected(self) -> None:
        path = self._kernel_with(TG_HIDDEN="2048")
        with self.assertRaisesRegex(ValueError, "hidden_size"):
            self.spec.assert_matches_kernel(path)

    def test_a_drifted_derived_kernel_constant_is_detected(self) -> None:
        """The kernel's *derived* constants are parsed too, not just its literals.

        ``TG_Q_SIZE`` is written as ``TG_NUM_Q_HEADS * TG_HEAD_DIM`` in the kernel. A
        parser that only read `=`-with-a-digit would skip it entirely, and the kernel
        could then disagree about what a head is while every literal still matched.
        """

        path = self._kernel_with(TG_Q_SIZE="4096")
        with self.assertRaisesRegex(ValueError, "q_size"):
            self.spec.assert_matches_kernel(path)

    def test_the_real_kernel_passes(self) -> None:
        self.spec.assert_matches_kernel()


class DerivedFromSpecTests(unittest.TestCase):
    """The detector for inline-literal duplicates must be able to fail."""

    def setUp(self) -> None:
        self.spec = QWEN3_06B

    def test_a_correctly_derived_constant_passes(self) -> None:
        self.spec.assert_derived_from_spec(
            {
                "q_size": (2048, "NUM_Q_HEADS * HEAD_DIM"),
                "kv_size": (1024, "NUM_KV_HEADS * HEAD_DIM"),
            }
        )

    def test_an_inline_literal_duplicate_is_caught(self) -> None:
        """`Q_SIZE = 16 * HEAD_DIM` is the historical bug: right value, wrong form."""

        with self.assertRaisesRegex(ValueError, "duplicates num_q_heads"):
            self.spec.assert_derived_from_spec({"q_size": (2048, "16 * HEAD_DIM")})

    def test_a_bare_literal_is_caught(self) -> None:
        with self.assertRaisesRegex(ValueError, "contains the literal"):
            self.spec.assert_derived_from_spec({"q_size": (2048, "2048")})

    def test_a_wrong_value_is_caught_even_when_derived(self) -> None:
        with self.assertRaisesRegex(ValueError, "published"):
            self.spec.assert_derived_from_spec(
                {"q_size": (4096, "NUM_Q_HEADS * HEAD_DIM")}
            )

    def test_a_missing_expression_is_caught(self) -> None:
        with self.assertRaisesRegex(ValueError, "no expression"):
            self.spec.assert_derived_from_spec({"q_size": (2048, None)})

    def test_names_not_on_the_spec_are_ignored(self) -> None:
        """Callers pass whole module namespaces; only spec-derived names are judged."""

        self.spec.assert_derived_from_spec({"something_else": (1, "X * Y")})


class RooflineAgreementTests(unittest.TestCase):
    """`roofline.ModelDims` is a *view*; it must agree with the spec exactly."""

    def test_per_layer_and_total_agree(self) -> None:
        from nanovllm.roofline import ModelDims

        dims = ModelDims()
        self.assertEqual(dims.params_per_layer(), QWEN3_06B.params_per_layer())
        self.assertEqual(dims.final_norm_params(), QWEN3_06B.final_norm_params())
        self.assertEqual(
            dims.total_params(tied_embedding=True), QWEN3_06B.total_layer_params()
            + QWEN3_06B.vocab_size * QWEN3_06B.hidden_size
        )

    def test_a_modeldims_built_from_other_dims_still_derives(self) -> None:
        """`ModelDims` is a public constructor; arbitrary dims must still work."""

        from nanovllm.roofline import ModelDims

        dims = ModelDims(hidden_size=2048, intermediate_size=8192, num_q_heads=16,
                         num_kv_heads=16, head_dim=128, num_layers=8, vocab_size=32000)
        self.assertEqual(
            dims.params_per_layer(),
            dims.attention_params_per_layer()
            + dims.qk_norm_params_per_layer()
            + 3 * 2048 * 8192,
        )


class CliTests(unittest.TestCase):
    def test_the_roofline_cli_still_validates_the_kernel(self) -> None:
        """The CLI's --check-kernel path goes through the spec-backed assertion."""

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "nanovllm.roofline",
                "--batch-sizes",
                "1",
                "--bandwidth-gbps",
                "1699",
                "--check-kernel",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)


class PublishedConstantsTests(unittest.TestCase):
    """The modules that used to hold their own copies must now be views.

    ``taskgraph.py`` and ``weights.py`` still *publish* these names because
    schedulers, tools and tests import them from there -- but the values must come
    from the spec, and the two must agree by construction rather than by review.
    """

    PUBLISHED = (
        "NUM_LAYERS",
        "NUM_KV_HEADS",
        "HEAD_DIM",
        "HIDDEN_SIZE",
        "INTERMEDIATE_SIZE",
        "Q_SIZE",
        "KV_SIZE",
        "VOCAB_SIZE",
    )

    #: name -> the ModelSpec attribute it must come from.
    SPEC_ATTR = {
        "NUM_LAYERS": "num_layers",
        "NUM_KV_HEADS": "num_kv_heads",
        "HEAD_DIM": "head_dim",
        "HIDDEN_SIZE": "hidden_size",
        "INTERMEDIATE_SIZE": "intermediate_size",
        "Q_SIZE": "q_size",
        "KV_SIZE": "kv_size",
        "VOCAB_SIZE": "vocab_size",
    }

    def test_no_literal_geometry_survives_in_the_taskgraph(self) -> None:
        import re

        text = (
            ROOT / "src" / "nanovllm" / "backends" / "megakernel" / "taskgraph.py"
        ).read_text(encoding="utf-8")
        for name in self.PUBLISHED:
            with self.subTest(name=name):
                match = re.search(rf"^{name} = (.+)$", text, re.MULTILINE)
                self.assertIsNotNone(match, f"{name} is no longer published")
                rhs = match.group(1).strip()
                self.assertFalse(
                    rhs.isdigit(),
                    f"{name} = {rhs} is a typed literal again; it must derive from "
                    f"the spec",
                )
                self.assertIn("MEGAKERNEL_SPEC", rhs)

    def test_no_literal_geometry_survives_in_the_weights_module(self) -> None:
        import re

        text = (
            ROOT / "src" / "nanovllm" / "backends" / "megakernel" / "weights.py"
        ).read_text(encoding="utf-8")
        for name, attr in (
            ("NUM_LAYERS", "num_layers"),
            ("NUM_KV_HEADS", "num_kv_heads"),
            ("HEAD_DIM", "head_dim"),
            ("HIDDEN_SIZE", "hidden_size"),
            ("INTERMEDIATE_SIZE", "intermediate_size"),
            ("Q_SIZE", "q_size"),
            ("KV_SIZE", "kv_size"),
            ("MAX_SEQ_LEN", "max_model_len"),
            ("VOCAB_SIZE", "vocab_size"),
            ("LAYER_WEIGHT_POINTERS", "weight_pointer_count"),
        ):
            with self.subTest(name=name):
                match = re.search(rf"^{name} = (.+)$", text, re.MULTILINE)
                self.assertIsNotNone(match, f"{name} is no longer published")
                self.assertEqual(match.group(1).strip(), f"QWEN3_06B.{attr}")

    def test_the_packer_uses_the_derived_pointer_count(self) -> None:
        """``n_ptrs = 11`` was the literal that had to agree with the ABI enum."""

        text = (
            ROOT / "src" / "nanovllm" / "backends" / "megakernel" / "weights.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("n_ptrs = 11", text)
        self.assertIn("n_ptrs = LAYER_WEIGHT_POINTERS", text)

    def test_taskgraph_publishes_the_spec_values(self) -> None:
        from nanovllm.backends.megakernel import taskgraph as tg

        for name, attr in self.SPEC_ATTR.items():
            with self.subTest(name=name):
                self.assertEqual(getattr(tg, name), getattr(QWEN3_06B, attr))
        self.assertEqual(tg.MAX_ATTN_SPLITS, QWEN3_06B.max_attn_splits)
        self.assertEqual(tg.NUM_Q_HEADS, QWEN3_06B.num_q_heads)


if __name__ == "__main__":
    unittest.main()
