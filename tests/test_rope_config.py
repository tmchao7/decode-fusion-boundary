"""CPU-safe tests for the shared RoPE config resolver.

The nano-vLLM Qwen3 model and the Megakernel weight loader used to resolve
``rope_theta`` independently with opposite precedence and different defaults,
so a checkpoint where the keys disagree -- or where both are absent -- silently
got two different cos/sin tables.  These tests pin the single shared rule.
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nanovllm.rope_config import (  # noqa: E402
    QWEN3_DEFAULT_ROPE_THETA,
    assert_supported_rope_type,
    resolve_rope_theta,
    resolve_rope_type,
)


class ResolveRopeThetaTests(unittest.TestCase):
    def test_shipped_qwen3_config(self) -> None:
        # The real Qwen3-0.6B config.json: top-level rope_theta, rope_scaling
        # is null.  This is the case every run actually hits.
        config = SimpleNamespace(rope_theta=1000000, rope_scaling=None)
        self.assertEqual(resolve_rope_theta(config), 1000000.0)

    def test_top_level_wins_over_nested(self) -> None:
        config = SimpleNamespace(
            rope_theta=1000000,
            rope_scaling={"rope_theta": 10000, "rope_type": "default"},
        )
        self.assertEqual(resolve_rope_theta(config), 1000000.0)

    def test_nested_is_honoured_when_toplevel_is_absent(self) -> None:
        config = SimpleNamespace(
            rope_scaling={"rope_theta": 500000, "rope_type": "default"}
        )
        self.assertEqual(resolve_rope_theta(config), 500000.0)

    def test_default_is_qwen3_theta_not_llama(self) -> None:
        # 1e6, not the 1e4 that the megakernel resolver used to default to.
        self.assertEqual(resolve_rope_theta(SimpleNamespace()), 1e6)
        self.assertEqual(
            resolve_rope_theta(SimpleNamespace()), QWEN3_DEFAULT_ROPE_THETA
        )

    def test_explicit_default_overrides(self) -> None:
        self.assertEqual(resolve_rope_theta(SimpleNamespace(), 10000.0), 10000.0)

    def test_rope_parameters_spelling(self) -> None:
        # Newer transformers rename the field; honour it so a config that only
        # carries the new name does not silently fall back to the default.
        self.assertEqual(
            resolve_rope_theta(SimpleNamespace(rope_parameters={"rope_theta": 1234.0})),
            1234.0,
        )

    def test_accepts_a_plain_dict(self) -> None:
        self.assertEqual(resolve_rope_theta({"rope_theta": 1000000}), 1000000.0)
        self.assertEqual(resolve_rope_theta({"rope_scaling": None}), 1e6)

    def test_bool_is_not_a_frequency(self) -> None:
        # ``rope_theta: true`` is a malformed config, not 1.0 Hz.
        self.assertEqual(resolve_rope_theta(SimpleNamespace(rope_theta=True)), 1e6)


class RopeTypeTests(unittest.TestCase):
    def test_default_and_absent_are_supported(self) -> None:
        self.assertIsNone(resolve_rope_type(SimpleNamespace()))
        self.assertIsNone(resolve_rope_type(SimpleNamespace(rope_scaling=None)))
        assert_supported_rope_type(SimpleNamespace(rope_scaling=None))
        assert_supported_rope_type(
            SimpleNamespace(rope_scaling={"rope_type": "default"})
        )

    def test_unsupported_scaling_raises(self) -> None:
        # Silently ignoring YaRN/linear/dynamic would produce unscaled
        # positions -- quiet wrong output is worse than an error.
        for rope_type in ("yarn", "linear", "dynamic", "longrope", "llama3"):
            with self.subTest(rope_type=rope_type):
                config = SimpleNamespace(rope_scaling={"rope_type": rope_type})
                self.assertEqual(resolve_rope_type(config), rope_type)
                with self.assertRaises(NotImplementedError):
                    assert_supported_rope_type(config)

    def test_legacy_type_key_is_read(self) -> None:
        config = SimpleNamespace(rope_scaling={"type": "linear"})
        self.assertEqual(resolve_rope_type(config), "linear")
        with self.assertRaises(NotImplementedError):
            assert_supported_rope_type(config)


class RopeWiringTests(unittest.TestCase):
    """Both backends must resolve RoPE through the one shared helper.

    The fix is only real if neither path keeps its own resolution: an override
    left in ``Qwen3Attention.__init__`` would restore the old "rope_scaling
    wins" precedence and diverge from the Megakernel loader again.
    """

    def test_qwen3_model_uses_the_shared_resolver(self) -> None:
        text = (REPO_ROOT / "src" / "nanovllm" / "models" / "qwen3.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("resolve_rope_theta(config)", text)
        self.assertIn("assert_supported_rope_type(config)", text)
        # No second, independent resolution in the attention module.
        self.assertNotIn('rope_scaling.get("rope_theta"', text)

    def test_megakernel_loader_uses_the_shared_resolver(self) -> None:
        text = (
            REPO_ROOT
            / "src"
            / "nanovllm"
            / "backends"
            / "megakernel"
            / "model.py"
        ).read_text(encoding="utf-8")
        self.assertIn("resolve_rope_theta", text)
        self.assertIn("assert_supported_rope_type", text)
        # The old hand-rolled top-level-first reader must be gone.
        self.assertNotIn('getattr(config, "rope_theta", None)', text)


if __name__ == "__main__":
    unittest.main()
