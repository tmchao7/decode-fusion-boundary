"""Resolve RoPE settings from a HuggingFace config, once, for every path.

The nano-vLLM Qwen3 model and the Megakernel weight loader must build the
*same* cos/sin tables.  They used to resolve ``rope_theta`` independently with
opposite precedence and different defaults, so any checkpoint where the two
keys disagree -- or where both are absent -- silently produced different RoPE
between the two backends.

This module is deliberately torch-free so the resolution rules are unit
testable on a laptop.

Precedence (HuggingFace semantics):

1. top-level ``rope_theta`` (the modern field, and what the shipped
   Qwen3-0.6B config.json uses, with ``rope_scaling: null``);
2. a nested ``rope_theta`` inside ``rope_scaling`` / ``rope_parameters``
   (older checkpoints that embedded it there);
3. ``default`` -- Qwen3's 1e6, not the 1e4 used by older Llama-style models.
"""

from __future__ import annotations

from typing import Any

QWEN3_DEFAULT_ROPE_THETA = 1_000_000.0

_NESTED_KEYS = ("rope_theta", "rope_parameters")
_SUPPORTED_ROPE_TYPES = (None, "default")


def _as_float(value: Any) -> float | None:
    # bool is an int subclass; a config flag is never a frequency.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def resolve_rope_theta(
    config: Any, default: float = QWEN3_DEFAULT_ROPE_THETA
) -> float:
    """Return the RoPE base frequency for ``config``.

    Accepts either an object (transformers config) or a plain dict so the
    resolution can be tested and reused without transformers.
    """

    getter = config.get if isinstance(config, dict) else lambda key, _d=None: getattr(
        config, key, _d
    )
    for key in _NESTED_KEYS:
        value = _as_float(getter(key))
        if value is not None:
            return value
    scaling = getter("rope_scaling") or getter("rope_parameters")
    if isinstance(scaling, dict):
        for key in _NESTED_KEYS:
            value = _as_float(scaling.get(key))
            if value is not None:
                return value
    return float(default)


def resolve_rope_type(config: Any) -> str | None:
    """Return the configured ``rope_type`` (``None`` when unset)."""

    getter = config.get if isinstance(config, dict) else lambda key, _d=None: getattr(
        config, key, _d
    )
    scaling = getter("rope_scaling") or getter("rope_parameters")
    if not isinstance(scaling, dict):
        return None
    rope_type = scaling.get("rope_type") or scaling.get("type")
    return None if rope_type is None else str(rope_type)


def assert_supported_rope_type(config: Any) -> None:
    """Raise for a RoPE scaling this project does not implement.

    YaRN / linear / dynamic / longrope / llama3 scaling was silently ignored:
    both code paths only implement unscaled RoPE, so such a checkpoint would
    produce quietly wrong positions instead of an error.  Fail loudly.
    """

    rope_type = resolve_rope_type(config)
    if rope_type is None or rope_type.lower() in _SUPPORTED_ROPE_TYPES:
        return
    raise NotImplementedError(
        f"rope_type={rope_type!r} is not supported: nano-vLLM's Qwen3 model and "
        "the Megakernel cos/sin tables only implement unscaled RoPE. Implement "
        "the scaling in both places together, or use a checkpoint with "
        "rope_type='default'."
    )
