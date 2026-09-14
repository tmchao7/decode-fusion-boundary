"""Qwen3-0.6B Megakernel backend and CUDA extension helpers."""

from .backend import Compatibility, MegakernelBackend, cuda_arch_for_capability
from .cache import PagedKVCacheSpec

__all__ = [
    "Compatibility",
    "Decoder",
    "MegakernelBackend",
    "PagedKVCacheSpec",
    "cuda_arch_for_capability",
    "generate",
    "load_weights",
    "weights_from_nano_model",
]


def __getattr__(name):
    if name in {"Decoder", "generate", "load_weights", "weights_from_nano_model"}:
        from .build import get_extension

        get_extension()
        from . import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
