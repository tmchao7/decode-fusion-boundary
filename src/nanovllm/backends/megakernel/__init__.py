"""Qwen3-0.6B Megakernel backend and CUDA extension helpers."""

from .backend import (
    Compatibility,
    MegakernelBackend,
    cuda_arch_for_capability,
    is_megakernel_backend,
    supports_no_sync_burst,
)
from .cache import PagedKVCacheSpec
from .taskgraph import (
    TG_MAX_BATCH,
    TaskGraphCompatibility,
    taskgraph_compatibility,
)

__all__ = [
    "Compatibility",
    "Decoder",
    "MegakernelBackend",
    "PagedKVCacheSpec",
    "TG_MAX_BATCH",
    "TaskGraphCompatibility",
    "cuda_arch_for_capability",
    "generate",
    "is_megakernel_backend",
    "load_weights",
    "supports_no_sync_burst",
    "taskgraph_compatibility",
    "weights_from_nano_model",
]


def __getattr__(name):
    if name in {"Decoder", "generate", "load_weights", "weights_from_nano_model"}:
        from .build import get_extension

        get_extension()
        from . import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
