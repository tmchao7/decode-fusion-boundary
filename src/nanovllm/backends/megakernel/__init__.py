"""Qwen3-0.6B Megakernel backend and CUDA extension helpers."""

from .backend import (
    Compatibility,
    MegakernelBackend,
    cuda_arch_for_capability,
    is_megakernel_backend,
)
from .cache import PagedKVCacheSpec
from .taskgraph import (
    TG_MAX_BATCH,
    TaskGraphCompatibility,
    taskgraph_compatibility,
)

__all__ = [
    "Compatibility",
    "MegakernelBackend",
    "PagedKVCacheSpec",
    "TG_MAX_BATCH",
    "TaskGraphCompatibility",
    "cuda_arch_for_capability",
    "is_megakernel_backend",
    "taskgraph_compatibility",
]
