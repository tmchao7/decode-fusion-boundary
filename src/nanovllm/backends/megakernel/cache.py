"""Shared paged KV-cache contract for nano-vLLM backends.

The contract mirrors ``ModelRunner.allocate_kv_cache``. K and V are passed as
separate views with shape ``[num_layers, num_blocks, block_size, num_kv_heads,
head_dim]``; the leading K/V dimension is owned by the parent cache tensor.
All offsets returned here are element offsets, not byte offsets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PagedKVCacheSpec:
    """Shape, stride, and logical-to-physical mapping rules for KV cache."""

    num_layers: int
    num_blocks: int
    block_size: int
    num_kv_heads: int
    head_dim: int
    dtype_itemsize: int

    def __post_init__(self) -> None:
        fields = (
            ("num_layers", self.num_layers),
            ("num_blocks", self.num_blocks),
            ("num_kv_heads", self.num_kv_heads),
            ("head_dim", self.head_dim),
            ("dtype_itemsize", self.dtype_itemsize),
        )
        for name, value in fields:
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the combined ``[K/V, layer, block, token, head, dim]`` cache."""

        return (
            2,
            self.num_layers,
            self.num_blocks,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
        )

    @property
    def head_stride(self) -> int:
        return self.head_dim

    @property
    def token_stride(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def block_stride(self) -> int:
        return self.block_size * self.token_stride

    @property
    def layer_stride(self) -> int:
        return self.num_blocks * self.block_stride

    @property
    def total_elements(self) -> int:
        return self.num_layers * self.layer_stride

    @property
    def total_bytes(self) -> int:
        return 2 * self.total_elements * self.dtype_itemsize

    def physical_slot(
        self, block_table: Sequence[int], logical_position: int
    ) -> tuple[int, int]:
        """Map a logical token position to ``(physical_block, block_offset)``."""

        if logical_position < 0:
            raise IndexError("position must be non-negative")
        if not block_table:
            raise ValueError("block table must not be empty")
        logical_block, block_offset = divmod(logical_position, self.block_size)
        if logical_block >= len(block_table):
            raise IndexError("position is outside the block table")
        physical_block = int(block_table[logical_block])
        if not 0 <= physical_block < self.num_blocks:
            raise ValueError("block table contains an invalid physical block")
        return physical_block, block_offset

    def element_offset(
        self,
        layer: int,
        block_table: Sequence[int],
        logical_position: int,
        kv_head: int,
        head_dim_index: int,
    ) -> int:
        """Return the flat element offset in one K or V cache view."""

        if not 0 <= layer < self.num_layers:
            raise IndexError("layer is outside the cache")
        if not 0 <= kv_head < self.num_kv_heads:
            raise IndexError("kv_head is outside the cache")
        if not 0 <= head_dim_index < self.head_dim:
            raise IndexError("head dimension is outside the cache")
        physical_block, block_offset = self.physical_slot(
            block_table, logical_position
        )
        return (
            layer * self.layer_stride
            + physical_block * self.block_stride
            + block_offset * self.token_stride
            + kv_head * self.head_stride
            + head_dim_index
        )
