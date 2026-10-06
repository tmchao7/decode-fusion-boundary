from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import AutoConfig


# ``megakernel`` -> the task-graph persistent kernel (docs/11/12/13)
# ``torch``      -> the standard nano-vLLM path.
VALID_DECODE_BACKENDS = {"torch", "megakernel"}


def validate_decode_backend(value: str) -> str:
    if value not in VALID_DECODE_BACKENDS:
        options = ", ".join(sorted(VALID_DECODE_BACKENDS))
        raise ValueError(f"decode_backend must be one of: {options}; got {value!r}")
    return value


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    # Scheduler controls for mixed prefill/decode service behavior. A value of
    # 0 keeps the legacy full token budget for prefill; positive values cap one
    # prefill iteration so long prompts cannot monopolize the engine.
    max_prefill_tokens_per_step: int = 0
    # When both queues are non-empty, run at most this many decode iterations
    # before allowing one prefill iteration. This preserves decode ITL while
    # still making progress on waiting requests.
    prefill_decode_ratio: int = 1
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    decode_backend: str = "torch"
    # Task-graph knobs. ``attn_splits`` selects the static attention split count
    # baked into the schedule; 1 is the validated default (docs/12 §3.4 wants
    # vectorization before more splits).
    megakernel_attn_splits: int = 1
    # Fusion granularity for the task-graph decoder: the step's instruction
    # stream is built as this many contiguous layer segments and each segment is
    # one cooperative launch. 1 = the single-megakernel baseline; its upper bound
    # is the model's layer count, so it is validated where the model is known
    # (TaskGraphDecoder), not here.
    fusion_groups: int = 1
    megakernel_verbose: bool = False
    megakernel_debug_bounds: bool = False
    hf_config: Any | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        self.decode_backend = validate_decode_backend(self.decode_backend)
        if self.max_prefill_tokens_per_step < 0:
            raise ValueError(
                "max_prefill_tokens_per_step must be >= 0; got "
                f"{self.max_prefill_tokens_per_step}"
            )
        if self.prefill_decode_ratio < 1:
            raise ValueError(
                "prefill_decode_ratio must be >= 1; got "
                f"{self.prefill_decode_ratio}"
            )
        if self.megakernel_attn_splits < 1:
            raise ValueError(
                "megakernel_attn_splits must be >= 1; got "
                f"{self.megakernel_attn_splits}"
            )
        if self.fusion_groups < 1:
            raise ValueError(
                f"fusion_groups must be >= 1; got {self.fusion_groups}"
            )
        if not os.path.isdir(self.model):
            raise ValueError(f"model directory does not exist: {self.model}")
        if self.max_num_batched_tokens < 1:
            raise ValueError(
                "max_num_batched_tokens must be >= 1; got "
                f"{self.max_num_batched_tokens}"
            )
        if self.max_num_seqs < 1:
            raise ValueError(f"max_num_seqs must be >= 1; got {self.max_num_seqs}")
        if self.max_model_len < 1:
            raise ValueError(f"max_model_len must be >= 1; got {self.max_model_len}")
        if self.gpu_memory_utilization <= 0.0 or self.gpu_memory_utilization > 1.0:
            raise ValueError(
                "gpu_memory_utilization must be in (0, 1]; got "
                f"{self.gpu_memory_utilization}"
            )
        if self.kvcache_block_size < 256 or self.kvcache_block_size % 256 != 0:
            raise ValueError(
                "kvcache_block_size must be a multiple of 256 and >= 256; got "
                f"{self.kvcache_block_size}"
            )
        if not 1 <= self.tensor_parallel_size <= 8:
            raise ValueError(
                "tensor_parallel_size must be in [1, 8]; got "
                f"{self.tensor_parallel_size}"
            )
        # Keep importing the optional server stack lazy so CPU-only scheduler
        # and configuration tests can run without transformers installed.
        from transformers import AutoConfig

        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
