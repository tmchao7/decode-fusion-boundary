import os
from dataclasses import dataclass
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
    megakernel_verbose: bool = False
    hf_config: AutoConfig | None = None
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
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
