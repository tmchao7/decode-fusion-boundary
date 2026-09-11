import os
from dataclasses import dataclass
from transformers import AutoConfig


VALID_DECODE_BACKENDS = {"torch", "megakernel"}
VALID_MEGAKERNEL_BATCH_KERNELS = {"scalar", "wmma"}


def validate_decode_backend(value: str) -> str:
    if value not in VALID_DECODE_BACKENDS:
        options = ", ".join(sorted(VALID_DECODE_BACKENDS))
        raise ValueError(f"decode_backend must be one of: {options}; got {value!r}")
    return value


def validate_megakernel_batch_kernel(value: str) -> str:
    if value not in VALID_MEGAKERNEL_BATCH_KERNELS:
        options = ", ".join(sorted(VALID_MEGAKERNEL_BATCH_KERNELS))
        raise ValueError(
            f"megakernel_batch_kernel must be one of: {options}; got {value!r}"
        )
    return value


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    decode_backend: str = "torch"
    megakernel_batch_kernel: str = "scalar"
    megakernel_burst: bool = True
    megakernel_burst_cap: int = 64
    megakernel_burst_batch: int = 8
    # When False (default), batch>1 decode falls back to cudagraph and only
    # batch=1 uses the megakernel persistent kernel — the hybrid architecture
    # validated by the crossover scan (docs/07 §9.6: megakernel wins batch=1,
    # loses batch>=2). Set True to force the megakernel batch (WMMA) path for
    # batch>1, used for correctness research (tools/correctness_wmma_vs_scalar).
    megakernel_batch_decode: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        self.decode_backend = validate_decode_backend(self.decode_backend)
        self.megakernel_batch_kernel = validate_megakernel_batch_kernel(
            self.megakernel_batch_kernel
        )
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
