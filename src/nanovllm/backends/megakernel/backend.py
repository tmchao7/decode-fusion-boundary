"""Optional Qwen3 task-graph Megakernel decode backend.

The CUDA extension is imported lazily so the regular nano-vLLM backend remains
usable on machines without CUDA, and so importing the package never triggers a
CUDA build. The mainline backend is the persistent task-graph kernel in
``csrc/taskgraph_kernel.cu``. It supports Qwen3-0.6B greedy decode with a
compile-time batch ceiling; callers should use :meth:`compatibility` before
invoking :meth:`decode` or :meth:`decode_batch`.
"""

from dataclasses import dataclass
from typing import Any

from nanovllm.backends.megakernel.taskgraph import taskgraph_compatibility


VALID_CUDA_ARCHES = {"sm_70", "sm_80", "sm_86", "sm_89", "sm_90", "sm_120"}
MAX_BATCH_SIZE = 8

# The public Megakernel name intentionally maps to one implementation only.
MEGAKERNEL_BACKENDS = ("megakernel",)


def is_megakernel_backend(value: Any) -> bool:
    """Whether ``value`` selects a megakernel decode path."""

    return value in MEGAKERNEL_BACKENDS



# Fixed-shape contract baked into the fused task-graph CUDA kernel. This is the
# Python-side source of truth for what the active backend was compiled for;
# compatibility() guards that the loaded nano-vLLM model matches it.
QWEN3_CONTRACT = {
    "model_type": "qwen3",
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "vocab_size": 151936,
    "max_model_len": 2048,
    "head_dim": 128,
}


def qwen3_contract_mismatch(config) -> str | None:
    """Name of the first :data:`QWEN3_CONTRACT` field the config violates.

    Shared by both megakernel paths: the frozen kernel and the task-graph kernel
    are compiled for the same Qwen3-0.6B shapes, so a mismatch means neither can
    run and the caller must stay on torch.
    """

    hf_config = _get_attr(config, "hf_config")
    if _get_attr(hf_config, "tie_word_embeddings", True) is not True:
        return "tie_word_embeddings"
    if _get_attr(hf_config, "attention_bias", False) is not False:
        return "attention_bias"
    for name, expected in QWEN3_CONTRACT.items():
        if name == "max_model_len":
            actual = _get_attr(config, "max_model_len", 2048)
        else:
            actual = (
                _get_attr(hf_config, "head_dim", None)
                if name == "head_dim"
                else _get_attr(hf_config, name)
            )
            if name == "head_dim" and actual is None:
                actual = hf_config.hidden_size // hf_config.num_attention_heads
        if actual != expected:
            return name
    return None



@dataclass(frozen=True)
class Compatibility:
    supported: bool
    reason: str = ""


def cuda_arch_for_capability(capability: tuple[int, int]) -> str:
    """Return the CUDA ``-arch`` value for a device capability.

    The Megakernel currently uses CUDA features available from Volta onward;
    the runtime backend still performs a stricter BF16 capability check before
    execution. Keeping this mapping separate makes build configuration easy to
    unit test without importing CUDA.
    """

    major, minor = capability
    arch = f"sm_{major}{minor}"
    if arch not in VALID_CUDA_ARCHES:
        raise ValueError(f"unsupported CUDA capability: {major}.{minor}")
    return arch


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


class MegakernelBackend:
    """Lazy adapter around the packaged single-token decoder."""

    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.decoder = None
        self.kv_cache = None
        self.disabled_reason = None

    @staticmethod
    def compatibility(config, seqs, is_prefill: bool) -> Compatibility:
        decision = taskgraph_compatibility(config, seqs, is_prefill)
        if not decision.supported:
            return Compatibility(False, decision.reason)

        mismatch = qwen3_contract_mismatch(config)
        if mismatch is not None:
            return Compatibility(False, f"unsupported Qwen config: {mismatch}")
        return Compatibility(True)

    def set_kv_cache(self, kv_cache) -> None:
        """Attach the nano-vLLM-owned ``[K/V, layer, block, token, head, dim]`` cache."""

        if kv_cache is None or kv_cache.dim() != 6 or kv_cache.shape[0] != 2:
            raise ValueError("kv_cache must have shape [2, layers, blocks, tokens, heads, dim]")
        self.kv_cache = kv_cache

    def disable(self, reason: str) -> None:
        self.disabled_reason = reason

    @property
    def ready(self) -> bool:
        return self.decoder is not None

    def ensure_ready(self) -> None:
        """Build the decoder eagerly so the first decode has no cold-start spike.

        Called at ModelRunner startup after the KV cache is attached. On any
        failure the caller should ``disable()`` so the torch path remains the
        integration fallback.
        """
        self._ensure_decoder()

    def _validate_cache_layout(self) -> None:
        """Validate nano-vLLM's paged cache against the kernel's offset contract.

        The fused kernel (``ldg_kv_ptr``) computes flat element offsets assuming
        a single contiguous ``[layers, blocks, block_size, kv_heads, head_dim]``
        K/V view per cache with layer stride ``num_blocks * block_stride``.
        ``PagedKVCacheSpec`` is the canonical description of that layout; check
        the actual attached tensors against it so any layout drift surfaces at
        init instead of producing silently-wrong attention.
        """
        from nanovllm.backends.megakernel.cache import PagedKVCacheSpec

        kv = self.kv_cache
        hf = self.config.hf_config
        head_dim = getattr(hf, "head_dim", None) or (
            hf.hidden_size // hf.num_attention_heads
        )
        spec = PagedKVCacheSpec(
            num_layers=hf.num_hidden_layers,
            num_blocks=kv.shape[2],
            block_size=self.config.kvcache_block_size,
            num_kv_heads=kv.shape[4],
            head_dim=kv.shape[5],
            dtype_itemsize=kv.element_size(),
        )
        expected = spec.shape  # (2, layers, blocks, bs, kv_heads, head_dim)
        if tuple(kv.shape) != expected:
            raise RuntimeError(
                f"paged KV cache shape {tuple(kv.shape)} does not match the "
                f"megakernel contract {expected}"
            )
        if not (kv[0].is_contiguous() and kv[1].is_contiguous()):
            raise RuntimeError(
                "paged KV cache views must be contiguous for the megakernel "
                "layer-stride offset math"
            )
        if kv[0].size(0) * spec.layer_stride != kv[0].numel():
            raise RuntimeError(
                "paged KV cache layer stride does not match the megakernel "
                "offset contract"
            )
        if head_dim != QWEN3_CONTRACT["head_dim"]:
            raise RuntimeError("Qwen head_dim differs from the megakernel contract")

    def _ensure_decoder(self):
        if self.disabled_reason:
            raise RuntimeError(self.disabled_reason)
        if self.decoder is not None:
            return
        if self.kv_cache is None:
            raise RuntimeError("nano-vLLM KV cache has not been attached")

        # Keep the extension optional and isolated from the normal import path.
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Megakernel requires a CUDA device")
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            raise RuntimeError(
                "Megakernel v1 requires an A100-class BF16 GPU (sm_80+); "
                f"detected sm_{capability[0]}{capability[1]}"
            )

        self._validate_cache_layout()

        from .build import get_extension

        get_extension()
        from .taskgraph import TaskGraphDecoder
        from .weights import weights_from_nano_model

        weights = weights_from_nano_model(self.model, self.config.hf_config)
        self.decoder = TaskGraphDecoder(
            weights,
            k_cache=self.kv_cache[0],
            v_cache=self.kv_cache[1],
            block_size=self.config.kvcache_block_size,
            num_cache_blocks=self.kv_cache.shape[2],
            max_model_len=_get_attr(self.config, "max_model_len", 2048),
            attn_splits=int(_get_attr(self.config, "megakernel_attn_splits", 1)),
            verbose=bool(_get_attr(self.config, "megakernel_verbose", False)),
        )

    def decode_batch(self, input_ids, positions, block_tables) -> list[int]:
        """Decode one token for each scheduler-selected sequence."""

        self._ensure_decoder()
        return self.decoder.decode_rows(input_ids, positions, block_tables)

    def decode(self, seq, block_table) -> list[int]:
        """Decode one token and return a one-element Python token-id list.

        The caller supplies the current nano-vLLM block table. The decoder
        writes the new K/V values into the scheduler-owned paged cache and
        reads prior positions through that same cache; no prompt replay or
        private contiguous cache is used.
        """

        self._ensure_decoder()
        return [
            int(
                self.decoder.step(
                    seq.last_token,
                    position=len(seq) - 1,
                    block_table=block_table,
                )
            )
        ]
