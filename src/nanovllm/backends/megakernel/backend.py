"""Optional Qwen3 Megakernel decode backend.

The CUDA extension is imported lazily so the regular nano-vLLM backend remains
usable on machines without CUDA, and so importing the package never triggers a
CUDA build. The first implementation intentionally handles only single-sequence
Qwen3-0.6B greedy decode; callers should use :meth:`compatibility` before
invoking :meth:`decode` and fall back to the normal model path otherwise.
"""

from dataclasses import dataclass
from typing import Any


VALID_CUDA_ARCHES = {"sm_70", "sm_80", "sm_86", "sm_89", "sm_90", "sm_120"}


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
        if is_prefill:
            return Compatibility(False, "Megakernel is decode-only; use the prefill backend")
        if len(seqs) != 1:
            return Compatibility(False, "Megakernel v1 supports one decode sequence")
        if _get_attr(config, "max_num_seqs", 1) != 1:
            return Compatibility(
                False,
                "Megakernel v1 requires max_num_seqs=1 while paged KV sync is pending",
            )
        if _get_attr(config, "tensor_parallel_size", 1) != 1:
            return Compatibility(False, "Megakernel v1 requires tensor_parallel_size=1")

        seq = seqs[0]
        if not bool(_get_attr(seq, "is_greedy", False)):
            temperature = _get_attr(seq, "temperature", None)
            if temperature is None or temperature > 1e-10:
                return Compatibility(False, "Megakernel v1 supports greedy sampling only")

        hf_config = _get_attr(config, "hf_config")
        expected = {
            "model_type": "qwen3",
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "vocab_size": 151936,
        }
        for name, value in expected.items():
            if _get_attr(hf_config, name) != value:
                return Compatibility(False, f"unsupported Qwen config: {name}")

        max_model_len = _get_attr(config, "max_model_len", 2048)
        if max_model_len > 2048:
            return Compatibility(False, "Megakernel v1 supports max_model_len <= 2048")
        return Compatibility(True)

    def set_kv_cache(self, kv_cache) -> None:
        """Attach the nano-vLLM-owned ``[K/V, layer, block, token, head, dim]`` cache."""

        if kv_cache is None or kv_cache.dim() != 6 or kv_cache.shape[0] != 2:
            raise ValueError("kv_cache must have shape [2, layers, blocks, tokens, heads, dim]")
        self.kv_cache = kv_cache

    def disable(self, reason: str) -> None:
        self.disabled_reason = reason

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

        from .build import get_extension

        get_extension()
        from .model import Decoder, weights_from_nano_model
        weights = weights_from_nano_model(self.model, self.config.hf_config)
        self.decoder = Decoder(
            weights=weights,
            tokenizer=None,
            verbose=False,
            k_cache=self.kv_cache[0],
            v_cache=self.kv_cache[1],
            block_size=self.config.kvcache_block_size,
            num_cache_blocks=self.kv_cache.shape[2],
        )

    def decode(self, seq, block_table) -> list[int]:
        """Decode one token and return a one-element Python token-id list.

        The caller supplies the current nano-vLLM block table. The decoder
        writes the new K/V values into the scheduler-owned paged cache and
        reads prior positions through that same cache; no prompt replay or
        private contiguous cache is used.
        """

        self._ensure_decoder()
        token_id = self.decoder.step(
            seq.last_token,
            position=len(seq) - 1,
            block_table=block_table,
        )
        return [int(token_id)]
