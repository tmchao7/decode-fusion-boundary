"""Optional Qwen3 Megakernel decode backend.

The CUDA extension is imported lazily so the regular nano-vLLM backend remains
usable on machines without CUDA, and so importing the package never triggers a
CUDA build. The initial paged implementation supports Qwen3-0.6B greedy decode
with a conservative batch limit; callers should use :meth:`compatibility`
before invoking :meth:`decode` or :meth:`decode_batch`.
"""

from dataclasses import dataclass
from typing import Any


VALID_CUDA_ARCHES = {"sm_70", "sm_80", "sm_86", "sm_89", "sm_90", "sm_120"}
MAX_BATCH_SIZE = 32

# Fixed-shape contract baked into the fused CUDA kernel (kernel.cu). This is
# the single Python-side source of truth for what the kernel was compiled for;
# compatibility() guards that the loaded nano-vLLM model matches it. If these
# ever change, kernel.cu's constexpr block (HIDDEN_SIZE, NUM_*_HEADS, HEAD_DIM,
# NUM_LAYERS at the launch site, LDG_VOCAB_SIZE) must change in lockstep.
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


def wmma_batch_limit(default: int = MAX_BATCH_SIZE) -> int:
    """Real batch limit of the WMMA batch kernel, or ``default`` if unknown.

    ``MAX_BATCH_SIZE`` is the *compile-time* ceiling (``-DLDG_MAX_BATCH``) and
    the binding's ``TORCH_CHECK`` accepts 1-32, but the kernel partitions its
    resident cooperative grid across sequences and aborts the launch unless each
    sequence gets ``LDG_ATTN_BLOCKS`` blocks.  With the default 80-block grid
    and ``LDG_ATTN_BLOCKS=8`` that is batch <= 10.  Querying the kernel keeps
    this guard in sync with the real occupancy instead of hard-coding 10.

    The scalar batch kernel shares the same grid-partitioning constraint, so
    this limit is used as a conservative bound for both paths (WMMA is more
    register-heavy, so its occupancy is at most the scalar kernel's).

    Returns ``default`` when the extension or CUDA is unavailable, which keeps
    the compatibility check usable on the CPU-only test path; the binding still
    rejects an unsupported batch with a clear error at execution time.
    """

    try:
        import torch

        if not torch.cuda.is_available():
            return default
        from .build import get_extension

        extension = get_extension()
        limit = int(extension.wmma_max_supported_batch())
    except Exception:
        return default
    return limit if 0 < limit <= default else default


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
        max_batch_size = min(
            _get_attr(config, "max_num_seqs", MAX_BATCH_SIZE), MAX_BATCH_SIZE
        )
        if not 1 <= len(seqs) <= max_batch_size:
            return Compatibility(
                False,
                f"Megakernel v1 supports batch sizes from 1 to {max_batch_size}",
            )
        # Both the scalar and WMMA batch kernels partition the cooperative
        # grid per sequence and need LDG_ATTN_BLOCKS blocks each, so they
        # support far fewer sequences than MAX_BATCH_SIZE.  The WMMA limit is
        # a conservative bound for the scalar kernel too (WMMA is more
        # register-heavy, so its occupancy is at most the scalar kernel's).
        # Reject here so the batch falls back to torch-cudagraph with a clear
        # reason instead of the launch aborting and the caller reading the
        # previous step's tokens.
        #
        # Query the limit only when the batch could exceed it.  The limit is
        # >= 1 for any launchable grid, so batch=1 always fits and the check is
        # vacuous there; meanwhile wmma_batch_limit() issues CUDA driver calls
        # (cudaGetDeviceProperties + cudaOccupancyMaxActiveBlocksPerMultiprocessor)
        # and compatibility() runs on every decode step.  Batch=1 is the path
        # that wins by removing per-step host work, so an unconditional query
        # would tax exactly the path this project optimized.
        if len(seqs) > 1:
            batch_limit = min(max_batch_size, wmma_batch_limit(max_batch_size))
            if len(seqs) > batch_limit:
                return Compatibility(
                    False,
                    f"Megakernel batch decode supports at most {batch_limit} "
                    "sequences (grid partitioned per sequence with "
                    "LDG_ATTN_BLOCKS=8 per sequence); use torch-cudagraph for "
                    "larger batches",
                )
        if _get_attr(config, "tensor_parallel_size", 1) != 1:
            return Compatibility(False, "Megakernel v1 requires tensor_parallel_size=1")

        for seq in seqs:
            if bool(_get_attr(seq, "is_greedy", False)):
                continue
            temperature = _get_attr(seq, "temperature", None)
            if temperature is None or temperature > 1e-10:
                return Compatibility(False, "Megakernel v1 supports greedy sampling only")

        # Hybrid architecture (docs/07 §9.6): megakernel wins batch=1 by
        # eliminating host launch overhead, but loses batch>=2 because the
        # fixed cooperative grid cannot saturate SMs for batched GEMMs
        # (§2/§3.5). Default: batch>1 falls back to cudagraph. Set
        # megakernel_batch_decode=True to force the megakernel batch path
        # (used by correctness research tools).
        if len(seqs) > 1 and not _get_attr(config, "megakernel_batch_decode", False):
            return Compatibility(
                False,
                "batch>1 falls back to the torch path unless "
                "megakernel_batch_decode is set. In this engine that fallback "
                "is eager (decode_backend='megakernel' disables CUDA-graph "
                "capture); the A/B comparison runs torch-cudagraph in a "
                "separate engine. Set megakernel_batch_decode=True to force "
                "the megakernel batch kernel.",
            )

        hf_config = _get_attr(config, "hf_config")
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
                return Compatibility(False, f"unsupported Qwen config: {name}")
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
            max_batch_size=min(
                _get_attr(self.config, "max_num_seqs", 1), MAX_BATCH_SIZE
            ),
        )

    def decode_batch(self, input_ids, positions, block_tables) -> list[int]:
        """Decode one token for each scheduler-selected sequence.

        Uses the WMMA tensor-core batch kernel when
        ``config.megakernel_batch_kernel == "wmma"`` (A/B switch); otherwise the
        scalar batch kernel.
        """

        self._ensure_decoder()
        if (
            getattr(self.config, "megakernel_batch_kernel", "scalar") == "wmma"
        ):
            return self.decoder.decode_paged_batch_wmma(
                input_ids, positions, block_tables
            )
        return self.decoder.step_batch(input_ids, positions, block_tables)

    def generate_paged_batch(
        self, seqs, block_tables, max_steps: int, eos_token: int
    ) -> list[list[int]]:
        """Run a no-sync burst for a batch of sequences.

        ``block_tables`` is a device int32 tensor built by
        ``ModelRunner.prepare_block_tables``. Returns the raw ``[batch][max_steps]``
        token log (entries past per-seq EOS are -1). The caller (scheduler
        ``postprocess_burst``) handles EOS / max_tokens truncation.

        Hybrid dispatch: batch=1 uses the persistent single-seq kernel
        (faster — no WMMA tile padding), batch>1 uses the WMMA batch kernel.
        """
        self._ensure_decoder()
        import torch

        batch_size = len(seqs)
        if batch_size == 1:
            # Persistent single-seq kernel: extract the 1D block table and
            # call generate_paged with device-side EOS.
            seq = seqs[0]
            block_table_1d = block_tables[0][: len(seq.block_table)]
            raw = self.decoder.generate_paged(
                seq.last_token,
                len(seq) - 1,
                block_table_1d,
                max_steps,
                eos_token,
            )
            return [raw]

        input_ids = torch.tensor(
            [seq.last_token for seq in seqs],
            dtype=torch.int64,
            device="cuda",
        )
        positions = torch.tensor(
            [len(seq) - 1 for seq in seqs],
            dtype=torch.int64,
            device="cuda",
        )
        return self.decoder.generate_paged_batch(
            input_ids,
            positions,
            block_tables,
            max_steps,
            eos_token,
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
