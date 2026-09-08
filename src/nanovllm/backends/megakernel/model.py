"""Weight loading and high-level decode API for Qwen3-0.6B."""

import math
import os
import struct

import torch

NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
Q_SIZE = 16 * HEAD_DIM  # 2048
KV_SIZE = 8 * HEAD_DIM  # 1024
MAX_SEQ_LEN = 2048
VOCAB_SIZE = 151936

_decode = torch.ops.nanovllm_megakernel_C.decode
_decode_paged = torch.ops.nanovllm_megakernel_C.decode_paged
_decode_paged_batch = torch.ops.nanovllm_megakernel_C.decode_paged_batch
_generate_nosync_paged = torch.ops.nanovllm_megakernel_C.generate_nosync_paged
LDG_LM_NUM_BLOCKS = int(os.getenv("LDG_LM_NUM_BLOCKS", "1280"))


def _resolve_rope_theta(config, default: float = 10000.0) -> float:
    """Resolve the RoPE base frequency from a HuggingFace/transformers config.

    Qwen3-0.6B stores ``rope_theta`` inside the ``rope_scaling`` dict (as
    ``rope_scaling={"rope_theta": 1000000, "rope_type": "default"}``) rather
    than as a top-level attribute.  nano-vLLM's Qwen3 model handles this in
    ``Qwen3Attention.__init__``; the Megakernel weight loader must do the
    same or the cos/sin tables will use the wrong frequency.
    """
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None:
        rope_scaling = getattr(config, "rope_scaling", None)
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", default)
        else:
            rope_theta = default
    return float(rope_theta)


def _rope_tables(rope_theta: float = 10000.0):
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    positions = torch.arange(MAX_SEQ_LEN, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos_table = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    sin_table = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    return cos_table, sin_table


def load_weights(model_name="Qwen/Qwen3-0.6B", verbose: bool = True):
    """Load Qwen3-0.6B weights from HuggingFace into GPU tensors."""
    if not verbose:
        import os

        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    if not verbose:
        hf_logging.set_verbosity_error()
        try:
            hf_logging.disable_progress_bar()
        except AttributeError:
            pass
        try:
            from huggingface_hub import logging as hf_hub_logging

            hf_hub_logging.set_verbosity_error()
        except Exception:
            pass

    if verbose:
        print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    state = model.state_dict()

    cos_table, sin_table = _rope_tables(
        _resolve_rope_theta(model.config)
    )

    # Per-layer weight list (11 tensors per layer, flattened)
    layer_weights = []
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}."
        layer_weights.extend(
            [
                state[p + "input_layernorm.weight"].contiguous(),
                state[p + "self_attn.q_proj.weight"].contiguous(),
                state[p + "self_attn.k_proj.weight"].contiguous(),
                state[p + "self_attn.v_proj.weight"].contiguous(),
                state[p + "self_attn.q_norm.weight"].contiguous(),
                state[p + "self_attn.k_norm.weight"].contiguous(),
                state[p + "self_attn.o_proj.weight"].contiguous(),
                state[p + "post_attention_layernorm.weight"].contiguous(),
                state[p + "mlp.gate_proj.weight"].contiguous(),
                state[p + "mlp.up_proj.weight"].contiguous(),
                state[p + "mlp.down_proj.weight"].contiguous(),
            ]
        )

    embed_weight = state["model.embed_tokens.weight"].contiguous()
    weights = dict(
        embed_weight=embed_weight,
        layer_weights=layer_weights,
        final_norm_weight=state["model.norm.weight"].contiguous(),
        lm_head_weight=embed_weight,  # tied embeddings
        cos_table=cos_table,
        sin_table=sin_table,
    )

    del model
    torch.cuda.empty_cache()
    return weights, tokenizer


def weights_from_nano_model(model, hf_config):
    """Create Megakernel weight views from a tensor-parallel nano-vLLM model.

    This keeps the integration from loading a second Hugging Face model. The
    first backend milestone is intentionally tensor_parallel_size=1, so every
    tensor has the full Qwen3-0.6B shape expected by the fixed-shape kernel.
    """

    qwen = model.model
    if qwen.embed_tokens.weight.dtype != torch.bfloat16:
        raise RuntimeError("Qwen Megakernel v1 requires BF16 model weights")
    if getattr(hf_config, "attention_bias", False):
        raise RuntimeError("Qwen Megakernel v1 requires Qwen QK normalization")

    layer_weights = []
    for layer in qwen.layers:
        attn = layer.self_attn
        qkv = attn.qkv_proj.weight
        q_weight, k_weight, v_weight = qkv.split(
            [attn.q_size, attn.kv_size, attn.kv_size], dim=0
        )
        gate_weight, up_weight = layer.mlp.gate_up_proj.weight.split(
            [INTERMEDIATE_SIZE, INTERMEDIATE_SIZE], dim=0
        )
        layer_weights.extend(
            [
                layer.input_layernorm.weight,
                q_weight,
                k_weight,
                v_weight,
                attn.q_norm.weight,
                attn.k_norm.weight,
                attn.o_proj.weight,
                layer.post_attention_layernorm.weight,
                gate_weight,
                up_weight,
                layer.mlp.down_proj.weight,
            ]
        )

    cos_table, sin_table = _rope_tables(
        _resolve_rope_theta(hf_config)
    )
    embed_weight = qwen.embed_tokens.weight
    return dict(
        embed_weight=embed_weight,
        layer_weights=layer_weights,
        final_norm_weight=qwen.norm.weight,
        lm_head_weight=model.lm_head.weight,
        cos_table=cos_table,
        sin_table=sin_table,
    )


def _pack_layer_weights(layer_weights: list[torch.Tensor]) -> torch.Tensor:
    """Pack 11-tensor-per-layer flat list into a device blob of LDGLayerWeights structs."""
    ptr_size = 8  # 64-bit pointers
    n_ptrs = 11
    struct_bytes = n_ptrs * ptr_size
    buf = bytearray(NUM_LAYERS * struct_bytes)
    for i in range(NUM_LAYERS):
        for j in range(n_ptrs):
            ptr = layer_weights[i * n_ptrs + j].data_ptr()
            struct.pack_into("Q", buf, (i * n_ptrs + j) * ptr_size, ptr)
    t = torch.frombuffer(buf, dtype=torch.uint8).cuda()
    return t


class Decoder:
    """Stateful decoder wrapping the Qwen Megakernel torch ops."""

    def __init__(
        self,
        weights=None,
        tokenizer=None,
        model_name="Qwen/Qwen3-0.6B",
        verbose: bool = True,
        k_cache=None,
        v_cache=None,
        block_size: int | None = None,
        num_cache_blocks: int | None = None,
        max_batch_size: int = 1,
    ):
        if weights is None:
            weights, tokenizer = load_weights(model_name, verbose=verbose)
        self.tokenizer = tokenizer
        self._position = 0
        self._max_batch_size = int(max_batch_size)
        if self._max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")

        # Keep references so tensors stay alive (prevents GC of weight memory).
        self._weights = weights

        # Model weights (read-only, shared across calls)
        self._embed_weight = weights["embed_weight"]
        self._final_norm_weight = weights["final_norm_weight"]
        self._lm_head_weight = weights["lm_head_weight"]
        self._cos_table = weights["cos_table"]
        self._sin_table = weights["sin_table"]
        self._layer_weights_packed = _pack_layer_weights(weights["layer_weights"])

        self._attn_scale = 1.0 / math.sqrt(HEAD_DIM)

        # KV cache. When views are provided, they must be nano-vLLM's
        # [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        # tensors; the Megakernel then uses the scheduler's block table.
        if (k_cache is None) != (v_cache is None):
            raise ValueError("k_cache and v_cache must be provided together")
        self._paged = k_cache is not None
        if self._paged:
            if k_cache.dim() != 5 or v_cache.shape != k_cache.shape:
                raise ValueError("paged KV cache views must have matching 5D shapes")
            if k_cache.dtype != torch.bfloat16 or v_cache.dtype != torch.bfloat16:
                raise ValueError("paged KV cache views must use BF16")
            self._k_cache = k_cache
            self._v_cache = v_cache
            self._block_size = int(block_size or k_cache.shape[2])
            self._num_cache_blocks = int(num_cache_blocks or k_cache.shape[1])
            if k_cache.shape[0] != NUM_LAYERS:
                raise ValueError("paged KV cache layer count does not match Qwen3-0.6B")
            if k_cache.shape[1] != self._num_cache_blocks:
                raise ValueError("num_cache_blocks does not match k_cache")
            if k_cache.shape[2] != self._block_size:
                raise ValueError("block_size does not match k_cache")
            if k_cache.shape[3:] != (NUM_KV_HEADS, HEAD_DIM):
                raise ValueError("paged KV cache head shape does not match Qwen3-0.6B")
        else:
            self._k_cache = torch.zeros(
                NUM_LAYERS,
                NUM_KV_HEADS,
                MAX_SEQ_LEN,
                HEAD_DIM,
                dtype=torch.bfloat16,
                device="cuda",
            )
            self._v_cache = torch.zeros_like(self._k_cache)

        # Scratch buffers are partitioned by sequence in paged batch mode.
        f32 = dict(dtype=torch.float32, device="cuda")
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        self._hidden = torch.empty((self._max_batch_size, HIDDEN_SIZE), **bf16)
        self._act = torch.empty((self._max_batch_size, HIDDEN_SIZE), **f32)
        self._res = torch.empty((self._max_batch_size, HIDDEN_SIZE), **f32)
        self._q = torch.empty((self._max_batch_size, Q_SIZE), **f32)
        self._k = torch.empty((self._max_batch_size, KV_SIZE), **f32)
        self._v = torch.empty((self._max_batch_size, KV_SIZE), **f32)
        self._attn_out = torch.empty((self._max_batch_size, Q_SIZE), **f32)
        self._mlp_inter = torch.empty(
            (self._max_batch_size, INTERMEDIATE_SIZE), **f32
        )
        self._norm_out = torch.empty((self._max_batch_size, HIDDEN_SIZE), **f32)
        self._bmax_vals = torch.empty(
            self._max_batch_size * LDG_LM_NUM_BLOCKS, **f32
        )
        self._bmax_idxs = torch.empty(
            self._max_batch_size * LDG_LM_NUM_BLOCKS,
            dtype=torch.int32,
            device="cuda",
        )
        self._out_token = torch.empty(
            self._max_batch_size, dtype=torch.int32, device="cuda"
        )

    def step(self, token_id: int, position: int | None = None, block_table=None) -> int:
        """Decode one token and return the next token id.

        Paged mode requires the caller to provide the logical position and the
        device block table for the current sequence. Legacy mode keeps the
        original contiguous-cache operator for the standalone reference tool.
        """
        if self._paged:
            if position is None or block_table is None:
                raise ValueError("paged decode requires position and block_table")
            _decode_paged(
                self._out_token,
                token_id,
                self._embed_weight,
                self._layer_weights_packed,
                self._final_norm_weight,
                self._lm_head_weight,
                self._cos_table,
                self._sin_table,
                self._k_cache,
                self._v_cache,
                self._hidden,
                self._act,
                self._res,
                self._q,
                self._k,
                self._v,
                self._attn_out,
                self._mlp_inter,
                self._norm_out,
                self._bmax_vals,
                self._bmax_idxs,
                block_table,
                NUM_LAYERS,
                position,
                self._block_size,
                self._num_cache_blocks,
                self._attn_scale,
            )
            # Single-sequence decode writes only element 0 of _out_token, which
            # is sized to _max_batch_size (and can be > 1). Index element 0 so
            # .item() works even after a multi-sequence batch shrinks to one.
            return int(self._out_token[0].item())
        _decode(
            self._out_token,
            token_id,
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            NUM_LAYERS,
            self._position,
            MAX_SEQ_LEN,
            self._attn_scale,
        )
        self._position += 1
        return int(self._out_token[0].item())

    def step_batch(self, input_token_ids, positions, block_tables) -> list[int]:
        """Decode one token for each sequence using one paged batch operator."""

        if not self._paged:
            raise RuntimeError("step_batch() requires paged KV cache mode")
        batch_size = int(input_token_ids.numel())
        if batch_size <= 0 or batch_size > self._max_batch_size:
            raise ValueError(
                f"batch size must be between 1 and {self._max_batch_size}"
            )
        if (
            positions.numel() != batch_size
            or block_tables.dim() != 2
            or block_tables.shape[0] != batch_size
        ):
            raise ValueError("batch metadata shapes do not match")
        _decode_paged_batch(
            self._out_token[:batch_size],
            input_token_ids,
            positions,
            block_tables,
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            NUM_LAYERS,
            self._block_size,
            self._num_cache_blocks,
            self._attn_scale,
        )
        return self._out_token[:batch_size].cpu().tolist()

    def reset(self):
        self._position = 0
        if not self._paged:
            self._k_cache.zero_()
            self._v_cache.zero_()

    @property
    def position(self) -> int:
        return self._position

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        if self._paged:
            raise RuntimeError("generate() is only available for contiguous reference mode")
        self.reset()
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        for tid in ids[:-1]:
            self.step(tid)
        _gen = torch.ops.nanovllm_megakernel_C.generate_nosync
        output_ids = _gen(
            ids[-1],
            max_tokens,
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            NUM_LAYERS,
            self._position,
            MAX_SEQ_LEN,
            self._attn_scale,
        )
        self._position += max_tokens
        out = output_ids.cpu().tolist()
        eos = self.tokenizer.eos_token_id
        if eos in out:
            out = out[: out.index(eos)]
        return self.tokenizer.decode(out, skip_special_tokens=True)

    def generate_paged(
        self,
        first_token_id: int,
        start_position: int,
        block_table,
        num_tokens: int,
    ) -> list[int]:
        """Generate ``num_tokens`` on-device with NO per-step CPU sync.

        All ``block_table`` blocks for positions
        ``[start_position, start_position + num_tokens)`` must already be
        allocated (the block table is fixed over the span). Returns the full
        token list after a single host sync.
        """
        if not self._paged:
            raise RuntimeError("generate_paged() requires paged KV cache mode")
        if not isinstance(block_table, torch.Tensor):
            block_table = torch.tensor(block_table, dtype=torch.int32, device="cuda")
        block_table = block_table.to(torch.int32).contiguous()
        output_ids = _generate_nosync_paged(
            first_token_id,
            num_tokens,
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            block_table,
            NUM_LAYERS,
            start_position,
            self._block_size,
            self._num_cache_blocks,
            self._attn_scale,
        )
        return output_ids.cpu().tolist()


def generate(prompt: str, max_tokens: int = 100, verbose: bool = True) -> str:
    """One-shot convenience: load model, generate, return text."""
    return Decoder(verbose=verbose).generate(prompt, max_tokens)
