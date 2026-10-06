"""Per-token byte budget and decode-roofline arithmetic for sparse (MoE) models.

Why this module exists
----------------------
For a *dense* decode step the interesting question is "how close is the kernel to
the weight-streaming wall", and ``nanovllm.roofline`` answers it.  For a *sparse*
decode step there is a second, prior question that ``roofline.py`` cannot answer:
**which bytes are actually read, and is the routing that selects them even
representable in a static instruction stream?**

At batch 1 a sparse decode step is pure streaming of

    [W] = attention projections + routers + (top_k / num_experts) x experts
          + lm_head  [+ embedding rows, only when untied]

and the answer to "can a fused kernel win" is decided by three arithmetic facts,
all computed here:

1. **the byte budget** -- 2.358 GB/token for OLMoE-1B-7B-0924, of which experts are
   68.3%, so experts are the only thing worth fighting over;
2. **the compute margin** -- a few GFLOP against ~200 TFLOP/s, i.e. ~10x of compute
   headroom on top of a 1.5 ms memory floor.  Decode is bandwidth-bound, so the
   architecture is a *memory-movement* problem, not a GEMM-tiling problem;
3. **the load balance** -- at ``batch == 1`` exactly ``top_k`` experts are active per
   token and each active expert receives exactly one token, so the per-expert token
   histogram is flat by construction.  That is why a *weight-major* layout needs no
   gather, no sort, and no dynamic work stealing at batch 1, and why those become
   necessary only as the batch grows.

Everything in this module is pure arithmetic on an explicit
:class:`MoEDims`, so a CPU-only checkout can re-derive every number in
``docs/22`` -- and the tests assert the derived values against the checkpoints
that were verified item-by-item from ``config.json``.

Provenance of the shipped preset
--------------------------------
:data:`OLMOE_1B_7B` was reconciled against the **actual checkpoint tensors**, not
against a prose description of the architecture.  ``model.safetensors``'s header
lists 3,219 tensors and the safetensors API reports
``parameters.BF16 = 6,919,161,856``; the per-tensor shapes were read out and every
one of them is reproduced by :meth:`MoEDims.total_params`.  That reconciliation
found **four** things worth recording, because each one changes a number this
module derives:

1. **OLMoE has per-head QK-norms** -- ``self_attn.q_norm.weight`` and
   ``self_attn.k_norm.weight``, 2048 params each per layer (16 heads x 128).  That is
   4,096 per layer, or **65,536 across 16 layers**, on top of the two RMSNorm vectors
   per layer a dense model already has.  There is also a final ``model.norm.weight``
   of 2,048.  A description that says "attention q/k/v/o + two RMSNorms" undercounts
   by 67,584 params and by the same number of streamed bytes.
2. **``num_attention_heads == num_key_value_heads == 16``**, i.e. MHA, so the KV
   width equals ``hidden`` and the KV read is 2x what a GQA(16/8) model would pay.
3. **``norm_topk_prob`` is ``false``**: the top-k routing weights are *not*
   renormalized.  The combine step must therefore consume the raw softmax scores of
   the selected experts.
4. **There is no grouped top-k** in OLMoE (``num_expert_groups`` does not appear in
   ``config.json``); ``MoEDims.num_expert_groups`` defaults to 1 for exactly that
   reason.

The one set of constants this module does **not** reproduce is the pair recorded in
``docs/MODELS.md`` §3.2 and ``configs/models.yaml``. Both are short, and the
shortfalls decompose exactly:

* per token, the recorded ``[W] = 2,357,723,136 B`` is **262,144 B** below the
  tensor-verified value -- two QK-norm tensors for each of the 16 layers
  (``2 x 16 x 2048 x 2 B``), which that byte table's attention row does not include
  and whose total the final-layer norm is charged for instead;
* for storage, the recorded ``13,838,057,472 B`` is **266,240 B** below
  ``2 x total_params`` = ``262,144`` (the QK norms) ``+ 4,096`` (the final
  ``model.norm``).

Two independently recorded numbers missing the same tensors is what makes this an
omission rather than arithmetic noise. It is 0.011% of ``[W]`` and moves no floor to
three significant figures -- but it is recorded because "OLMoE has per-head QK
norms" is a behavioural difference, not a rounding footnote: a kernel must read and
apply two more tensors per layer than a dense model would, and they sit on the
critical path between the projection and attention.

Bandwidth is never assumed here: :func:`decode_budget` takes it as an argument, and
callers are expected to pass this machine's *measured* read bandwidth
(``results/bandwidth.json`` via ``nanovllm.roofline``).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

# =============================================================================
# Model dimensions
# =============================================================================


@dataclass(frozen=True)
class MoEDims:
    """Dimensions of one sparse transformer, in the units the checkpoint uses.

    ``num_q_heads`` and ``num_kv_heads`` are deliberately separate: OLMoE is
    multi-head (16/16, so a 1:1 q:kv ratio) while the Qwen3 family is grouped-query
    (16/8).  The KV traffic formula depends on ``num_kv_heads``, and with MHA it is
    1024 wide per token per layer at ``hidden = 2048`` -- do not substitute the
    dense-project default.
    """

    name: str = "OLMoE-1B-7B-0924"
    num_layers: int = 16
    hidden_size: int = 2048
    num_q_heads: int = 16
    num_kv_heads: int = 16  # MHA: equals num_q_heads for OLMoE
    head_dim: int = 128
    num_experts: int = 64
    top_k: int = 8
    expert_ffn_dim: int = 1024
    vocab_size: int = 50304
    tie_word_embeddings: bool = False  # OLMoE config.json: false
    #: Per-head QK RMSNorm (``self_attn.q_norm`` / ``k_norm``). OLMoE has it; a
    #: dense Qwen3 does not. Each is ``num_heads * head_dim`` params per layer.
    qk_norm: bool = True
    #: num_experts is split into this many groups for grouped top-k routing. OLMoE
    #: does **not** use grouping (the key is absent from config.json), so 1.
    num_expert_groups: int = 1
    #: Experts selected per group when ``num_expert_groups > 1``.
    top_k_per_group: int = 0
    #: Renormalize the selected routing weights to sum to 1. OLMoE sets this to
    #: **false**, so the combine step consumes raw softmax scores.
    norm_topk_prob: bool = False

    # -- derived sizes -----------------------------------------------------

    @property
    def q_size(self) -> int:
        return self.num_q_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def num_matrix_tiles(self) -> int:
        """Output columns of one expert's gate/up pair, in 16-column WMMA tiles."""

        return 2 * self.expert_ffn_dim // 16

    def validate(self) -> None:
        if self.num_layers < 1 or self.hidden_size < 1 or self.vocab_size < 1:
            raise ValueError("layers/hidden/vocab must be positive")
        if self.num_experts < 1:
            raise ValueError("num_experts must be positive")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError(
                f"top_k must be in 1..num_experts; got {self.top_k}/{self.num_experts}"
            )
        if self.qk_norm and self.q_size != self.hidden_size:
            # The norm weight is laid out as num_heads x head_dim, so the product
            # must tile `hidden`; otherwise norm_params_per_layer undercounts.
            raise ValueError(
                f"qk_norm layout needs num_q_heads*head_dim == hidden_size; "
                f"got {self.q_size} != {self.hidden_size}"
            )
        if self.num_expert_groups > self.num_experts:
            raise ValueError(
                f"num_expert_groups={self.num_expert_groups} exceeds "
                f"num_experts={self.num_experts}"
            )
        if self.num_expert_groups > 1:
            if self.top_k % self.num_expert_groups != 0:
                raise ValueError(
                    f"top_k={self.top_k} is not divisible by "
                    f"num_expert_groups={self.num_expert_groups}"
                )
            if self.top_k_per_group and (
                self.top_k_per_group * self.num_expert_groups != self.top_k
            ):
                raise ValueError(
                    f"top_k_per_group={self.top_k_per_group} x "
                    f"num_expert_groups={self.num_expert_groups} != top_k={self.top_k}"
                )

    # -- parameter counts --------------------------------------------------

    def attention_params_per_layer(self) -> int:
        """q + k + v + o projections (norms are counted separately)."""

        qkv = self.hidden_size * (self.q_size + 2 * self.kv_size)
        o_proj = self.q_size * self.hidden_size
        return qkv + o_proj

    def norms_params_per_layer(self) -> int:
        """All RMSNorm vectors in one layer.

        Two stream norms (input, post-attention) plus, when :attr:`qk_norm`, the two
        per-head QK norms.  OLMoE's checkpoint confirms all four exist as separate
        ``[hidden]`` tensors; omitting the QK pair is the single most common way a
        hand-written budget undercounts this model.
        """

        norms = 2 * self.hidden_size
        if self.qk_norm:
            norms += 2 * self.hidden_size
        return norms

    def final_norm_params(self) -> int:
        """The output RMSNorm applied after the last layer."""

        return self.hidden_size

    def router_params_per_layer(self) -> int:
        """The router is one ``num_experts x hidden`` matrix; untied from anything."""

        return self.num_experts * self.hidden_size

    def experts_params_per_layer(self) -> int:
        """All experts, gate + up + down: ``num_experts * 3 * hidden * ffn``."""

        return self.num_experts * 3 * self.hidden_size * self.expert_ffn_dim

    def active_experts_params_per_layer(self) -> int:
        """Only the ``top_k`` experts a single token reads."""

        return self.top_k * 3 * self.hidden_size * self.expert_ffn_dim

    def lm_head_params(self) -> int:
        return self.vocab_size * self.hidden_size

    def embedding_params(self) -> int:
        return self.vocab_size * self.hidden_size

    def total_params(self) -> int:
        """All storage: layers + embedding table + untied lm_head + final norm.

        Verified against the checkpoint: this reproduces safetensors' reported
        ``parameters.BF16 = 6,919,161,856`` for OLMoE exactly.
        """

        per_layer = (
            self.attention_params_per_layer()
            + self.norms_params_per_layer()
            + self.router_params_per_layer()
            + self.experts_params_per_layer()
        )
        embedding = 0 if self.tie_word_embeddings else self.embedding_params()
        return (
            self.num_layers * per_layer
            + embedding
            + self.lm_head_params()
            + self.final_norm_params()
        )

    def active_params_per_token(self) -> int:
        """Parameters that cross HBM for one token (the ``[W]`` count).

        Excludes the embedding *table*: a lookup reads ``batch`` rows, so only the
        rows are charged.  Includes the full lm_head, because every vocabulary row
        is needed to produce logits, and every RMSNorm vector, because each is read
        once per token.
        """

        per_layer = (
            self.attention_params_per_layer()
            + self.norms_params_per_layer()
            + self.router_params_per_layer()
            + self.active_experts_params_per_layer()
        )
        return (
            self.num_layers * per_layer
            + self.lm_head_params()
            + self.final_norm_params()
        )


#: ``allenai/OLMoE-1B-7B-0924`` as verified from ``config.json`` (see module docstring).
OLMOE_1B_7B = MoEDims()


# =============================================================================
# Byte budget
# =============================================================================

#: The four recorded byte components of ``[W]``, in the order docs/22 tabulates them.
BUDGET_PARTS = ("attention", "router", "experts", "lm_head")

#: How many contiguous ``tile_n`` one instruction may pack
#: (``schedule._TILE_SLOTS_PER_INST``). Note this is the *instruction-stream* packing
#: limit set by the existing encoding; the in-kernel shared-memory budget is tighter
#: for MoE (see ``docs/22`` §3), so the kernel must re-derive its own limit.
_TILES_PER_INST = 7


@dataclass(frozen=True)
class ByteBudget:
    """Bytes one token must stream, split by component.

    ``embedding_rows`` is reported separately and is **not** part of ``weight_bytes``:
    a tied model reads one row per token, and an untied model still does not stream
    the table (it streams the separate lm_head instead).
    """

    bytes_per_param: int
    attention: int
    router: int
    experts: int
    lm_head: int
    embedding_rows: int
    mv_bytes_per_expert: int
    #: Total checkpoint storage, reported so ``[W]`` cannot be misread as "the model".
    storage_bytes: int = 0

    @property
    def weight_bytes(self) -> int:
        """``[W]``: exactly the sum of the four streamed components."""

        return self.attention + self.router + self.experts + self.lm_head

    def public_bytes(self, kv_bytes: int) -> int:
        """``[T]``: ``[W]`` plus one token's KV read."""

        return self.weight_bytes + int(kv_bytes)

    def share(self, part: str) -> float:
        """Fraction of ``[W]`` held by one component (``BUDGET_PARTS``)."""

        if part not in BUDGET_PARTS:
            raise ValueError(f"unknown component {part!r}; expected {BUDGET_PARTS}")
        total = self.weight_bytes
        return 0.0 if total == 0 else getattr(self, part) / total

    def as_dict(self) -> dict[str, int | float]:
        return {
            "attention": self.attention,
            "router": self.router,
            "experts": self.experts,
            "lm_head": self.lm_head,
            "weight_bytes": self.weight_bytes,
            "storage_bytes": self.storage_bytes,
            "embedding_rows": self.embedding_rows,
            "mv_bytes_per_expert": self.mv_bytes_per_expert,
        }


def kv_bytes_per_token(dims: MoEDims, ctx: int, bytes_per_param: int) -> int:
    """KV read for one token at context ``ctx``.

    ``2`` because K and V; ``num_kv_heads * head_dim`` is the KV width.  At MHA
    (``num_kv_heads == num_q_heads``) this is ``2 * layers * hidden * ctx *
    bytes_per_param`` -- 128 KiB/token per context step, i.e. 0.134 GB at ctx=1024
    for OLMoE, which is 5.4% of ``[T]``.
    """

    dims.validate()
    if ctx < 0:
        raise ValueError(f"ctx must be >= 0; got {ctx}")
    return 2 * dims.num_layers * int(ctx) * dims.kv_size * bytes_per_param


def byte_budget(dims: MoEDims = OLMOE_1B_7B, *, bytes_per_param: int = 2) -> ByteBudget:
    """Streamed bytes per token, by component, at ``bytes_per_param`` each.

    ``bytes_per_param = 2`` is fp16/bf16; pass ``1`` or ``0.5``-style integers via
    the ``quantized_bytes_per_param`` helper for a quantized arm.  The router is a
    full read regardless of quantization because it is needed before any expert is
    selected (that is why router cost is a floor on the quantization win).
    """

    dims.validate()
    if bytes_per_param < 1:
        raise ValueError(f"bytes_per_param must be >= 1; got {bytes_per_param}")
    scale = int(bytes_per_param)
    per_layer_attention = dims.attention_params_per_layer() + dims.norms_params_per_layer()
    return ByteBudget(
        bytes_per_param=scale,
        attention=scale * dims.num_layers * per_layer_attention,
        router=scale * dims.num_layers * dims.router_params_per_layer(),
        experts=scale * dims.num_layers * dims.active_experts_params_per_layer(),
        lm_head=scale * dims.lm_head_params(),
        embedding_rows=scale * dims.hidden_size,
        mv_bytes_per_expert=scale * 3 * dims.hidden_size * dims.expert_ffn_dim,
        storage_bytes=scale * dims.total_params(),
    )


# =============================================================================
# Roofline and parallelism
# =============================================================================


@dataclass(frozen=True)
class DecodeBudget:
    """Everything docs/22 needs for one (dims, batch, ctx, bandwidth) point."""

    dims: MoEDims
    batch_size: int
    ctx: int
    bandwidth_gbps: float
    bytes: ByteBudget
    kv_bytes: int
    flops: float
    compute_floor_ms: float
    compute_backend: str
    compute_peak_tflops: float

    @property
    def weight_floor_ms(self) -> float:
        """``[W] / bandwidth`` -- the weights-only floor at batch 1."""

        return self.bytes.weight_bytes / (self.bandwidth_gbps * 1e9) * 1000.0

    @property
    def total_floor_ms(self) -> float:
        """``[T] / bandwidth``: the floor including KV, still at batch 1."""

        return self.bytes.public_bytes(self.kv_bytes) / (self.bandwidth_gbps * 1e9) * 1000.0

    @property
    def active_weight_floor_ms(self) -> float:
        """Floor if the gather/materialization is *not* eliminated.

        This is the comparison that motivates the architecture: a fixed-shape
        grouped-GEMM path must copy the selected experts before computing
        (read + write + re-read), tripling the expert traffic.  The fused kernel
        reads the same weights once.
        """

        gather = 3 * self.bytes.experts
        fixed = self.bytes.weight_bytes - self.bytes.experts + gather
        return (fixed + self.kv_bytes) / (self.bandwidth_gbps * 1e9) * 1000.0

    @property
    def gather_saving_ms(self) -> float:
        """How much of the floor a gather-free design buys back, in ms/token."""

        return self.active_weight_floor_ms - self.total_floor_ms

    @property
    def expert_share(self) -> float:
        return self.bytes.share("experts")

    @property
    def bandwidth_attainment(self) -> float:
        """Ratio of the achieved rate to this machine's measured achievable rate.

        Always 1.0 for a modeled floor: it exists so a *measured* per-token latency
        can be divided in and reported next to the floor (``docs/02`` requires the
        distance to the roofline to be stated, not implied).
        """

        return 1.0


def decode_budget(
    dims: MoEDims = OLMOE_1B_7B,
    *,
    batch_size: int = 1,
    ctx: int = 1024,
    bandwidth_gbps: float,
    bytes_per_param: int = 2,
    compute_peak_tflops: float = 209.5,
) -> DecodeBudget:
    """Build the byte/roofline budget for one decode configuration.

    ``bandwidth_gbps`` is required and keyword-only on purpose: there is no default
    peak, because a spec number as a roofline denominator is the mistake this repo
    keeps having to re-fix (A100: 2039 spec vs 1699 measured; 5090: 1792 spec vs
    1559.2 measured).
    """

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1; got {batch_size}")
    if bandwidth_gbps <= 0:
        raise ValueError(f"bandwidth_gbps must be positive; got {bandwidth_gbps}")
    budget = byte_budget(dims, bytes_per_param=bytes_per_param)
    ctx_value = int(ctx)
    flops = flops_per_token(dims, batch_size=batch_size, ctx=ctx_value)
    return DecodeBudget(
        dims=dims,
        batch_size=int(batch_size),
        ctx=ctx_value,
        bandwidth_gbps=float(bandwidth_gbps),
        bytes=budget,
        kv_bytes=kv_bytes_per_token(dims, ctx_value, bytes_per_param),
        flops=flops,
        compute_floor_ms=flops / (compute_peak_tflops * 1e12) * 1000.0,
        compute_backend="tensor_core",
        compute_peak_tflops=float(compute_peak_tflops),
    )


def flops_per_token(dims: MoEDims, *, batch_size: int = 1, ctx: int = 0) -> float:
    """Multiply-accumulates x2 for one decode step, sparse (``top_k`` experts only).

    At batch 1 and OLMoE this is ~2.4 GFLOP against a ~200 TFLOP/s tensor-core peak:
    about 0.012 ms, i.e. two orders of magnitude below the 1.5 ms memory floor.  That
    ratio is the *reason* the architecture treats this as a streaming problem.  The
    same FLOPs on CUDA cores (no MMA) would be ~0.3 ms at 8 TFLOP/s FP32, still below
    the floor -- so MMA is required but is never the limiter.
    """

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1; got {batch_size}")
    if ctx < 0:
        raise ValueError(f"ctx must be >= 0; got {ctx}")
    tokens = int(batch_size)
    per_layer_experts = 2.0 * dims.top_k * 3 * dims.hidden_size * dims.expert_ffn_dim
    per_layer_attention = 2.0 * (
        dims.hidden_size * (dims.q_size + 2 * dims.kv_size)
        + dims.q_size * dims.hidden_size
    )
    # QK^T and PV over the cached context: 2 matmuls, each 2 * q_size * ctx.
    attention_scores = 2.0 * 2.0 * dims.q_size * ctx
    lm_head = 2.0 * dims.vocab_size * dims.hidden_size
    return float(tokens) * (
        dims.num_layers
        * (per_layer_experts + per_layer_attention + attention_scores)
        + lm_head
    )


# =============================================================================
# Work units: tiles, slots, and the batch-1 balance argument
# =============================================================================


@dataclass(frozen=True)
class WorkPlan:
    """Tile counts and instruction counts for one batch-1 decode step.

    These are the quantities that decide whether 170 SMs can be fed.  All of them
    are pure geometry; none of them depends on the router's output.
    """

    tile_cols: int
    expert_slots: int
    tiles_per_layer: dict[str, int]
    tiles_total: int
    instructions_per_sm_per_wave: dict[str, int]
    serial_levels: int
    k_steps_per_level: int

    def as_dict(self) -> dict[str, object]:
        return {
            "tile_cols": self.tile_cols,
            "expert_slots": self.expert_slots,
            "tiles_per_layer": dict(self.tiles_per_layer),
            "tiles_total": self.tiles_total,
            "instructions_per_sm_per_wave": dict(self.instructions_per_sm_per_wave),
            "serial_levels": self.serial_levels,
            "k_steps_per_level": self.k_steps_per_level,
        }


#: The five serial dependency levels one MoE layer contributes.  Mirrors the opcode
#: order in docs/22 §3; the count is what bounds cross-SM barrier cost.
MOE_LEVELS_PER_LAYER = ("attn", "router", "expert_gateup", "expert_down", "combine")

#: Terminal levels after the last layer: LM_HEAD tiles then the argmax wave.
TERMINAL_LEVELS = ("lm_head", "lm_argmax")


def plan_work(
    dims: MoEDims = OLMOE_1B_7B,
    *,
    batch_size: int = 1,
    num_sms: int = 170,
    tile_cols: int = 16,
) -> WorkPlan:
    """Count tiles, slots and per-SM instruction passes for one batch-1 step.

    The unit of work is one **16-column tile of an output matrix**, matching the
    existing instruction encoding (one instruction packs a *contiguous run* of
    ``tile_n``).  Because every active expert at batch 1 has exactly one token, the
    expert waves are pure "stream 16 columns of this expert's weight" work: no token
    gather, no padding, no per-expert token histogram.
    """

    dims.validate()
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1; got {batch_size}")
    if num_sms < 1:
        raise ValueError(f"num_sms must be >= 1; got {num_sms}")
    if tile_cols < 1:
        raise ValueError(f"tile_cols must be >= 1; got {tile_cols}")

    slots = dims.top_k * batch_size
    per_layer = {
        # q/k/v are three projections over one output matrix; count their columns.
        "qkv": (dims.q_size + 2 * dims.kv_size) // tile_cols,
        "attn": dims.num_kv_heads * batch_size,
        "o_proj": dims.q_size // tile_cols,
        "router": max(1, dims.num_experts // tile_cols),
        # gate and up are two separate [ffn, hidden] matrices per expert, so the
        # tile count is 2 * ffn / 16 per expert -- not ffn / 16.
        "expert_gateup": slots * (2 * dims.expert_ffn_dim // tile_cols),
        "expert_down": slots * (dims.expert_ffn_dim // tile_cols),
        "combine": batch_size,
    }
    tiles_total = dims.num_layers * sum(per_layer.values())
    tiles_total += dims.vocab_size // tile_cols  # LM_HEAD
    tiles_total += 1  # LM_ARGMAX

    passes: dict[str, int] = {}
    for name in ("expert_gateup", "expert_down"):
        per_slot = -(-per_layer[name] // slots // num_sms) if slots else 0
        passes[name] = max(1, per_slot) * slots
    for name in ("qkv", "o_proj", "router", "combine"):
        count = per_layer[name]
        packed = max(1, -(-count // _TILES_PER_INST))
        passes[name] = max(1, -(-packed // num_sms))
    # LM_HEAD is the only wave that needs many passes per SM.
    lm_packed = max(1, -(-(dims.vocab_size // tile_cols) // _TILES_PER_INST))
    passes["lm_head"] = max(1, -(-lm_packed // num_sms))

    serial_levels = dims.num_layers * len(MOE_LEVELS_PER_LAYER) + len(TERMINAL_LEVELS)
    # The longest K loop in the step: expert down contracts over expert_ffn_dim with
    # a 16-wide tile per K step. Every other projection contracts over hidden.
    k_steps = max(dims.hidden_size, dims.expert_ffn_dim) // tile_cols

    return WorkPlan(
        tile_cols=tile_cols,
        expert_slots=slots,
        tiles_per_layer={**per_layer, "lm_head": dims.vocab_size // tile_cols},
        tiles_total=tiles_total,
        instructions_per_sm_per_wave=passes,
        serial_levels=serial_levels,
        k_steps_per_level=k_steps,
    )


#: The five serial dependency levels one MoE layer contributes.  Mirrors the opcode
#: order in docs/22 §3; the count is what bounds cross-SM barrier cost.
MOE_LEVELS_PER_LAYER = ("attn", "router", "expert_gateup", "expert_down", "combine")

#: Terminal levels after the last layer: LM_HEAD tiles then the argmax wave.
TERMINAL_LEVELS = ("lm_head", "lm_argmax")


def expert_token_load(batch_size: int, dims: MoEDims = OLMOE_1B_7B) -> dict[str, int]:
    """Per-expert token histogram bounds at ``batch_size``.

    The batch-1 result is the load-bearing one:

    * ``slots == top_k`` and ``max == min == 1``, so every active expert carries
      exactly one token -- the histogram is flat *by construction*, not by luck;
    * therefore ``runtime == max_e`` and there is nothing for a work-stealing queue
      to balance, which is why the batch-1 design needs neither a sort nor a drain.

    As ``batch`` grows the same bound degrades to ``ceil(batch * top_k /
    num_experts)`` on average but ``batch`` in the worst case, which is exactly when
    dynamic claiming starts paying.
    """

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1; got {batch_size}")
    dims.validate()
    slots = batch_size * dims.top_k
    if batch_size == 1:
        return {
            "slots": slots,
            "min_tokens_per_active_expert": 1,
            "max_tokens_per_active_expert": 1,
            "expected_tokens_per_expert": 0,  # exact: 1 for active, 0 for the rest
            "balanced_by_construction": 1,
        }
    return {
        "slots": slots,
        "min_tokens_per_active_expert": 1,
        "max_tokens_per_active_expert": batch_size,
        "expected_tokens_per_expert": 0,
        "balanced_by_construction": 0,
    }


# =============================================================================
# Reporting
# =============================================================================


def render_report(budget: DecodeBudget, plan: WorkPlan | None = None) -> str:
    """Human-readable one-page summary; mirrors docs/22 §0."""

    bytes_ = budget.bytes
    lines = [
        f"# Sparse decode budget: {budget.dims.name}",
        "",
        f"batch={budget.batch_size}  ctx={budget.ctx}  "
        f"bandwidth={budget.bandwidth_gbps:.1f} GB/s (measured)",
        "",
        "| component | bytes/token | share of [W] |",
        "|---|---:|---:|",
    ]
    for part in BUDGET_PARTS:
        lines.append(
            f"| {part} | {getattr(bytes_, part):,} | {bytes_.share(part):.1%} |"
        )
    lines += [
        f"| **[W] total** | **{bytes_.weight_bytes:,}** | 100% |",
        "",
        f"KV read @ctx={budget.ctx}: {budget.kv_bytes:,} B "
        f"({budget.kv_bytes / bytes_.public_bytes(budget.kv_bytes):.1%} of [T])",
        f"[T] total: {bytes_.public_bytes(budget.kv_bytes):,} B",
        "",
        f"floor_W = {budget.weight_floor_ms:.3f} ms/token",
        f"floor_T = {budget.total_floor_ms:.3f} ms/token",
        f"floor if experts are materialized (gather path) = "
        f"{budget.active_weight_floor_ms:.3f} ms/token "
        f"-> gather costs {budget.gather_saving_ms:.3f} ms/token",
        "",
        f"sparse FLOPs/step = {budget.flops / 1e9:.3f} GFLOP -> "
        f"{budget.compute_floor_ms:.4f} ms at {budget.compute_peak_tflops:.1f} TFLOP/s "
        f"({budget.compute_floor_ms / budget.total_floor_ms:.2%} of the memory floor)",
    ]
    if plan is not None:
        lines += [
            "",
            f"expert slots = {plan.expert_slots}; tiles/step = {plan.tiles_total:,}; "
            f"serial levels = {plan.serial_levels}",
            "",
            "| wave | tiles/layer | instruction passes/SM |",
            "|---|---:|---:|",
        ]
        for name, count in plan.tiles_per_layer.items():
            lines.append(
                f"| {name} | {count} | "
                f"{plan.instructions_per_sm_per_wave.get(name, '-')} |"
            )
    return "\n".join(lines) + "\n"


# =============================================================================
# CLI: re-derive docs/22 §0 on any machine, with an explicit denominator
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sparse (MoE) per-token byte budget and decode roofline. Every output "
            "number is config arithmetic; the bandwidth is an input, never a spec "
            "default."
        )
    )
    parser.add_argument(
        "--bandwidth-gbps",
        type=float,
        default=None,
        help=(
            "measured read bandwidth. Defaults to the mean of "
            "results/bandwidth.json (written by RUN_MODE=bandwidth), which is the "
            "only acceptable denominator. Passing a spec rate here is a choice the "
            "caller has to make explicitly."
        ),
    )
    parser.add_argument("--bandwidth-json", default=None)
    parser.add_argument("--batch-sizes", default="1", help="comma-separated")
    parser.add_argument("--ctx", type=int, default=1024)
    parser.add_argument(
        "--bytes-per-param",
        type=int,
        default=2,
        help="2 = fp16/bf16, 1 = int8/fp8; the quantization arm (see docs/22 §5)",
    )
    parser.add_argument("--num-sms", type=int, default=170)
    parser.add_argument("--json", dest="json_path")
    return parser


def main(argv: list[str] | None = None) -> int:
    from nanovllm.roofline import (
        DEFAULT_BANDWIDTH_JSON,
        load_measured_bandwidth,
    )

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.bandwidth_gbps is not None and args.bandwidth_json is not None:
        parser.error("--bandwidth-gbps and --bandwidth-json are mutually exclusive")
    if args.bandwidth_gbps is not None:
        bandwidth = float(args.bandwidth_gbps)
        source = "explicit --bandwidth-gbps"
    else:
        path = Path(args.bandwidth_json or DEFAULT_BANDWIDTH_JSON)
        try:
            bandwidth = load_measured_bandwidth(path)
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        source = str(path)

    batch_sizes = [int(part) for part in args.batch_sizes.split(",") if part.strip()]
    if not batch_sizes or any(size < 1 for size in batch_sizes):
        parser.error("--batch-sizes must be positive integers")

    plan = plan_work(OLMOE_1B_7B, batch_size=batch_sizes[0], num_sms=args.num_sms)
    budgets = [
        decode_budget(
            OLMOE_1B_7B,
            batch_size=size,
            ctx=args.ctx,
            bandwidth_gbps=bandwidth,
            bytes_per_param=args.bytes_per_param,
        )
        for size in batch_sizes
    ]
    print(f"roofline denominator: {bandwidth:.1f} GB/s  ({source})")
    print(render_report(budgets[0], plan))
    if len(budgets) > 1:
        print("| batch | floor_W ms | floor_T ms | FLOPs/step |")
        print("|---:|---:|---:|---:|")
        for size, budget in zip(batch_sizes, budgets):
            print(
                f"| {size} | {budget.weight_floor_ms:.3f} | "
                f"{budget.total_floor_ms:.3f} | {budget.flops / 1e9:.3f} G |"
            )
    if args.json_path:
        payload = {
            "model": OLMOE_1B_7B.name,
            "bandwidth_gbps": bandwidth,
            "bandwidth_source": source,
            "ctx": args.ctx,
            "bytes_per_param": args.bytes_per_param,
            "plan": plan.as_dict(),
            "budgets": [
                {
                    "batch_size": budget.batch_size,
                    "bytes": budget.bytes.as_dict(),
                    "kv_bytes": budget.kv_bytes,
                    "flops": budget.flops,
                    "weight_floor_ms": budget.weight_floor_ms,
                    "total_floor_ms": budget.total_floor_ms,
                    "gather_floor_ms": budget.active_weight_floor_ms,
                    "gather_saving_ms": budget.gather_saving_ms,
                    "expert_share": budget.expert_share,
                }
                for budget in budgets
            ],
        }
        Path(args.json_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
