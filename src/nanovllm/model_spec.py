"""The one place model geometry is defined.

Why this module exists
---------------------
Before it, one model's dimensions were written down in four independent places --
the CUDA kernel's ``constexpr`` block, ``backends/megakernel/taskgraph.py``,
``backends/megakernel/weights.py`` and ``roofline.ModelDims`` (with a fifth copy
growing inside ``moe_budget``).  Only one drift guard existed
(:func:`nanovllm.roofline.assert_kernel_constants`), and none of them was a
*source*: every consumer re-declared the same numbers.

That is survivable while there is exactly one model.  It stops being survivable the
moment a second model shares the same task graph, which is the whole point of
``docs/22``.  Two failure modes motivate the consolidation, and both are **silent**:

1. a copy drifts, and the failure surfaces as a wrong number rather than an error;
2. a derived constant is written as an *inline literal* instead of a reference --
   ``Q_SIZE = 16 * HEAD_DIM`` -- so it keeps producing the right value while the
   named constant it duplicates moves.  For this repo that is not hypothetical:
   ``weights.py`` had exactly ``Q_SIZE = 16 * HEAD_DIM`` and
   ``KV_SIZE = 8 * HEAD_DIM``.

:meth:`ModelSpec.assert_derived_from_spec` exists specifically for (2).  A plain
``==`` on the numbers cannot see the difference between a derived and a
coincidentally-equal literal, so callers hand in the raw expression as well and the
check is done on what the expression *references*.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, fields
from typing import Any

BYTES_PER_PARAM_BF16 = 2


@dataclass(frozen=True)
class ModelSpec:
    """Geometry the task-graph kernel is compiled for.

    Every field is a *requirement*, not a preference: the kernel's ``constexpr``
    block must equal these and :meth:`assert_matches_kernel` is how that is
    enforced.

    ``max_model_len`` and ``max_attn_splits`` live here rather than in the backend
    because both are baked into kernel-side storage: the RoPE table has
    ``max_model_len`` rows, and ``o_partial`` / ``lse_partial`` are indexed by
    ``max_attn_splits``.
    """

    name: str
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    #: RoPE table rows; also the engine's ``max_model_len`` ceiling.
    max_model_len: int = 2048
    #: Capacity of ``o_partial`` / ``lse_partial``, mirrored by
    #: ``schedule.default_geometry``'s ``max_attn_splits``.
    max_attn_splits: int = 4
    #: Tied embeddings change the *streamed* weight bytes, not the kernel shapes.
    tie_word_embeddings: bool = True
    #: Weight matrices the FFN contributes per layer: gate, up and down for a gated
    #: MLP. Each one is its own device pointer in the layer ABI.
    ffn_weight_matrices: int = 3
    #: RMSNorm vectors *inside* a layer (the input and post-attention pair). A layer's
    #: per-head ``q_norm`` / ``k_norm`` are counted separately via :attr:`qk_norm`;
    #: the final output norm is not part of a layer at all.
    layer_norms: int = 2
    #: Per-head QK RMSNorm. Qwen3-0.6B has it despite being grouped-query; so does
    #: OLMoE. It changes both the layer's pointer count and its streamed bytes.
    qk_norm: bool = True

    # -- validation --------------------------------------------------------

    def assert_valid(self) -> None:
        """Reject geometry the task-graph kernel cannot represent."""

        for f in fields(self):
            if f.type != "int":
                continue
            value = getattr(self, f.name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{f.name} must be a positive int; got {value!r}")
        if self.num_q_heads % self.num_kv_heads:
            raise ValueError(
                f"num_q_heads ({self.num_q_heads}) must be a multiple of "
                f"num_kv_heads ({self.num_kv_heads}); the kernel's GQA mapping "
                f"needs an integral group size"
            )
        group_size = self.group_size
        if group_size > 16 or (group_size & (group_size - 1)):
            raise ValueError(
                f"group size {group_size} (= num_q_heads / num_kv_heads) must be a "
                f"power of two no larger than the 16-wide WMMA tile; the kernel's "
                f"per-head projection pipeline has no path for it"
            )
        # Every projection is `hidden x (heads*head_dim)` shaped and the kernel
        # tiles it with 16-wide WMMA N tiles. A width that does not divide leaves a
        # tail tile the schedule silently never emits.
        for label, width in (("q", self.q_size), ("kv", self.kv_size)):
            if width % 16:
                raise ValueError(
                    f"{label}_size={width} is not a multiple of the 16-wide "
                    f"WMMA N tile"
                )
        for label, width in (
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
        ):
            if width % 16:
                raise ValueError(f"{label}={width} is not a multiple of 16")
        if self.head_dim % 16:
            raise ValueError(f"head_dim={self.head_dim} is not a multiple of 16")

    # -- derived sizes -----------------------------------------------------

    @property
    def q_size(self) -> int:
        return self.num_q_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def group_size(self) -> int:
        """Query heads per KV head (1 for multi-head attention)."""

        return self.num_q_heads // self.num_kv_heads

    @property
    def num_bar_arrays(self) -> int:
        """Distinct barrier arrays the dense task graph synchronises through."""

        return 8

    @property
    def weight_pointer_count(self) -> int:
        """Device pointers one layer's weight ABI carries.

        Per layer, in ``tg_abi.h``'s declared order: the attention projections
        (q, k, v, o), the layer's RMSNorm vectors, its optional per-head QK norms,
        and ``ffn_weight_matrices`` FFN matrices (gate, up, down).  Qwen3-0.6B is
        11 = 4 + 2 + 2 (its ``q_norm`` / ``k_norm``, which it *does* have even
        though it is grouped-query) + 3.

        This belongs on the spec because that struct hard-locks the count at 11,
        while a sparse model needs a second level of indirection (a per-expert
        pointer table).  Making the count *derived* is what lets ``docs/22``'s MoE
        opcodes land without the struct and its ABI test drifting apart.
        """

        return (
            4
            + self.layer_norms
            + (2 if self.qk_norm else 0)
            + self.ffn_weight_matrices
        )

    @property
    def qk_norm_params_per_layer(self) -> int:
        """Parameters in the per-head ``q_norm`` / ``k_norm`` pair.

        Each vector is ``head_dim`` wide, **not** ``head_dim * num_q_heads``: Qwen3
        normalises the *head dimension* of each head, so the tensor is ``(128,)`` for
        Qwen3-0.6B and ``(128,)`` for OLMoE alike (verified against the instantiated
        model and against the OLMoE checkpoint's tensors).  Both counts are
        plausible, so this is written down where it can be checked rather than
        inferred: using ``num_q_heads * head_dim`` overcounts by 2,048 per layer, and
        using a single vector undercounts by 128.
        """

        if not self.qk_norm:
            return 0
        return 2 * self.head_dim

    def attention_params_per_layer(self) -> int:
        """q, k, v and o projections plus the layer's own two RMSNorm vectors.

        Excludes :meth:`qk_norm_params_per_layer` (added by
        :meth:`params_per_layer`) and :meth:`final_norm_params` (one tensor for the
        whole model, so folding it in here would charge it ``num_layers`` times).
        """

        qkv = self.hidden_size * (self.q_size + 2 * self.kv_size)
        o_proj = self.q_size * self.hidden_size
        norms = self.layer_norms * self.hidden_size
        return qkv + o_proj + norms

    def final_norm_params(self) -> int:
        """The output RMSNorm, applied once after the last layer.

        Deliberately **not** divided across layers.  ``roofline.ModelDims`` amortises
        it into ``params_per_layer``; that reports the same *total* but makes the
        per-layer figure 2,048 too high and describes the model as carrying one more
        norm vector per layer than it has.
        """

        return self.hidden_size

    def ffn_params_per_layer(self) -> int:
        """FFN parameters per layer: ``ffn_weight_matrices`` x hidden x intermediate.

        For Qwen3-0.6B that is gate, up and down, each 1024 x 3072.
        """

        return self.ffn_weight_matrices * self.hidden_size * self.intermediate_size

    def params_per_layer(self) -> int:
        """Every parameter a layer streams: attention + layer norms + QK norms + FFN.

        For Qwen3-0.6B this is 15,730,944, reproduced exactly from the checkpoint's
        own per-layer tensor list (11 tensors, counted on the instantiated model)::

            q_proj              2,048 x 1,024 = 2,097,152
            k_proj              1,024 x 1,024 = 1,048,576
            v_proj              1,024 x 1,024 = 1,048,576
            o_proj              1,024 x 2,048 = 2,097,152
            input + post norm        2 x 1,024 =     2,048
            q_norm + k_norm         2 x   128 =       256
            gate + up + down    3 x 1,024 x 3,072 = 9,437,184
        """

        return (
            self.attention_params_per_layer()
            + self.qk_norm_params_per_layer
            + self.ffn_params_per_layer()
        )

    def total_layer_params(self) -> int:
        """``num_layers * params_per_layer() + final_norm_params()``.

        The total is where this agrees with ``roofline.ModelDims``; the two differ
        only in which side of the division the final norm sits on, so any *total*
        comparison must use this method rather than a per-layer one.
        """

        return self.num_layers * self.params_per_layer() + self.final_norm_params()

    def bytes_per_layer_weights(
        self, *, bytes_per_param: int = BYTES_PER_PARAM_BF16
    ) -> int:
        return self.params_per_layer() * int(bytes_per_param)

    # -- kernel-side contract ---------------------------------------------

    def kernel_constants(self) -> dict[str, int]:
        """``name -> value`` that the kernel's ``constexpr`` block must match.

        Keys are the names :func:`nanovllm.roofline.parse_kernel_constants` reads,
        so this dict is directly comparable to that parser's output -- including the
        kernel's *derived* ``TG_Q_SIZE`` / ``TG_KV_SIZE``, which is what makes an
        inline literal in the kernel detectable.
        """

        return {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_q_heads": self.num_q_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "num_layers": self.num_layers,
            "vocab_size": self.vocab_size,
            "q_size": self.q_size,
            "kv_size": self.kv_size,
        }

    def kernel_defines(self) -> dict[str, int]:
        """``-D`` defines that pin the kernel's geometry to this spec.

        ``build.py`` already injects ``-DTG_BLOCK_SIZE`` / ``-DTG_B_STAGES``; these
        are the same mechanism one level up.  Until they are wired in, the kernel's
        own ``#ifndef`` fallbacks are the live values and
        :meth:`assert_matches_kernel` is what keeps them honest.
        """

        return {
            "TG_HIDDEN": self.hidden_size,
            "TG_INTERMEDIATE": self.intermediate_size,
            "TG_NUM_Q_HEADS": self.num_q_heads,
            "TG_NUM_KV_HEADS": self.num_kv_heads,
            "TG_HEAD_DIM": self.head_dim,
            "TG_NUM_LAYERS": self.num_layers,
            "TG_VOCAB": self.vocab_size,
        }

    def assert_matches_kernel(self, source: str | Any | None = None) -> None:
        """Raise unless the checked-in CUDA kernel is compiled for this spec."""

        from nanovllm.roofline import DEFAULT_KERNEL_SOURCE, parse_kernel_constants

        target = DEFAULT_KERNEL_SOURCE if source is None else source
        actual = parse_kernel_constants(target)
        mismatches = {
            key: (want, actual.get(key))
            for key, want in self.kernel_constants().items()
            if actual.get(key) != want
        }
        if mismatches:
            detail = ", ".join(
                f"{key}: spec={want} kernel={got}"
                for key, (want, got) in sorted(mismatches.items())
            )
            raise ValueError(f"{self.name} disagrees with {target}: {detail}")

    def assert_derived_from_spec(self, published: dict[str, tuple[Any, Any]]) -> None:
        """Raise unless every published constant was *derived* from this spec.

        ``published`` maps ``name -> (value, raw_expression)``.  The expression is
        required because a value-only check is blind to the failure this catches::

            NUM_Q_HEADS = 16
            Q_SIZE = 16 * HEAD_DIM      # right answer today, duplicate literal

        Both forms reference ``HEAD_DIM``, so *which names appear* cannot separate
        them.  What separates them is the **literals**: ``16`` is a hard-coded copy
        of ``num_q_heads``, while ``NUM_Q_HEADS`` carries the name.  So the check is
        on the expression tree -- reject a bare literal where the spec has a field,
        and reject a literal that duplicates a field's value while that field's
        canonical name is absent from the expression.
        """

        field_names = {f.name for f in fields(self)}
        problems: list[str] = []
        for name, (value, expression) in published.items():
            expected = getattr(self, name, None)
            if expected is None:
                continue
            if value != expected:
                problems.append(f"{name}: published {value!r}, spec says {expected!r}")
                continue
            if expression is None:
                problems.append(f"{name}: no expression supplied to check derivation")
                continue
            problems.extend(self._derivation_problems(name, expression, field_names))
        if problems:
            raise ValueError(
                f"{self.name}: published constants must derive from the spec: "
                + "; ".join(problems)
            )

    def _derivation_problems(
        self, name: str, expression: Any, field_names: set[str]
    ) -> list[str]:
        if isinstance(expression, str):
            try:
                tree: ast.AST = ast.parse(expression, mode="eval")
            except SyntaxError:
                return [f"{name}={expression!r} is not a parseable expression"]
        elif isinstance(expression, ast.AST):
            tree = expression
        else:
            return [f"{name}: expression must be source text or an AST"]

        if isinstance(tree, ast.Constant):
            return [
                f"{name}={ast.unparse(tree)!r} is a bare literal; it must reference "
                f"the field {name!r}"
            ]

        seen_names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        problems: list[str] = []
        for node in ast.walk(tree):
            # A subscript (TG_DIMS["head_dim"]) is a genuine reference to the spec.
            if not isinstance(node, ast.Constant) or not isinstance(node.value, int):
                continue
            if node.value in (0, 1):
                continue
            for field_name in field_names:
                if getattr(self, field_name) != node.value:
                    continue
                canonical = field_name.upper()
                if canonical in seen_names:
                    continue
                problems.append(
                    f"{name}={ast.unparse(tree)!r} contains the literal "
                    f"{node.value}, which duplicates {field_name}="
                    f"{node.value} while {canonical!r} is not referenced"
                )
        return problems

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)




# ---------------------------------------------------------------------------
# The models this repo knows about.
# ---------------------------------------------------------------------------

#: Qwen3-0.6B -- the dense task-graph target (``docs/13``). Values are byte-identical
#: to the four copies they replace; consolidating them changed no number.
QWEN3_06B = ModelSpec(
    name="Qwen3-0.6B",
    num_layers=28,
    hidden_size=1024,
    intermediate_size=3072,
    num_q_heads=16,
    num_kv_heads=8,
    head_dim=128,
    vocab_size=151936,
    max_model_len=2048,
    max_attn_splits=4,
    tie_word_embeddings=True,
)

#: OLMoE-1B-7B-0924 -- the sparse target (``docs/22``). Declared here so the
#: *shared* geometry (layers, hidden, heads, vocab) has one home. Its expert/FFN
#: arithmetic stays in ``nanovllm.moe_budget``, because a sparse FFN is a different
#: parameterisation (`num_experts x expert_ffn_dim`), not a second value of
#: ``intermediate_size``. Note ``num_key_value_heads == num_attention_heads``: OLMoE
#: is multi-head, not grouped-query, which doubles its per-token KV read.
OLMOE_1B_7B = ModelSpec(
    name="OLMoE-1B-7B-0924",
    num_layers=16,
    hidden_size=2048,
    intermediate_size=1024,
    num_q_heads=16,
    num_kv_heads=16,
    head_dim=128,
    vocab_size=50304,
    max_model_len=4096,
    max_attn_splits=4,
    tie_word_embeddings=False,
)

#: Everything the compile step can be asked for.
KNOWN_SPECS = {spec.name: spec for spec in (QWEN3_06B, OLMOE_1B_7B)}
