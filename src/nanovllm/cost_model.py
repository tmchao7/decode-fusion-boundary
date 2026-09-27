"""Structural cost model for one task-graph decode step.

``docs/12`` §1.2 modeled the step as ``waves x K-steps/wave x time-per-K-step``
with 197 waves and 64 K-steps each, and fit 857 ns per K-step from the measured
10.8 ms/step. This module rebuilds that model from the **actual schedule**
instead of a hand-counted wave formula, which changes three things:

1. **Waves are dependency levels.** Grouping the decoded instruction stream by
   dependency depth gives 170 levels at ``attn_splits=1`` (each level exactly one
   opcode -- asserted, not assumed) instead of 197: the hand count included
   ``ATTN_COMBINE``, which emits no instruction at ``attn_splits=1``. At
   ``attn_splits=2`` the combine wave exists and the count is 198.
2. **A wave can need several rounds.** A wave with more instructions than SMs
   serializes: the LM head emits 1357 instructions, i.e. 13 rounds over 108 SMs,
   which the "64 K-steps per wave" rule did not account for.
3. **K is per-opcode.** ``FFN_DOWN`` contracts over ``intermediate`` (3072), so
   it is 192 K-steps, not 64 -- a 3x term in every one of the 28 layers.

With those corrections, one calibration point (10.8 ms/step at 108 SMs) predicts
the **documented 1-SM measurement of 275 ms/step to within ~5%**, which is the
out-of-sample check in ``tests/test_cost_model.py``. The model is calibrated for
the wave-serialized regime; it does **not** explain the batch-parallel collapse
at ``BS>=2`` (``docs/07`` §2.3), and it says so rather than extrapolating.

Nothing here needs a GPU: the structure comes from ``schedule.py``, and the only
measured input is the per-K-step cost, which callers pass in with its provenance.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

from nanovllm.backends.megakernel import schedule as tg
from nanovllm.backends.megakernel import verifier as V

#: K dimension per GEMM opcode, named by the geometry field it contracts over.
_GEMM_K_FIELD = {
    tg.TgOp.QKV_PROJ: "hidden",
    tg.TgOp.O_PROJ: "hidden",
    tg.TgOp.FFN_UPGATE_ACT: "hidden",
    tg.TgOp.FFN_DOWN: "intermediate",
    tg.TgOp.LM_HEAD: "hidden",
}

#: Waves with no fixed contraction: attention tracks the KV length (one step at
#: the ctx=1 baseline), and the elementwise waves have no serial K loop.
_NON_GEMM_K_STEPS = {
    "ATTN_DECODE": 1,  # one KV step at the ctx=1 baseline
    "ATTN_COMBINE": 0,  # a reduction over KV heads, no serial K loop
    "ADD_RMSNORM": 0,
    "LM_ARGMAX": 0,
    "NOP": 0,
}


@dataclass(frozen=True)
class Wave:
    """One dependency level: a homogeneous group of instructions."""

    level: int
    name: str
    instructions: int
    rounds: int
    k_steps: int

    @property
    def k_total(self) -> int:
        return self.rounds * self.k_steps


@dataclass(frozen=True)
class StepStructure:
    waves: tuple[Wave, ...]
    instructions: int

    @property
    def levels(self) -> int:
        """Dependency depth == number of serial waves."""

        return len(self.waves)

    @property
    def rounds(self) -> int:
        return sum(wave.rounds for wave in self.waves)

    @property
    def k_steps(self) -> int:
        return sum(wave.k_total for wave in self.waves)

    def k_steps_by_wave(self) -> dict[str, int]:
        totals: dict[str, int] = defaultdict(int)
        for wave in self.waves:
            totals[wave.name] += wave.k_total
        return dict(sorted(totals.items()))

    def rounds_by_wave(self) -> dict[str, int]:
        totals: dict[str, int] = defaultdict(int)
        for wave in self.waves:
            totals[wave.name] += wave.rounds
        return dict(sorted(totals.items()))


def dependency_levels(
    prepared: V.PreparedSchedule,
) -> list[list[V.DecodedInstruction]]:
    """Group instructions by depth in the wait graph.

    Depth is the longest chain of instructions a consumer transitively waits for,
    so everything at depth *d* can run concurrently with its siblings and strictly
    before anything at depth *d+1* -- which is what makes a level a wave. Computed
    with Kahn's algorithm: a cycle (which the verifier reports separately) leaves
    nodes unprocessed, so it raises instead of returning something meaningless.
    """

    flat = [inst for stream in prepared.streams for inst in stream]
    producers: dict[V.BarKey, list[int]] = defaultdict(list)
    for position, inst in enumerate(flat):
        for key, count in inst.arrivals:
            producers[key].extend([position] * count)

    # Deduplicate: one instruction can supply several of the arrivals a consumer
    # waits for, and in-degree must count it once.
    dependencies = [
        sorted(
            {
                producer
                for condition in inst.waits
                for producer in producers.get(condition.key, [])[: condition.threshold]
            }
        )
        for inst in flat
    ]

    children: dict[int, list[int]] = defaultdict(list)
    in_degree = [len(parents) for parents in dependencies]
    for node, parents in enumerate(dependencies):
        for parent in parents:
            children[parent].append(node)

    depth = [1] * len(flat)
    ready = deque(node for node in range(len(flat)) if in_degree[node] == 0)
    processed = 0
    while ready:
        node = ready.popleft()
        processed += 1
        for child in children[node]:
            depth[child] = max(depth[child], depth[node] + 1)
            in_degree[child] -= 1
            if in_degree[child] == 0:
                ready.append(child)
    if processed != len(flat):
        raise RuntimeError(
            f"dependency depth did not converge ({processed}/{len(flat)} nodes): "
            "the wait graph has a cycle -- run the verifier to see it"
        )

    levels: dict[int, list[V.DecodedInstruction]] = defaultdict(list)
    for position, inst in enumerate(flat):
        levels[depth[position]].append(inst)
    return [levels[key] for key in sorted(levels)]


def _wave_k_steps(name: str, geo: tg.TgGeometry) -> int:
    """Serial K steps contributed by one instruction of this wave."""

    for opcode, field in _GEMM_K_FIELD.items():
        if V.opcode_name(opcode) == name:
            return max(1, int(getattr(geo, field)) // tg.TgTiling.qkv_bk)
    if name in _NON_GEMM_K_STEPS:
        return _NON_GEMM_K_STEPS[name]
    raise ValueError(f"unknown wave {name!r}: the model has no K for it")


def structure_from_schedule(
    sched,
    num_sms: int,
    geo: tg.TgGeometry,
) -> StepStructure:
    """Build the wave/K-step structure of one decode step."""

    if num_sms < 1:
        raise ValueError(f"num_sms must be >= 1; got {num_sms}")
    return structure_from_prepared(V.prepare(sched, num_sms), geo)


def structure_from_segments(
    segments: Iterable[object],
    num_sms: int,
    geo: tg.TgGeometry,
) -> StepStructure:
    """Structure of a multi-launch step (``fusion_groups > 1``)."""

    if num_sms < 1:
        raise ValueError(f"num_sms must be >= 1; got {num_sms}")
    return structure_from_prepared(V.prepare_segments(segments, num_sms), geo)


def structure_from_prepared(
    prepared: V.PreparedSchedule,
    geo: tg.TgGeometry,
) -> StepStructure:
    """Build the wave/K-step structure from decoded streams.

    Only ``hidden``/``intermediate`` and the 16-wide K tile matter, so the model
    is insensitive to how tiles are packed into instructions.
    """

    num_sms = len(prepared.streams)
    levels = dependency_levels(prepared)

    waves: list[Wave] = []
    for level, group in enumerate(levels, start=1):
        names = {inst.name for inst in group}
        if len(names) != 1:
            raise RuntimeError(
                f"dependency level {level} mixes waves {sorted(names)}; the "
                "level-is-a-wave assumption does not hold"
            )
        name = names.pop()
        instructions = len(group)
        waves.append(
            Wave(
                level=level,
                name=name,
                instructions=instructions,
                rounds=-(-instructions // num_sms),  # ceil
                k_steps=_wave_k_steps(name, geo),
            )
        )

    return StepStructure(waves=tuple(waves), instructions=sum(len(g) for g in levels))


def predict_ms(structure: StepStructure, k_step_ns: float) -> float:
    """Predicted device time per decode step, in milliseconds."""

    if k_step_ns <= 0:
        raise ValueError(f"k_step_ns must be > 0; got {k_step_ns}")
    return structure.k_steps * k_step_ns / 1e6


def fit_k_step_ns(structure: StepStructure, measured_ms: float) -> float:
    """Back out the per-K-step cost from a measurement of this structure.

    This is a *calibration*, not a prediction: the constant only describes the
    kernel version and shape the measurement came from. Its use is to predict
    other configurations of that same kernel.
    """

    if structure.k_steps <= 0:
        raise ValueError("structure has no K steps to calibrate against")
    if measured_ms <= 0:
        raise ValueError(f"measured_ms must be > 0; got {measured_ms}")
    return measured_ms * 1e6 / structure.k_steps


# ---------------------------------------------------------------------------
# Pre-registered predictions for the fusion-granularity experiment
# ---------------------------------------------------------------------------

#: Measured on the archived ``kernel.cu`` line: 336 full-grid barriers cost
#: 0.547 ms, i.e. 1.63 us per grid sync (docs/07 §3.4). A segment boundary is a
#: launch boundary -- a drain of the previous launch plus a fresh grid start -- so
#: two grid syncs is the documented *lower* bound for one extra segment.
GRID_SYNC_US = 1.63

#: Documented host residual for the cudagraph line: 2.2-2.8 ms/step across the
#: ~200 launches a step takes, i.e. ~11 us of host work per launch. Used as the
#: *upper* bound for a segment boundary.
HOST_LAUNCH_US = 11.0


def segment_delta_ms(
    groups: Iterable[int],
    *,
    base_ms: float,
    per_segment_us: float,
) -> dict[int, float]:
    """``T(g) = T(1) + (g-1) * per_segment`` for each requested granularity."""

    if base_ms <= 0:
        raise ValueError(f"base_ms must be > 0; got {base_ms}")
    if per_segment_us < 0:
        raise ValueError(f"per_segment_us must be >= 0; got {per_segment_us}")
    return {g: base_ms + (g - 1) * per_segment_us / 1000.0 for g in groups if g >= 1}


def preregistered_granularity_prediction(
    *,
    base_device_ms: float,
    groups: Iterable[int] = (1, 2, 4, 7, 14, 28),
) -> dict[str, object]:
    """The prediction the fusion-granularity sweep is meant to test.

    The verifier proves segmentation conserves the instruction stream and its wave
    structure (``g in {1,2,4,7,14,28}`` all give 170 levels and 4522 instructions
    at BS=1), so the model has exactly one channel for a granularity effect: extra
    launch boundaries. Bounded between a grid-sync pair and a host launch, ``g=28``
    costs at most ~16% of a 1.89 ms/step device time -- hence the pre-registered
    expectation in ``docs/14`` §5: flat-to-monotone-worse, and a valley would
    refute the claim that the bars already serialize the same boundaries.
    """

    ordered = sorted(g for g in groups if g >= 1)
    if not ordered:
        raise ValueError("groups must contain at least one granularity >= 1")
    lower = segment_delta_ms(ordered, base_ms=base_device_ms, per_segment_us=2 * GRID_SYNC_US)
    upper = segment_delta_ms(ordered, base_ms=base_device_ms, per_segment_us=HOST_LAUNCH_US)
    worst = ordered[-1]
    bound = (upper[worst] - base_device_ms) / base_device_ms
    return {
        "base_device_ms": base_device_ms,
        "groups": ordered,
        "device_ms_lower_bound": lower,
        "device_ms_upper_bound": upper,
        "worst_case_relative_change": bound,
        "refuted_if_measured_delta_exceeds": bound,
        "note": (
            "the instruction stream and wave structure are invariant in g "
            "(verifier), so a larger measured effect comes from something the "
            "model does not represent: launch/drain cost beyond these bounds, or "
            "lost L2 reuse across segment boundaries"
        ),
    }
