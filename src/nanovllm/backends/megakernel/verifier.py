"""Static verification of a task-graph decode schedule.

A cooperative launch is correct only if an invariant the CUDA compiler cannot
check holds: **every consumer's wait is satisfied by a producer that is reachable
before the consumer stalls**. ``schedule.py`` emits that instruction stream;
this module decides whether the emitted stream can deadlock, and how strongly it
can say so.

Two detectors and one localizer
-------------------------------

``execution`` and ``arrivals`` are independent **detectors**: either can reject a
schedule the other accepts. ``wait_graph`` is a **localizer** -- on this kernel's
schedule family it never fires without ``execution`` also failing (measured: 720
sampled mutants across six geometries, rejected 564 times by ``execution`` and
156 by ``arrivals``, never by ``wait_graph``), but when a deadlock does happen it
names the instruction cycle instead of reporting "these SMs stalled". Read the
per-check rejection counts in ``AdversarialReport.rejected_checks`` rather than
assuming all three are load-bearing.

1. ``execution`` -- greedy-maximal simulation. Each SM is a program counter that
   advances when its next instruction's waits are satisfied. Repeating a full
   scan until no SM can progress is a maximal execution, because bar counters are
   monotone (executing an instruction never disables another wait). A shortfall
   is therefore a real deadlock, not a simulation artifact.
2. ``wait_graph`` -- the wait relation between instructions must be acyclic. This
   is a different argument from (1): it does not depend on the greedy pick order,
   and it can name a cycle whose every member is individually reachable.
3. ``arrivals`` -- for every bar, the produced count must exactly cover the
   largest threshold anyone waits on. Over-arrival means someone arrives twice
   (a real hazard the kernel's counters would mask); under-arrival is a deadlock
   that (1) usually reports first.

Adversarial mutation
--------------------

``adversarial_report`` derives schedules that *must* be rejected -- a dropped
arrival, a raised threshold, a shifted bar key, a duplicated arrival, a
truncated per-SM stream -- and counts how many the verifier accepts anyway. The
metric that matters is **false accepts**, and it must be zero: that is the same
claim AutoMegaKernel's schedule-IR validator makes at a much larger scale
(7,160 adversarial schedules, zero false accepts). ``tools/verify_schedule.py``
reports the counts for this kernel.

Everything here is pure Python and torch-free, so it runs in the CPU test suite.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Iterator, NamedTuple

from nanovllm.backends.megakernel import schedule as tg

# ---------------------------------------------------------------------------
# Instruction decoding. The field indices mirror ``csrc/taskgraph_kernel.cu``;
# ``tests/test_taskgraph_abi.py`` parses the CUDA header and locks the opcodes,
# and ``tests/test_schedule_verifier.py`` locks the field use below.
# ---------------------------------------------------------------------------

#: AttnDecode's arrival index doubles as a target selector: bit 31 set means the
#: arrival belongs to the *combine* bar rather than the attention bar.
_COMBINE_BIT = 1 << 31


class BarKey(NamedTuple):
    """A single counter in one of the global bar arrays.

    A tuple subclass on purpose: callers (and the ABI tests) index bar state as
    ``bars[("layer", 1)]``, and ``BarKey("layer", 1)`` must be the same key.
    """

    kind: str
    index: int


@dataclass(frozen=True)
class Condition:
    """``bar[key] >= threshold``."""

    key: BarKey
    threshold: int


@dataclass(frozen=True)
class DecodedInstruction:
    opcode: int
    fields: tuple[int, ...]
    waits: tuple[Condition, ...]
    arrivals: tuple[tuple[BarKey, int], ...]

    @property
    def name(self) -> str:
        return opcode_name(self.opcode)

    @property
    def describe(self) -> str:
        produced = ", ".join(f"{k.kind}[{k.index}]+={n}" for k, n in self.arrivals)
        consumed = ", ".join(f"{c.key.kind}[{c.key.index}]>={c.threshold}" for c in self.waits)
        return f"{self.name}(wait: {consumed or '-'}; arrive: {produced or '-'})"


def opcode_name(opcode: int) -> str:
    """Public opcode -> name lookup (``op<N>`` for unknown opcodes)."""

    return _OPCODE_NAMES.get(opcode, f"op{opcode}")


_OPCODE_NAMES = {
    tg.TgOp.NOP: "NOP",
    tg.TgOp.QKV_PROJ: "QKV_PROJ",
    tg.TgOp.ATTN_DECODE: "ATTN_DECODE",
    tg.TgOp.ATTN_COMBINE: "ATTN_COMBINE",
    tg.TgOp.O_PROJ: "O_PROJ",
    tg.TgOp.FFN_UPGATE_ACT: "FFN_UPGATE_ACT",
    tg.TgOp.FFN_DOWN: "FFN_DOWN",
    tg.TgOp.ADD_RMSNORM: "ADD_RMSNORM",
    tg.TgOp.LM_HEAD: "LM_HEAD",
    tg.TgOp.LM_ARGMAX: "LM_ARGMAX",
}


def decode_instruction(inst: Iterable[int]) -> DecodedInstruction:
    """Turn one raw instruction row into its waits and arrivals.

    NOP (and any opcode without dependencies) decodes to no waits and no
    arrivals rather than raising: the verifier must be able to look at a partially
    built stream.
    """

    fields = tuple(int(value) for value in inst)
    if len(fields) < tg.TG_INST_WIDTH:
        raise ValueError(
            f"instruction needs {tg.TG_INST_WIDTH} ints, got {len(fields)}"
        )
    op = fields[0]
    waits: tuple[Condition, ...] = ()
    arrivals: tuple[tuple[BarKey, int], ...] = ()

    if op == tg.TgOp.QKV_PROJ:
        waits = (Condition(BarKey("layer", fields[5]), fields[6]),)
        arrivals = ((BarKey("qkv", fields[7]), fields[9]),)
    elif op == tg.TgOp.ATTN_DECODE:
        waits = (Condition(BarKey("qkv", fields[7]), fields[8]),)
        target = fields[9]
        kind = "combine" if target & _COMBINE_BIT else "attn"
        arrivals = ((BarKey(kind, target & ~_COMBINE_BIT), 1),)
    elif op == tg.TgOp.ATTN_COMBINE:
        waits = (Condition(BarKey("combine", fields[5]), fields[6]),)
        arrivals = ((BarKey("attn", fields[7]), 1),)
    elif op == tg.TgOp.O_PROJ:
        waits = (Condition(BarKey("attn", fields[6]), fields[7]),)
        arrivals = ((BarKey("oproj", fields[8]), fields[9]),)
    elif op == tg.TgOp.FFN_UPGATE_ACT:
        waits = (Condition(BarKey("oproj", fields[6]), fields[7]),)
        arrivals = ((BarKey("silu", fields[8]), fields[9]),)
    elif op == tg.TgOp.FFN_DOWN:
        waits = (Condition(BarKey("silu", fields[6]), fields[7]),)
        arrivals = ((BarKey("ffn_down", fields[8]), fields[9]),)
    elif op == tg.TgOp.ADD_RMSNORM:
        waits = (
            Condition(BarKey("oproj", fields[3]), fields[4]),
            Condition(BarKey("ffn_down", fields[5]), fields[6]),
        )
        arrivals = ((BarKey("layer", fields[7]), 1),)
    elif op == tg.TgOp.LM_HEAD:
        waits = (Condition(BarKey("layer", fields[6]), fields[7]),)
        arrivals = ((BarKey("lm_head", fields[8]), fields[9]),)
    elif op == tg.TgOp.LM_ARGMAX:
        waits = (Condition(BarKey("lm_head", fields[6]), fields[7]),)

    return DecodedInstruction(
        opcode=op, fields=fields, waits=waits, arrivals=arrivals
    )


# ---------------------------------------------------------------------------
# Stream extraction
# ---------------------------------------------------------------------------


def _per_sm_streams(sched, num_sms: int) -> list[list[DecodedInstruction]]:
    """The decoded instruction stream of every SM, in program order."""

    if num_sms < 1:
        raise ValueError(f"num_sms must be >= 1; got {num_sms}")
    if num_sms > len(sched.num_inst_per_sm):
        raise ValueError(
            f"schedule has streams for {len(sched.num_inst_per_sm)} SMs, "
            f"asked for {num_sms}"
        )
    width = tg.TG_INST_WIDTH
    streams: list[list[DecodedInstruction]] = []
    for sm in range(num_sms):
        base = sm * sched.max_inst * width
        count = sched.num_inst_per_sm[sm]
        streams.append(
            [
                decode_instruction(
                    sched.flat_inst_buf[base + i * width : base + (i + 1) * width]
                )
                for i in range(count)
            ]
        )
    return streams


def _produced_counts(streams: Iterable[Iterable[DecodedInstruction]]) -> dict[BarKey, int]:
    """Total produced count per bar across every stream."""

    totals: dict[BarKey, int] = defaultdict(int)
    for stream in streams:
        for inst in stream:
            for key, count in inst.arrivals:
                totals[key] += count
    return dict(totals)


def _needed_thresholds(
    streams: Iterable[Iterable[DecodedInstruction]],
) -> dict[BarKey, int]:
    """Largest threshold any instruction waits on, per bar."""

    thresholds: dict[BarKey, int] = defaultdict(int)
    for stream in streams:
        for inst in stream:
            for condition in inst.waits:
                key = condition.key
                thresholds[key] = max(thresholds[key], condition.threshold)
    return dict(thresholds)


# ---------------------------------------------------------------------------
# Check 1: greedy-maximal execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedSchedule:
    """Decoded streams, so the three checks do not decode four times."""

    streams: tuple[tuple[DecodedInstruction, ...], ...]

    @property
    def total(self) -> int:
        return sum(len(stream) for stream in self.streams)


def prepare(sched, num_sms: int) -> PreparedSchedule:
    return PreparedSchedule(
        streams=tuple(tuple(stream) for stream in _per_sm_streams(sched, num_sms))
    )


@dataclass
class ExecutionResult:
    done: int
    total: int
    bars: dict[BarKey, int]
    pc: list[int]
    stalled_sms: tuple[int, ...] = ()

    @property
    def completed(self) -> bool:
        return self.done == self.total

    @property
    def shortfall(self) -> int:
        return self.total - self.done


def execute_prepared(
    prepared: PreparedSchedule,
    bars: dict[BarKey, int] | None = None,
) -> ExecutionResult:
    """Run a maximal execution of a prepared schedule.

    ``bars`` may be pre-seeded and is **mutated in place** and returned, so a
    caller can carry counters across consecutive segments of one decode step --
    which is exactly the property segmented fusion relies on.
    """

    if bars is None:
        bars = defaultdict(int)
    streams = prepared.streams
    num_sms = len(streams)
    total = prepared.total
    pc = [0] * num_sms
    done = 0

    def ready(inst: DecodedInstruction) -> bool:
        return all(bars[condition.key] >= condition.threshold for condition in inst.waits)

    while done < total:
        progressed = False
        for sm in range(num_sms):
            if pc[sm] >= len(streams[sm]):
                continue
            inst = streams[sm][pc[sm]]
            if ready(inst):
                for key, count in inst.arrivals:
                    bars[key] += count
                pc[sm] += 1
                done += 1
                progressed = True
        if not progressed:
            break

    stalled = tuple(
        sm for sm in range(num_sms) if pc[sm] < len(streams[sm])
    )
    return ExecutionResult(done=done, total=total, bars=bars, pc=pc, stalled_sms=stalled)


def execute(
    sched,
    num_sms: int,
    bars: dict[BarKey, int] | None = None,
) -> ExecutionResult:
    """Convenience wrapper: prepare then execute."""

    return execute_prepared(prepare(sched, num_sms), bars)


# ---------------------------------------------------------------------------
# Check 2: acyclic wait graph
# ---------------------------------------------------------------------------


def wait_graph_cycle(
    sched, num_sms: int
) -> list[tuple[int, int]] | None:
    """Convenience wrapper: prepare then check."""

    return wait_graph_cycle_prepared(prepare(sched, num_sms))


def wait_graph_cycle_prepared(
    prepared: PreparedSchedule,
) -> list[tuple[int, int]] | None:
    """Return one dependency cycle in the flattening order, or ``None``.

    Instruction *u* depends on instruction *v* when *v* could supply an arrival
    *u* waits for. Concretely: a wait ``bar X >= t`` is linked to the first *t*
    arrivals to *X* in the flattening order, **including ones that sit later than
    *u*** -- the kernel's counters are device-wide, so "later in program order" is
    not "unavailable".

    How strong that is: the arrival audit accepts a schedule only when every bar
    produces exactly as many arrivals as its largest threshold, and under that
    condition the first *t* arrivals *are* all the producers of *X* -- the sound
    over-approximation ("*u* may need any of them"), so any cycle is a real
    deadlock. On a schedule whose bars over-produce, the slice is a strict subset
    and this check can only under-report; over-production is what
    ``audit_arrivals`` is for.

    So a cycle is always a real deadlock, while the abstraction can miss one that
    needs a more precise accounting. ``execution`` stays the primary detector and
    this is the localizer that names the cycle.
    """

    streams = prepared.streams
    flat: list[DecodedInstruction] = [
        inst for stream in streams for inst in stream
    ]
    location: list[tuple[int, int]] = [
        (sm, index) for sm, stream in enumerate(streams) for index in range(len(stream))
    ]

    # Producers of each bar, expanded by arrival count and kept in order.
    producers: dict[BarKey, list[int]] = defaultdict(list)
    for position, inst in enumerate(flat):
        for key, count in inst.arrivals:
            producers[key].extend([position] * count)

    adjacency: dict[int, list[int]] = defaultdict(list)
    for position, inst in enumerate(flat):
        for condition in inst.waits:
            for producer in producers.get(condition.key, [])[: condition.threshold]:
                adjacency[position].append(producer)

    # Iterative DFS with colours (0 = unvisited, 1 = on stack, 2 = done).
    colour = [0] * len(flat)
    parent: dict[int, int] = {}
    for root in range(len(flat)):
        if colour[root]:
            continue
        colour[root] = 1
        stack = [(root, iter(adjacency.get(root, ())))]
        while stack:
            node, children = stack[-1]
            advanced = False
            for child in children:
                if colour[child] == 1:
                    # `child` is an ancestor of `node` on the DFS stack, so walk
                    # the parent chain up from `node` to close the cycle.
                    path = [node]
                    while path[-1] != child:
                        path.append(parent[path[-1]])
                    path.append(node)
                    path.reverse()
                    return [location[i] for i in path]
                if colour[child] == 0:
                    colour[child] = 1
                    parent[child] = node
                    stack.append((child, iter(adjacency.get(child, ()))))
                    advanced = True
                    break
            if not advanced:
                colour[node] = 2
                stack.pop()
    return None


# ---------------------------------------------------------------------------
# Check 3: arrival accounting
# ---------------------------------------------------------------------------


@dataclass
class ArrivalAudit:
    over: dict[BarKey, tuple[int, int]] = field(default_factory=dict)
    under: dict[BarKey, tuple[int, int]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.over and not self.under

    def describe(self) -> str:
        parts = []
        for key, (produced, needed) in sorted(self.over.items()):
            parts.append(f"{key.kind}[{key.index}] produces {produced} but needs {needed}")
        for key, (produced, needed) in sorted(self.under.items()):
            parts.append(f"{key.kind}[{key.index}] produces {produced} < {needed}")
        return "; ".join(parts) if parts else "balanced"


def audit_arrivals(sched, num_sms: int) -> ArrivalAudit:
    """Convenience wrapper: prepare then audit."""

    return audit_arrivals_prepared(prepare(sched, num_sms))


def audit_arrivals_prepared(prepared: PreparedSchedule) -> ArrivalAudit:
    """Every bar's produced count must equal its largest consumed threshold.

    Equality (not ``>=``) is deliberate: over-arrival is a double-arrive bug that
    the kernel's monotone counters would happily mask, so it must be reported
    here rather than tolerated.
    """

    streams = prepared.streams
    produced = _produced_counts(streams)
    needed = _needed_thresholds(streams)
    audit = ArrivalAudit()
    for key in sorted(set(produced) | set(needed)):
        have = produced.get(key, 0)
        want = needed.get(key, 0)
        if have > want:
            audit.over[key] = (have, want)
        elif have < want:
            audit.under[key] = (have, want)
    return audit


# ---------------------------------------------------------------------------
# The verifier
# ---------------------------------------------------------------------------


@dataclass
class VerificationReport:
    ok: bool
    num_sms: int
    instructions: int
    completed: int
    stalled_sms: tuple[int, ...]
    cycle: list[tuple[int, int]] | None
    arrival_audit: ArrivalAudit
    failures: list[str] = field(default_factory=list)

    def describe(self) -> str:
        verdict = "PASS" if self.ok else "FAIL"
        lines = [
            f"{verdict}: {self.instructions} instructions over {self.num_sms} SMs, "
            f"{self.completed} executed",
        ]
        # The failures already carry the cycle and the arrival detail; repeating
        # them here made every report print the same thing twice.
        lines.extend(f"  {failure}" for failure in self.failures)
        return "\n".join(lines)


def prepare_segments(
    segments: Iterable[object], num_sms: int
) -> PreparedSchedule:
    """Concatenate per-SM streams of back-to-back launches.

    Segments of one decode step are separate kernel launches, so each starts only
    after the previous completed (a launch is an implicit barrier) and the bar
    counters carry across. Concatenating the per-SM streams reproduces exactly
    that, which is the property layer-contiguous fusion relies on: segment k+1's
    first QKV waits on the ``layer`` counter segment k's last ADD_RMSNORM
    produced.
    """

    per_segment = [prepare(segment, num_sms) for segment in segments]
    if not per_segment:
        raise ValueError("prepare_segments needs at least one segment")
    return PreparedSchedule(
        streams=tuple(
            tuple(
                inst
                for segment in per_segment
                for inst in segment.streams[sm]
            )
            for sm in range(num_sms)
        )
    )


def verify(
    sched, num_sms: int, bars: dict[BarKey, int] | None = None
) -> VerificationReport:
    """Run every check on one schedule."""

    return verify_prepared(prepare(sched, num_sms), bars)


def verify_segmented(
    segments: Iterable[object],
    num_sms: int,
    bars: dict[BarKey, int] | None = None,
) -> VerificationReport:
    """Run every check on a multi-launch step."""

    return verify_prepared(prepare_segments(segments, num_sms), bars)


def verify_prepared(
    prepared: PreparedSchedule, bars: dict[BarKey, int] | None = None
) -> VerificationReport:
    """Run every check on a prepared (possibly concatenated) schedule.

    ``bars`` seeds the execution check so a segmented step can be verified with
    the counters its earlier segments produced.
    """

    result = execute_prepared(prepared, dict(bars) if bars else None)
    cycle = wait_graph_cycle_prepared(prepared)
    audit = audit_arrivals_prepared(prepared)

    failures: list[str] = []
    if not result.completed:
        # Name what each stalled SM is blocked on: "SMs 3,7 stalled" costs a
        # debugging session, "SM 3 waits on layer[12]>=56" does not.
        stalls = [
            f"SM {sm} at #{result.pc[sm]}: {prepared.streams[sm][result.pc[sm]].describe}"
            for sm in result.stalled_sms[:3]
            if result.pc[sm] < len(prepared.streams[sm])
        ]
        failures.append(
            f"deadlock: {result.shortfall} instruction(s) never became ready "
            f"(stalled SMs {result.stalled_sms})"
            + ("; " + " | ".join(stalls) if stalls else "")
        )
    if cycle is not None:
        failures.append(f"wait-graph cycle at {cycle}")
    if not audit.ok:
        failures.append(f"arrival accounting: {audit.describe()}")

    return VerificationReport(
        ok=not failures,
        num_sms=len(prepared.streams),
        instructions=prepared.total,
        completed=result.done,
        stalled_sms=result.stalled_sms,
        cycle=cycle,
        arrival_audit=audit,
        failures=failures,
    )


# ---------------------------------------------------------------------------
# Adversarial mutation
# ---------------------------------------------------------------------------


class MutatedSchedule:
    """A schedule with one instruction field (or stream length) damaged."""

    __slots__ = ("flat_inst_buf", "num_inst_per_sm", "max_inst", "bar_dims", "label")

    def __init__(self, sched, label: str) -> None:
        self.flat_inst_buf = list(sched.flat_inst_buf)
        self.num_inst_per_sm = list(sched.num_inst_per_sm)
        self.max_inst = sched.max_inst
        self.bar_dims = sched.bar_dims
        self.label = label


def _mutation_candidates(sched, num_sms: int) -> list[tuple[str, int, int]]:
    """``(kind, sm, index)`` of every mutation that would change the buffer.

    A no-op mutant proves nothing -- an opcode with no count field cannot have
    its count doubled -- so candidates are filtered from the raw fields *before*
    anything is copied. Filtering first also keeps the population cheap to count:
    copying the whole flat buffer once per candidate would be tens of millions of
    element copies for a single configuration.
    """

    width = tg.TG_INST_WIDTH
    candidates: list[tuple[str, int, int]] = []
    for sm in range(num_sms):
        base = sm * sched.max_inst * width
        for index in range(sched.num_inst_per_sm[sm]):
            start = base + index * width
            fields = sched.flat_inst_buf[start : start + width]
            op = fields[0]
            if any(fields[offset] for offset in _arrival_field_offsets(op)):
                candidates.append(("drop_arrival", sm, index))
            if any(fields[offset] for offset in _arrival_count_offsets(op)):
                candidates.append(("double_arrival", sm, index))
            if _arrival_index_offsets(op):
                candidates.append(("shift_bar", sm, index))
            if _wait_threshold_offsets(op):
                candidates.append(("raise_threshold", sm, index))
    for sm in range(num_sms):
        if sched.num_inst_per_sm[sm] >= 2:
            candidates.append(("truncate_stream", sm, 0))
    return candidates


def mutant_count(sched, num_sms: int) -> int:
    """Size of the adversarial population for one configuration."""

    return len(_mutation_candidates(sched, num_sms))


def _apply_mutation(sched, num_sms: int, kind: str, sm: int, index: int) -> MutatedSchedule:
    width = tg.TG_INST_WIDTH
    base = sm * sched.max_inst * width + index * width
    mutant = MutatedSchedule(sched, f"{kind}:sm{sm}:i{index}")
    op = sched.flat_inst_buf[base]

    if kind == "drop_arrival":
        for offset in _arrival_field_offsets(op):
            mutant.flat_inst_buf[base + offset] = 0
    elif kind == "double_arrival":
        for offset in _arrival_count_offsets(op):
            mutant.flat_inst_buf[base + offset] *= 2
    elif kind == "shift_bar":
        for offset in _arrival_index_offsets(op):
            mutant.flat_inst_buf[base + offset] += 1
    elif kind == "raise_threshold":
        for offset in _wait_threshold_offsets(op):
            mutant.flat_inst_buf[base + offset] += 1
    elif kind == "truncate_stream":
        mutant.num_inst_per_sm[sm] = max(1, sched.num_inst_per_sm[sm] // 2)
    else:
        raise ValueError(f"unknown mutation kind: {kind}")
    return mutant


def mutants(
    sched, num_sms: int, *, sample: int | None = None
) -> Iterator[MutatedSchedule]:
    """Adversarial schedules that must be rejected. Deterministic by construction.

    ``sample`` bounds how many are yielded, taking every k-th candidate with a
    fixed stride. A full sweep is thousands of mutants per configuration (each
    needing three checks), so offline sweeps sample; a coverage number that
    changed between runs would not be evidence, hence the fixed stride.
    """

    if sample is not None and sample < 1:
        raise ValueError(f"sample must be >= 1 or None; got {sample}")
    for kind, sm, index in _sample_candidates(
        _mutation_candidates(sched, num_sms), sample
    ):
        yield _apply_mutation(sched, num_sms, kind, sm, index)


def _sample_candidates(
    candidates: list[tuple[str, int, int]], sample: int | None
) -> list[tuple[str, int, int]]:
    """Take ``sample`` candidates, **stratified by mutation kind**.

    A plain stride over the concatenated list would silently drop whole classes:
    ``truncate_stream`` candidates are appended after tens of thousands of
    per-instruction ones, so an evenly spaced sample of 60 never reaches them and
    the "every kind was exercised" claim would be false. Each kind therefore gets
    a quota of at least one, and the stride inside a kind stays fixed so the
    selection is reproducible.
    """

    if sample is None or sample >= len(candidates):
        return candidates

    by_kind: dict[str, list[tuple[str, int, int]]] = {}
    for candidate in candidates:
        by_kind.setdefault(candidate[0], []).append(candidate)

    total = len(candidates)
    chosen: list[tuple[str, int, int]] = []
    for kind in sorted(by_kind):
        pool = by_kind[kind]
        quota = min(len(pool), max(1, round(sample * len(pool) / total)))
        if quota >= len(pool):
            chosen.extend(pool)
            continue
        step = len(pool) / quota
        chosen.extend(pool[int(index * step)] for index in range(quota))

    if len(chosen) > sample:  # rounding overshoot: thin out deterministically
        step = len(chosen) / sample
        chosen = [chosen[int(index * step)] for index in range(sample)]
    return chosen


def _arrival_field_offsets(opcode: int) -> tuple[int, ...]:
    return {
        tg.TgOp.QKV_PROJ: (7, 9),
        tg.TgOp.ATTN_DECODE: (9,),
        tg.TgOp.ATTN_COMBINE: (7,),
        tg.TgOp.O_PROJ: (8, 9),
        tg.TgOp.FFN_UPGATE_ACT: (8, 9),
        tg.TgOp.FFN_DOWN: (8, 9),
        tg.TgOp.ADD_RMSNORM: (7,),
        tg.TgOp.LM_HEAD: (8, 9),
    }.get(opcode, ())


def _arrival_count_offsets(opcode: int) -> tuple[int, ...]:
    return {
        tg.TgOp.QKV_PROJ: (9,),
        tg.TgOp.O_PROJ: (9,),
        tg.TgOp.FFN_UPGATE_ACT: (9,),
        tg.TgOp.FFN_DOWN: (9,),
        tg.TgOp.LM_HEAD: (9,),
    }.get(opcode, ())


def _arrival_index_offsets(opcode: int) -> tuple[int, ...]:
    return {
        tg.TgOp.QKV_PROJ: (7,),
        tg.TgOp.ATTN_DECODE: (9,),
        tg.TgOp.ATTN_COMBINE: (7,),
        tg.TgOp.O_PROJ: (8,),
        tg.TgOp.FFN_UPGATE_ACT: (8,),
        tg.TgOp.FFN_DOWN: (8,),
        tg.TgOp.ADD_RMSNORM: (7,),
        tg.TgOp.LM_HEAD: (8,),
    }.get(opcode, ())


def _wait_threshold_offsets(opcode: int) -> tuple[int, ...]:
    return {
        tg.TgOp.QKV_PROJ: (6,),
        tg.TgOp.ATTN_DECODE: (8,),
        tg.TgOp.ATTN_COMBINE: (6,),
        tg.TgOp.O_PROJ: (7,),
        tg.TgOp.FFN_UPGATE_ACT: (7,),
        tg.TgOp.FFN_DOWN: (7,),
        tg.TgOp.ADD_RMSNORM: (4, 6),
        tg.TgOp.LM_HEAD: (7,),
        tg.TgOp.LM_ARGMAX: (7,),
    }.get(opcode, ())


@dataclass
class AdversarialReport:
    accepted_valid: int
    valid_total: int
    rejected_mutants: int
    mutant_total: int
    candidate_total: int = 0
    false_accepts: list[str] = field(default_factory=list)
    accepted_mutants: list[str] = field(default_factory=list)
    rejected_checks: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.valid_total == self.accepted_valid and not self.false_accepts

    @property
    def sampled(self) -> bool:
        return self.mutant_total < self.candidate_total

    def describe(self) -> str:
        scope = (
            f"{self.mutant_total} of {self.candidate_total} mutants"
            if self.sampled
            else f"all {self.mutant_total} mutants"
        )
        return (
            f"valid schedules accepted {self.accepted_valid}/{self.valid_total}; "
            f"rejected {self.rejected_mutants}/{scope} "
            f"(false accepts: {len(self.false_accepts)})"
        )


def adversarial_report(
    schedules: Iterable[tuple[str, object, int]],
    *,
    sample: int | None = None,
) -> AdversarialReport:
    """Verify real schedules and their mutants; count false accepts.

    ``schedules`` yields ``(label, schedule, num_sms)``. Mutants are generated per
    schedule, so the adversarial set scales with the geometry matrix instead of
    being a fixed handful. ``sample`` bounds the per-schedule mutant count.
    """

    report = AdversarialReport(
        accepted_valid=0, valid_total=0, rejected_mutants=0, mutant_total=0
    )
    for label, sched, num_sms in schedules:
        report.candidate_total += mutant_count(sched, num_sms)
        report.valid_total += 1
        verdict = verify(sched, num_sms)
        if verdict.ok:
            report.accepted_valid += 1
        else:
            report.false_accepts.append(f"valid schedule rejected: {label}: {verdict.failures}")

        for mutant in mutants(sched, num_sms, sample=sample):
            report.mutant_total += 1
            outcome = verify(mutant, num_sms)
            if outcome.ok:
                report.false_accepts.append(f"{label} / {mutant.label}")
                report.accepted_mutants.append(mutant.label)
            else:
                report.rejected_mutants += 1
                first = outcome.failures[0].split(":")[0]
                report.rejected_checks[first] = report.rejected_checks.get(first, 0) + 1
    return report
