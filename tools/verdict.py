"""Compute the high-batch Megakernel verdict from the A100 probe outputs.

docs/09 defines three experiments with hard thresholds.  This tool applies
them mechanically so the verdict is reproducible rather than hand-derived from
a log, and so the thresholds can be unit tested on synthetic evidence.

Inputs (all written by ``RUN_MODE=roofline`` / ``RUN_MODE=account``)::

    results/a100_roofline_probe.json   # bandwidth, real weights, host, barrier
    results/a100_lower_bound.json      # device lower bound probe + host-free graph
    results/a100_account_breakdown.json  # per-step host/device ledger

The decision hierarchy follows docs/09:

1. **Bandwidth calibration first.**  If the measured per-step device time is
   below the weight floor, the timing cannot cover a whole step (or the device
   exceeds its spec bandwidth).  Until the measured attainable bandwidth is
   known, no headroom claim is trustworthy and the verdict is
   ``NEEDS_CALIBRATION``.
2. **Device lower bound is the gate.**  ``<= 1.5`` ms/step means the batch>1
   device side is recoverable and the R3 rewrite is worth attempting; ``> 3``
   ms/step means the high-batch crossover is not achievable and the rewrite
   must stop.
3. **Measured vs probe** distinguishes "headroom is in the device side" from
   "the device side is already there, so only host overhead remains".

Honesty rules baked in: the verdict only claims what the data supports.  A
recoverable probe does *not* by itself promise a speedup unless the measured
Megakernel device time is clearly above the probe floor; and the device-side
statement is always conditional on the calibration outcome.

Usage::

    python -m tools.verdict --roofline results/a100_roofline_probe.json \
        --lower-bound results/a100_lower_bound.json \
        --account results/a100_account_breakdown.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nanovllm.roofline import HardwareSpec, ModelDims, RooflineModel  # noqa: E402

# docs/09 section 2 thresholds, in ms per decode step.
RECOVERABLE_MS = 1.5
STOP_MS = 3.0

# A measured device time within this factor of the probe floor means the device
# side is already at its lower bound, so there is no device-side work left.
DEVICE_AT_FLOOR_RATIO = 1.25

VERDICT_NEEDS_DATA = "NEEDS_DATA"
VERDICT_NEEDS_CALIBRATION = "NEEDS_CALIBRATION"
VERDICT_DEVICE_RECOVERABLE = "DEVICE_RECOVERABLE"
VERDICT_DEVICE_AT_FLOOR = "DEVICE_AT_FLOOR"
VERDICT_DEVICE_UNREACHABLE = "DEVICE_UNREACHABLE"
VERDICT_DEVICE_BORDERLINE = "DEVICE_BORDERLINE"
VERDICT_HOST_BOUND_OPPORTUNITY = "HOST_BOUND_OPPORTUNITY"
VERDICT_DEVICE_BOUND_CUDA_GRAPH = "DEVICE_BOUND_CUDA_GRAPH"

# If the measured host-free per-step time is this close to the probe floor, the
# cudagraph path has no host overhead worth removing and no device headroom.
DEVICE_BOUND_RATIO = 1.25


@dataclass
class Calibration:
    measured_bandwidth_gbps: float | None = None
    spec_bandwidth_gbps: float | None = None
    implied_bandwidth_gbps: float | None = None
    contradiction_present: bool = False
    resolved: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class DeviceEvidence:
    probe_ms_per_step: float | None = None
    probe_batch_sizes: list[int] = field(default_factory=list)
    # The probe streams per-layer weights only (no embed / lm_head).  The LM
    # head is a full-matrix read every decode step and must be part of the
    # floor, so the verdict adds its bandwidth-equivalent time analytically.
    lm_head_stream_ms: float | None = None
    full_device_floor_ms: float | None = None
    measured_ms_per_step: float | None = None
    measured_backend: str | None = None
    measured_batch: int | None = None
    measured_host_residual_ms: float | None = None
    barrier_ms_per_step: float | None = None
    # Experiment C: the host-free graph replay, i.e. the measured opponent a
    # batch>1 persistent megakernel would have to beat once the host is gone.
    host_free_ms_per_step: float | None = None
    host_free_batches: list[int] = field(default_factory=list)
    host_inclusive_ms_per_step: float | None = None


@dataclass
class HostOverlay:
    measured_host_ms_per_step: float | None = None
    host_inclusive_ms_per_step: float | None = None
    host_free_ms_per_step: float | None = None
    # The measured device work a persistent kernel would replace.
    remaining_device_ms_per_step: float | None = None
    # True when removing the per-step host round trip (always at least one
    # cudaGraphLaunch + sampling + scheduler step) would save a material amount
    # of time.  This is a statement about the host *overlay*, not a claim that
    # measured cudagraph is host-bound device-wise.
    host_overlay_material: bool | None = None
    # True only if the no-arithmetic device floor is below the host-free time it
    # would have to beat, i.e. a kernel launched with no host round trip at all
    # would win on device work.
    megakernel_could_win: bool | None = None


@dataclass
class Verdict:
    verdict: str
    headline: str
    rationale: list[str]
    actions: list[str]
    calibration: Calibration
    device: DeviceEvidence
    host_overlay: HostOverlay = field(default_factory=HostOverlay)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "headline": self.headline,
            "rationale": self.rationale,
            "actions": self.actions,
            "calibration": asdict(self.calibration),
            "device": asdict(self.device),
            "host_overlay": asdict(self.host_overlay),
        }


def _load(path: str | None) -> dict | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def _bandwidth_gbps(roofline: dict | None) -> float | None:
    if not roofline:
        return None
    measured = (roofline.get("bandwidth") or {}).get("best_gbps")
    if measured:
        return float(measured)
    return (roofline.get("roofline_with_measured_bandwidth") or {}).get(
        "bandwidth_gbps"
    )


def _probe_ms_per_step(lower_bound: dict | None) -> tuple[float | None, list[int]]:
    if not lower_bound:
        return None, []
    rows = ((lower_bound.get("lower_bound_probe") or {}).get("rows")) or []
    usable = [
        row
        for row in rows
        if isinstance(row.get("ms_per_step_mean"), (int, float))
        and row["ms_per_step_mean"] > 0
    ]
    if not usable:
        return None, []
    batches = [int(row.get("batch_size") or 0) for row in usable]
    # The floor is batch-independent (weights are shared), so take the fastest
    # observed row as the probe's answer.
    return min(float(row["ms_per_step_mean"]) for row in usable), batches


def _measured_device_row(
    account: dict | None, model: RooflineModel
) -> tuple[dict | None, float | None]:
    """Pick the measured batch>1 row with the largest device time.

    That is the row a batch>1 fused kernel would have to replace, so it is the
    right comparison for "is the device side already at the floor?".
    """

    if not account:
        return None, None
    rows = account.get("rows") or []
    # Megakernel rows when present, else any batch>1 row with a measured GPU time.
    candidates = [
        row
        for row in rows
        if int(row.get("batch_size") or 0) > 1
        and row.get("device_source") == "measured"
        and (row.get("ms_per_step_gpu") or 0) > 0
    ]
    if not candidates:
        return None, None
    megakernel = [row for row in candidates if "megakernel" in str(row.get("backend_actual"))]
    pool = megakernel or candidates
    row = max(pool, key=lambda item: item["ms_per_step_gpu"])
    floor_ms = (
        model.weight_bytes()
        / (model.hardware.effective_bandwidth_gbps() * 1e9)
        * 1000.0
    )
    implied = model.weight_bytes() / (row["ms_per_step_gpu"] / 1000.0) / 1e9
    return row, implied if row["ms_per_step_gpu"] < floor_ms else None


def _host_free_evidence(
    lower_bound: dict | None,
) -> tuple[float | None, float | None, list[int]]:
    """Return (host_free_ms_per_step, host_inclusive_ms_per_step, batches)."""

    if not lower_bound:
        return None, None, []
    rows = lower_bound.get("host_free_graph") or []
    free, inclusive, batches = [], [], []
    for row in rows:
        value = row.get("ms_per_step_host_free")
        if isinstance(value, (int, float)) and value > 0:
            free.append(float(value))
            batches.append(int(row.get("batch_size") or 0))
            host_inclusive = row.get("ms_per_step_host_inclusive")
            if isinstance(host_inclusive, (int, float)) and host_inclusive > 0:
                inclusive.append(float(host_inclusive))
    if not free:
        return None, None, []
    return (
        min(free),
        min(inclusive) if inclusive else None,
        batches,
    )


def _build_host_overlay(
    *,
    host_free_ms: float | None,
    host_inclusive_ms: float | None,
    probe_ms: float | None,
) -> HostOverlay:
    """Quantify the measured opponent: how much of cudagraph is host overhead.

    ``host_inclusive - host_free`` is the per-step cost that exists only because
    the host must drive each step.  A persistent kernel removes exactly that
    and nothing else, so a batch>1 megakernel can only win if its own device
    work is at most ``host_free``.
    """

    overlay = HostOverlay(
        host_inclusive_ms_per_step=host_inclusive_ms,
        host_free_ms_per_step=host_free_ms,
    )
    if host_free_ms is None:
        return overlay
    if host_inclusive_ms is not None:
        overlay.measured_host_ms_per_step = host_inclusive_ms - host_free_ms
        overlay.host_overlay_material = overlay.measured_host_ms_per_step > 0.5
    overlay.remaining_device_ms_per_step = host_free_ms
    if probe_ms is not None:
        # A persistent kernel still has to beat the host-free graph time, which
        # already contains the same device work without the per-step host round
        # trip.  It wins on device work only if the no-arithmetic floor is below
        # that time even after allowing this margin for implementation loss.
        overlay.megakernel_could_win = probe_ms * DEVICE_BOUND_RATIO < host_free_ms
    return overlay


def compute_verdict(
    *,
    roofline: dict | None = None,
    lower_bound: dict | None = None,
    account: dict | None = None,
    model: RooflineModel | None = None,
    recoverable_ms: float = RECOVERABLE_MS,
    stop_ms: float = STOP_MS,
) -> Verdict:
    # Qwen3-0.6B has tie_word_embeddings=True (the LM head weight IS the
    # embedding weight), so the roofline must use tied_embedding=True to
    # avoid double-counting the 0.311 GB embedding/lm-head matrix.
    model = model or RooflineModel(ModelDims(), HardwareSpec(), tied_embedding=True)
    spec_bw = model.hardware.effective_bandwidth_gbps()
    weight_gb = model.weight_bytes() / 1e9
    spec_floor_ms = model.weight_bytes() / (spec_bw * 1e9) * 1000.0

    measured_bw = _bandwidth_gbps(roofline)
    probe_ms, probe_batches = _probe_ms_per_step(lower_bound)
    measured_row, implied_bw = _measured_device_row(account, model)
    host_free_ms, host_inclusive_ms, host_free_batches = _host_free_evidence(
        lower_bound
    )

    # The lower-bound probe streams per-layer weights only (it skips
    # embed_tokens and lm_head in _weight_layout).  The LM head is a full
    # matrix read every decode step and must be part of the device floor.
    # Add its bandwidth-equivalent time analytically using the measured (or
    # spec) bandwidth.  With tied_embedding=True the embedding_bytes() IS
    # the lm_head weight.
    lm_head_bytes = model.embedding_bytes()
    bw_for_correction = measured_bw if measured_bw else spec_bw
    lm_head_stream_ms = (
        lm_head_bytes / (bw_for_correction * 1e9) * 1000.0
        if lm_head_bytes and bw_for_correction
        else None
    )
    full_device_floor_ms = (
        probe_ms + lm_head_stream_ms
        if probe_ms is not None and lm_head_stream_ms is not None
        else probe_ms
    )

    calibration = Calibration(
        measured_bandwidth_gbps=measured_bw,
        spec_bandwidth_gbps=spec_bw,
        implied_bandwidth_gbps=implied_bw,
        contradiction_present=implied_bw is not None and implied_bw > spec_bw,
    )
    # The contradiction counts as resolved once the attainable bandwidth is
    # measured: either it explains the implied rate, or it proves the timing
    # only covered part of the step.
    calibration.resolved = measured_bw is not None
    if measured_bw is None:
        calibration.notes.append(
            "no measured bandwidth: run RUN_MODE=roofline and pass "
            "--roofline results/a100_roofline_probe.json"
        )
    elif calibration.contradiction_present:
        reachable = measured_bw >= implied_bw
        calibration.notes.append(
            f"recorded device time implies {implied_bw:.0f} GB/s while the "
            f"measured attainable bandwidth is {measured_bw:.0f} GB/s: that rate "
            f"is {'reachable' if reachable else 'NOT reachable'} on this machine, "
            "so the recorded timing "
            + (
                "covers the whole step and the device really is at the HBM wall"
                if reachable
                else "cannot cover a whole step (it only timed part of the kernels)"
            )
        )

    device = DeviceEvidence(
        probe_ms_per_step=probe_ms,
        probe_batch_sizes=probe_batches,
        lm_head_stream_ms=lm_head_stream_ms,
        full_device_floor_ms=full_device_floor_ms,
        measured_ms_per_step=(
            float(measured_row["ms_per_step_gpu"]) if measured_row else None
        ),
        measured_backend=(
            str(measured_row.get("backend_actual")) if measured_row else None
        ),
        measured_batch=(int(measured_row["batch_size"]) if measured_row else None),
        measured_host_residual_ms=(
            float(measured_row["host_residual_ms"]) if measured_row else None
        ),
        barrier_ms_per_step=(
            float((roofline or {}).get("barrier", {}).get("ms_per_step_estimate"))
            if (roofline or {}).get("barrier", {}).get("ms_per_step_estimate")
            else None
        ),
        host_free_ms_per_step=host_free_ms,
        host_free_batches=host_free_batches,
        host_inclusive_ms_per_step=host_inclusive_ms,
    )
    overlay = _build_host_overlay(
        host_free_ms=host_free_ms,
        host_inclusive_ms=host_inclusive_ms,
        probe_ms=full_device_floor_ms,
    )

    rationale: list[str] = []
    actions: list[str] = []

    if probe_ms is None:
        rationale.append(
            "no device lower-bound probe result: the decisive experiment has not "
            "run, so no high-batch verdict can be issued"
        )
        actions.append("run RUN_MODE=roofline and pass --lower-bound")
        if not calibration.resolved:
            return Verdict(
                VERDICT_NEEDS_DATA,
                "Insufficient data: the device lower-bound probe has not run.",
                rationale,
                actions,
                calibration,
                device,
                overlay,
            )
        return Verdict(
            VERDICT_NEEDS_DATA,
            "Insufficient data: bandwidth is calibrated but the device "
            "lower-bound probe has not run.",
            rationale,
            actions,
            calibration,
            device,
            overlay,
        )

    rationale.append(
        f"device lower bound probe: {probe_ms:.3f} ms/step "
        f"(no arithmetic, 28 layers of weight streaming + 12 barriers/layer; "
        f"batches {probe_batches or 'n/a'})"
    )
    if lm_head_stream_ms is not None:
        rationale.append(
            f"LM head stream correction: {lm_head_stream_ms:.3f} ms/step "
            f"(the probe skips lm_head; {lm_head_bytes / 1e9:.3f} GB at "
            f"{bw_for_correction:.0f} GB/s)"
        )
        rationale.append(
            f"full device floor (probe + lm_head): {full_device_floor_ms:.3f} ms/step"
        )
    rationale.append(
        f"roofline weight floor: {weight_gb:.3f} GB/step at {spec_bw:.0f} GB/s = "
        f"{spec_floor_ms:.3f} ms (tied_embedding={model.tied_embedding})"
    )
    if device.barrier_ms_per_step is not None:
        rationale.append(
            f"barrier budget from the microbenchmark: "
            f"{device.barrier_ms_per_step:.3f} ms/step (336 grid syncs)"
        )

    if not calibration.resolved:
        actions.append(
            "resolve the calibration first: the recorded device time implies "
            f"{implied_bw:.0f} GB/s, above the {spec_bw:.0f} GB/s spec"
            if calibration.contradiction_present
            else "run RUN_MODE=roofline for the bandwidth section"
        )
        return Verdict(
            VERDICT_NEEDS_CALIBRATION,
            f"Device side is {full_device_floor_ms:.3f} ms/step, but the bandwidth "
            "contradiction is unresolved, so no headroom claim is trustworthy yet.",
            rationale,
            actions,
            calibration,
            device,
            overlay,
        )

    # Use the FULL device floor (probe + lm_head correction) for the threshold
    # comparison: the probe only streams per-layer weights, but the LM head is
    # a full-matrix read every decode step and must be part of the floor.
    floor_ms = full_device_floor_ms if full_device_floor_ms is not None else probe_ms

    if floor_ms > stop_ms:
        rationale.append(
            f"full device floor {floor_ms:.3f} ms/step > stop threshold {stop_ms:.3f} ms/step: "
            "a fused kernel cannot even stream the weights faster than this, so "
            "the batch>1 device side is not recoverable"
        )
        actions += [
            "stop the batch>1 kernel rewrite (docs/07 section 8)",
            "record the measured floor in docs/07 section 10 and docs/08",
            "redirect effort to the per-step host overhead, which both backends pay",
        ]
        return Verdict(
            VERDICT_DEVICE_UNREACHABLE,
            f"High-batch device-side crossover is NOT achievable: the no-arithmetic "
            f"device floor is already {floor_ms:.3f} ms/step.",
            rationale,
            actions,
            calibration,
            device,
            overlay,
        )

    if floor_ms > recoverable_ms:
        rationale.append(
            f"full device floor {floor_ms:.3f} ms/step sits between the recoverable "
            f"({recoverable_ms:.1f} ms) and stop ({stop_ms:.1f} ms) thresholds"
        )
        actions.append(
            "only attempt R3 with an explicit, narrower justification "
            "(e.g. also fixing attention/LM head)"
        )
        return Verdict(
            VERDICT_DEVICE_UNREACHABLE if floor_ms > stop_ms else VERDICT_DEVICE_BORDERLINE,
            f"High-batch device side is borderline: {floor_ms:.3f} ms/step.",
            rationale,
            actions,
            calibration,
            device,
            overlay,
        )

    rationale.append(
        f"full device floor {floor_ms:.3f} ms/step <= recoverable threshold "
        f"{recoverable_ms:.1f} ms/step: a fused kernel could plausibly reach the "
        "HBM wall at batch>1"
    )
    if device.measured_ms_per_step is None:
        rationale.append(
            "no measured batch>1 device time available, so the remaining gap "
            "between the current kernel and the probe floor is unknown"
        )
        actions.append(
            "run RUN_MODE=account and pass --account to size the device-side gap"
        )
    else:
        gap = device.measured_ms_per_step / floor_ms if floor_ms > 0 else float("inf")
        rationale.append(
            f"measured batch>1 device time {device.measured_ms_per_step:.3f} ms/step "
            f"({device.measured_backend}, batch={device.measured_batch}) = "
            f"{gap:.1f}x the full device floor"
        )
        if gap <= DEVICE_AT_FLOOR_RATIO:
            if device.measured_host_residual_ms is not None:
                rationale.append(
                    f"but the recoverable cost is still host-side: "
                    f"{device.measured_host_residual_ms:.3f} ms/step on that row"
                )
            actions.append(
                "the device side is already at its floor; the remaining win is "
                "host overhead, not kernel work"
            )
            return Verdict(
                VERDICT_DEVICE_AT_FLOOR,
                f"Device side is already at the floor ({gap:.1f}x probe); "
                "only host overhead remains.",
                rationale,
                actions,
                calibration,
                device,
                overlay,
            )
        actions += [
            "size the R3 rewrite against the probe floor and the barrier budget",
            "keep the change behind megakernel_batch_decode=True and re-run "
            "tools/correctness_wmma_vs_scalar.py (PASS/CASCADE/FAIL)",
        ]

    # Experiment C decides whether there is anything to win at all: a persistent
    # kernel removes per-step host cost and nothing else, so it can only beat the
    # host-free graph time, and only if its own device work is at most that.
    if overlay.host_free_ms_per_step is not None:
        rationale.append(
            f"host-free graph replay: {overlay.host_free_ms_per_step:.3f} ms/step "
            f"(batches {device.host_free_batches or 'n/a'})"
        )
        if overlay.measured_host_ms_per_step is not None:
            rationale.append(
                f"measured per-step host overlay: "
                f"{overlay.measured_host_ms_per_step:.3f} ms "
                f"(host-inclusive {overlay.host_inclusive_ms_per_step:.3f} ms)"
            )
        if overlay.megakernel_could_win:
            actions.append(
                "a host-free persistent kernel has room only if its device work "
                f"stays at or below {overlay.remaining_device_ms_per_step:.3f} "
                "ms/step, which is the probe's job to prove"
            )
            return Verdict(
                VERDICT_HOST_BOUND_OPPORTUNITY,
                f"Host-bound opportunity: the device floor "
                f"({probe_ms:.3f} ms/step) is below the host-free graph time "
                f"({overlay.host_free_ms_per_step:.3f} ms/step), so a kernel with "
                "no per-step host round trip can win on device work.",
                rationale,
                actions,
                calibration,
                device,
                overlay,
            )
        actions.append(
            f"the host-free graph already runs at {overlay.host_free_ms_per_step:.3f} "
            f"ms/step against a {probe_ms:.3f} ms/step device floor, so a persistent "
            "batch>1 kernel has no margin to exploit"
        )
        if overlay.host_overlay_material:
            return Verdict(
                VERDICT_DEVICE_BOUND_CUDA_GRAPH,
                f"No device-side win is available: the host-free graph "
                f"({overlay.host_free_ms_per_step:.3f} ms/step) is at or below the "
                f"probe floor ({probe_ms:.3f} ms/step).",
                rationale,
                actions,
                calibration,
                device,
                overlay,
            )

    return Verdict(
        VERDICT_DEVICE_RECOVERABLE,
        f"Device-side recovery is plausible: probe floor {probe_ms:.3f} ms/step "
        f"vs a {stop_ms:.1f} ms stop threshold.",
        rationale,
        actions,
        calibration,
        device,
        overlay,
    )


def render_markdown(verdict: Verdict) -> str:
    lines = [
        f"# High-batch Megakernel verdict: `{verdict.verdict}`",
        "",
        f"**{verdict.headline}**",
        "",
        "## Evidence",
        "",
    ]
    lines += [f"- {item}" for item in verdict.rationale] or ["- (none)"]
    lines += ["", "## Actions", ""]
    lines += [f"{index}. {item}" for index, item in enumerate(verdict.actions, start=1)] or [
        "1. (none)"
    ]
    calibration = verdict.calibration
    lines += [
        "",
        "## Calibration",
        "",
        f"- measured attainable bandwidth: "
        f"{'n/a' if calibration.measured_bandwidth_gbps is None else f'{calibration.measured_bandwidth_gbps:.0f} GB/s'}",
        f"- spec bandwidth: {calibration.spec_bandwidth_gbps:.0f} GB/s",
        f"- implied by recorded device time: "
        f"{'n/a' if calibration.implied_bandwidth_gbps is None else f'{calibration.implied_bandwidth_gbps:.0f} GB/s'}",
        f"- contradiction present: {calibration.contradiction_present}",
        f"- resolved: {calibration.resolved}",
    ]
    for note in calibration.notes:
        lines.append(f"- note: {note}")
    overlay = verdict.host_overlay

    def _ms(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f} ms"

    lines += [
        "",
        "## Host overlay (experiment C)",
        "",
        f"- host-inclusive graph replay: {_ms(overlay.host_inclusive_ms_per_step)}",
        f"- host-free graph replay: {_ms(overlay.host_free_ms_per_step)}",
        f"- measured host overlay: {_ms(overlay.measured_host_ms_per_step)}",
        f"- host overlay material (>0.5 ms): {overlay.host_overlay_material}",
        f"- megakernel could win on device work: {overlay.megakernel_could_win}",
    ]
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--roofline", default=None, help="a100_roofline_probe.json")
    parser.add_argument("--lower-bound", dest="lower_bound", default=None,
                        help="a100_lower_bound.json")
    parser.add_argument("--account", default=None,
                        help="a100_account_breakdown.json")
    parser.add_argument("--results-dir", default=None,
                        help="default the three inputs to this results/ directory")
    parser.add_argument("--bandwidth-gbps", type=float, default=None,
                        help="override the model's spec bandwidth")
    parser.add_argument("--json", dest="json_out")
    parser.add_argument("--markdown", dest="markdown_out")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    results_dir = Path(args.results_dir) if args.results_dir else None

    def resolve(explicit: str | None, filename: str) -> str | None:
        if explicit:
            return explicit
        if results_dir and (results_dir / filename).exists():
            return str(results_dir / filename)
        return None

    roofline = _load(resolve(args.roofline, "a100_roofline_probe.json"))
    lower_bound = _load(resolve(args.lower_bound, "a100_lower_bound.json"))
    account = _load(resolve(args.account, "a100_account_breakdown.json"))
    if roofline is None and lower_bound is None and account is None:
        print(
            "no probe inputs found; run RUN_MODE=roofline / RUN_MODE=account first "
            "(or pass --results-dir)",
            file=sys.stderr,
        )
        return 2

    hardware = HardwareSpec()
    if args.bandwidth_gbps:
        hardware = HardwareSpec(bandwidth_gbps=args.bandwidth_gbps)
    model = RooflineModel(ModelDims(), hardware)
    verdict = compute_verdict(
        roofline=roofline, lower_bound=lower_bound, account=account, model=model
    )
    markdown = render_markdown(verdict)
    print(markdown)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(verdict.to_dict(), indent=2), encoding="utf-8"
        )
    if args.markdown_out:
        Path(args.markdown_out).write_text(markdown, encoding="utf-8")

    # Exit 0 means "there is something real to chase" (a device-side recovery,
    # a host-overlay opportunity, or a device already at its floor where the
    # remaining win is host-side).  Everything else -- including
    # DEVICE_BOUND_CUDA_GRAPH, where the rewrite has nothing to beat -- is 1.
    return 0 if verdict.verdict in (
        VERDICT_DEVICE_RECOVERABLE,
        VERDICT_DEVICE_AT_FLOOR,
        VERDICT_HOST_BOUND_OPPORTUNITY,
    ) else 1


if __name__ == "__main__":
    sys.exit(main())
