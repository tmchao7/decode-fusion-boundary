"""Measure the achievable HBM read bandwidth on the current device.

Why this exists: the roofline model in ``src/nanovllm/roofline.py`` has a
*peak* number (A100-SXM4-80GB spec = 2039 GB/s) and the verdict machinery
compares measured kernel time against a lower bound derived from it. The
achievable fraction of peak is device- and clock-dependent, so the denominator
must be **measured on the machine that produced the measurement** -- otherwise
every "headroom" number is off by the peak-vs-achievable gap (on A100 that gap
is ~17%: 2039 spec vs 1699 measured).

The archived ``kernel.cu`` line had ``tools/bench_roofline_a100.py`` for this,
but that tool lives only on branch ``codex/archive/frozen-kernel`` and pulls in
archived probes. This is a small, architecture-neutral replacement that runs on
any CUDA device (A100 / 5090 / ...) using nothing but torch + optional pynvml.

Two numbers come out:

* ``achieved_gbps`` -- measured streaming read bandwidth (mean / stddev / min /
  max over the timed iterations, so the dispersion is reported too);
* ``bandwidth_gbps`` -- the device's *spec* bandwidth, derived from the memory
  clock and bus width via pynvml when available; ``achieved_fraction`` is the
  ratio and is ``null`` when the spec cannot be read (never guessed).

Usage on the server::

    python -m tools.bench_memory_bandwidth --bytes 1.2e9 --iterations 20 \
        --json results/bandwidth.json

Only the pure helpers below are unit-tested locally (the CUDA path needs a
GPU); the GPU path is a single large reduction, which is the simplest kernel
that streams a buffer exactly once.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

DEFAULT_BYTES = 1.2e9  # ~ the Qwen3-0.6B weight footprint per decode step
DEFAULT_ITERATIONS = 20
DEFAULT_WARMUP = 5


def parse_byte_count(value: str) -> int:
    """Parse ``1.2e9`` / ``1200MB`` / ``2GiB`` into a byte count."""

    text = value.strip()
    if not text:
        raise ValueError("byte count must not be empty")
    suffixes = {
        "b": 1,
        "kb": 10**3,
        "mb": 10**6,
        "gb": 10**9,
        "kib": 2**10,
        "mib": 2**20,
        "gib": 2**30,
    }
    lowered = text.lower()
    for suffix, scale in sorted(suffixes.items(), key=lambda kv: -len(kv[0])):
        if lowered.endswith(suffix) and len(lowered) > len(suffix):
            number = lowered[: -len(suffix)]
            break
    else:
        suffix, scale, number = "b", 1, lowered
    try:
        count = float(number) * scale
    except ValueError as exc:
        raise ValueError(f"not a byte count: {value!r}") from exc
    if count < 1:
        raise ValueError(f"byte count must be >= 1; got {value!r}")
    return int(count)


def bandwidth_gbps(total_bytes: int, seconds: float) -> float | None:
    """GB/s (10^9 bytes/s) for one timed pass, or ``None`` if unusable."""

    if total_bytes <= 0 or seconds <= 0:
        return None
    return total_bytes / seconds / 1e9


def achieved_fraction(achieved: float | None, spec: float | None) -> float | None:
    """Fraction of spec bandwidth actually achieved; unknown stays unknown."""

    if achieved is None or spec is None or spec <= 0:
        return None
    return achieved / spec


def device_spec_bandwidth_gbps(
    memory_clock_mhz: float | None, bus_width_bits: int | None
) -> float | None:
    """HBM spec bandwidth from the memory clock and bus width.

    ``GB/s = clock_Hz * (bus_width / 8) / 1e9``. DDR transfers twice per clock,
    hence the factor of 2. Missing inputs return ``None`` rather than a guess.
    """

    if not memory_clock_mhz or not bus_width_bits:
        return None
    if memory_clock_mhz <= 0 or bus_width_bits <= 0:
        return None
    bytes_per_second = (memory_clock_mhz * 1e6) * 2 * (bus_width_bits / 8)
    return bytes_per_second / 1e9


def summarize_bandwidth(samples: list[float]) -> dict[str, Any]:
    """Distribution of per-iteration GB/s (reuses the shared statistics)."""

    from nanovllm.metrics import summarize

    return summarize(samples)


def read_pynvml_spec(index: int = 0) -> dict[str, Any]:
    """Best-effort device spec via pynvml; every field may be ``None``."""

    spec: dict[str, Any] = {
        "name": None,
        "memory_clock_mhz": None,
        "bus_width_bits": None,
        "spec_gbps": None,
    }
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            spec["name"] = pynvml.nvmlDeviceGetName(handle)
            try:
                spec["memory_clock_mhz"] = float(
                    pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                )
            except Exception:  # noqa: BLE001 - optional field
                pass
            try:
                spec["bus_width_bits"] = int(
                    pynvml.nvmlDeviceGetMemoryBusWidth(handle)
                )
            except Exception:  # noqa: BLE001 - optional field
                pass
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001 - pynvml is optional
        pass
    spec["spec_gbps"] = device_spec_bandwidth_gbps(
        spec["memory_clock_mhz"], spec["bus_width_bits"]
    )
    return spec


def measure_read_bandwidth(
    total_bytes: int,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    warmup: int = DEFAULT_WARMUP,
) -> dict[str, Any]:
    """Stream ``total_bytes`` with a single reduction per iteration.

    A full-buffer reduction is the simplest kernel that reads every byte once
    with no writes, so its runtime bounds the achievable read bandwidth from
    above. Requires CUDA.
    """

    # Validate before importing torch so the argument checks stay testable on a
    # machine without CUDA (the CPU test suite runs on a laptop).
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1; got {iterations}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0; got {warmup}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("measuring bandwidth requires a CUDA device")

    # 4 bytes per element (float32) keeps the read pattern contiguous and
    # vectorizable; the reduction accumulates in float32.
    elements = max(1, total_bytes // 4)
    buffer = torch.ones(elements, dtype=torch.float32, device="cuda")
    actual_bytes = elements * 4

    for _ in range(warmup):
        buffer.sum()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        buffer.sum()
        end.record()
        torch.cuda.synchronize()
        value = bandwidth_gbps(actual_bytes, start.elapsed_time(end) / 1000.0)
        if value is not None:
            samples.append(value)

    spec = read_pynvml_spec()
    summary = summarize_bandwidth(samples)
    return {
        "requested_bytes": total_bytes,
        "measured_bytes": actual_bytes,
        "iterations": iterations,
        "warmup": warmup,
        "device": spec["name"] or torch.cuda.get_device_name(0),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "spec_memory_clock_mhz": spec["memory_clock_mhz"],
        "spec_bus_width_bits": spec["bus_width_bits"],
        "spec_gbps": spec["spec_gbps"],
        "achieved_gbps": summary,
        "achieved_fraction_of_spec": achieved_fraction(
            summary["mean"], spec["spec_gbps"]
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bytes", default=str(DEFAULT_BYTES))
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--json", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    total_bytes = parse_byte_count(args.bytes)
    payload = measure_read_bandwidth(
        total_bytes, iterations=args.iterations, warmup=args.warmup
    )
    achieved = payload["achieved_gbps"]
    print(
        f"device: {payload['device']} ({payload['sm_count']} SMs)\n"
        f"measured read bandwidth: mean {achieved['mean']:.1f} GB/s "
        f"(min {achieved['min']:.1f}, max {achieved['max']:.1f}, "
        f"cov {achieved['cov']:.3%}, n={achieved['count']})\n"
        f"spec bandwidth: "
        + (
            f"{payload['spec_gbps']:.1f} GB/s -> achieved "
            f"{payload['achieved_fraction_of_spec']:.1%} of spec"
            if payload["spec_gbps"]
            else "unknown (pynvml unavailable); no fraction reported"
        )
    )
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
