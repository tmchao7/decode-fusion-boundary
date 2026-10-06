"""Request timing, aggregation, comparison, and optional GPU telemetry.

The timing values are kept in seconds internally and converted to milliseconds
only at serialization time. This avoids unit drift while the engine records
multiple phases of a request.
"""

from __future__ import annotations

import json
import random
import statistics
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable


def percentile(values: Iterable[float], quantile: float) -> float | None:
    """Return an inclusive linear-interpolated percentile, or ``None``."""

    values = sorted(float(value) for value in values)
    if not values:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    fraction = index - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def _sample_stddev(values: list[float]) -> float | None:
    """Sample standard deviation, ``None`` when it is undefined (n < 2)."""

    if len(values) < 2:
        return None
    return statistics.stdev(values)


def summarize(values: Iterable[float], scale: float = 1.0) -> dict[str, float | None]:
    """Distribution summary for one pooled sample.

    Reports absolute dispersion (``stddev``, ``cov``, ``min``/``max``) next to
    the percentiles, not just the mean: "report all results, not just the best"
    and "never report ratios without absolute values" (Hoefler & Belli,
    https://spcl.inf.ethz.ch/Teaching/2024-dphpc/hoefler-scientific-benchmarking.pdf).
    ``cov`` is the coefficient of variation (stddev / |mean|), which is the
    number to quote when asked "how stable is this measurement?".
    """

    values = [float(value) * scale for value in values]
    mean = statistics.fmean(values) if values else None
    stddev = _sample_stddev(values)
    return {
        "count": len(values) or None,
        "mean": mean,
        "stddev": stddev,
        "cov": (
            None
            if stddev is None or mean is None or mean == 0.0
            else stddev / abs(mean)
        ),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "p5": percentile(values, 0.05),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def summarize_runs(
    values: Iterable[float],
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Cross-run dispersion plus a bootstrap 95% confidence interval.

    ``summarize`` pools every token of every run into one sample, which hides the
    *between-run* variance — exactly the variance a reviewer asks about
    ("your 1.31x: is it significant?"). This function takes one scalar per run
    and reports its dispersion.

    The interval is a deterministic percentile bootstrap rather than a
    ``mean ± 1.96 σ`` normal interval, because the number of runs is small and
    normality must not be assumed (Hoefler & Belli #5/#6). The fixed ``seed``
    keeps reports reproducible.
    """

    values = [float(value) for value in values]
    summary: dict[str, Any] = dict(summarize(values))
    n = len(values)
    if n < 2:
        # No interval is possible, so do not report how many resamples would
        # have been drawn: `resamples` describes the interval that is here.
        summary["ci95_low"] = None
        summary["ci95_high"] = None
        summary["ci95_half_width"] = None
        summary["resamples"] = None
        return summary

    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choices(values, k=n)) for _ in range(resamples)
    )
    low = percentile(means, 0.025)
    high = percentile(means, 0.975)
    summary["ci95_low"] = low
    summary["ci95_high"] = high
    summary["ci95_half_width"] = None if low is None or high is None else (high - low) / 2.0
    summary["resamples"] = resamples
    return summary



class CudaEventTimer:
    """Lazy CUDA-event timer used only during explicit metrics collection."""

    def __init__(self):
        self.enabled = False
        self.started = False
        self.stopped = False
        self.start_event = None
        self.end_event = None
        try:
            import torch

            if torch.cuda.is_available():
                self.start_event = torch.cuda.Event(enable_timing=True)
                self.end_event = torch.cuda.Event(enable_timing=True)
                self.enabled = True
        except Exception:
            self.enabled = False

    def start(self) -> None:
        if self.enabled:
            self.start_event.record()
            self.started = True
            self.stopped = False

    def stop(self) -> None:
        if self.enabled and self.started:
            self.end_event.record()
            self.stopped = True

    def elapsed_seconds(self) -> float | None:
        if not self.enabled or not self.started or not self.stopped:
            return None
        try:
            self.end_event.synchronize()
            return self.start_event.elapsed_time(self.end_event) / 1000.0
        except (RuntimeError, ValueError):
            return None


@dataclass
class RequestMetrics:
    request_id: int
    prompt_tokens: int
    arrival_time: float
    token_times: list[float]
    output_tokens: int
    prefill_time: float
    decode_time: float
    backend_requested: str
    backend_actual: str | None = None
    fallback_reason: str | None = None
    error: str | None = None
    cuda_prefill_time: float = 0.0
    cuda_decode_time: float = 0.0

    def _ttft(self) -> float | None:
        if not self.token_times:
            return None
        return self.token_times[0] - self.arrival_time

    def _e2e(self) -> float | None:
        if not self.token_times:
            return None
        return self.token_times[-1] - self.arrival_time

    def _itl(self) -> list[float]:
        return [
            current - previous
            for previous, current in zip(self.token_times, self.token_times[1:])
        ]

    def to_dict(self) -> dict[str, Any]:
        ttft = self._ttft()
        e2e = self._e2e()
        itl = self._itl()
        tpot = None
        if e2e is not None and self.output_tokens > 1:
            tpot = (e2e - (ttft or 0.0)) / (self.output_tokens - 1)
        return {
            "request_id": self.request_id,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "arrival_time": self.arrival_time,
            "first_token_time": self.token_times[0] if self.token_times else None,
            "token_times": list(self.token_times),
            "ttft_ms": None if ttft is None else ttft * 1000.0,
            "itl_ms": [value * 1000.0 for value in itl],
            "tpot_ms": None if tpot is None else tpot * 1000.0,
            "prefill_time_ms": self.prefill_time * 1000.0,
            "decode_time_ms": self.decode_time * 1000.0,
            "cuda_prefill_time_ms": self.cuda_prefill_time * 1000.0,
            "cuda_decode_time_ms": self.cuda_decode_time * 1000.0,
            "e2e_ms": None if e2e is None else e2e * 1000.0,
            "prefill_tok_s": _safe_rate(self.prompt_tokens, self.prefill_time),
            "decode_tok_s": _safe_rate(self.output_tokens, self.decode_time),
            "output_tok_s": _safe_rate(self.output_tokens, e2e),
            "backend_requested": self.backend_requested,
            "backend_actual": self.backend_actual,
            "fallback_reason": self.fallback_reason,
            "error": self.error,
        }


def _safe_rate(tokens: int, seconds: float | None) -> float | None:
    if seconds is None or seconds <= 0.0 or tokens <= 0:
        return None
    return tokens / seconds



class MetricsCollector:
    """Mutable recorder used by ``LLMEngine.generate_with_metrics``."""

    def __init__(self, backend_requested: str, megakernel_attn_splits: int = 1):
        self.backend_requested = backend_requested
        self.megakernel_attn_splits = int(megakernel_attn_splits)
        self.requests: dict[int, RequestMetrics] = {}
        self.time_origin: float | None = None
        self.step_count = 0
        self.prefill_steps = 0
        self.decode_steps = 0
        self.prefill_time = 0.0
        self.decode_time = 0.0
        self.cuda_prefill_time = 0.0
        self.cuda_decode_time = 0.0
        self.pending_cuda_steps = []
        self._cuda_finalized = False

    def add_request(self, request_id: int, prompt_tokens: int, arrival_time: float) -> None:
        if self.time_origin is None:
            self.time_origin = arrival_time
        arrival_time -= self.time_origin
        self.requests[request_id] = RequestMetrics(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            arrival_time=arrival_time,
            token_times=[],
            output_tokens=0,
            prefill_time=0.0,
            decode_time=0.0,
            backend_requested=self.backend_requested,
        )

    def record_step(
        self,
        seqs,
        is_prefill: bool,
        duration: float,
        backend_actual: str | None,
        fallback_reason: str | None,
        cuda_timer: CudaEventTimer | None = None,
    ) -> None:
        self.step_count += 1
        if is_prefill:
            self.prefill_steps += 1
            self.prefill_time += duration
        else:
            self.decode_steps += 1
            self.decode_time += duration
        for seq in seqs:
            request = self.requests.get(seq.seq_id)
            if request is None:
                continue
            if is_prefill:
                request.prefill_time += duration
            else:
                request.decode_time += duration
            if backend_actual:
                request.backend_actual = _merge_backend_names(
                    request.backend_actual, backend_actual
                )
            if fallback_reason and request.fallback_reason is None:
                request.fallback_reason = fallback_reason
        if cuda_timer is not None:
            self.pending_cuda_steps.append(
                ([seq.seq_id for seq in seqs], is_prefill, cuda_timer)
            )

    def _finalize_cuda_timing(self) -> None:
        if self._cuda_finalized:
            return
        self._cuda_finalized = True
        for request_ids, is_prefill, timer in self.pending_cuda_steps:
            duration = timer.elapsed_seconds()
            if duration is None:
                continue
            if is_prefill:
                self.cuda_prefill_time += duration
            else:
                self.cuda_decode_time += duration
            for request_id in request_ids:
                request = self.requests.get(request_id)
                if request is None:
                    continue
                if is_prefill:
                    request.cuda_prefill_time += duration
                else:
                    request.cuda_decode_time += duration

    def record_token(self, request_id: int, timestamp: float) -> None:
        request = self.requests.get(request_id)
        if request is None:
            return
        if self.time_origin is None:
            self.time_origin = timestamp
        request.token_times.append(timestamp - self.time_origin)
        request.output_tokens += 1

    def mark_error(self, request_id: int, error: str) -> None:
        request = self.requests.get(request_id)
        if request is not None:
            request.error = error

    def report(
        self,
        wall_time: float | None = None,
        scheduler_stats: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._finalize_cuda_timing()
        summary = aggregate_requests(self.requests.values(), wall_time=wall_time)
        summary["prefill_tok_s"] = _safe_rate(
            summary["prompt_tokens"], self.prefill_time
        )
        summary["decode_tok_s"] = _safe_rate(
            summary["output_tokens"], self.decode_time
        )
        summary.update(
            {
                "step_count": self.step_count,
                "prefill_steps": self.prefill_steps,
                "decode_steps": self.decode_steps,
                "prefill_time_ms_total": self.prefill_time * 1000.0,
                "decode_time_ms_total": self.decode_time * 1000.0,
                "cuda_prefill_time_ms_total": self.cuda_prefill_time * 1000.0,
                "cuda_decode_time_ms_total": self.cuda_decode_time * 1000.0,
                "megakernel_attn_splits": self.megakernel_attn_splits
                if self.backend_requested == "megakernel"
                else 1,
            }
        )
        if scheduler_stats is not None:
            summary["scheduler"] = dict(scheduler_stats)
        return {
            "summary": summary,
            "requests": [
                request.to_dict()
                for request in sorted(self.requests.values(), key=lambda item: item.request_id)
            ],
        }


def _merge_backend_names(current: str | None, new: str) -> str:
    if not current or current == new:
        return new
    names = current.split("+")
    if new not in names:
        names.append(new)
    return "+".join(names)


def aggregate_requests(
    requests: Iterable[RequestMetrics],
    wall_time: float | None = None,
) -> dict[str, Any]:
    requests = list(requests)
    serialized = [request.to_dict() for request in requests]
    completed = [request for request in serialized if request["e2e_ms"] is not None]
    # The request-level fields are already milliseconds.  Convert to seconds for
    # the internal accounting (the throughput ratios below want seconds), then
    # ask summarize() for milliseconds again via scale=1000.0 at the call sites.
    ttft_values = [request["ttft_ms"] / 1000.0 for request in completed if request["ttft_ms"] is not None]
    tpot_values = [request["tpot_ms"] / 1000.0 for request in completed if request["tpot_ms"] is not None]
    itl_values = [
        value / 1000.0
        for request in completed
        for value in request["itl_ms"]
    ]
    e2e_values = [request["e2e_ms"] / 1000.0 for request in completed]
    prefill_values = [request["prefill_time_ms"] / 1000.0 for request in completed]
    decode_values = [request["decode_time_ms"] / 1000.0 for request in completed]
    cuda_prefill_values = [
        request["cuda_prefill_time_ms"] / 1000.0 for request in completed
    ]
    cuda_decode_values = [
        request["cuda_decode_time_ms"] / 1000.0 for request in completed
    ]
    prompt_tokens = sum(request["prompt_tokens"] for request in completed)
    output_tokens = sum(request["output_tokens"] for request in completed)
    if wall_time is None and completed:
        first_arrival = min(request.arrival_time for request in requests)
        last_token = max(
            request.token_times[-1]
            for request in requests
            if request.token_times
        )
        wall_time = max(0.0, last_token - first_arrival)
    wall_time = wall_time or 0.0

    fallback_reasons = [
        request["fallback_reason"]
        for request in serialized
        if request["fallback_reason"]
    ]
    return {
        "request_count": len(requests),
        "completed_requests": len(completed),
        "wall_time_s": wall_time,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        # Request-level latency is stored in milliseconds; summarize() is asked
        # for milliseconds, so it needs scale=1000.0 because the values above
        # were converted to seconds for the throughput ratios.
        "ttft_ms": summarize(ttft_values, scale=1000.0),
        "tpot_ms": summarize(tpot_values, scale=1000.0),
        "itl_ms": summarize(itl_values, scale=1000.0),
        "e2e_ms": summarize(e2e_values, scale=1000.0),
        "prefill_time_ms": summarize(prefill_values, scale=1000.0),
        "decode_time_ms": summarize(decode_values, scale=1000.0),
        "cuda_prefill_time_ms": summarize(cuda_prefill_values, scale=1000.0),
        "cuda_decode_time_ms": summarize(cuda_decode_values, scale=1000.0),
        "prefill_tok_s": _safe_rate(prompt_tokens, sum(prefill_values)),
        "decode_tok_s": _safe_rate(output_tokens, sum(decode_values)),
        "output_tok_s": _safe_rate(output_tokens, wall_time),
        "peak_memory_allocated_bytes": None,
        "peak_memory_reserved_bytes": None,
        "fallback_count": len(fallback_reasons),
        "fallback_reasons": fallback_reasons,
        "backend_actuals": sorted(
            {
                name
                for request in serialized
                if request["backend_actual"]
                for name in request["backend_actual"].split("+")
            }
        ),
    }



def request_metrics_from_dict(data: dict[str, Any]) -> RequestMetrics:
    """Rehydrate a serialized request for merging independent benchmark runs."""

    return RequestMetrics(
        request_id=int(data["request_id"]),
        prompt_tokens=int(data["prompt_tokens"]),
        arrival_time=float(data["arrival_time"]),
        token_times=[float(value) for value in data.get("token_times", [])],
        output_tokens=int(data["output_tokens"]),
        prefill_time=float(data.get("prefill_time_ms", 0.0)) / 1000.0,
        decode_time=float(data.get("decode_time_ms", 0.0)) / 1000.0,
        backend_requested=str(data["backend_requested"]),
        backend_actual=data.get("backend_actual"),
        fallback_reason=data.get("fallback_reason"),
        error=data.get("error"),
        cuda_prefill_time=float(data.get("cuda_prefill_time_ms", 0.0)) / 1000.0,
        cuda_decode_time=float(data.get("cuda_decode_time_ms", 0.0)) / 1000.0,
    )


def merge_metric_reports(reports: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-run reports while preserving request-level distributions."""

    reports = list(reports)
    requests = [
        request_metrics_from_dict(request)
        for report in reports
        for request in report.get("requests", [])
    ]
    wall_time = sum(
        float(report.get("summary", {}).get("wall_time_s", 0.0) or 0.0)
        for report in reports
    )
    summary = aggregate_requests(requests, wall_time=wall_time)
    for key in (
        "step_count",
        "prefill_steps",
        "decode_steps",
        "prefill_time_ms_total",
        "decode_time_ms_total",
        "cuda_prefill_time_ms_total",
        "cuda_decode_time_ms_total",
    ):
        summary[key] = sum(
            float(report.get("summary", {}).get(key, 0.0) or 0.0)
            for report in reports
        )
    summary["prefill_tok_s"] = _safe_rate(
        summary["prompt_tokens"], summary["prefill_time_ms_total"] / 1000.0
    )
    summary["decode_tok_s"] = _safe_rate(
        summary["output_tokens"], summary["decode_time_ms_total"] / 1000.0
    )
    return {
        "summary": summary,
        "requests": [request.to_dict() for request in requests],
    }


def compare_metric_summaries(
    baseline: dict[str, Any], optimized: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Compare scalar throughput and latency summaries without fake gains."""

    metric_paths = {
        "prefill_tok_s": ("prefill_tok_s", "higher"),
        "decode_tok_s": ("decode_tok_s", "higher"),
        "output_tok_s": ("output_tok_s", "higher"),
        "peak_memory_allocated_bytes": ("peak_memory_allocated_bytes", "lower"),
        "ttft_ms_p50": (("ttft_ms", "p50"), "lower"),
        "tpot_ms_p50": (("tpot_ms", "p50"), "lower"),
        "itl_ms_p50": (("itl_ms", "p50"), "lower"),
        "e2e_ms_p50": (("e2e_ms", "p50"), "lower"),
    }
    comparison = {}
    for name, (path, direction) in metric_paths.items():
        base = _read_path(baseline, path)
        opt = _read_path(optimized, path)
        comparable = base is not None and opt is not None and base != 0.0 and opt != 0.0
        item = {
            "baseline": base,
            "optimized": opt,
            "delta": None if base is None or opt is None else opt - base,
            "improvement_pct": None,
            "speedup": None,
            "comparable": comparable,
        }
        if comparable:
            if direction == "higher":
                item["improvement_pct"] = (opt - base) / base * 100.0
                item["speedup"] = opt / base
            else:
                item["improvement_pct"] = (base - opt) / base * 100.0
                item["speedup"] = base / opt
        comparison[name] = item
    return comparison


def _read_path(data: dict[str, Any], path: str | tuple[str, str]) -> float | None:
    if isinstance(path, tuple):
        value = (data.get(path[0]) or {}).get(path[1])
    else:
        value = data.get(path)
    return None if value is None else float(value)


@dataclass
class GpuTelemetrySampler:
    """Best-effort NVML sampler; unavailable telemetry never fails a run."""

    interval_s: float = 0.1
    samples: list[dict[str, float]] = field(default_factory=list)
    available: bool = False
    error: str | None = None
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _handle: Any = field(default=None, init=False, repr=False)
    _nvml: Any = field(default=None, init=False, repr=False)

    def start(self) -> "GpuTelemetrySampler":
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.available = True
        except Exception as exc:  # optional dependency/device path
            self.error = str(exc)
            return self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 4))
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                utilization = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                memory = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                self.samples.append(
                    {
                        "gpu_utilization_pct": float(utilization.gpu),
                        "memory_utilization_pct": float(utilization.memory),
                        "memory_used_bytes": float(memory.used),
                        "power_watts": float(self._nvml.nvmlDeviceGetPowerUsage(self._handle)) / 1000.0,
                        "temperature_c": float(self._nvml.nvmlDeviceGetTemperature(self._handle, 0)),
                    }
                )
            except Exception as exc:
                self.error = str(exc)
                break
            self._stop.wait(self.interval_s)

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {"available": self.available, "error": self.error, "samples": 0}
        return {
            "available": self.available,
            "error": self.error,
            "samples": len(self.samples),
            **{
                f"max_{key}": max(sample[key] for sample in self.samples)
                for key in self.samples[0]
            },
            "mean_gpu_utilization_pct": statistics.fmean(
                sample["gpu_utilization_pct"] for sample in self.samples
            ),
        }


@contextmanager
def nvtx_range(name: str):
    """Push/pop an NVTX range when CUDA NVTX is available."""

    nvtx = None
    try:
        import torch

        if torch.cuda.is_available() and hasattr(torch.cuda, "nvtx"):
            nvtx = torch.cuda.nvtx
            nvtx.range_push(name)
    except Exception:
        nvtx = None
    try:
        yield
    finally:
        if nvtx is not None:
            nvtx.range_pop()


def environment_metadata() -> dict[str, Any]:
    metadata = {"timestamp_utc": datetime.now(timezone.utc).isoformat()}
    try:
        import torch

        metadata.update(
            {
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
            }
        )
        if torch.cuda.is_available():
            metadata["gpu_name"] = torch.cuda.get_device_name()
            metadata["compute_capability"] = list(torch.cuda.get_device_capability())
            try:
                import pynvml

                pynvml.nvmlInit()
                driver = pynvml.nvmlSystemGetDriverVersion()
                metadata["driver_version"] = (
                    driver.decode() if isinstance(driver, bytes) else str(driver)
                )
                pynvml.nvmlShutdown()
            except Exception:
                metadata["driver_version"] = None
    except Exception as exc:
        metadata["torch_error"] = str(exc)
    return metadata


def dump_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
