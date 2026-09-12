"""Analytical roofline model for LLM decode steps.

The point of this module is to turn "can a fused kernel beat CUDA Graph at
large batch?" into arithmetic instead of speculation.  For a *decode* step the
answer is bounded by two hardware limits:

* memory: the weights must be streamed from HBM once per step (there is no
  reuse across the tiny M dimension), plus KV cache and activations;
* compute: ``2 * batch * params`` FLOPs against the tensor-core ceiling.

This matters because the arithmetic intensity of a decode step is so low that
the memory term dominates for a wide range of realistic batch sizes.  For
Qwen3-0.6B in bf16 the weight footprint is ~1.5 GB, which at a measured
A100 bandwidth of ~1.5-1.6 TB/s is ~1 ms per step -- and the observed
cudagraph decode kernel time is ~0.75 ms per step.  In other words the
existing batched/cuBLAS path is already at the weight-streaming wall, so a
megakernel cannot win on the device side: both are bounded by the same HBM
bandwidth, and the only recoverable cost is the per-step host overhead.

Everything here is pure Python (no torch, no CUDA) so it can be unit tested on
a laptop.  The model dimensions are checked against ``kernel.cu`` at import
time of :func:`assert_kernel_constants` so the two cannot silently drift.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

BYTES_PER_PARAM_BF16 = 2

DEFAULT_KERNEL_SOURCE = (
    Path(__file__).resolve().parent
    / "backends"
    / "megakernel"
    / "csrc"
    / "kernel.cu"
)


@dataclass(frozen=True)
class ModelDims:
    """Qwen3-0.6B dimensions, mirroring the ``kernel.cu`` constants."""

    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_q_heads: int = 16
    num_kv_heads: int = 8
    head_dim: int = 128
    num_layers: int = 28
    vocab_size: int = 151936

    @property
    def q_size(self) -> int:
        return self.num_q_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_kv_heads * self.head_dim

    def params_per_layer(self) -> int:
        """Attention (q/k/v/o) + MLP (gate/up/down) + two RMSNorm vectors."""

        qkv = self.hidden_size * (self.q_size + 2 * self.kv_size)
        o_proj = self.q_size * self.hidden_size
        mlp = 3 * self.hidden_size * self.intermediate_size
        norms = 2 * self.hidden_size
        return qkv + o_proj + mlp + norms

    def params_embedding(self) -> int:
        return self.vocab_size * self.hidden_size

    def params_lm_head(self, tied_embedding: bool = False) -> int:
        """LM head parameters; ``tied_embedding`` folds it into the embedding."""

        return 0 if tied_embedding else self.vocab_size * self.hidden_size

    def total_params(self, tied_embedding: bool = False) -> int:
        return (
            self.num_layers * self.params_per_layer()
            + self.params_embedding()
            + self.params_lm_head(tied_embedding)
        )


@dataclass(frozen=True)
class HardwareSpec:
    """Peak (or measured) device limits.  Defaults are A100-SXM4-80GB spec."""

    name: str = "A100-SXM4-80GB"
    bandwidth_gbps: float = 1555.0
    peak_tflops: float = 312.0
    compute_efficiency: float = 1.0

    def effective_bandwidth_gbps(self) -> float:
        return float(self.bandwidth_gbps)

    def effective_tflops(self) -> float:
        return float(self.peak_tflops) * float(self.compute_efficiency)


def parse_kernel_constants(source: str | Path = DEFAULT_KERNEL_SOURCE) -> dict[str, int]:
    """Extract the model constants a megakernel build uses from ``kernel.cu``."""

    text = Path(source).read_text(encoding="utf-8")
    wanted = {
        "hidden_size": "HIDDEN_SIZE",
        "intermediate_size": "INTERMEDIATE_SIZE",
        "num_q_heads": "NUM_Q_HEADS",
        "num_kv_heads": "NUM_KV_HEADS",
        "head_dim": "HEAD_DIM",
    }
    found: dict[str, int] = {}
    for field, symbol in wanted.items():
        match = re.search(
            rf"^constexpr int {symbol}\s*=\s*(\d+)\s*;", text, re.MULTILINE
        )
        if match is None:
            raise ValueError(f"could not find constant {symbol} in {source}")
        found[field] = int(match.group(1))
    vocab = re.search(
        r"^constexpr int LDG_VOCAB_SIZE\s*=\s*(\d+)\s*;", text, re.MULTILINE
    )
    if vocab is None:
        raise ValueError(f"could not find LDG_VOCAB_SIZE in {source}")
    found["vocab_size"] = int(vocab.group(1))
    return found


def assert_kernel_constants(
    dims: ModelDims, source: str | Path = DEFAULT_KERNEL_SOURCE
) -> None:
    """Raise if :class:`ModelDims` no longer matches the checked-in kernel."""

    actual = parse_kernel_constants(source)
    expected = asdict(dims)
    mismatches = {
        key: (expected[key], value)
        for key, value in actual.items()
        if expected.get(key) != value
    }
    if mismatches:
        detail = ", ".join(
            f"{key}: model={want} kernel={got}"
            for key, (want, got) in sorted(mismatches.items())
        )
        raise ValueError(f"ModelDims disagrees with {source}: {detail}")


class RooflineModel:
    """Decode-step lower bounds for a dense transformer on one device."""

    def __init__(
        self,
        dims: ModelDims | None = None,
        hardware: HardwareSpec | None = None,
        *,
        tied_embedding: bool = False,
        bytes_per_param: int = BYTES_PER_PARAM_BF16,
        attn_blocks: int = 8,
        kv_read_amp: int | None = None,
    ) -> None:
        self.dims = dims or ModelDims()
        self.hardware = hardware or HardwareSpec()
        self.tied_embedding = bool(tied_embedding)
        self.bytes_per_param = int(bytes_per_param)
        self.attn_blocks = max(1, int(attn_blocks))
        # KV read amplification: each query-head block re-reads the whole KV
        # range.  This is the *paged megakernel* contract; the batched cuBLAS
        # path reads KV once per query head too, but a non-split attention
        # kernel (amp=1) is the optimistic bound.
        self.kv_read_amp = max(1, int(kv_read_amp or attn_blocks))

    # -- traffic ---------------------------------------------------------

    def weight_bytes(self) -> int:
        """Bytes that every query token must stream from HBM in one step.

        This is the load_tokens term of the decode roofline: every weight is
        touched by every query token.  It includes the LM head (which is the
        single largest matrix) and the embedding table, because at the batch
        sizes where embedding rows stop being negligible the step is already
        compute-bound.  Use :meth:`weight_bytes_excl_embedding` for the term
        that is independent of batch and sequence length.
        """

        return self.dims.total_params(self.tied_embedding) * self.bytes_per_param

    def weight_bytes_excl_embedding(self) -> int:
        """Weight bytes ignoring the embedding table (batch-independent core)."""

        return self.dims.total_params(self.tied_embedding) * self.bytes_per_param - (
            self.dims.vocab_size * self.dims.hidden_size * self.bytes_per_param
        )

    def embedding_bytes(self) -> int:
        return self.dims.vocab_size * self.dims.hidden_size * self.bytes_per_param

    def weight_bytes_breakdown(self) -> dict[str, int]:
        d = self.dims
        scale = self.bytes_per_param
        layers = d.num_layers * d.params_per_layer()
        lm_head = d.params_lm_head(self.tied_embedding)
        return {
            "layers_bytes": layers * scale,
            "lm_head_bytes": lm_head * scale,
            "embedding_bytes": self.embedding_bytes(),
            "weight_bytes_total": (layers + lm_head) * scale + self.embedding_bytes(),
        }

    def kv_bytes(self, batch_size: int, seq_len: int) -> int:
        """KV cache traffic, including per-query-block read amplification."""

        per_seq = (
            2
            * self.dims.num_layers
            * seq_len
            * self.dims.kv_size
            * self.bytes_per_param
        )
        return int(batch_size) * per_seq * self.kv_read_amp

    def kv_bytes_unamplified(self, batch_size: int, seq_len: int) -> int:
        per_seq = (
            2
            * self.dims.num_layers
            * seq_len
            * self.dims.kv_size
            * self.bytes_per_param
        )
        return int(batch_size) * per_seq

    def activation_bytes(self, batch_size: int) -> int:
        """Intermediate activation round-trips if layers are not fused."""

        d = self.dims
        hidden = d.hidden_size
        inter = d.intermediate_size
        # per layer: hidden in/out, q/k/v/attn_out, gate+up+silu+down
        per_seq = (
            2 * hidden  # residual stream in + out
            + d.q_size
            + 2 * d.kv_size
            + d.q_size  # attention output
            + 3 * inter  # gate, up, down inputs
            + 2 * hidden  # normalized copies
        )
        return int(batch_size) * d.num_layers * per_seq * 4  # fp32 scratch

    def memory_bytes(self, batch_size: int, seq_len: int) -> int:
        return self.weight_bytes() + self.kv_bytes(batch_size, seq_len)

    # -- flops -----------------------------------------------------------

    def flops_per_step(self, batch_size: int, seq_len: int) -> float:
        d = self.dims
        batch = int(batch_size)
        tokens = batch * d.num_layers
        projections = 2.0 * tokens * (
            d.hidden_size * (d.q_size + 2 * d.kv_size)
            + d.q_size * d.hidden_size
            + 3 * d.hidden_size * d.intermediate_size
        )
        attention = 2.0 * 2.0 * d.num_q_heads * d.head_dim * seq_len * batch * d.num_layers
        lm_head = 2.0 * batch * d.vocab_size * d.hidden_size
        return projections + attention + lm_head

    # -- roofline --------------------------------------------------------

    def lower_bound_ms(self, batch_size: int = 1, seq_len: int = 1) -> float:
        """Best possible per-step device time: ``max(memory, compute)``."""

        mem_s = self.memory_bytes(batch_size, seq_len) / (
            self.hardware.effective_bandwidth_gbps() * 1e9
        )
        compute_s = self.flops_per_step(batch_size, seq_len) / (
            self.hardware.effective_tflops() * 1e12
        )
        return max(mem_s, compute_s) * 1000.0

    def memory_bound_ms(self, batch_size: int = 1, seq_len: int = 1) -> float:
        return self.memory_bytes(batch_size, seq_len) / (
            self.hardware.effective_bandwidth_gbps() * 1e9
        ) * 1000.0

    def compute_bound_ms(self, batch_size: int = 1, seq_len: int = 1) -> float:
        return self.flops_per_step(batch_size, seq_len) / (
            self.hardware.effective_tflops() * 1e12
        ) * 1000.0

    def arithmetic_intensity(self, batch_size: int = 1, seq_len: int = 1) -> float:
        mem = self.memory_bytes(batch_size, seq_len)
        return self.flops_per_step(batch_size, seq_len) / mem if mem else 0.0

    def is_memory_bound(
        self, batch_size: int = 1, seq_len: int = 1, *, ignore_kv: bool = False
    ) -> bool:
        if ignore_kv:
            return self.weight_bytes() / (
                self.hardware.effective_bandwidth_gbps() * 1e9
            ) * 1000.0 >= self.compute_bound_ms(batch_size, seq_len)
        return self.memory_bound_ms(batch_size, seq_len) >= self.compute_bound_ms(
            batch_size, seq_len
        )

    def compute_bound_batch(self, seq_len: int = 1) -> float:
        """Batch where weight streaming and tensor-core compute break even.

        This ignores the KV term, whose read amplification grows linearly with
        batch in the split-KV megakernel layout; including it only pushes the
        crossover higher, so this is a lower bound on the true crossover.
        """

        weights_ms = self.weight_bytes() / (
            self.hardware.effective_bandwidth_gbps() * 1e9
        ) * 1000.0
        flops_per_seq = self.flops_per_step(1, seq_len)
        tflops = self.hardware.effective_tflops() * 1e12
        # batch * flops_per_seq / tflops == weights_ms / 1000
        return weights_ms / 1000.0 * tflops / flops_per_seq

    def attainable_tok_s(self, batch_size: int, seq_len: int = 1) -> float:
        """Tokens/s if the step ran exactly at the roofline lower bound."""

        bound = self.lower_bound_ms(batch_size, seq_len)
        return 0.0 if bound <= 0 else 1000.0 * batch_size / bound

    def min_steps_per_sec(self, batch_size: int, seq_len: int = 1) -> float:
        bound = self.lower_bound_ms(batch_size, seq_len)
        return 0.0 if bound <= 0 else 1000.0 / bound


@dataclass(frozen=True)
class StepAccounting:
    """Per-step wall/device split derived from benchmark metrics."""

    backend: str
    batch_size: int
    ms_per_step_wall: float
    ms_per_step_gpu: float
    host_residual_ms: float
    gpu_fraction: float


def account_step(
    *,
    backend: str,
    batch_size: int,
    ms_per_token_wall: float,
    kernel_ms_per_step: float,
) -> StepAccounting:
    """Split one decode step into device time and host residual.

    A decode step emits exactly one token per running sequence, so the wall
    time of a step is ``ms_per_token_wall * batch_size``.  The GPU part is
    measured directly (nsys kernel time or ``cuda_decode_time_ms_total /
    decode_steps``).
    """

    wall = float(ms_per_token_wall) * int(batch_size)
    gpu = float(kernel_ms_per_step)
    residual = wall - gpu
    return StepAccounting(
        backend=backend,
        batch_size=int(batch_size),
        ms_per_step_wall=wall,
        ms_per_step_gpu=gpu,
        host_residual_ms=residual,
        gpu_fraction=(gpu / wall) if wall > 0 else 0.0,
    )


def step_accounting_from_metrics(
    *, backend: str, batch_size: int, metrics: dict
) -> StepAccounting | None:
    """Build :class:`StepAccounting` from a ``compare.py`` metrics summary.

    Accepts either the raw per-run summary or a merged summary; both expose
    ``tpot_ms`` and the decode-step totals.
    """

    summary = metrics.get("summary", metrics)
    decode_steps = float(summary.get("decode_steps", 0) or 0)
    if decode_steps <= 0:
        return None
    wall_total = float(summary.get("decode_time_ms_total", 0.0) or 0.0)
    gpu_total = float(summary.get("cuda_decode_time_ms_total", 0.0) or 0.0)
    wall_per_step = wall_total / decode_steps
    gpu_per_step = gpu_total / decode_steps
    if wall_per_step <= 0:
        return None
    residual = wall_per_step - gpu_per_step
    return StepAccounting(
        backend=backend,
        batch_size=int(batch_size),
        ms_per_step_wall=wall_per_step,
        ms_per_step_gpu=gpu_per_step,
        host_residual_ms=residual,
        gpu_fraction=gpu_per_step / wall_per_step,
    )


def gap_table(
    model: RooflineModel,
    batch_sizes: list[int],
    *,
    seq_len: int = 1,
    kernel_ms_per_step: float | None = None,
    kernel_backend: str = "torch-cudagraph",
    ms_per_token_wall: dict[int, float] | None = None,
) -> list[dict]:
    """Per-batch roofline / measured / host split rows for a report.

    When a measured ``kernel_ms_per_step`` is supplied we also solve for the
    bandwidth that measurement implies.  If the measured time is *below* the
    modeled lower bound, the model's hardware parameters (or its KV
    read-amplification assumption) are wrong on this machine -- that is a
    calibration finding, not a win for the kernel, and it is recorded as
    ``calibration_needed`` instead of being silently reported as >100% of
    peak bandwidth.
    """

    wall = ms_per_token_wall or {}
    peak_bw = model.hardware.effective_bandwidth_gbps() * 1e9
    # Batch-independent core traffic: weights without the embedding table and
    # without the KV read amplification.  This is the floor a fused kernel
    # shares with the cuBLAS path; the gap to it is genuine device headroom.
    core_bytes = model.weight_bytes_excl_embedding()
    core_floor_ms = core_bytes / peak_bw * 1000.0
    weights = model.weight_bytes()
    rows: list[dict] = []
    for batch in batch_sizes:
        lower = model.lower_bound_ms(batch, seq_len)
        row: dict = {
            "batch_size": batch,
            "weight_bytes": weights,
            "memory_bound_ms": model.memory_bound_ms(batch, seq_len),
            "compute_bound_ms": model.compute_bound_ms(batch, seq_len),
            "lower_bound_ms": lower,
            "lower_bound_excl_kv_amp_ms": max(
                core_floor_ms, model.compute_bound_ms(batch, seq_len)
            ),
            "core_weight_floor_ms": core_floor_ms,
            "attainable_tok_s": model.attainable_tok_s(batch, seq_len),
            "memory_bound": model.is_memory_bound(batch, seq_len),
            "calibration_needed": False,
            "headroom_ms": 0.0,
            "headroom_fraction": 0.0,
        }
        observed: float | None = None
        if kernel_ms_per_step is not None:
            observed = max(float(kernel_ms_per_step), 0.0)
            row["measured_kernel_ms_per_step"] = observed
            mem_at_batch = model.memory_bytes(batch, seq_len)
            if observed > 0:
                row["kernel_implied_bandwidth_gbps"] = mem_at_batch / (
                    observed / 1000.0
                ) / 1e9
            if observed < lower:
                row["calibration_needed"] = True
                row["effective_lower_bound_ms"] = observed
            else:
                row["effective_lower_bound_ms"] = lower
                row["headroom_ms"] = observed - lower
                row["headroom_fraction"] = (observed - lower) / observed
        if batch in wall:
            acc = account_step(
                backend=kernel_backend,
                batch_size=batch,
                ms_per_token_wall=wall[batch],
                kernel_ms_per_step=observed or 0.0,
            )
            effective = row.get("effective_lower_bound_ms", lower)
            row.update(
                {
                    "ms_per_step_wall": acc.ms_per_step_wall,
                    "host_residual_ms": acc.host_residual_ms,
                    "gpu_fraction": acc.gpu_fraction,
                    "wall_gap_to_effective_lower_bound": acc.ms_per_step_wall
                    / effective
                    if effective > 0
                    else 0.0,
                    "ceiling_tok_s_if_host_removed": (
                        1000.0 * batch / effective if effective > 0 else 0.0
                    ),
                }
            )
        rows.append(row)
    return rows


def render_markdown(rows: list[dict], model: RooflineModel) -> str:
    hw = model.hardware
    lines = [
        f"# Decode-step roofline ({hw.name}, {hw.effective_bandwidth_gbps():.0f} GB/s, "
        f"{hw.effective_tflops():.0f} TFLOP/s)",
        "",
        f"weight bytes/step = {model.weight_bytes() / 1e9:.3f} GB "
        f"(tied_embedding={model.tied_embedding}, attn_blocks={model.attn_blocks}, "
        f"kv_read_amp={model.kv_read_amp})",
        "",
        "| batch | mem bound ms | compute bound ms | lower bound ms | tok/s @ bound | bound |",
        "|------:|-------------:|-----------------:|---------------:|--------------:|:------|",
    ]
    for row in rows:
        lines.append(
            "| {batch_size} | {memory_bound_ms:.3f} | {compute_bound_ms:.3f} | "
            "{lower_bound_ms:.3f} | {attainable_tok_s:.1f} | {kind} |".format(
                kind="memory" if row["memory_bound"] else "compute", **row
            )
        )
    if any("ms_per_step_wall" in row for row in rows):
        lines += [
            "",
            "| batch | measured kernel ms/step | implied BW GB/s | wall ms/step | host residual ms | GPU share | wall/effective floor | ceiling tok/s (host removed) |",
            "|------:|------------------------:|----------------:|-------------:|-----------------:|----------:|---------------------:|-----------------------------:|",
        ]
        for row in rows:
            if "ms_per_step_wall" not in row:
                continue
            lines.append(
                "| {batch_size} | {measured_kernel_ms_per_step:.3f} | "
                "{kernel_implied_bandwidth_gbps:.0f} | {ms_per_step_wall:.3f} | "
                "{host_residual_ms:.3f} | {gpu_fraction:.1%} | "
                "{wall_gap_to_effective_lower_bound:.2f}x | "
                "{ceiling_tok_s_if_host_removed:.0f} |".format(**row)
            )
    if any(row["calibration_needed"] for row in rows):
        lines += [
            "",
            "> WARNING: measured kernel ms/step is below the modeled lower bound for "
            "some batches. The roofline parameters (bandwidth, KV read amplification) "
            "must be calibrated on the target machine -- do not read this as kernel "
            "headroom.",
        ]
    return "\n".join(lines) + "\n"


def _parse_int_list(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def _parse_wall_pairs(value: str | None) -> dict[int, float]:
    """Parse ``1:2.94,8:0.429`` into ``{1: 2.94, 8: 0.429}``."""

    if not value:
        return {}
    pairs: dict[int, float] = {}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"expected batch:ms pair, got {part!r}")
        batch, ms = part.split(":", 1)
        pairs[int(batch)] = float(ms)
    return pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Decode-step roofline for the Qwen3-0.6B megakernel target."
    )
    parser.add_argument("--batch-sizes", type=_parse_int_list, default=[1, 8, 32])
    parser.add_argument("--seq-len", type=int, default=224)
    parser.add_argument("--bandwidth-gbps", type=float, default=None)
    parser.add_argument("--peak-tflops", type=float, default=None)
    parser.add_argument("--tied-embedding", action="store_true")
    parser.add_argument("--attn-blocks", type=int, default=8)
    parser.add_argument(
        "--kv-read-amp",
        type=int,
        default=None,
        help="KV re-reads per query-head block (default: --attn-blocks)",
    )
    parser.add_argument(
        "--kernel-ms-per-step",
        type=float,
        default=None,
        help="measured pure-GPU decode kernel time for one step (nsys)",
    )
    parser.add_argument(
        "--ms-per-token-wall",
        default=None,
        help="measured wall ms/token per sequence, as batch:ms pairs, e.g. 1:2.94,8:0.429",
    )
    parser.add_argument("--kernel-source", default=str(DEFAULT_KERNEL_SOURCE))
    parser.add_argument(
        "--check-kernel",
        action="store_true",
        help="assert ModelDims still matches kernel.cu constants",
    )
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args(argv)

    dims = ModelDims()
    assert_kernel_constants(dims, args.kernel_source)
    hw = HardwareSpec(
        bandwidth_gbps=args.bandwidth_gbps or HardwareSpec().bandwidth_gbps,
        peak_tflops=args.peak_tflops or HardwareSpec().peak_tflops,
    )
    model = RooflineModel(
        dims,
        hw,
        tied_embedding=args.tied_embedding,
        attn_blocks=args.attn_blocks,
        kv_read_amp=args.kv_read_amp,
    )
    wall = _parse_wall_pairs(args.ms_per_token_wall)
    rows = gap_table(
        model,
        args.batch_sizes,
        seq_len=args.seq_len,
        kernel_ms_per_step=args.kernel_ms_per_step,
        ms_per_token_wall=wall,
    )
    print(render_markdown(rows, model))
    print(
        "compute-bound batch (peak {} TFLOP/s): {:.1f}".format(
            hw.effective_tflops(), model.compute_bound_batch(args.seq_len)
        )
    )
    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(
                {
                    "hardware": asdict(hw),
                    "dims": asdict(dims),
                    "weight_bytes": model.weight_bytes(),
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
