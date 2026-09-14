"""Plot the megakernel-vs-cudagraph crossover curve from bench_burst_batch JSON.

Produces a two-panel figure:
  - Left:  decode ms/token vs batch size (log-y), cudagraph vs megakernel,
           with the crossover at batch=1 annotated and the megakernel
           fixed-grid plateau / serial-burst region marked.
  - Right: speedup (cudagraph / megakernel) vs batch size (log-y), with the
           1.0 break-even line and the "megakernel wins" zone shaded.

Single-launch batches (<= SAFE_BURST_BATCH) are drawn solid; batches that
split into serial bursts (> SAFE_BURST_BATCH) are drawn dashed to honestly
show the fixed-grid architecture ceiling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help="results/crossover_scan.json")
    parser.add_argument("--output", default="results/crossover_plot.png")
    parser.add_argument("--title", default="MegaKernel vs CUDA Graph decode crossover (Qwen3-0.6B, A100)")
    args = parser.parse_args()

    with open(args.json_path, encoding="utf-8") as fh:
        data = json.load(fh)

    safe = data.get("safe_burst_batch", 8)
    batches, cg_ms, bu_ms, speedup, single = [], [], [], [], []
    for r in data["records"]:
        if r["burst"] is None:
            continue
        batches.append(r["batch"])
        cg_ms.append(r["cudagraph"]["ms_per_tok_min"])
        bu_ms.append(r["burst"]["ms_per_tok_min"])
        speedup.append(r["speedup"])
        single.append(r["batch"] <= safe)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    cg_color, bu_color = "#1f77b4", "#d62728"

    # --- Panel 1: ms/token vs batch ---
    ax1.plot(batches, cg_ms, "o-", color=cg_color, lw=2.2, ms=7,
             label="CUDA Graph (cuBLAS batched)")
    # megakernel: solid for single-burst, dashed for serial-burst
    bu_single_b = [b for b, s in zip(batches, single) if s]
    bu_single_m = [m for m, s in zip(bu_ms, single) if s]
    bu_serial_b = [b for b, s in zip(batches, single) if not s]
    bu_serial_m = [m for m, s in zip(bu_ms, single) if not s]
    ax1.plot(bu_single_b, bu_single_m, "s-", color=bu_color, lw=2.2, ms=7,
             label="MegaKernel burst (single launch)")
    if bu_serial_b:
        ax1.plot(bu_serial_b, bu_serial_m, "s--", color=bu_color, lw=2.0, ms=7,
                 label="MegaKernel burst (serial bursts)")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("batch size", fontsize=12)
    ax1.set_ylabel("decode ms / token (min-of-runs)", fontsize=12)
    ax1.set_title("Per-token latency vs batch", fontsize=12)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=9, loc="upper right")
    # annotate crossover
    if batches and bu_ms[0] < cg_ms[0]:
        ax1.annotate(
            f"crossover\nbatch=1: MegaKernel {speedup[0]:.2f}x",
            xy=(batches[0], bu_ms[0]),
            xytext=(batches[0] * 2.2, bu_ms[0] * 0.35),
            fontsize=9, ha="left",
            arrowprops=dict(arrowstyle="->", color=bu_color, lw=1.3),
        )
    # annotate plateau
    plateau_b = [b for b in batches if 8 <= b <= 32]
    if plateau_b:
        ax1.axvspan(8, 32, color=bu_color, alpha=0.06)
        ax1.text(16, min(bu_ms) * 0.8, "fixed-grid\nplateau\n(~6.5 ms/tok)",
                 fontsize=8, ha="center", color=bu_color, alpha=0.8)

    # --- Panel 2: speedup vs batch ---
    ax2.axhline(1.0, color="black", lw=1.0, ls="--", alpha=0.6)
    ax2.fill_between([min(batches) * 0.7, max(batches) * 1.3], 1.0, 100,
                     color="#2ca02c", alpha=0.10, label="MegaKernel wins")
    sp_single_b = [b for b, s in zip(batches, single) if s]
    sp_single_s = [s for s, x in zip(speedup, single) if x]
    sp_serial_b = [b for b, s in zip(batches, single) if not s]
    sp_serial_s = [s for s, x in zip(speedup, single) if not x]
    ax2.plot(sp_single_b, sp_single_s, "s-", color=bu_color, lw=2.2, ms=7,
             label="single launch")
    if sp_serial_b:
        ax2.plot(sp_serial_b, sp_serial_s, "s--", color=bu_color, lw=2.0, ms=7,
                 label="serial bursts")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("batch size", fontsize=12)
    ax2.set_ylabel("speedup (CG ms/tok ÷ MegaKernel ms/tok)", fontsize=12)
    ax2.set_title("MegaKernel speedup vs batch", fontsize=12)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=9, loc="upper right")
    for b, s in zip(batches, speedup):
        ax2.annotate(f"{s:.2f}x", (b, s), textcoords="offset points",
                     xytext=(6, 4), fontsize=7.5, color=bu_color)

    fig.suptitle(args.title, fontsize=13, y=1.02)
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"plot written to {out}")

    # markdown table for docs
    print("\n| batch | CG ms/tok | MegaKernel ms/tok | speedup | verdict |")
    print("|------:|----------:|------------------:|--------:|:--------|")
    for b, c, m, s, r in zip(batches, cg_ms, bu_ms, speedup, data["records"]):
        v = r["verdict"]
        note = f" ({r['note']})" if r["note"] else ""
        print(f"| {b} | {c:.3f} | {m:.3f} | {s:.2f}x | {v}{note} |")


if __name__ == "__main__":
    main()
