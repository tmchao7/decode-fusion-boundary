"""Create PNG summaries from bench_megakernel.py JSON output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _number(value, default=0.0):
    return default if value is None else float(value)


def _label(experiment):
    workload = experiment["workload"]
    return f"{experiment['backend']}\np{workload['prompt_length']} b{workload['batch_size']}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    parser.add_argument("--output-dir", default="results/plots")
    args = parser.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("plot_metrics.py requires matplotlib on the server") from exc

    payload = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    experiments = payload.get("experiments", [])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [_label(item) for item in experiments]
    positions = list(range(len(experiments)))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    ax = axes[0, 0]
    ax.bar(positions, [_number(item["summary"].get("output_tok_s")) for item in experiments])
    ax.set_title("Output throughput")
    ax.set_ylabel("tokens/s")
    ax.set_xticks(positions, labels, rotation=75, ha="right")

    ax = axes[0, 1]
    for metric, color in (("ttft_ms", "tab:blue"), ("tpot_ms", "tab:orange"), ("itl_ms", "tab:green")):
        ax.plot(
            positions,
            [_number(item["summary"].get(metric, {}).get("p50"), float("nan")) for item in experiments],
            marker="o",
            label=f"{metric} p50",
            color=color,
        )
    ax.set_title("Latency p50")
    ax.set_ylabel("milliseconds")
    ax.set_xticks(positions, labels, rotation=75, ha="right")
    ax.legend()

    ax = axes[1, 0]
    memory = [
        _number(item["summary"].get("peak_memory_allocated_bytes")) / 2**30
        for item in experiments
    ]
    ax.bar(positions, memory)
    ax.set_title("Peak allocated CUDA memory")
    ax.set_ylabel("GiB")
    ax.set_xticks(positions, labels, rotation=75, ha="right")

    ax = axes[1, 1]
    plotted = False
    for comparison in payload.get("comparisons", []):
        if comparison.get("status") != "comparable":
            continue
        speedup = comparison["metrics"].get("output_tok_s", {}).get("speedup")
        if speedup is None:
            continue
        label = f"{comparison['optimized_backend']} p{comparison['prompt_length']} b{comparison['batch_size']}"
        ax.bar(label, speedup)
        plotted = True
    ax.axhline(1.0, color="black", linewidth=1)
    ax.set_title("Output throughput speedup")
    ax.set_ylabel("optimized / baseline")
    if plotted:
        ax.tick_params(axis="x", rotation=75)
    else:
        ax.text(0.5, 0.5, "No comparable Megakernel result", ha="center", va="center")
        ax.set_xticks([])

    fig.suptitle("nano-vLLM inference metrics")
    fig.savefig(output_dir / "metrics_overview.png", dpi=160)
    plt.close(fig)

    # Plot one request's token timeline when raw token timestamps are present.
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    plotted = False
    for experiment in experiments:
        requests = experiment.get("requests", [])
        if not requests or not requests[0].get("token_times"):
            continue
        times = requests[0]["token_times"]
        origin = times[0]
        ax.plot(
            range(1, len(times) + 1),
            [(value - origin) * 1000.0 for value in times],
            marker=".",
            label=_label(experiment),
        )
        plotted = True
    if plotted:
        ax.set_title("First-request token timeline")
        ax.set_xlabel("output token index")
        ax.set_ylabel("milliseconds after first token")
        ax.legend(fontsize="small")
    else:
        ax.text(0.5, 0.5, "No token timeline in JSON", ha="center", va="center")
        ax.set_axis_off()
    fig.savefig(output_dir / "token_timeline.png", dpi=160)
    plt.close(fig)

    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
