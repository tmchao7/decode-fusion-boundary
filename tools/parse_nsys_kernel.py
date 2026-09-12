"""Parse nsys sqlite to summarize GPU kernel time by name.

Used to compare megakernel vs cudagraph kernel-time breakdown across batch
sizes. Reads the .sqlite produced by ``nsys profile -o <name>`` and prints
per-kernel invocation count, total ns, average ns, and the overall total.

``--memory`` additionally reports GPU memory-throughput counters when the
profile collected them.  Be aware that ``nsys profile --trace=cuda`` alone does
**not** collect DRAM counters -- those need GPU metrics sampling
(``--gpu-metrics-device`` / ``--enable-gpu-metrics`` depending on the nsys
version).  Rather than printing a silent zero, this tool reports the metric as
unavailable and says which tables it did find.

``--decode-steps`` plus ``--roofline`` turns the kernel total into an implied
bandwidth, which is how docs/07 §10 checks whether the measured per-step kernel
time is even physically possible for the 1.503 GB/step weight traffic::

    python -m tools.parse_nsys_kernel results/nsys_b8_cg.sqlite \
        --decode-steps 49 --memory --roofline
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict

# Substrings that identify memory-throughput tables across nsys versions.
MEMORY_TABLE_HINTS = (
    "GPU_MEMORY",
    "MEMORY_BW",
    "DRAM",
    "MIG_MEMORY",
    "CUPTI_ACTIVITY_KIND_MEMORY",
)


def parse(path: str, name_filter: str = "") -> dict:
    db = sqlite3.connect(path)
    rows = db.execute(
        "SELECT K.demangledName, K.end - K.start, K.gridX, K.blockX, "
        "K.registersPerThread, S.value "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL K "
        "JOIN StringIds S ON S.id = K.demangledName "
        "WHERE K.end > K.start"
    ).fetchall()
    agg = defaultdict(lambda: {"count": 0, "total_ns": 0, "samples": []})
    for demangled_id, dur_ns, grid_x, block_x, regs, name in rows:
        short = name.split("(")[0].split("<")[0]
        if name_filter and name_filter not in short:
            continue
        a = agg[short]
        a["count"] += 1
        a["total_ns"] += dur_ns
        a["samples"].append(dur_ns)
    db.close()
    # compute average + sort by total
    result = []
    for k, v in agg.items():
        v["avg_ns"] = v["total_ns"] / v["count"] if v["count"] else 0
        v["min_ns"] = min(v["samples"]) if v["samples"] else 0
        result.append((k, v))
    result.sort(key=lambda x: -x[1]["total_ns"])
    grand_total = sum(v["total_ns"] for _, v in result)
    return {"kernels": result, "grand_total_ns": grand_total,
            "grand_total_ms": grand_total / 1e6}


def list_tables(path: str) -> list[str]:
    db = sqlite3.connect(path)
    tables = [
        row[0]
        for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    ]
    db.close()
    return tables


def parse_memory(path: str) -> dict:
    """Report DRAM/memory throughput if the profile actually captured it.

    Returns a dict with ``available=False`` and the candidate tables that were
    checked when no memory table exists, so a missing metric is never confused
    with a measured zero.
    """

    tables = list_tables(path)
    candidates = [
        table
        for table in tables
        if any(hint in table.upper() for hint in MEMORY_TABLE_HINTS)
    ]
    if not candidates:
        return {
            "available": False,
            "reason": "no memory-throughput table in this profile",
            "candidate_tables_checked": list(MEMORY_TABLE_HINTS),
            "tables_present": tables,
            "hint": "re-profile with GPU metrics sampling enabled "
            "(--gpu-metrics-device / --enable-gpu-metrics) or fall back to "
            "tools.bench_roofline_a100 --sections bandwidth for a measured "
            "attainable-bandwidth number",
        }

    db = sqlite3.connect(path)
    columns = {
        table: [row[1] for row in db.execute(f"PRAGMA table_info({table})")]
    }
    db.close()
    return {
        "available": True,
        "tables": candidates,
        "columns": columns,
        "note": "raw tables only; decode the counter columns with the nsys "
        "schema for your version before trusting absolute GB/s",
    }


def roofline_context(
    kernel_ms_per_step: float,
    *,
    weight_bytes: float | None = None,
    spec_bandwidth_gbps: float | None = None,
) -> dict:
    """Convert a per-step kernel time into an implied weight-streaming rate."""

    if weight_bytes is None or spec_bandwidth_gbps is None:
        from nanovllm.roofline import HardwareSpec, ModelDims, RooflineModel

        model = RooflineModel(ModelDims(), HardwareSpec())
        weight_bytes = model.weight_bytes()
        spec_bandwidth_gbps = model.hardware.effective_bandwidth_gbps()
    if kernel_ms_per_step <= 0:
        return {"valid": False, "reason": "kernel_ms_per_step must be positive"}
    implied = weight_bytes / (kernel_ms_per_step / 1000.0) / 1e9
    return {
        "valid": True,
        "weight_bytes": weight_bytes,
        "spec_bandwidth_gbps": spec_bandwidth_gbps,
        "implied_bandwidth_gbps": implied,
        "spec_fraction": implied / spec_bandwidth_gbps
        if spec_bandwidth_gbps
        else None,
        "physically_possible": implied <= spec_bandwidth_gbps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite_path")
    parser.add_argument("--filter", default="", help="substring filter on kernel name")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--decode-steps", type=int, default=0,
                        help="if set, also report per-decode-step total (grand_total / steps)")
    parser.add_argument("--memory", action="store_true",
                        help="report GPU memory-throughput tables if present")
    parser.add_argument("--roofline", action="store_true",
                        help="translate per-step kernel time into implied weight bandwidth")
    args = parser.parse_args()

    data = parse(args.sqlite_path, args.filter)
    print(f"=== {args.sqlite_path} ===")
    print(f"grand total GPU kernel time: {data['grand_total_ms']:.3f} ms "
          f"({len(data['kernels'])} distinct kernels)")
    per_step_ms = None
    if args.decode_steps:
        per_step_ms = data["grand_total_ms"] / args.decode_steps
        print(f"per decode-step: {per_step_ms:.3f} ms/step")
    print(f"{'kernel':60s} {'count':>6s} {'total ms':>10s} {'avg us':>10s} {'min us':>10s}")
    for name, v in data["kernels"][:args.top]:
        print(f"{name[:60]:60s} {v['count']:>6d} {v['total_ns']/1e6:>9.3f}  "
              f"{v['avg_ns']/1e3:>9.2f}  {v['min_ns']/1e3:>9.2f}")

    if args.memory:
        memory = parse_memory(args.sqlite_path)
        print("\n=== memory throughput ===")
        if not memory["available"]:
            print(f"UNAVAILABLE: {memory['reason']}")
            print(f"hint: {memory['hint']}")
        else:
            print(f"tables: {', '.join(memory['tables'])}")
            print(f"note: {memory['note']}")

    if args.roofline:
        if per_step_ms is None:
            print("\n--roofline needs --decode-steps; skipping")
        else:
            context = roofline_context(per_step_ms)
            print("\n=== roofline check ===")
            print(f"weight bytes/step: {context['weight_bytes'] / 1e9:.3f} GB "
                  f"(Qwen3-0.6B bf16, model, excludes embedding rows)")
            print(f"implied weight bandwidth: {context['implied_bandwidth_gbps']:.0f} GB/s "
                  f"({context['spec_fraction']:.0%} of the "
                  f"{context['spec_bandwidth_gbps']:.0f} GB/s A100 spec)")
            if not context["physically_possible"]:
                print("WARNING: implied bandwidth exceeds the device spec; this "
                      "kernel-time total cannot cover a full decode step. Check "
                      "which kernels the filter captured before trusting it.")


if __name__ == "__main__":
    main()
