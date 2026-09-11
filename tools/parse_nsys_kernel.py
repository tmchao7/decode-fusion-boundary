"""Parse nsys sqlite to summarize GPU kernel time by name.

Used to compare megakernel vs cudagraph kernel-time breakdown across batch
sizes. Reads the .sqlite produced by ``nsys profile -o <name>`` and prints
per-kernel invocation count, total ns, average ns, and the overall total.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite_path")
    parser.add_argument("--filter", default="", help="substring filter on kernel name")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--decode-steps", type=int, default=0,
                        help="if set, also report per-decode-step total (grand_total / steps)")
    args = parser.parse_args()

    data = parse(args.sqlite_path, args.filter)
    print(f"=== {args.sqlite_path} ===")
    print(f"grand total GPU kernel time: {data['grand_total_ms']:.3f} ms "
          f"({len(data['kernels'])} distinct kernels)")
    if args.decode_steps:
        print(f"per decode-step: {data['grand_total_ms'] / args.decode_steps:.3f} ms/step")
    print(f"{'kernel':60s} {'count':>6s} {'total ms':>10s} {'avg us':>10s} {'min us':>10s}")
    for name, v in data["kernels"][:args.top]:
        print(f"{name[:60]:60s} {v['count']:>6d} {v['total_ns']/1e6:>9.3f}  "
              f"{v['avg_ns']/1e3:>9.2f}  {v['min_ns']/1e3:>9.2f}")


if __name__ == "__main__":
    main()
