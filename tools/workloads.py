"""Load and validate benchmark workload specifications."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_workloads(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    performance = payload.get("performance", {})
    prompt_lengths = performance.get("prompt_lengths", [])
    if not prompt_lengths or any(int(length) <= 0 for length in prompt_lengths):
        raise ValueError("performance.prompt_lengths must contain positive integers")
    for item in payload.get("correctness", []):
        for required in ("id", "category", "prompt", "max_tokens"):
            if required not in item:
                raise ValueError(f"correctness workload missing {required!r}")
        if int(item["max_tokens"]) <= 0:
            raise ValueError(f"invalid max_tokens for {item['id']!r}")
    for item in performance.get("batch_mixes", []):
        lengths = item.get("prompt_lengths", [])
        if not item.get("id") or not lengths or any(int(length) <= 0 for length in lengths):
            raise ValueError("each performance.batch_mixes item needs id and positive prompt_lengths")
    return payload


def select_correctness_workloads(
    payload: dict[str, Any], workload_ids: list[str] | None = None
) -> list[dict[str, Any]]:
    items = payload.get("correctness", [])
    if workload_ids is None:
        return items
    selected = {item["id"] for item in items if item["id"] in workload_ids}
    missing = set(workload_ids) - selected
    if missing:
        raise ValueError(f"unknown correctness workload(s): {', '.join(sorted(missing))}")
    return [item for item in items if item["id"] in selected]
