"""Small shared CLI helpers for the tools in this directory.

``parse_int_list`` lived in ``tools/bench_fusion_granularity.py`` and was imported
from there by ``tools/verify_schedule.py``, which made a verification tool depend
on a benchmarking tool for its argument parsing. One definition, no coupling.
"""

from __future__ import annotations


def parse_int_list(value: str, *, name: str) -> list[int]:
    """Parse ``"1,2,4"`` into positive ints, rejecting empties and duplicates."""

    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{name} must be a non-empty comma-separated list")
    try:
        values = [int(item) for item in items]
    except ValueError as exc:
        raise ValueError(f"{name} must contain integers; got {value!r}") from exc
    for parsed in values:
        if parsed < 1:
            raise ValueError(f"{name} values must be >= 1; got {parsed}")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not repeat values; got {values}")
    return values
