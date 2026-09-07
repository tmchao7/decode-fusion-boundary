"""Helpers for initializing local tensor-parallel process groups."""

from __future__ import annotations

import os
import socket


_ENV_PORT = "NANOVLLM_DIST_PORT"


def _parse_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise ValueError(f"{_ENV_PORT} must be an integer port, got {value!r}") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"{_ENV_PORT} must be between 1 and 65535, got {port}")
    return port


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def build_init_method() -> str:
    """Return a local TCP rendezvous URL without relying on a fixed port.

    ``NANOVLLM_DIST_PORT`` remains available for controlled multi-process runs,
    while the default chooses an ephemeral local port for each ``LLM`` engine.
    The selected URL is created once by the parent and passed to all tensor
    parallel workers, so sequential benchmark backends do not collide on a
    stale hard-coded port.
    """

    value = os.getenv(_ENV_PORT)
    port = _parse_port(value) if value else _find_free_port()
    return f"tcp://127.0.0.1:{port}"
