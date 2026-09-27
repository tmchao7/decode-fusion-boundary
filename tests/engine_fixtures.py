"""Shared CPU-only fixtures for engine tests (scheduler / KV block layout).

Importing this module needs ``xxhash`` and ``numpy`` (``BlockManager``'s
dependencies) but no torch, no ``transformers`` and no CUDA.

The fixtures deliberately build the **production** objects:

* a real :class:`~nanovllm.config.Config` (so ``__post_init__`` validation still
  runs) with ``transformers`` replaced in ``sys.modules`` for the duration of the
  call -- this is the only dependency the config needs and it is what the
  megakernel admission path would otherwise force on a CPU-only box;
* a real :class:`~nanovllm.engine.scheduler.Scheduler`, hence a real
  :class:`~nanovllm.engine.block_manager.BlockManager`;
* a real :class:`~nanovllm.engine.sequence.Sequence`.

The previous phase-0 tests replaced the scheduler's imports with
``SimpleNamespace`` stubs, which meant the "scheduler behavior" tests actually
exercised a hand-built object graph. Using production constructors keeps the
behavior tests honest and removes the stub.
"""

from __future__ import annotations

import sys
import types

from nanovllm.config import Config
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

BLOCK_SIZE = 256


class _FakeHfConfig:
    """The subset of a HF config ``Config.__post_init__`` reads."""

    max_position_embeddings = 4096


def _fake_transformers_module() -> types.ModuleType:
    module = types.ModuleType("transformers")
    module.AutoConfig = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda *args, **kwargs: _FakeHfConfig()
    )
    return module


def make_config(model_dir, **overrides) -> Config:
    """Build a real :class:`Config` without requiring ``transformers``."""

    saved = sys.modules.get("transformers")
    sys.modules["transformers"] = _fake_transformers_module()
    try:
        return Config(model=str(model_dir), **overrides)
    finally:
        if saved is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = saved


def make_scheduler(model_dir, *, num_blocks: int = 64, **overrides) -> Scheduler:
    """Real ``Scheduler`` (and therefore a real ``BlockManager``)."""

    config = make_config(model_dir, num_kvcache_blocks=num_blocks, **overrides)
    return Scheduler(config)


def make_sequence(token_ids, **sampling) -> Sequence:
    """Greedy sequence with EOS ignored, so tests drive completion themselves."""

    params = SamplingParams(
        **{"temperature": 0.0, "max_tokens": 8, "ignore_eos": True, **sampling}
    )
    return Sequence(list(token_ids), params)
