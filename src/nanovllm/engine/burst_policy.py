"""No-sync burst eligibility policy (torch-free, unit-testable).

Kept out of ``llm_engine`` and ``model_runner`` because both import torch, which
would make this policy untestable on a CPU-only checkout.

Background: a burst runs up to ``max_steps`` decode steps inside one
cooperative kernel with device-side EOS.  The EOS flag handed to the kernel is
batch-wide because a cooperative grid cannot shrink mid-burst, so:

* a batch where *every* sequence ignores EOS may burst by passing the ``-1``
  sentinel, which the update kernels read as "never deactivate";
* a batch where *no* sequence ignores EOS may burst with the real token id;
* a *mixed* batch must fall back, because one device flag cannot express
  per-sequence behaviour.

That first case is what makes the megakernel-vs-cudagraph comparison fair: the
controlled benchmark suites set ``ignore_eos=True`` to force a fixed token
count, so before this the benchmark could never exercise the burst path and the
headline table silently measured the step-wise path instead.
"""

from __future__ import annotations

from typing import Any, Sequence


def burst_batch_eligibility(config: Any, seqs: Sequence[Any]) -> tuple[bool, str]:
    """Whether ``seqs`` may run as one no-sync megakernel burst.

    Returns ``(eligible, reason)``; ``reason`` is empty when eligible.
    """

    if not getattr(config, "megakernel_burst", False):
        return False, "megakernel_burst is disabled"
    if getattr(config, "decode_backend", None) != "megakernel":
        return False, "decode backend is not megakernel"
    if getattr(config, "megakernel_batch_kernel", None) != "wmma":
        return False, "batch kernel is not wmma"
    if not seqs:
        return False, "no running sequences"
    for seq in seqs:
        if not getattr(seq, "is_greedy", False):
            return False, "burst requires greedy sampling"
    ignore_flags = [bool(getattr(seq, "ignore_eos", False)) for seq in seqs]
    if any(ignore_flags) and not all(ignore_flags):
        return False, "mixed ignore_eos batch cannot share one device EOS flag"
    if len(seqs) > 1 and not getattr(config, "megakernel_batch_decode", False):
        return (
            False,
            "batch>1 defaults to cudagraph unless megakernel_batch_decode is set",
        )
    return True, ""


def burst_eos_token(config: Any, seqs: Sequence[Any]) -> int:
    """Device-side EOS argument for a burst: ``-1`` when the batch ignores EOS."""

    if seqs and all(getattr(seq, "ignore_eos", False) for seq in seqs):
        return -1
    return int(getattr(config, "eos", -1))
