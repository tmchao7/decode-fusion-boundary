"""JIT compilation of the Qwen task-graph Megakernel CUDA extension."""

import os

from torch.utils.cpp_extension import load

_module = None
_DIR = os.path.dirname(os.path.abspath(__file__))
_CSRC = os.path.join(_DIR, "csrc")
SUPPORTED_ARCHES = {"sm_70", "sm_80", "sm_86", "sm_89", "sm_90", "sm_120"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def cuda_arch_for_device() -> str:
    """Resolve the target architecture on the CUDA host."""

    override = os.getenv("NANOVLLM_CUDA_ARCH")
    if override:
        arch = override if override.startswith("sm_") else f"sm_{override}"
        if arch not in SUPPORTED_ARCHES:
            raise RuntimeError(f"unsupported CUDA architecture override: {arch}")
        return arch

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Megakernel compilation requires CUDA; set NANOVLLM_CUDA_ARCH "
            "when compiling on a device-less build host"
        )
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    if arch not in SUPPORTED_ARCHES:
        raise RuntimeError(f"unsupported CUDA architecture: {arch}")
    return arch


def taskgraph_kernel_flags() -> list[str]:
    """Flags for the task-graph kernel.

    The ``#ifndef`` fallbacks in ``taskgraph_kernel.cu`` must equal these
    defaults; ``tests/test_taskgraph_abi.py`` enforces that synchronization.

    The grid size and the batch size are runtime launch-descriptor fields
    (``desc.num_sms`` / ``desc.BS``), so there is no ``-DTG_NUM_BLOCKS`` or
    ``-DTG_MAX_BATCH`` here: they were passed while the kernel ignored them.
    """

    return [
        f"-DTG_BLOCK_SIZE={_env_int('TG_BLOCK_SIZE', 256)}",
        f"-DTG_B_STAGES={_env_int('TG_B_STAGES', 8)}",
    ]


def cuda_flags() -> list[str]:
    flags = [
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        f"-arch={cuda_arch_for_device()}",
        f"-I{_CSRC}",
    ]
    if os.getenv("NANOVLLM_CUDA_FAST_MATH") == "1":
        flags.append("--use_fast_math")
    if os.getenv("NANOVLLM_CUDA_SANITIZE") == "1":
        flags.append("-lineinfo")
    return flags + taskgraph_kernel_flags()


def get_extension():
    """Build (or return cached) the task-graph CUDA extension."""

    global _module
    if _module is not None:
        return _module

    _module = load(
        name="nanovllm_megakernel_C",
        sources=[
            os.path.join(_CSRC, "torch_bindings.cpp"),
            os.path.join(_CSRC, "taskgraph_kernel.cu"),
        ],
        extra_cuda_cflags=cuda_flags(),
        extra_cflags=[f"-I{_CSRC}"],
        verbose=False,
    )
    return _module


if __name__ == "__main__":
    get_extension()
    print("Megakernel extension build: OK")
