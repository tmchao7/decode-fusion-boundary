"""JIT compilation of the Qwen Megakernel CUDA extension."""

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
    """Resolve the target architecture on the CUDA host.

    ``NANOVLLM_CUDA_ARCH`` is useful on build/login nodes where the visible
    device is not the eventual execution device. Otherwise use the current
    CUDA device capability. A100 is the primary target; V100 is a best-effort
    build target while the runtime adapter remains conservative and gates
    execution based on precision support.
    """

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


def kernel_flags() -> list[str]:
    """Return tunable kernel flags for the selected GPU family."""

    return [
        f"-DLDG_NUM_BLOCKS={_env_int('LDG_NUM_BLOCKS', 128)}",
        f"-DLDG_BLOCK_SIZE={_env_int('LDG_BLOCK_SIZE', 512)}",
        f"-DLDG_LM_NUM_BLOCKS={_env_int('LDG_LM_NUM_BLOCKS', 1280)}",
        f"-DLDG_LM_BLOCK_SIZE={_env_int('LDG_LM_BLOCK_SIZE', 384)}",
        f"-DLDG_LM_ROWS_PER_WARP={_env_int('LDG_LM_ROWS_PER_WARP', 2)}",
        f"-DLDG_ATTN_BLOCKS={_env_int('LDG_ATTN_BLOCKS', 8)}",
        f"-DLDG_PREFETCH_QK={_env_int('LDG_PREFETCH_QK', 0)}",
        f"-DLDG_PREFETCH_THREAD_STRIDE={_env_int('LDG_PREFETCH_THREAD_STRIDE', 10)}",
        f"-DLDG_PREFETCH_DOWN={_env_int('LDG_PREFETCH_DOWN', 1)}",
        f"-DLDG_PREFETCH_ELEM_STRIDE={_env_int('LDG_PREFETCH_ELEM_STRIDE', 1)}",
        f"-DLDG_PREFETCH_BLOCK_STRIDE={_env_int('LDG_PREFETCH_BLOCK_STRIDE', 1)}",
        f"-DLDG_PREFETCH_GATE={_env_int('LDG_PREFETCH_GATE', 1)}",
        f"-DLDG_PREFETCH_UP={_env_int('LDG_PREFETCH_UP', 1)}",
        "-DLDG_USE_UINT4",
        "-DLDG_ATTENTION_VEC4",
        "-DLDG_WEIGHT_LDCS",
        "-DLDG_MLP_SMEM",
    ]


def cuda_flags() -> list[str]:
    return [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        f"-arch={cuda_arch_for_device()}",
        f"-I{_CSRC}",
    ] + kernel_flags()


def get_extension():
    """Build (or return cached) the CUDA extension."""
    global _module
    if _module is not None:
        return _module

    _module = load(
        name="nanovllm_megakernel_C",
        sources=[
            os.path.join(_CSRC, "torch_bindings.cpp"),
            os.path.join(_CSRC, "kernel.cu"),
        ],
        extra_cuda_cflags=cuda_flags(),
        extra_cflags=[f"-I{_CSRC}"],
        verbose=False,
    )
    return _module
