"""Public nano-vLLM API with lazy imports for optional backends."""

__all__ = ["LLM", "SamplingParams"]


def __getattr__(name):
    if name == "LLM":
        from nanovllm.llm import LLM

        return LLM
    if name == "SamplingParams":
        from nanovllm.sampling_params import SamplingParams

        return SamplingParams
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
