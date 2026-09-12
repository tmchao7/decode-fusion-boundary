from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        # Real validation, not `assert`: assertions are stripped under -O, and
        # max_tokens <= 0 never satisfies the engine's completion condition, so
        # it would decode until the KV cache runs out instead of failing fast.
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

    @property
    def is_greedy(self) -> bool:
        return self.temperature <= 1e-10
