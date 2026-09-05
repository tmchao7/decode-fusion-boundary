import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float()
        greedy = temperatures <= 1e-10
        safe_temperatures = temperatures.clamp_min(1e-10).unsqueeze(dim=1)
        scaled_logits = logits / safe_temperatures
        probs = torch.softmax(scaled_logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        greedy_tokens = logits.argmax(dim=-1)
        return torch.where(greedy, greedy_tokens, sample_tokens)
