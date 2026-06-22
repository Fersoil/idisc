import math
from typing import Sequence

import torch
from torch import nn


class LoRALinear(nn.Module):

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.lora_B(self.lora_A(self.drop(x)))


def inject_lora(
    root: nn.Module,
    targets: Sequence[str] = ("qkv", "proj"),
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> int:
    wanted = set(targets)
    to_wrap = [
        (module, name, child)
        for module in root.modules()
        for name, child in module.named_children()
        if name in wanted and isinstance(child, nn.Linear)
    ]
    for module, name, child in to_wrap:
        setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
    return len(to_wrap)
