import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Wrap an existing nn.Linear with a LoRA delta.
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
    ):
        super().__init__()

        in_features = linear.in_features
        out_features = linear.out_features

        self.linear = linear

        for param in self.linear.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scale = alpha / rank

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        delta = (x @ self.lora_A.T) @ self.lora_B.T
        return base + delta * self.scale


def inject_lora(
    vit_model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
) -> nn.Module:
    replaced = 0

    for name, module in vit_model.named_modules():
        if name.endswith("attn"):
            if hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear):
                module.qkv = LoRALinear(module.qkv, rank=rank, alpha=alpha)
                replaced += 1

            elif (
                hasattr(module, "q")
                and hasattr(module, "v")
                and isinstance(module.q, nn.Linear)
                and isinstance(module.v, nn.Linear)
            ):
                module.q = LoRALinear(module.q, rank=rank, alpha=alpha)
                module.v = LoRALinear(module.v, rank=rank, alpha=alpha)
                replaced += 2

    print(
        f"  LoRA injected into {replaced} attention projection(s) "
        f"(rank={rank}, alpha={alpha})"
    )

    return vit_model
