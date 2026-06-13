# src/models/bnneck.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class BNNeck(nn.Module):
    """BatchNorm neck used to project features to ReID embeddings."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 256,
        use_proj: bool = True,
    ):
        super().__init__()

        if use_proj:
            self.proj = nn.Linear(in_dim, out_dim, bias=False)
            neck_dim = out_dim
        else:
            self.proj = nn.Identity()
            neck_dim = in_dim

        self.bn = nn.BatchNorm1d(neck_dim)
        self.bn.bias.requires_grad_(False)

        self.out_dim = neck_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.bn(x)
        return F.normalize(x, p=2, dim=1)
