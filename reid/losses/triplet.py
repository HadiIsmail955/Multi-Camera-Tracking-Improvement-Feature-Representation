# src/losses/triplet.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class BHTripletLoss(nn.Module):
    """
    Batch Hard Triplet Loss.

    For each anchor:
        hardest positive = farthest sample with same label
        hardest negative = closest sample with different label

    Assumes embeddings are already L2-normalized.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        dot = embeddings @ embeddings.T
        dist = 2.0 - 2.0 * dot
        dist = dist.clamp(min=0.0)

        label_mat = labels.unsqueeze(1) == labels.unsqueeze(0)

        pos_dist = dist.clone()
        pos_dist[~label_mat] = -1e9
        hardest_pos = pos_dist.max(dim=1).values

        neg_dist = dist.clone()
        neg_dist[label_mat] = 1e9
        hardest_neg = neg_dist.min(dim=1).values

        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        return loss.mean()
