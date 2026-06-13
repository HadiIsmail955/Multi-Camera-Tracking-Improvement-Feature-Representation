import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss for single-view embeddings.

    Assumes embeddings are L2-normalized.
    """

    def __init__(self, temperature: float = 0.07, eps: float = 1e-12):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        bsz = embeddings.size(0)
        if bsz < 2:
            return embeddings.new_tensor(0.0)

        sim = (embeddings @ embeddings.T) / self.temperature

        # Exclude self-comparisons from denominator.
        logits_mask = torch.ones_like(sim, dtype=torch.bool)
        logits_mask.fill_diagonal_(False)

        # Positive pairs: same label, excluding self-pairs.
        pos_mask = labels.unsqueeze(1).eq(labels.unsqueeze(0)) & logits_mask

        # Stable log-softmax over valid (non-self) pairs only.
        row_max = sim.max(dim=1, keepdim=True).values
        exp_sim = torch.exp(sim - row_max) * logits_mask
        log_prob = (
            sim - row_max - torch.log(exp_sim.sum(dim=1, keepdim=True) + self.eps)
        )

        pos_count = pos_mask.sum(dim=1)
        valid = pos_count > 0
        if not valid.any():
            return embeddings.new_tensor(0.0)

        mean_log_prob_pos = (log_prob * pos_mask).sum(dim=1) / (pos_count + self.eps)
        loss = -mean_log_prob_pos[valid].mean()
        return loss
