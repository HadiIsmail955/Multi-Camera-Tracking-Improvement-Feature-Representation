import torch
import torch.nn as nn
import torch.nn.functional as F


class OcclusionAwareLoss(nn.Module):
    """
    Occlusion consistency loss for ReID embeddings.

    Applies random feature dropping in embedding space and enforces consistency
    between original and perturbed embeddings via cosine similarity.
    """

    def __init__(
        self,
        feature_drop_prob: float = 0.25,
    ):
        super().__init__()
        if not (0.0 <= feature_drop_prob < 1.0):
            raise ValueError("feature_drop_prob must be in [0, 1).")

        self.feature_drop_prob = feature_drop_prob

    def _occlusion_consistency(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.feature_drop_prob <= 0.0:
            return embeddings.new_tensor(0.0)

        keep_prob = 1.0 - self.feature_drop_prob
        drop_mask = (torch.rand_like(embeddings) < keep_prob).to(embeddings.dtype)

        # Keep expected scale stable before re-normalization.
        dropped = embeddings * drop_mask / keep_prob
        dropped = F.normalize(dropped, p=2, dim=1)

        cos = F.cosine_similarity(embeddings, dropped, dim=1)
        return (1.0 - cos).mean()

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        emb = F.normalize(embeddings, p=2, dim=1)

        loss_consistency = self._occlusion_consistency(embeddings=emb)
        return loss_consistency
