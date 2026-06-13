import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """
    Additive Angular Margin (ArcFace) loss.

    Expects:
    - normalized embeddings from the model head
    - classifier weights from the model
    """

    def __init__(
        self,
        scale: float = 30.0,
        margin: float = 0.5,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        classifier_weight: torch.Tensor,
    ) -> torch.Tensor:
        emb = F.normalize(embeddings, p=2, dim=1)
        w = F.normalize(classifier_weight, p=2, dim=1)

        cosine = F.linear(emb, w)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=0.0, max=1.0))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0.0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = (one_hot * phi + (1.0 - one_hot) * cosine) * self.scale
        return F.cross_entropy(logits, labels)
