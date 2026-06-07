import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.dino_backbone import DINOBackbone


class DINOv2ReID(nn.Module):

    def __init__(
        self,
        num_classes,
        embedding_dim=512,
        dino_type="vit_b",
        unfreeze_last_blocks=0,
        dropout=0.1,
        freeze_backbone=True,
    ):
        super().__init__()

        self.backbone = DINOBackbone(
            dino_type=dino_type,
            unfreeze_last_blocks=unfreeze_last_blocks,
            freeze=freeze_backbone,
        )

        backbone_dim = self.backbone.embed_dim

        self.projector = nn.Sequential(
            nn.Linear(backbone_dim, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, embedding_dim, bias=False),
        )

        self.bnneck = nn.BatchNorm1d(embedding_dim)
        self.bnneck.bias.requires_grad_(False)

        self.classifier = nn.Linear(
            embedding_dim,
            num_classes,
            bias=False,
        )

        self._init_params()

    def _init_params(self):
        for m in self.projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        nn.init.constant_(self.bnneck.weight, 1.0)
        nn.init.constant_(self.bnneck.bias, 0.0)

        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x):
        features = self.backbone(x)

        if isinstance(features, dict):
            features = features["x_norm_clstoken"]

        feat = self.projector(features)

        embedding = F.normalize(feat, p=2, dim=1)

        bn_feat = self.bnneck(feat)
        bn_embedding = F.normalize(bn_feat, p=2, dim=1)

        logits = self.classifier(bn_feat)

        return {
            "features": feat,
            "embedding": embedding,
            "bn_features": bn_feat,
            "bn_embedding": bn_embedding,
            "logits": logits,
        }