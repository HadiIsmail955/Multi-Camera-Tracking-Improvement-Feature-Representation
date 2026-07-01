# src/models/osnet.py

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from reid.models.bnneck import BNNeck


class OSNetBaseline(nn.Module):
    """
    OSNet baseline loaded through torchreid.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        weight_path: str | None = None,
    ):
        super().__init__()

        try:
            import torchreid
            from torchreid import utils as torchreid_utils
        except ImportError as exc:
            raise ImportError(
                "torchreid not installed. Run: pip install torchreid"
            ) from exc

        self.backbone = torchreid.models.build_model(
            name="osnet_x1_0",
            num_classes=num_classes,
            pretrained=pretrained,
        )

        if weight_path:
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f"OSNet weight file not found: {weight_path}")

            print(f"  Loading OSNet ReID weights from: {weight_path}")
            torchreid_utils.load_pretrained_weights(self.backbone, weight_path)

        self.backbone.classifier = nn.Identity()

        backbone_dim = 512

        self.head = BNNeck(
            in_dim=backbone_dim,
            out_dim=256,
            use_proj=False,
        )

        self.classifier = nn.Linear(backbone_dim, num_classes, bias=False)

        self.embed_dim = 256
        self.backbone_dim = backbone_dim
        self.use_raw_inference = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)

        if self.use_raw_inference and not self.training:
            return F.normalize(feat, p=2, dim=1)

        return self.head(feat)

    def forward_train(self, x: torch.Tensor):
        feat = self.backbone(x)
        emb = self.head(feat)
        logits = self.classifier(emb)
        return emb, logits
