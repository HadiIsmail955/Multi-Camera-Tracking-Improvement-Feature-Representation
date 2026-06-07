import torch
import torch.nn as nn


class DINOBackbone(nn.Module):
    def __init__(
        self,
        dino_type="vit_b",
        unfreeze_last_blocks=0,
        freeze=True,
    ):
        super().__init__()

        dino_models = {
            "vit_b": "dinov2_vitb14",
            "vit_l": "dinov2_vitl14",
            "vit_g": "dinov2_vitg14",
        }

        assert dino_type in dino_models, f"Unknown dino_type: {dino_type}"

        self.encoder = torch.hub.load(
            "facebookresearch/dinov2",
            dino_models[dino_type],
            trust_repo=True,
        )

        self.embed_dim = self.encoder.embed_dim

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

        if unfreeze_last_blocks > 0:
            blocks = getattr(self.encoder, "blocks", None)

            assert blocks is not None, "DINO encoder has no transformer blocks"
            assert unfreeze_last_blocks <= len(blocks)

            for block in blocks[-unfreeze_last_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True

    def forward(self, x):
        out = self.encoder.forward_features(x)
        return out["x_norm_clstoken"]