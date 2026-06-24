from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from reid.models.bnneck import BNNeck
from reid.models.lora import inject_lora


class DINOv2Model(nn.Module):
    """
    DINOv2 with configurable backbone variant.
    """

    def __init__(
        self,
        num_classes: int,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        embed_dim: int = 256,
        feature_mode: str = "cls",
        model_name: str = "dinov2_vitb14",
    ):
        super().__init__()

        if feature_mode not in {"cls", "cls_patchavg"}:
            raise ValueError(
                f"Invalid feature_mode: {feature_mode!r}. "
                "Choose 'cls' or 'cls_patchavg'."
            )
        if model_name not in {"dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"}:
            raise ValueError(
                f"Invalid model_name: {model_name!r}. "
                "Choose 'dinov2_vits14', 'dinov2_vitb14', or 'dinov2_vitl14'."
            )

        self.feature_mode = feature_mode
        self.model_name = model_name

        print(f"  Loading DINOv2 ({model_name}) from torch.hub...")

        load_errors = []
        model_fallbacks = {
            "dinov2_vits14": ["dinov2_vits14", "dinov2_vits14_reg"],
            "dinov2_vitb14": ["dinov2_vitb14", "dinov2_vitb14_reg"],
            "dinov2_vitl14": ["dinov2_vitl14", "dinov2_vitl14_reg"],
        }
        hub_candidates = [
            ("facebookresearch/dinov2", entry) for entry in model_fallbacks[model_name]
        ]

        self.vit: nn.Module | None = None
        loaded_entry: str | None = None

        for repo, entry in hub_candidates:
            try:
                self.vit = cast(
                    nn.Module,
                    torch.hub.load(
                        repo,
                        entry,
                        pretrained=True,
                        verbose=False,
                    ),
                )
                loaded_entry = entry
                print(f"  [OK] Loaded {entry} from {repo}")
                break

            except Exception as exc:
                load_errors.append(f"{repo}:{entry} -> {exc}")

        if self.vit is None:
            raise RuntimeError(
                "Unable to load DINOv2 from torch.hub. Tried:\n  - "
                + "\n  - ".join(load_errors)
            )

        if use_lora:
            for param in self.vit.parameters():
                param.requires_grad = False

            self.vit = inject_lora(
                self.vit,
                rank=lora_rank,
                alpha=lora_alpha,
            )
        else:
            for param in self.vit.parameters():
                param.requires_grad = True
            print("  Full fine-tuning: all ViT parameters unfrozen")

        vit_embed_dim = getattr(self.vit, "embed_dim", None)
        if isinstance(vit_embed_dim, int) and vit_embed_dim > 0:
            backbone_dim = vit_embed_dim
        elif loaded_entry is not None and "vitl" in loaded_entry:
            backbone_dim = 1024
        elif loaded_entry is not None and "vits" in loaded_entry:
            backbone_dim = 384
        else:
            backbone_dim = 768

        feat_dim = backbone_dim if self.feature_mode == "cls" else backbone_dim * 2

        self.head = BNNeck(
            in_dim=feat_dim,
            out_dim=embed_dim,
            use_proj=False,
        )

        self.classifier = nn.Linear(self.head.out_dim, num_classes, bias=False)
        self.embed_dim = self.head.out_dim

    def _resize_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        assert self.vit is not None
        patch_embed = getattr(self.vit, "patch_embed", None)
        proj = getattr(patch_embed, "proj", None)
        kernel_size = getattr(proj, "kernel_size", None)

        if isinstance(kernel_size, tuple) and len(kernel_size) == 2:
            patch_h, patch_w = kernel_size
            h, w = x.shape[-2], x.shape[-1]

            if (h % patch_h != 0) or (w % patch_w != 0):
                new_h = max(patch_h, (h // patch_h) * patch_h)
                new_w = max(patch_w, (w // patch_w) * patch_w)

                x = F.interpolate(
                    x,
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                )

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.vit is not None
        x = self._resize_if_needed(x)
        feat = self._extract_features(x)
        emb = self.head(feat)
        return emb

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        assert self.vit is not None

        forward_features = getattr(self.vit, "forward_features", None)
        if not callable(forward_features):
            raise RuntimeError("DINOv2 backbone does not expose forward_features().")

        feats = forward_features(x)

        if not isinstance(feats, dict):
            raise RuntimeError("DINOv2 forward_features() must return a dict.")

        cls_token = feats.get("x_norm_clstoken")
        patch_tokens = feats.get("x_norm_patchtokens")

        if not isinstance(cls_token, torch.Tensor):
            raise RuntimeError("DINOv2 features missing x_norm_clstoken.")

        if self.feature_mode == "cls":
            return cls_token

        if not isinstance(patch_tokens, torch.Tensor):
            raise RuntimeError("DINOv2 features missing x_norm_patchtokens.")

        patch_avg = patch_tokens.mean(dim=1)
        return torch.cat([cls_token, patch_avg], dim=1)

    def forward_train(self, x: torch.Tensor):
        emb = self.forward(x)
        logits = self.classifier(emb)
        return emb, logits
