# src/models/dinov3_lora.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import cast

from reid.models.bnneck import BNNeck
from reid.models.lora import inject_lora


class DINOv3Model(nn.Module):
    """
    DINOv3 ViT-B with LoRA adapters.
    """

    DINOV3_DIM = 768

    def __init__(
        self,
        num_classes: int,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        embed_dim: int = 256,
    ):
        super().__init__()

        print("  Loading DINOv3 ViT-B from torch.hub...")

        load_errors = []
        hub_candidates = [
            ("facebookresearch/dinov3", "dinov3_vitb16"),
            ("facebookresearch/dinov3", "dinov3_vitb14"),
            ("facebookresearch/dinov3", "dinov3_vitb"),
            ("facebookresearch/dinov2", "dinov2_vitb14"),
        ]

        self.vit: nn.Module | None = None
        loaded_repo = None

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
                loaded_repo = repo
                print(f"  [OK] Loaded {entry} from {repo}")
                break

            except Exception as exc:
                load_errors.append(f"{repo}:{entry} -> {exc}")

        if self.vit is None:
            raise RuntimeError(
                "Unable to load DINOv3/DINOv2 from torch.hub. Tried:\n  - "
                + "\n  - ".join(load_errors)
            )

        assert self.vit is not None

        if loaded_repo == "facebookresearch/dinov2":
            print(
                "  [NOTE] DINOv3 weights were unavailable; "
                "using DINOv2 ViT-B/14 fallback."
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

        self.head = BNNeck(
            in_dim=self.DINOV3_DIM,
            out_dim=embed_dim,
            use_proj=True,
        )

        self.classifier = nn.Linear(embed_dim, num_classes, bias=False)
        self.embed_dim = embed_dim

    def _resize_if_needed(self, x: torch.Tensor) -> torch.Tensor:
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
        cls_token = self.vit(x)
        emb = self.head(cls_token)
        return emb

    def forward_train(self, x: torch.Tensor):
        emb = self.forward(x)
        logits = self.classifier(emb)
        return emb, logits
