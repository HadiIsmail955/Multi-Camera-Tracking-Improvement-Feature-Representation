import os
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BNNeck(nn.Module):
    """BatchNorm neck used to project features to ReID embeddings.

    Feature flow:
        input -> Linear (optional) -> BatchNorm1d -> L2-normalize

    Args:
        in_dim: Backbone feature dimension.
        out_dim: Output embedding dimension.
        use_proj: If True, apply a linear projection from `in_dim` to `out_dim`.
    """

    def __init__(self, in_dim: int, out_dim: int = 256, use_proj: bool = True):
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


class OSNetBaseline(nn.Module):
    """
    OSNet loaded via torchreid.
    Outputs a 512-dim L2-normalised embedding.

    Install: pip install torchreid
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
            self.backbone = torchreid.models.build_model(
                name='osnet_x1_0',
                num_classes=num_classes,
                pretrained=pretrained,
            )
            if weight_path:
                if not os.path.exists(weight_path):
                    raise FileNotFoundError(
                        f"OSNet weight file not found: {weight_path}"
                    )
                print(f"  Loading OSNet ReID weights from: {weight_path}")
                torchreid_utils.load_pretrained_weights(self.backbone, weight_path)
                
            self.backbone.classifier = nn.Identity()
            backbone_dim = 512
        except ImportError:
            raise ImportError(
                "torchreid not installed. Run: pip install torchreid"
            )

        self.head = BNNeck(in_dim=backbone_dim, out_dim=256, use_proj=True)
        
        self.classifier = nn.Linear(256, num_classes, bias=False)
        self.embed_dim  = 256
        self.backbone_dim = backbone_dim
        self.use_raw_inference = False

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : [B, 3, H, W]
        Returns:
            emb : [B, D]  L2-normalised embedding
        """
        feat = self.backbone(x)          # [B, 512]
        if self.use_raw_inference and not self.training:
            emb = F.normalize(feat, p=2, dim=1)
        else:
            emb = self.head(feat)        # [B, 256]  L2-normalised
        return emb

    def forward_train(self, x: torch.Tensor):
        """Returns (embedding, logits) for combined CE + metric loss training."""
        # During training, use projection head
        feat = self.backbone(x)          # [B, 512]
        emb  = self.head(feat)           # [B, 256]  L2-normalised
        logits = self.classifier(emb)
        return emb, logits

class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear with a LoRA delta:
        output = W x  +  (B @ A) * scale

    Only A and B are trainable; original W is frozen.

    Args:
        linear : the original nn.Linear to wrap
        rank   : LoRA rank r
        alpha  : LoRA scaling factor (scale = alpha / rank)
    """

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        in_features  = linear.in_features
        out_features = linear.out_features

        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scale  = alpha / rank

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base  = self.linear(x)
        delta = (x @ self.lora_A.T) @ self.lora_B.T
        return base + delta * self.scale


def inject_lora(vit_model: nn.Module, rank: int = 8, alpha: float = 16.0):

    replaced = 0
    for name, module in vit_model.named_modules():
        if name.endswith('attn'):
            if hasattr(module, 'qkv') and isinstance(module.qkv, nn.Linear):
                module.qkv = LoRALinear(module.qkv, rank=rank, alpha=alpha)
                replaced += 1
            # Some DINOv3 variants expose separate q_proj, v_proj
            elif hasattr(module, 'q') and isinstance(module.q, nn.Linear):
                module.q = LoRALinear(module.q, rank=rank, alpha=alpha)
                module.v = LoRALinear(module.v, rank=rank, alpha=alpha)
                replaced += 2
    print(f"  LoRA injected into {replaced} attention projection(s)  "
          f"(rank={rank}, alpha={alpha})")
    return vit_model


class DINOv3LoRA(nn.Module):
    """
    DINOv3 ViT-B with LoRA adapters in all attention QKV projections.

    Trainable parameters:
        - LoRA A/B matrices in every attention block
        - BNNeck (projection head)
        - CE classifier head

    Frozen parameters:
        - All original ViT weights (patch embed, LayerNorm, FFN, etc.)

    Args:
        num_classes : number of training identities (for CE loss head)
        lora_rank   : LoRA rank r (start with 8, scale to 16 if underfitting)
        lora_alpha  : LoRA scaling factor
        embed_dim   : final embedding dimension after projection head
    """

    DINOV3_DIM = 768   # ViT-B CLS token dimension

    def __init__(
        self,
        num_classes : int,
        lora_rank   : int   = 8,
        lora_alpha  : float = 16.0,
        embed_dim   : int   = 256,
    ):
        super().__init__()

        print("  Loading DINOv3 ViT-B from torch.hub...")
        load_errors = []
        hub_candidates = [
            ('facebookresearch/dinov3', 'dinov3_vitb16'),
            ('facebookresearch/dinov3', 'dinov3_vitb14'),
            ('facebookresearch/dinov3', 'dinov3_vitb'),
            ('facebookresearch/dinov2', 'dinov2_vitb14'),
        ]
        self.vit = None
        loaded_repo = None
        loaded_entry = None
        for repo, entry in hub_candidates:
            try:
                self.vit = torch.hub.load(
                    repo,
                    entry,
                    pretrained=True,
                    verbose=False,
                )
                loaded_repo = repo
                loaded_entry = entry
                print(f"  [OK] Loaded {entry} from {repo}")
                break
            except Exception as exc:
                load_errors.append(f"{repo}:{entry} -> {exc}")

        if self.vit is None:
            raise RuntimeError(
                "Unable to load DINOv3/DINOv2 from torch.hub. Tried:\n  - "
                + "\n  - ".join(load_errors)
            )

        if loaded_repo == 'facebookresearch/dinov2':
            print("  [NOTE] DINOv3 weights were unavailable; using DINOv2 ViT-B/14 fallback.")

        # Freeze all base ViT parameters
        for p in self.vit.parameters():
            p.requires_grad = False

        # Inject LoRA into attention layers
        self.vit = inject_lora(self.vit, rank=lora_rank, alpha=lora_alpha)

        # Projection head: CLS (768) -> 512 -> 256
        self.head = BNNeck(
            in_dim=self.DINOV3_DIM,
            out_dim=embed_dim,
            use_proj=True,
        )

        # CE classifier head (training only)
        self.classifier = nn.Linear(embed_dim, num_classes, bias=False)
        self.embed_dim  = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : [B, 3, H, W]  — expects 256×128 input
        Returns:
            emb : [B, 256]  L2-normalised embedding
        """
        
        patch_embed = getattr(self.vit, 'patch_embed', None)
        proj = getattr(patch_embed, 'proj', None)
        kernel_size = getattr(proj, 'kernel_size', None)
        if isinstance(kernel_size, tuple) and len(kernel_size) == 2:
            patch_h, patch_w = kernel_size
            h, w = x.shape[-2], x.shape[-1]
            if (h % patch_h != 0) or (w % patch_w != 0):
                new_h = max(patch_h, (h // patch_h) * patch_h)
                new_w = max(patch_w, (w // patch_w) * patch_w)
                x = F.interpolate(
                    x,
                    size=(new_h, new_w),
                    mode='bilinear',
                    align_corners=False,
                )

        cls_token = self.vit(x)          # [B, 768]  CLS token
        emb       = self.head(cls_token) # [B, 256]  L2-normalised
        return emb

    def forward_train(self, x: torch.Tensor):
        """Returns (embedding, logits) for combined CE + metric loss training."""
        emb    = self.forward(x)
        logits = self.classifier(emb)
        return emb, logits


class DINOv2FullFT(nn.Module):
    """DINOv2 ViT-B/14 with full fine-tuning (all parameters trainable)."""

    DINOV2_DIM = 768

    def __init__(self, num_classes: int, embed_dim: int = 256):
        super().__init__()

        print("  Loading DINOv2 ViT-B/14 from torch.hub (full fine-tuning)...")
        load_errors = []
        hub_candidates = [
            ('facebookresearch/dinov2', 'dinov2_vitb14'),
            ('facebookresearch/dinov2', 'dinov2_vitb14_reg'),
            ('facebookresearch/dinov2', 'dinov2_vitb16'),
        ]

        self.vit = None
        for repo, entry in hub_candidates:
            try:
                self.vit = torch.hub.load(
                    repo,
                    entry,
                    pretrained=True,
                    verbose=False,
                )
                print(f"  [OK] Loaded {entry} from {repo}")
                break
            except Exception as exc:
                load_errors.append(f"{repo}:{entry} -> {exc}")

        if self.vit is None:
            raise RuntimeError(
                "Unable to load DINOv2 from torch.hub. Tried:\n  - "
                + "\n  - ".join(load_errors)
            )

        for p in self.vit.parameters():
            p.requires_grad = True

        self.head = BNNeck(
            in_dim=self.DINOV2_DIM,
            out_dim=embed_dim,
            use_proj=True,
        )
        self.classifier = nn.Linear(embed_dim, num_classes, bias=False)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized embedding for inference."""
        patch_embed = getattr(self.vit, 'patch_embed', None)
        proj = getattr(patch_embed, 'proj', None)
        kernel_size = getattr(proj, 'kernel_size', None)
        if isinstance(kernel_size, tuple) and len(kernel_size) == 2:
            patch_h, patch_w = kernel_size
            h, w = x.shape[-2], x.shape[-1]
            if (h % patch_h != 0) or (w % patch_w != 0):
                new_h = max(patch_h, (h // patch_h) * patch_h)
                new_w = max(patch_w, (w // patch_w) * patch_w)
                x = F.interpolate(
                    x,
                    size=(new_h, new_w),
                    mode='bilinear',
                    align_corners=False,
                )

        feat = self.vit(x)      # [B, 768] CLS token
        emb = self.head(feat)   # [B, embed_dim] L2-normalized
        return emb

    def forward_train(self, x: torch.Tensor):
        """Returns (embedding, logits) for CE + metric loss training."""
        emb = self.forward(x)
        logits = self.classifier(emb)
        return emb, logits


def build_model(
    backbone: str,
    num_classes: int,
    lora_rank: int = 8,
    osnet_weight_path: str | None = None,
) -> nn.Module:
    """
    Build and return the chosen backbone model.

    Args:
        backbone    : 'osnet' | 'dinov3' | 'dinov2'
        num_classes : number of training identities (751 for Market-1501 train)
        lora_rank         : LoRA rank (dinov3 only)
        osnet_weight_path : optional local checkpoint path for OSNet ReID
                    pretrained weights

    Returns:
        model : nn.Module with .forward(x) -> L2-norm embedding
                             and .forward_train(x) -> (emb, logits)
    """
    if backbone == 'osnet':
        print(f"\nBuilding OSNet baseline  (num_classes={num_classes})")
        model = OSNetBaseline(
            num_classes=num_classes,
            pretrained=(osnet_weight_path is None),
            weight_path=osnet_weight_path,
        )
    elif backbone == 'dinov3':
        print(f"\nBuilding DINOv3+LoRA  (num_classes={num_classes}, rank={lora_rank})")
        model = DINOv3LoRA(
            num_classes=num_classes,
            lora_rank=lora_rank,
            lora_alpha=lora_rank * 2.0,
        )
    elif backbone == 'dinov2':
        print(f"\nBuilding DINOv2 full fine-tuning  (num_classes={num_classes})")
        model = DINOv2FullFT(
            num_classes=num_classes,
            embed_dim=256,
        )
    else:
        raise ValueError(
            f"Unknown backbone: '{backbone}'. Choose 'osnet', 'dinov3', or 'dinov2'."
        )
    return model


def param_summary(model: nn.Module, name: str) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    pct       = 100.0 * trainable / total if total > 0 else 0.0

    print(f"\n  [{name}] Parameter summary")
    print(f"    Total      : {total:>12,}")
    print(f"    Trainable  : {trainable:>12,}  ({pct:.2f}%)")
    print(f"    Frozen     : {frozen:>12,}")
    return {'model': name, 'total': total, 'trainable': trainable,
            'frozen': frozen, 'trainable_pct': round(pct, 4)}


def main(args):
    print("\nStep 4: Feature Extractor")
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 751   # Market-1501 train identities

    summaries = []

    for backbone in ['osnet', 'dinov3', 'dinov2']:
        try:
            model = build_model(backbone, num_classes=num_classes,
                                lora_rank=args.lora_rank)
            model = model.to(device)
            model.eval()

            # Forward pass sanity check
            dummy = torch.randn(4, 3, 256, 128, device=device)
            with torch.no_grad():
                emb    = model(dummy)
                _, log = model.forward_train(dummy)

            print(f"\n  Sanity check [{backbone}]:")
            print(f"    Input  : {list(dummy.shape)}")
            print(f"    Embed  : {list(emb.shape)}   "
                  f"norm={emb.norm(dim=1).mean():.4f} (should be ~1.0)")
            print(f"    Logits : {list(log.shape)}")

            s = param_summary(model, backbone)
            summaries.append(s)
        except Exception as e:
            print(f"\n  [SKIP] {backbone}: {e}")
            summaries.append({'model': backbone, 'error': str(e)})

    # Save summary
    out = 'backbone_summary.txt'
    with open(out, 'w') as f:
        f.write("Backbone Parameter Summary\n")
        f.write("=" * 40 + "\n")
        for s in summaries:
            f.write(str(s) + "\n")
    print(f"\n[Saved] {out}")

    print("\n[Step 4 Complete]")
    print("Next step: python step5_training.py --backbone dinov2")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Step 4: Feature Extractor')
    parser.add_argument('--backbone',  type=str, default='dinov3',
                        choices=['osnet', 'dinov3', 'dinov2'])
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='LoRA rank for DINOv3 (default: 8)')
    args = parser.parse_args()
    main(args)