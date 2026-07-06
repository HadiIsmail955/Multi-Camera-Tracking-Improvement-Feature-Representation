import torch.nn as nn

from reid.models.osnet import OSNetBaseline
from reid.models.dinov3 import DINOv3Model
from reid.models.dinov2 import DINOv2Model


def build_model(
    backbone: str,
    num_classes: int,
    lora_rank: int = 8,
    use_lora: bool = True,
    osnet_weight_path: str | None = None,
    dino_feature_mode: str = "cls",
    dino_model_name: str = "dinov2_vitb14",
) -> nn.Module:
    if backbone in ["osnet", "osnet_ain"]:
        print(f"\nBuilding {backbone.upper()} baseline  (num_classes={num_classes})")
        
        torchreid_name = "osnet_ain_x1_0" if backbone == "osnet_ain" else "osnet_x1_0"
        
        return OSNetBaseline(
            num_classes=num_classes,
            pretrained=(osnet_weight_path is None),
            weight_path=osnet_weight_path,
            model_name=torchreid_name  
        )

    if backbone == "dinov3":
        mode = f"LoRA rank={lora_rank}" if use_lora else "full fine-tune"
        print(f"\nBuilding DINOv3  (num_classes={num_classes}, {mode})")
        return DINOv3Model(
            num_classes=num_classes,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_rank * 2.0,
        )

    if backbone == "dinov2":
        mode = f"LoRA rank={lora_rank}" if use_lora else "full fine-tune"
        print(
            f"\nBuilding DINOv2  "
            f"(num_classes={num_classes}, {mode}, "
            f"model_name={dino_model_name}, feature_mode={dino_feature_mode})"
        )
        return DINOv2Model(
            num_classes=num_classes,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_rank * 2.0,
            embed_dim=256,
            feature_mode=dino_feature_mode,
            model_name=dino_model_name,
        )

    raise ValueError(
        f"Unknown backbone: {backbone!r}. Choose 'osnet', 'dinov3', or 'dinov2'."
    )
