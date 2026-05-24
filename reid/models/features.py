from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import timm

from reid import consts
from reid import utils
from torchreid.reid.utils.feature_extractor import (
    FeatureExtractor as TorchReIDFeatureExtractor,
)


@dataclass(slots=True)
class ExtractorConfig:
    backbone: str
    device: str = "auto"
    batch_size: int = 32
    image_size: tuple[int, int] = (256, 128)
    checkpoint: str | None = None
    model_name: str | None = None
    normalize: bool = True
    verbose: bool = False


class FeatureExtractor:
    """Unified feature extractor for re-identification backbones.

    Supported backbone presets:
    - osnet-ain
    - osnet-ibn
    - transreid
    - dinov2-base, dinov2-small
    - dinov3-base, dinov3-small

    The preset names map to concrete model implementations, while
    ``model_name`` can override the default timm/torchreid backbone.
    """

    def __init__(self, config: ExtractorConfig):
        self.config = config
        self.device = utils.set_device(config.device)
        self.backbone_key = config.backbone.lower()
        self.model_name = config.model_name or consts.MODELS.get(self.backbone_key)
        if self.model_name is None:
            raise ValueError(
                f"Unsupported backbone '{config.backbone}'. "
                f"Available presets: {', '.join(sorted(consts.MODELS))}"
            )

        self._model = self._build_model()

    def _build_model(self):
        if self.model_name.startswith("osnet_"):
            if TorchReIDFeatureExtractor is None:
                raise RuntimeError(
                    "torchreid could not be imported. Install the project "
                    "dependencies, including gdown, then retry."
                )
            return TorchReIDFeatureExtractor(
                model_name=self.model_name,
                model_path=self.config.checkpoint or "",
                image_size=self.config.image_size,
                device=str(self.device),
                verbose=self.config.verbose,
            )

        model = timm.create_model(
            self.model_name,
            pretrained=self.config.checkpoint is None,
            num_classes=0,
            global_pool="avg",
        )
        model.eval()
        model.to(self.device)
        if self.config.checkpoint:
            self._load_checkpoint(model, self.config.checkpoint)
        return model

    @staticmethod
    def _load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        model.load_state_dict(checkpoint, strict=False)

    def extract_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from a tensor batch shaped ``(B, C, H, W)``."""

        if batch.dim() == 3:
            batch = batch.unsqueeze(0)
        if batch.dim() != 4:
            raise ValueError("Input batch must have shape (B, C, H, W) or (C, H, W)")

        batch = batch.to(self.device)
        with torch.no_grad():
            if isinstance(self._model, torch.nn.Module):
                autocast_context = get_autocast(self.device)
                with autocast_context:
                    embeddings = self._model(batch)
            else:
                embeddings = self._model(batch)

        embeddings = embeddings.float()
        if self.config.normalize:
            embeddings = F.normalize(embeddings, dim=1)
        return embeddings.cpu()

def get_autocast(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    elif device.type == "mps":
        return torch.autocast(device_type="mps", dtype=torch.float16)
    return torch.autocast(device_type="cpu", dtype=torch.float32, enabled=False)
