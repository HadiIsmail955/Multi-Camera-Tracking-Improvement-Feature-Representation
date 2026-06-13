# src/models/__init__.py

from reid.models.factory import build_model
from reid.models.bnneck import BNNeck
from reid.models.osnet import OSNetBaseline
from reid.models.dinov2 import DINOv2Model
from reid.models.dinov3 import DINOv3Model

__all__ = [
    "build_model",
    "BNNeck",
    "OSNetBaseline",
    "DINOv2Model",
    "DINOv3Model",
]
