from typing import Final

# 384 x 192 or 256 x 128 
# The original OSNet paper uses 256 x 128, while the DINOv2 paper uses 384 x 192.
IMAGE_HEIGHT: Final[int] = 256
IMAGE_WIDTH: Final[int] = 128

MEAN: Final[list[float]] = [0.485, 0.456, 0.406]
STD: Final[list[float]] = [0.229, 0.224, 0.225]
