import torch
from torchvision import transforms
import torchvision.transforms.functional as TF


class ResizeWithAspectPad:
    def __init__(self, size=224, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, image):
        w, h = image.size

        scale = self.size / max(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        image = TF.resize(image, (new_h, new_w))

        pad_left = (self.size - new_w) // 2
        pad_top = (self.size - new_h) // 2
        pad_right = self.size - new_w - pad_left
        pad_bottom = self.size - new_h - pad_top

        image = TF.pad(
            image,
            padding=[pad_left, pad_top, pad_right, pad_bottom],
            fill=self.fill
        )

        return image