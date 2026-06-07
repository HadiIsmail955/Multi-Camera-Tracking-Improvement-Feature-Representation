import torchvision.transforms as T
from reid.utils.ResizeWithAspectPad import ResizeWithAspectPad

class ReIDTransform:
    def __init__(self, backbone="dinov2", img_size=224, train=True):
        if backbone.lower() in ["dino", "dinov2", "dino3"]:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        elif backbone.lower() == "clip":
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if train:
            self.transform = T.Compose([
                ResizeWithAspectPad(img_size),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(0.2, 0.2, 0.2, 0.05),
                T.ToTensor(),
                T.RandomErasing(
                    p=0.25,
                    scale=(0.02, 0.20),
                    ratio=(0.3, 3.3),
                    value="random",
                ),
                T.Normalize(mean=mean, std=std),
            ])
        else:
            self.transform = T.Compose([
                ResizeWithAspectPad(img_size),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])

    def __call__(self, image):
        return self.transform(image)