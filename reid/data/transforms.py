from torchvision import transforms

from reid.consts import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MEAN,
    STD,
)


def build_train_transform(
    image_height: int | None = None, 
    image_width: int | None = None, 
    random_erasing: bool = True,
) -> transforms.Compose:

    if image_height is None:
        image_height = IMAGE_HEIGHT
    
    if image_width is None:
        image_width = IMAGE_WIDTH

    _transforms = [
        transforms.Resize((image_height, image_width)),
        transforms.Pad(10),
        transforms.RandomCrop((image_height, image_width)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        ),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
    if random_erasing: 
        _transforms.append(        
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),
        )

    return transforms.Compose(_transforms)


def build_eval_transform(   
    image_height: int | None = None, 
    image_width: int | None = None, 
) -> transforms.Compose:

    if image_height is None:
        image_height = IMAGE_HEIGHT
    
    if image_width is None:
        image_width = IMAGE_WIDTH
        
    return transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])