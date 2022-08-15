import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def create_augmentation_transform(augmentation_config: dict):
    transform = A.Sequential([A.HorizontalFlip(p=0.5)], p=1.0)
    return transform


def create_additional_mask_transform(target_mask_size: int):
    transform = A.Sequential(
        [A.Resize(target_mask_size, target_mask_size), ToTensorV2(transpose_mask=True)],
        p=1.0,
    )
    return transform
