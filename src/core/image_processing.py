import albumentations as A


def create_scaling_transform(side_size: int, random_seed: int):
    transform = A.Sequential(
        [
            A.SmallestMaxSize(side_size, always_apply=True),
            A.CenterCrop(height=side_size, width=side_size, always_apply=True),
        ],
        p=1.0,
    )
    return transform
