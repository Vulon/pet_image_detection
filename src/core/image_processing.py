import imgaug as ia
import imgaug.augmenters as iaa
import albumentations as A



def create_scaling_sequence(side_size : int, random_seed: int):
    crop_sequence = iaa.Sequential(
        [
            iaa.Resize(
                {"shorter-side": side_size, "longer-side": "keep-aspect-ratio"},
                random_state=random_seed,
            ),
            iaa.CenterCropToSquare(random_state=random_seed),
        ],
        random_state=random_seed,
    )
    return crop_sequence
    

def create_scaling_transform(side_size : int, random_seed: int):
    transform = A.Sequential([
        A.SmallestMaxSize(side_size, always_apply=True),
        A.CenterCrop(height=side_size, width=side_size, always_apply=True)
    ], p=1.0)
    return transform

