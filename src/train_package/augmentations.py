import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def create_augmentation_transform(augmentation_config: dict):
    transform = A.Sequential([A.HorizontalFlip(p=0.5)], p=1.0)
    return transform


def _create_rotate_sequence(augmentations_config: dict):
    augmentations_config = {
        "rotate_angle": 45,
        "rotate_p": 0.05,
        "flip_p": 0.05,
        "rotate_sequence_p": 1.0,
    }
    transform = A.Sequential(
        [
            A.Rotate(
                limit=augmentations_config["rotate_angle"],
                p=augmentations_config["rotate_p"],
            ),
            A.HorizontalFlip(p=augmentations_config["flip_p"]),
        ],
        p=augmentations_config["rotate_sequence_p"],
    )
    return transform


def _create_noise_sequence(augmentations_config: dict):
    augmentations_config = {
        "gauss_noise_var_limit": [64, 512],
        "gauss_noise_p": 0.1,
        "iso_color_shift": [0.01, 0.1],
        "iso_intensity": [0.1, 0.5],
        "iso_p": 0.1,
        "noise_sequence_p": 1.0,
    }
    transform = A.Sequential(
        [
            A.GaussNoise(
                var_limit=augmentations_config["gauss_noise_var_limit"],
                p=augmentations_config["gauss_noise_p"],
            ),
            A.ISONoise(
                color_shift=augmentations_config["iso_color_shift"],
                intensity=augmentations_config["iso_intensity"],
                p=augmentations_config["iso_p"],
            ),
        ],
        p=augmentations_config["noise_sequence_p"],
    )
    return transform


def _create_graphical_sequence(augmentations_config: dict):
    augmentations_config = {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.2,
        "color_jitter_p": 0.4,
        "CLAHE_p": 0.1,
        "graphical_sequence_p": 1.0,
    }
    transform = A.Sequential(
        [
            A.ColorJitter(
                brightness=augmentations_config["brightness"],
                contrast=augmentations_config["contrast"],
                saturation=augmentations_config["saturation"],
                hue=augmentations_config["hue"],
                p=augmentations_config["color_jitter_p"],
            ),
            A.CLAHE(clip_limit=4.0, p=augmentations_config["CLAHE_p"]),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        ],
        p=augmentations_config["graphical_sequence_p"],
    )
    return transform


def _create_compression_sequence(augmentations_config: dict):
    augmentations_config = {
        "downscale_scale_min": 0.6,
        "downscale_scale_max": 0.6,
        "downscale_p": 0.1,
        "image_compression_p": 0.1,
        "compression_sequence_p": 1.0,
    }
    transform = A.Sequential(
        [
            A.Downscale(
                scale_min=augmentations_config["downscale_scale_min"],
                scale_max=augmentations_config["downscale_scale_max"],
                p=augmentations_config["downscale_p"],
            ),
            A.ImageCompression(
                quality_lower=80,
                quality_upper=100,
                p=augmentations_config["image_compression_p"],
            ),
        ],
        p=augmentations_config["compression_sequence_p"],
    )
    return transform


def _create_dropout_sequence(augmentations_config: dict):
    augmentations_config = {
        "pixel_dropout_dropout_prob": 0.01,
        "pixel_dropout_p": 0.1,
        "grid_distortion_p": 0.1,
        "optical_distortion_p": 0.2,
        "perspective_p": 0.1,
        "dropout_sequence_p": 1.0,
    }
    transform = A.Sequential(
        [
            A.PixelDropout(
                dropout_prob=augmentations_config["pixel_dropout_dropout_prob"],
                p=augmentations_config["pixel_dropout_p"],
            ),
            A.GridDistortion(p=augmentations_config["grid_distortion_p"]),
            A.OpticalDistortion(
                p=augmentations_config["optical_distortion_p"],
                distort_limit=0.2,
                shift_limit=0.2,
            ),
            A.Perspective(scale=[0.05, 0.1], p=augmentations_config["perspective_p"]),
        ],
        p=augmentations_config["dropout_sequence_p"],
    )
    return transform


def _create_blur_sequence(augmentations_config: dict):
    augmentations_config = {
        "ringing_overshoot_p": 0.2,
        "sharpen_p": 0.05,
        "unsharp_mask_p": 0.05,
        "blur_sequence_p": 1.0,
    }
    transform = A.Sequential(
        [
            A.RingingOvershoot(
                blur_limit=(7, 11), p=augmentations_config["ringing_overshoot_p"]
            ),
            A.Sharpen(p=augmentations_config["sharpen_p"]),
            A.UnsharpMask(p=augmentations_config["unsharp_mask_p"], blur_limit=[5, 13]),
        ],
        p=augmentations_config["blur_sequence_p"],
    )
    return transform


def create_train_sequence(augmentations_config: dict):

    transform = A.Sequential(
        [
            _create_rotate_sequence(augmentations_config),
            _create_noise_sequence(augmentations_config),
            _create_graphical_sequence(augmentations_config),
            _create_compression_sequence(augmentations_config),
            _create_dropout_sequence(augmentations_config),
            _create_blur_sequence(augmentations_config),
        ]
    )
    return transform


def create_additional_mask_transform(target_mask_size: int):
    transform = A.Sequential(
        [A.Resize(target_mask_size, target_mask_size), ToTensorV2(transpose_mask=True)],
        p=1.0,
    )
    return transform
