import dataclasses
import re
from dataclasses import dataclass

import yaml


@dataclass
class DatasetConfig:
    coco_categories_map: dict
    coco_train_annotations_url: str
    coco_archive_train_filename: str
    coco_archive_val_filename: str
    raw_files_folder: str
    dataset_files_folder: str

    train_fracture: float
    val_fracture: float
    test_fracture: float

    image_size: int


@dataclass
class ModelConfig:
    pretrained_name: str
    target_categories: list[int]


@dataclass
class TrainingConfig:
    downloaded_datasets_folder: str
    train_mode: str
    container_data_folder: str

    evaluation_strategy: str
    eval_steps: int
    train_batch_size: int
    val_batch_size: int
    epochs: int
    warmup_epochs: int
    learning_rate: float
    weight_decay: float
    save_steps: int
    fp16: bool
    gradient_accumulation_steps: int
    eval_accumulation_steps: int
    logging_strategy: str
    logging_steps: int
    tensorboard_logs_directory: str
    trainer_checkpoint: str

    test_metrics_path: str

    data_portion: float


@dataclass
class AugmentationsConfig:
    rotate_angle: int
    rotate_p: float
    flip_p: float
    rotate_sequence_p: float

    gauss_noise_var_limit: list[int]
    gauss_noise_p: float
    iso_color_shift: list[float]
    iso_intensity: list[float]
    iso_p: float
    noise_sequence_p: float

    brightness: float
    contrast: float
    saturation: float
    hue: float
    color_jitter_p: float
    CLAHE_p: float
    graphical_sequence_p: float

    downscale_scale_min: float
    downscale_scale_max: float
    downscale_p: float
    image_compression_p: float
    compression_sequence_p: float

    pixel_dropout_dropout_prob: float
    pixel_dropout_p: float
    grid_distortion_p: float
    optical_distortion_p: float
    perspective_p: float
    dropout_sequence_p: float

    ringing_overshoot_p: float
    sharpen_p: float
    unsharp_mask_p: float
    blur_sequence_p: float


@dataclass
class ScoringConfig:
    masks_names: dict
    model_path: str


@dataclass
class MlflowConfig:
    artifact_storage_bucket: str
    host: str
    port: str
    key_name: str


@dataclass
class BaseConfig:
    random_seed: int
    project_name: str
    version: str
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    augmentations: AugmentationsConfig
    score: ScoringConfig
    mlflow: MlflowConfig


def __dataclass_from_dict(klass, d):
    try:
        fieldtypes = {f.name: f.type for f in dataclasses.fields(klass)}
        return klass(**{f: __dataclass_from_dict(fieldtypes[f], d[f]) for f in d})
    except:
        return d  # Not a dataclass field


def get_config_from_dvc() -> BaseConfig:
    import dvc.api

    params = dvc.api.params_show()
    config = __dataclass_from_dict(BaseConfig, params)
    return config


def get_config_from_yaml(yaml_path: str) -> BaseConfig:
    with open(yaml_path, "r") as file:
        yaml_dict = yaml.safe_load(file)
        config = __dataclass_from_dict(BaseConfig, yaml_dict)
        return config


def get_terraform_variable(terraform_variables_file_path: str, variable_name: str):
    with open(terraform_variables_file_path, "r") as file:
        data = file.read()
    for line in data.split("variable"):
        regex_result = re.search('"\w+"', line)
        if regex_result is None:
            continue

        if regex_result.group(0) == f'"{variable_name}"':
            variable_body = line[regex_result.end() :]
            variable_default_value = re.search(
                'default\s+=\s+"\w+"', variable_body
            ).group(0)
            variable_default_value = (
                variable_default_value.split("=")[-1].replace('"', "").strip()
            )
            return variable_default_value
