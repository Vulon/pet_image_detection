import datetime
import json
import os
import sys

import mlflow
import mlflow.pytorch
import torch
import yaml
from augmentations import (
    create_additional_mask_transform,
    create_augmentation_transform,
)
from dataset import SegmentationDataset
from metrics import create_compute_metrics_function
from model import get_feature_extractor, get_model_class
from transformers import Trainer, TrainingArguments


def create_loss_function():
    bce = torch.nn.BCEWithLogitsLoss()

    def loss_function(true_mask, predicted_mask):
        dice_loss = (
            1
            - (2 * true_mask * predicted_mask).sum()
            / (true_mask**2 + predicted_mask**2).sum()
        )
        return bce(true_mask, predicted_mask) + dice_loss

    return loss_function


def download_files(output_folder: str):
    check_flag = True
    files_to_check = [
        "train_images.h5py",
        "train_masks.h5py",
        "val_images.h5py",
        "val_masks.h5py",
        "test_images.h5py",
        "test_masks.h5py",
    ]
    for file_name in files_to_check:
        if not os.path.exists(os.path.join(output_folder, file_name)):
            check_flag = False
    if not check_flag:
        # TODO add loading for cloud training
        pass


def load_datasets(
    files_folder: str,
    feature_extractor,
    augmentation_transform,
    additional_mask_transform,
) -> tuple[SegmentationDataset, SegmentationDataset, SegmentationDataset]:
    train_dataset = SegmentationDataset(
        os.path.join(files_folder, "train_images.h5py"),
        os.path.join(files_folder, "train_masks.h5py"),
        feature_extractor,
        augmentation_transform,
        additional_mask_transform,
    )
    val_dataset = SegmentationDataset(
        os.path.join(files_folder, "val_images.h5py"),
        os.path.join(files_folder, "val_masks.h5py"),
        feature_extractor,
        None,
        additional_mask_transform,
    )
    test_dataset = SegmentationDataset(
        os.path.join(files_folder, "test_images.h5py"),
        os.path.join(files_folder, "test_masks.h5py"),
        feature_extractor,
        None,
        additional_mask_transform,
    )
    return train_dataset, val_dataset, test_dataset


def get_cli_train_mode_argument() -> str:
    import argparse

    parser = argparse.ArgumentParser(description="Train model script")
    parser.add_argument(
        "mode", type=str, default="local", help="Training mode: local or cloud"
    )

    args = parser.parse_args()
    return args.mode


if __name__ == "__main__":
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    train_mode = get_cli_train_mode_argument()
    if train_mode == "local":
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        # config_path = os.path.join(project_root, "params.yaml")
    elif train_mode == "cloud":
        project_root = os.path.dirname(__file__)
        # config_path = os.path.join(project_root, "params.yaml")
    else:
        raise Exception(
            f"Training mode argument {train_mode} must be 'local' or 'cloud'"
        )
    package_path = os.path.dirname(__file__)
    config_path = os.path.join(package_path, "params.yaml")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    cls = get_model_class(config["model"]["pretrained_name"])
    loss_function = create_loss_function()
    model = cls(
        config["model"]["pretrained_name"],
        loss_function,
        config["model"]["target_categories"],
    )

    feature_extractor = get_feature_extractor(config["model"]["pretrained_name"])
    augmentation_transform = create_augmentation_transform(config["augmentations"])
    mask_transform = create_additional_mask_transform(
        config["augmentations"]["mask_size"]
    )

    if train_mode == "local":
        files_folder = os.path.join(project_root, "data", "datasets")
    else:
        files_folder = config["training"]["container_data_folder"]
        download_files(files_folder)

    train_dataset, val_dataset, test_dataset = load_datasets(
        files_folder, feature_extractor, augmentation_transform, mask_transform
    )

    args = TrainingArguments(
        output_dir=os.path.join(project_root, "output"),
        evaluation_strategy="epoch",
        # eval_steps=config['training']['eval_steps'],
        per_device_train_batch_size=config["training"]["train_batch_size"],
        per_device_eval_batch_size=config["training"]["val_batch_size"],
        num_train_epochs=config["training"]["epochs"],
        seed=config["random_seed"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        save_steps=config["training"]["save_steps"],
        fp16=config["training"]["fp16"],
        # gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        # eval_accumulation_steps=config['training']['eval_accumulation_steps'],
        logging_strategy=config["training"]["logging_strategy"],
        logging_dir=os.path.join(
            project_root, config["training"]["tensorboard_logs_directory"]
        ),
        dataloader_drop_last=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=create_compute_metrics_function(),
    )
    trainer.save_model(os.path.join(project_root, config["score"]["model_path"]))
    model_name = f"{config['model']['pretrained_name']}_{config['version']}"
    run_name = f"model_name_{datetime.datetime.now().strftime('%Y_%m_%d')}"
    host = config["mlflow"]["host"]
    if host == "0.0.0.0":
        host = "127.0.0.1"

    mlflow.set_tracking_uri(f"http://{host}:{config['mlflow']['port']}")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
        project_root, "keys", config["mlflow"]["key_name"]
    )
    mlflow.set_experiment(model_name)

    with mlflow.start_run(run_name=run_name) as run:
        trainer.train(
            config["training"]["trainer_checkpoint"]
            if config["training"]["trainer_checkpoint"]
            else None
        )
        metrics = trainer.evaluate(test_dataset)
        mlflow.log_metrics(metrics)
        mlflow.log_params(config["model"])
        # mlflow.log_params(config["training"])

        # mlflow.pytorch.log_model(
        #     model, artifact_path=f"gs://{ config['mlflow']['artifact_storage_bucket'] }"
        # )

        mlflow.pytorch.log_model(model, artifact_path=model_name)
        model_uri = f"runs:/{run.info.run_id}/{model_name}"
        print("Trying to register model. URI", model_uri, "model name", model_name)
        mlflow.register_model(model_uri, model_name)

    with open(
        os.path.join(project_root, config["training"]["test_metrics_path"]), "w"
    ) as file:
        json.dump(metrics, file)
