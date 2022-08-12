import os
import sys
import torch
import yaml
from model import get_model_class, get_feature_extractor
from dataset import SegmentationDataset
from augmentations import create_augmentation_transform, create_additional_mask_transform
from transformers import Trainer, TrainingArguments

def create_loss_function():
    return torch.nn.BCELoss()

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
        if not os.path.exists( os.path.join(output_folder, file_name) ):
            check_flag = False
    if not check_flag:
        pass
    

def load_datasets(files_folder : str, feature_extractor, augmentation_transform, additional_mask_transform)-> tuple[SegmentationDataset, SegmentationDataset, SegmentationDataset]:
    train_dataset = SegmentationDataset(
        os.path.join(files_folder, "train_images.h5py"),
        os.path.join(files_folder, "train_masks.h5py"),
        feature_extractor,
        augmentation_transform,
        additional_mask_transform
    )
    val_dataset = SegmentationDataset(
        os.path.join(files_folder, "val_images.h5py"),
        os.path.join(files_folder, "val_masks.h5py"),
        feature_extractor,
        None,
        additional_mask_transform
    )
    test_dataset = SegmentationDataset(
        os.path.join(files_folder, "test_images.h5py"),
        os.path.join(files_folder, "test_masks.h5py"),
        feature_extractor,
        None,
        additional_mask_transform
    )
    return train_dataset, val_dataset, test_dataset
    

if __name__ == "__main__":
    with open("./params.yaml", 'r') as file:
        config = yaml.safe_load(file)
    cls = get_model_class(config["model"]["pretrained_name"])
    loss_function = create_loss_function()
    model = cls(config["model"]["pretrained_name"], loss_function, config['model']['target_categories'])

    feature_extractor = get_feature_extractor(config["model"]["pretrained_name"])
    augmentation_transform = create_augmentation_transform(config['augmentations'])
    mask_transform = create_additional_mask_transform(config['augmentations']['mask_size'])
    
    if config["training"]['train_mode'] == "local":
        project_root = os.path.dirname(os.path.dirname(__file__))
        project_root = os.path.dirname(project_root)
        files_folder = os.path.join(project_root, "data", "datasets")
        

    elif config["training"]['train_mode'] == "cloud":
        project_root = os.path.dirname(__file__)
        files_folder = config["training"]["container_data_folder"]

        download_files(files_folder)

    train_dataset, val_dataset, test_dataset = load_datasets(
        files_folder,
        feature_extractor,
        augmentation_transform,
        mask_transform
    )
    
    args = TrainingArguments(
        output_dir=os.path.join(project_root, "output"),
        evaluation_strategy="epoch",
        # eval_steps=config['training']['eval_steps'],
        per_device_train_batch_size=config['training']['train_batch_size'],
        per_device_eval_batch_size=config['training']['val_batch_size'],
        num_train_epochs=config['training']['epochs'],
        seed=config['random_seed'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        save_steps=config['training']['save_steps'],
        fp16=config['training']['fp16'],
        # gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        # eval_accumulation_steps=config['training']['eval_accumulation_steps'],
        logging_strategy=config['training']['logging_strategy'],
        logging_dir=os.path.join(project_root, config['training']['tensorboard_logs_directory']),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # compute_metrics=create_compute_metrics_function(),
    )
    trainer.train(
        config['training']['trainer_checkpoint'] if config['training']['trainer_checkpoint'] else None
    )
    