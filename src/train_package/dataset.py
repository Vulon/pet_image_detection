import albumentations as A
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(
        self,
        images_file_path: str,
        masks_file_path: str,
        feature_extractor,
        transform: A.DualTransform,
        additional_segmentation_transform: A.DualTransform = None,
    ) -> None:
        super().__init__()
        self.images = h5py.File(images_file_path, "r")
        self.masks = h5py.File(masks_file_path, "r")
        self.keys = sorted([item for item in self.images.keys()])
        self.feature_extractor = feature_extractor
        self.transform = transform
        self.additional_segmentation_transform = additional_segmentation_transform

    def __del__(self):
        self.close()

    def __len__(self):
        return len(self.keys)

    def apply_feature_extractor(self, image: np.ndarray) -> torch.Tensor:
        value = self.feature_extractor(image=image, return_tensors="pt")
        image_tensor = value["image"]
        print("IMage tensor", image_tensor.shape)

        return image_tensor

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        key = self.keys[index]
        image = self.images[key][:]
        mask = self.masks[key][:]
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)

        else:
            transformed = {"image": image, "mask": mask}
        if self.additional_segmentation_transform is not None:
            transformed["mask"] = self.additional_segmentation_transform(
                image=transformed["image"], mask=transformed["mask"]
            )["mask"]

        transformed["image"] = self.apply_feature_extractor(transformed["image"])
        transformed["labels"] = transformed["mask"]

        del transformed["mask"]

        return transformed

    def close(self):
        self.images.close()
        self.masks.close()
