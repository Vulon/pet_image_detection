import torch
from torch.utils.data import Dataset
import albumentations as A
import h5py


class SegmentationDataset(Dataset):
    def __init__(self, images_file_path: str, masks_file_path: str, transform: A.DualTransform) -> None:
        super().__init__()
        self.images = h5py.File(images_file_path, 'r')
        self.masks = h5py.File(masks_file_path, "r")
        self.transform = transform
    
    def __del__(self):
        self.images.close()
        self.masks.close()

    def __len__(self):
        return len(self.images.keys())
    
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = self.images[index]
        mask = self.masks[index]
        return self.transform(image=image, mask=mask)
