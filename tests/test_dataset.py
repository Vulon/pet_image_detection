import os
import shutil
import tempfile
from unittest import TestCase

import albumentations as A
import h5py
import numpy as np

from src.train_package.dataset import SegmentationDataset


class TestSegmentationDataset(TestCase):
    def test_dataset(self):
        folder = tempfile.mkdtemp()

        try:
            image_file = h5py.File(os.path.join(folder, "image.h5py"), "w")
            mask_file = h5py.File(os.path.join(folder, "mask.h5py"), "w")
            image = np.zeros((16, 16))
            image[:8, :8] = 1
            transform = A.CenterCrop(16, 16)
            mask = np.zeros((16, 16))
            mask[:8, :8] = 1
            mask[8:, 8:] = 1
            image_file["a"] = image
            mask_file["a"] = mask
            image_file.close()
            mask_file.close()

            dataset = SegmentationDataset(
                os.path.join(folder, "image.h5py"),
                os.path.join(folder, "mask.h5py"),
                transform,
                None,
            )
            output = dataset[0]
            self.assertIn("image", output.keys())
            self.assertIn("labels", output.keys())
            self.assertListEqual(
                image.flatten().tolist(), output["image"].flatten().tolist()
            )
            self.assertListEqual(
                mask.flatten().tolist(), output["labels"].flatten().tolist()
            )
            dataset.close()

        finally:
            shutil.rmtree(folder)
