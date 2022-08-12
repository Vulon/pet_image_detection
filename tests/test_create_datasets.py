from unittest import TestCase
import numpy as np
from src.stages.create_datasets import draw_polygon_mask, split_data, build_all_masks, create_dataset_from_annotation_lines


class Test(TestCase):
    def test_split_data(self):
        array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        train, val, test = split_data(array, 0.7, 0.2, 0.1, 52)
        self.assertListEqual(train, [0, 1, 2, 3, 4, 5, 6])
        self.assertListEqual(val, [7, 8])
        self.assertListEqual(test, [9])

        train, val, test = split_data(array, 0.9, 0.1, 0, 52)
        self.assertListEqual(train, [0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.assertListEqual(val, [9])
        self.assertListEqual(test, [])

    def test_draw_polygon_mask(self):
        mask = np.zeros((16, 16))
        mask[1 : 15, 1: 15] = 1
        polygon = [ 1, 1, 1, 14, 14, 14, 14, 1 ]
        output = draw_polygon_mask(polygon, 16, 16)

        self.assertListEqual(mask.flatten().tolist(), output.flatten().tolist())
        mask = np.zeros((32, 32))
        mask[1 : 4, 1: 4] = 1
        polygon = [ 1.01, 0.99, 1.02, 3.02, 2.99, 3.001, 3, 1 ]
        output = draw_polygon_mask(polygon, 32, 32)

        self.assertListEqual(mask.flatten().tolist(), output.flatten().tolist())


        mask = np.zeros((8, 8))
        mask[1: 7, 1: 7] = 1
        polygon = [ 1, 0.99, 1, 6.02, 5.99, 6, 6.03, 1.004 ]
        output = draw_polygon_mask(polygon, 8, 8)

        self.assertListEqual(mask.flatten().tolist(), output.flatten().tolist())

        mask = np.zeros((8, 8))
        mask[1: 4, 1: 7 ] = 1
        polygon = [ 1, 1, 6, 1, 6, 3, 1, 3 ]
        output = draw_polygon_mask(polygon, 8, 8)

        self.assertListEqual(mask.flatten().tolist(), output.flatten().tolist())

    def test_build_all_masks(self):
        target_categories = [1, 2]
        mask_1 = np.zeros((8, 8))
        mask_1[1: 7, 1: 7] = 1
        mask_2 = np.zeros((8, 8))
        mask_2[1: 4, 1: 7 ] = 1
        annotations = [
            { "category" : 0, "annotation": [5, 5, 6, 6] },
            { "category" : 1, "annotation": [ 1, 0.99, 1, 6.02, 5.99, 6, 6.03, 1.004 ] },
            { "category" : 2, "annotation": [ 1, 1, 6, 1, 6, 3, 1, 3 ] },
        ]
        output = build_all_masks(annotations, target_categories, 8, 8)
        mask = np.stack([mask_1, mask_2], axis=2)

        self.assertListEqual( mask.flatten().tolist(), output.flatten().tolist() )

    def test_create_dataset_from_annotation_lines(self):




