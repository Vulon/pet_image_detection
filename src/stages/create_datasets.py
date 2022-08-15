import json
import os
import random
import shutil
import sys
import tempfile
import time

import cv2
import h5py
import numpy as np
import requests
import tqdm


def load_image(image_file_path: str):
    if os.path.exists(image_file_path):
        image = cv2.imread(image_file_path)
        return image
    else:
        return


def split_data(
    data_lines: list[dict],
    train_fracture: float,
    val_fracture: float,
    test_fracture: float,
    random_seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    train_count = int(
        train_fracture
        * len(data_lines)
        / (train_fracture + val_fracture + test_fracture)
    )
    val_count = int(
        val_fracture * len(data_lines) / (train_fracture + val_fracture + test_fracture)
    )
    test_count = int(
        test_fracture
        * len(data_lines)
        / (train_fracture + val_fracture + test_fracture)
    )

    # random.shuffle(data_lines, random=random_seed)
    print("Train", train_count, "Val", val_count)
    train_lines = data_lines[:train_count]
    val_lines = data_lines[train_count : train_count + val_count]
    test_lines = data_lines[train_count + val_count :]
    return train_lines, val_lines, test_lines


def draw_polygon_mask(polygon: list[int], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width))

    polygon = np.array(np.round(polygon), dtype=int).reshape((-1, 2))
    mask = cv2.fillPoly(mask, pts=[polygon], color=1)
    return mask


def build_all_masks(
    annotations: list[dict], target_categories: list[int], width: int, height: int
) -> np.ndarray:
    output_masks = np.zeros((height, width, len(target_categories)))
    for i, category_id in enumerate(target_categories):
        mask = np.zeros((height, width))
        for anno in annotations:
            if anno["category"] == category_id:
                mask = mask + draw_polygon_mask(
                    anno["annotation"], width=width, height=height
                )
        output_masks[:, :, i] = mask
    return output_masks.astype(np.uint8)


def generate_image_name(url: str, id: int):
    extension = str(os.path.basename(url)).split(".")[-1]
    return f"{id}.{extension}"


def create_dataset_from_annotation_lines(
    image_data_lines: list[dict],
    image_dataset_filename: str,
    masks_dataset_filename: str,
    target_categories: list[int],
    scaling_sequence,
):
    folder = tempfile.mkdtemp()
    try:
        urls = [item["url"] for item in image_data_lines]
        image_ids = [item["id"] for item in image_data_lines]
        image_filenames = [
            generate_image_name(image_url, image_id)
            for image_url, image_id in zip(urls, image_ids)
        ]
        async_download_files(urls, image_filenames, folder, semaphore_value=50)

        with h5py.File(image_dataset_filename, "w") as image_file:
            with h5py.File(masks_dataset_filename, "w") as mask_file:
                for line in tqdm.tqdm(image_data_lines):
                    image_id = str(line["id"])
                    try:
                        image_path = generate_image_name(line["url"], image_id)
                        image = load_image(os.path.join(folder, image_path))
                        if image is None:
                            print("Image", line["url"], "was not loaded")
                            continue
                        masks = build_all_masks(
                            line["annotations"],
                            target_categories,
                            line["width"],
                            line["height"],
                        )
                        transformed = scaling_sequence(image=image, mask=masks)

                        image_file.create_dataset(
                            image_id,
                            data=transformed["image"],
                            dtype="uint8",
                            compression="gzip",
                        )
                        mask_file.create_dataset(
                            image_id,
                            data=transformed["mask"],
                            dtype="uint8",
                            compression="gzip",
                        )
                        mask_file[image_id].attrs["mask_points"] = transformed[
                            "mask"
                        ].sum()
                        mask_file[image_id].attrs["original_points"] = masks.sum()

                    except Exception as e:
                        print("Exception. image url: ", line["url"], e)
    finally:
        shutil.rmtree(folder)


if __name__ == "__main__":
    project_root = os.environ["DVC_ROOT"]
    sys.path.append(project_root)
    from src.config import get_config_from_dvc
    from src.core.download_utlis import async_download_files
    from src.core.image_processing import create_scaling_transform

    config = get_config_from_dvc()

    with open(
        os.path.join(
            project_root, config.dataset.raw_files_folder, "parsed_annotations.json"
        ),
        "r",
    ) as input_file:
        full_data_lines = json.load(input_file)

    train_lines, val_lines, test_lines = split_data(
        full_data_lines,
        config.dataset.train_fracture,
        config.dataset.val_fracture,
        config.dataset.test_fracture,
        config.random_seed,
    )

    target_categories = sorted(config.dataset.coco_categories_map.keys())

    datasets_folder = os.path.join(project_root, config.dataset.dataset_files_folder)

    scaling_sequence = create_scaling_transform(
        config.dataset.image_size, config.random_seed
    )

    print("Started val images download")
    start = time.time()
    create_dataset_from_annotation_lines(
        val_lines,
        os.path.join(datasets_folder, "val_images.h5py"),
        os.path.join(datasets_folder, "val_masks.h5py"),
        target_categories,
        scaling_sequence,
    )
    print("Finished val images download in", time.time() - start)
    start = time.time()
    print("Started test images download")
    create_dataset_from_annotation_lines(
        test_lines,
        os.path.join(datasets_folder, "test_images.h5py"),
        os.path.join(datasets_folder, "test_masks.h5py"),
        target_categories,
        scaling_sequence,
    )
    print("Finished test images download in", time.time() - start)
    start = time.time()
    print("Started train images download")
    create_dataset_from_annotation_lines(
        train_lines,
        os.path.join(datasets_folder, "train_images.h5py"),
        os.path.join(datasets_folder, "train_masks.h5py"),
        target_categories,
        scaling_sequence,
    )
    print("Finished train images download in", time.time() - start)

    print("Download finished")
