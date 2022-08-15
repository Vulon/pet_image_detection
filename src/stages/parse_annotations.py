import json
import os
import sys

import numpy as np


def extract_needed_annotations(
    annotations_lines: list[dict], target_categories: list[int]
) -> dict[list[dict]]:
    parsed_annotations = dict()
    for item in annotations_lines:
        if (
            item["category_id"] not in target_categories
            or "segmentation" not in item.keys()
        ):
            continue
        annotations_list = parsed_annotations.get(item["image_id"], [])
        for polygon in item["segmentation"]:
            annotations_list.append(
                {"annotation": polygon, "category": item["category_id"]}
            )
        parsed_annotations[item["image_id"]] = annotations_list
    return parsed_annotations


def filter_images(
    image_data_lines: list[dict], parsed_annotations: dict[list[dict]]
) -> list[dict]:
    images_data = []
    for data_line in image_data_lines:
        if data_line["id"] not in parsed_annotations.keys():
            continue
        parsed_image_data = {
            "url": data_line["coco_url"],
            "width": data_line["width"],
            "height": data_line["height"],
            "id": data_line["id"],
            "annotations": parsed_annotations[data_line["id"]],
        }
        images_data.append(parsed_image_data)
    return images_data


def parse_annotations_file(
    input_file_path: str, target_categories: list[int]
) -> list[dict]:
    with open(input_file_path, "r") as file:
        data = json.load(file)
    annotations = extract_needed_annotations(data["annotations"], target_categories)
    parsed_images_data = filter_images(data["images"], annotations)
    return parsed_images_data


if __name__ == "__main__":
    project_root = os.environ["DVC_ROOT"]
    sys.path.append(project_root)
    from src.config import get_config_from_dvc

    config = get_config_from_dvc()
    raw_files_folder = os.path.join(project_root, config.dataset.raw_files_folder)

    parsed_train_annotations = parse_annotations_file(
        os.path.join(raw_files_folder, "train_anno.json"),
        config.dataset.coco_categories_map.keys(),
    )
    parsed_val_annotations = parse_annotations_file(
        os.path.join(raw_files_folder, "val_anno.json"),
        config.dataset.coco_categories_map.keys(),
    )
    output = parsed_train_annotations + parsed_val_annotations

    with open(
        os.path.join(raw_files_folder, "parsed_annotations.json"), "w"
    ) as output_file:
        json.dump(output, output_file)
