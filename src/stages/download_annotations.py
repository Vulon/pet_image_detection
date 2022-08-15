import json
import os
import shutil
import sys
import tempfile
import zipfile

import requests


def extract_file(archive: zipfile.ZipFile, archive_filename: str, output_filename: str):
    data = archive.read(archive_filename)

    with open(output_filename, "w") as json_file:
        json.dump(json.loads(data.decode()), json_file)


if __name__ == "__main__":
    project_root = os.environ["DVC_ROOT"]
    sys.path.append(project_root)
    from src.config import get_config_from_dvc
    from src.core.download_utlis import download_archive

    config = get_config_from_dvc()

    temp_folder = tempfile.mkdtemp()
    archive = None
    try:
        datasets_folder = os.path.join(project_root, config.dataset.raw_files_folder)
        os.makedirs(datasets_folder, exist_ok=True)
        archive = download_archive(
            config.dataset.coco_train_annotations_url, temp_folder
        )
        extract_file(
            archive,
            config.dataset.coco_archive_train_filename,
            os.path.join(datasets_folder, "train_anno.json"),
        )
        extract_file(
            archive,
            config.dataset.coco_archive_val_filename,
            os.path.join(datasets_folder, "val_anno.json"),
        )
    finally:
        if archive is not None:
            archive.close()
        shutil.rmtree(temp_folder)
