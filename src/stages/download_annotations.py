import json
import requests
import shutil
import tempfile
import os
import zipfile
import sys


def download_archive(url: str):
    response = requests.get(url)
    with open( os.path.join(temp_folder, "archive.zip"), "wb" ) as file:
        file.write(response.content)

    archive = zipfile.ZipFile( os.path.join(temp_folder, "archive.zip") , 'r')
    return archive

def extract_file(archive: zipfile.ZipFile, archive_filename: str, output_filename: str):
    data = archive.read(archive_filename)
    
    with open(output_filename, "w") as json_file:
        json.dump(json.loads(data.decode()), json_file)


if __name__ == "__main__":
    project_root = os.environ["DVC_ROOT"]
    sys.path.append(project_root)
    from src.config import get_config_from_dvc
    

    config = get_config_from_dvc()

    temp_folder = tempfile.mkdtemp()
    archive = None
    try:
        archive = download_archive(config.dataset.coco_train_annotations_url)
        extract_file(archive, config.dataset.coco_archive_train_filename, os.path.join(config.dataset.raw_files_folder, "train_anno.json"))
        extract_file(archive, config.dataset.coco_archive_val_filename, os.path.join(config.dataset.raw_files_folder, "val_anno.json"))
    finally:
        if archive is not None:
            archive.close()
        shutil.rmtree(temp_folder)



