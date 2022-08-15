import tempfile
import zipfile

from src.core.download_utlis import CloudClient


def archive_files(input_file_paths: list[str], archive_output_file: str):
    with zipfile.ZipFile(archive_output_file, "w") as archive:
        for path in input_file_paths:
            archive.write(path)
