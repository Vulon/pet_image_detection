import asyncio
import os
import zipfile

import aiofiles
import aiohttp
import requests
from google.cloud import storage


def download_archive(url: str, temp_folder: str):
    response = requests.get(url)
    with open(os.path.join(temp_folder, "archive.zip"), "wb") as file:
        file.write(response.content)

    archive = zipfile.ZipFile(os.path.join(temp_folder, "archive.zip"), "r")
    return archive


def async_download_files(
    urls: list[str], file_names: list[str], output_folder: str, semaphore_value: int = 5
):

    sema = asyncio.BoundedSemaphore(semaphore_value)

    async def fetch_file(url: str, output_basename: str):

        async with sema, aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.read()
                else:
                    return

        async with aiofiles.open(
            os.path.join(output_folder, output_basename), "wb"
        ) as outfile:
            await outfile.write(data)

    loop = asyncio.new_event_loop()
    tasks = [
        loop.create_task(fetch_file(url, name)) for url, name in zip(urls, file_names)
    ]
    print("Started downloading", len(tasks), "files")
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()


class CloudClient:
    def __init__(self, project_name: str, default_upload_bucket: str = None) -> None:
        self.default_upload_bucket = default_upload_bucket
        self.storage_client = storage.Client(project_name)

    def upload_file(
        self, input_file_path: str, output_cloud_path: str, bucket_name: str = None
    ):
        if not bucket_name:
            bucket_name = self.default_upload_bucket
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(output_cloud_path)
        blob.upload_from_file(input_file_path)

    def __del__(self):
        self.storage_client.close()
