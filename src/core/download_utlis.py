import os
import requests
import zipfile
import asyncio
import aiohttp
import aiofiles


def download_archive(url: str, temp_folder: str):
    response = requests.get(url)
    with open( os.path.join(temp_folder, "archive.zip"), "wb" ) as file:
        file.write(response.content)

    archive = zipfile.ZipFile( os.path.join(temp_folder, "archive.zip") , 'r')
    return archive



def async_download_files(urls : list[str], file_names: list[str], output_folder: str, semaphore_value: int = 5):
    
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
    tasks = [loop.create_task(fetch_file(url, name)) for url, name in zip(urls, file_names)]
    print("Started downloading", len(tasks), "files")
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()