import requests
import os
import logging
from google.cloud import storage
from google.oauth2 import service_account
from hashlib import sha512
from preprocessing import load_and_save_masks_and_captions
import shutil
import mimetypes
from zipfile import ZipFile

def upload_file_to_presigned_url(file_path, signed_url):    
    with open(file_path, 'rb') as file:
        headers = {'Content-Type': 'application/octet-stream'}
        response = requests.put(signed_url, data=file, headers=headers)

    if response.status_code == 200:
        print("File uploaded successfully.")
        return True
    else:
        print(f"File upload failed: {response.text}")
        return False

def generate_signed_put_url(bucket_name, blob_name, expiration):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    url = blob.generate_signed_url(
        version="v4",
        expiration=expiration,
        method="PUT",
        content_type="application/octet-stream",
    )
    return url

def generate_download_signed_url_v4(bucket_name, blob_name, expiration_time):

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    url = blob.generate_signed_url(
        version="v4",
        # This URL is valid for 15 minutes
        expiration=expiration_time,
        # Allow GET requests using this URL.
        method="GET",
    )
    return url

def url_local_fn(url):
    return sha512(url.encode()).hexdigest() + ".zip"


def download_file(url):
    fn = url_local_fn(url)
    if not os.path.exists(fn):
        print("Downloading instance data... from", url)
        # stream chunks of the file to disk
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(fn, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    else:
        print("Using disk cache...")

    return fn

def clean_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def clean_directories(paths):
    for path in paths:
        clean_directory(path)


def random_seed():
    return int.from_bytes(os.urandom(2), "big")


def extract_zip_and_flatten(zip_path, output_path):
    # extract zip contents, flattening any paths present within it
    with ZipFile(str(zip_path), "r") as zip_ref:
        for zip_info in zip_ref.infolist():
            if zip_info.filename[-1] == "/" or zip_info.filename.startswith(
                "__MACOSX"
            ):
                continue
            mt = mimetypes.guess_type(zip_info.filename)
            if mt and mt[0] and (mt[0].startswith("image/") or mt[0].startswith("text/")):
                zip_info.filename = os.path.basename(zip_info.filename)
                zip_ref.extract(zip_info, output_path)

def download_data():
    instance_data_url = os.getenv("DATA_URL")
    instance_data_folder = os.getenv("INSTANCE_DIR")
    output_dir = os.getenv("OUTPUT_DIR", "checkpoints")
    resolution = int(os.getenv("RESOLUTION", 512))
    enable_preprocessing = int(os.getenv("PREPROCESSING", "0")) == 1
    use_face = int(os.getenv("FACE", "0")) == 1
    clean_directories([instance_data_folder, output_dir])
    instance_data=download_file(instance_data_url)
    extract_zip_and_flatten(instance_data, instance_data_folder)
    if enable_preprocessing:
        load_and_save_masks_and_captions(instance_data_folder, instance_data_folder+"/preprocessing", target_size=resolution, use_face_detection_instead=use_face)


    

if __name__ == '__main__':
    download_data()