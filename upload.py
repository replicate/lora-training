import requests
import os
import logging
from google.cloud import storage
from google.oauth2 import service_account
from hashlib import sha512

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

def main():
    # Replace these values with your own
    model_bucket_name = '<>'
    file_key = 'model/xxxx.safetensors'
    expiration_time = 7*24*3600  # Time in seconds; 3600 seconds = 1 hour
    up_signed_url = generate_signed_put_url(model_bucket_name, file_key, expiration_time)
    print("\nSigned PUT URL for uploading a file:")
    print(up_signed_url)
    
    
    signed_url = generate_download_signed_url_v4(model_bucket_name, file_key, expiration_time)
    print("\nSigned GET URL for uploading a file:")
    print(signed_url)
    # instance_data=download_file("")
    # print(instance_data)
    # upload_file_to_presigned_url("huydeptrai.safetensors",up_signed_url)


if __name__ == '__main__':
    main()