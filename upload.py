import requests
import os
import logging
from google.cloud import storage
from google.oauth2 import service_account

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


def main():
    # Replace these values with your own
    data_bucket_name = '<>'
    model_bucket_name = '<>'
    file_key = 'data/test.zip'
    expiration_time = 7*24*3600  # Time in seconds; 3600 seconds = 1 hour
    signed_url = generate_signed_put_url(data_bucket_name, file_key, expiration_time)
    print("\nSigned PUT URL for uploading a file:")
    print(signed_url)
    upload_file_to_presigned_url('hanabunny_school.zip', signed_url)


if __name__ == '__main__':
    main()