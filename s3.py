import boto3
from botocore.exceptions import NoCredentialsError, BotoCoreError, ClientError
import requests
import os
import logging

def create_presigned_upload_url(bucket_name, key, expiration=3600):
    s3_client = boto3.client('s3')
    try:
        response = s3_client.generate_presigned_post(
            Bucket=bucket_name,
            Key=key,
            ExpiresIn=expiration
        )
    except NoCredentialsError as e:
        print(e)
        return None

    return response

def upload_file_to_s3_presigned_url(file_path, url, fields):    
    with open(file_path, 'rb') as file:
        files = {'file': (file.name, file)}
        response = requests.post(url, data=fields, files=files)
    
    if response.status_code == 204:
        return True
    else:
        print("File upload failed:", response.text)
        return False

def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

def create_s3_bucket(bucket_name, region=None):
    try:
        if region is None:
            s3_client = boto3.client('s3')
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client = boto3.client('s3', region_name=region)
            location = {'LocationConstraint': region}
            s3_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration=location)
            
        print(f"Bucket {bucket_name} created successfully.")
    except (BotoCoreError, ClientError) as e:
        print(f"Error creating bucket {bucket_name}: {e}")

def main():
    # Replace these values with your own
    bucket_name = 'data'
    file_key = 'huy/data/test.zip'
    create_s3_bucket('data')
    create_s3_bucket('model')
    expiration_time = 31*24*3600  # Time in seconds; 3600 seconds = 1 hour

    # Generate the pre-signed URL for file upload
    presigned_url = create_presigned_upload_url(bucket_name, file_key, expiration_time)

    if presigned_url:
        print("\nPre-signed POST URL for uploading a file:")
        print(presigned_url)
    else:
        print("Failed to generate pre-signed URL.")
    
    # upload_file_to_s3_presigned_url('hanabunny_school.zip', presigned_url['url'], presigned_url['fields'])
    upload_file('hanabunny_school.zip', 'data', 'hanabunny_school.zip')


if __name__ == '__main__':
    main()