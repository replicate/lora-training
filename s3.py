import boto3
from botocore.exceptions import NoCredentialsError
import requests

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

def main():
    # Replace these values with your own
    bucket_name = 'data-training'
    file_key = 'huy/data/test.zip'
    expiration_time = 31*24*3600  # Time in seconds; 3600 seconds = 1 hour

    # Generate the pre-signed URL for file upload
    presigned_url = create_presigned_upload_url(bucket_name, file_key, expiration_time)

    if presigned_url:
        print("\nPre-signed POST URL for uploading a file:")
        print(presigned_url)
    else:
        print("Failed to generate pre-signed URL.")
    
    upload_file_to_s3_presigned_url('hanabunny_school.zip', presigned_url['url'], presigned_url['fields'])


if __name__ == '__main__':
    main()