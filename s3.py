import boto3
from botocore.exceptions import NoCredentialsError
import requests

def create_presigned_upload_url(bucket_name, key, expiration=3600):
    """
    Parameters
    ----------
    bucket_name : str
        Name of the S3 bucket
    key : str
        Path to the file in S3 including the file name
    expiration : int, optional
        Time in seconds for the pre-signed URL to be valid, defaults to 1 hour

    Returns
    -------
    str
        Pre-signed URL for uploading a file to the specified bucket and key
    """
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
    """
    Upload a file to S3 using presigned_post_data generated via AWS S3.generate_presigned_post()
    
    Parameters
    ----------
    file_path : str
        Path to the local file to be uploaded
    presigned_post_data : dict
        A dictionary containing pre-signed POST data generated by AWS S3.generate_presigned_post()
        
    Returns
    -------
    bool: True if the file is uploaded successfully, False otherwise
    """
    form_data = fields
    
    with open(file_path, 'rb') as file:
        files = {'file': (file.name, file)}
        response = requests.post(url, data=form_data, files=files)
    
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