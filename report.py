import os
import json
import requests

def send_training_report(data):
    url = os.environ["REPORT_URL"]
    bearer_token = os.environ["REPORT_TOKEN"]
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(f"\nResponse status: {response.status_code}")
    print(f"\nResponse body: {response.text}")
    return response