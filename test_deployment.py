import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get credentials from environment variables
SCORING_URI = os.getenv('AZURE_ML_ENDPOINT', 'https://sentiment-endpoint-48a4f2d5.centralindia.inference.ml.azure.com/score')
API_KEY = os.getenv('AZURE_ML_API_KEY', '')

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# Test data
test_cases = [
    "This is absolutely amazing! Best product ever!",
    "Terrible experience. Worst purchase I've made.",
    "It's okay, nothing special really."
]

for text in test_cases:
    data = {"text": text}
    
    response = requests.post(
        SCORING_URI,
        headers=headers,
        json=data
    )
    
    print(f"\nText: {text}")
    print(f"Response: {response.json()}")
