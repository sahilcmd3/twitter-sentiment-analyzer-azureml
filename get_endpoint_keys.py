"""
Get Azure ML endpoint keys
"""
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
SUBSCRIPTION_ID = os.getenv('AZURE_SUBSCRIPTION_ID')
RESOURCE_GROUP = os.getenv('AZURE_RESOURCE_GROUP')
WORKSPACE_NAME = os.getenv('AZURE_WORKSPACE_NAME')
ENDPOINT_NAME = os.getenv('AZURE_ENDPOINT_NAME')

# Authenticate
credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME
)

# Get endpoint details
endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
print(f"Endpoint: {endpoint.name}")
print(f"Scoring URI: {endpoint.scoring_uri}")

# Get keys
keys = ml_client.online_endpoints.get_keys(ENDPOINT_NAME)
print(f"\n{'='*60}")
print("🔑 Authentication Keys:")
print(f"{'='*60}")
print(f"Primary Key: {keys.primary_key}")
print(f"Secondary Key: {keys.secondary_key}")
print(f"{'='*60}")

print("\n✅ Test your endpoint with:")
print(f"\ncurl -X POST {endpoint.scoring_uri} \\")
print(f'  -H "Content-Type: application/json" \\')
print(f'  -H "Authorization: Bearer {keys.primary_key}" \\')
print(f'  -d \'{{"text": "This is amazing!"}}\'')
