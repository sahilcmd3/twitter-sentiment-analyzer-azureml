"""
Deploy sentiment analysis model to Azure ML
"""
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)
from azure.identity import DefaultAzureCredential
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
SUBSCRIPTION_ID = os.getenv('AZURE_SUBSCRIPTION_ID')
RESOURCE_GROUP = os.getenv('AZURE_RESOURCE_GROUP')
WORKSPACE_NAME = os.getenv('AZURE_WORKSPACE_NAME')
ENDPOINT_NAME = os.getenv('AZURE_ENDPOINT_NAME')  # Must be globally unique
DEPLOYMENT_NAME = "sentiment-bert-deployment"

def main():
    print("🚀 Starting Azure ML Deployment...")
    
    # Authenticate
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME
    )
    
    print(f"✅ Connected to workspace: {WORKSPACE_NAME}")
    
    # Create endpoint
    print(f"\n📍 Creating endpoint: {ENDPOINT_NAME}")
    endpoint = ManagedOnlineEndpoint(
        name=ENDPOINT_NAME,
        description="Twitter Sentiment Analysis API",
        auth_mode="key"
    )
    
    try:
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        print(f"✅ Endpoint created: {ENDPOINT_NAME}")
    except Exception as e:
        print(f"ℹ️  Endpoint might already exist: {e}")
    
    # Create environment
    print("\n🔧 Creating environment...")
    env = Environment(
        name="sentiment-env",
        description="Environment for sentiment analysis",
        conda_file="environment.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
    )
    
    ml_client.environments.create_or_update(env)
    print("✅ Environment created")
    
    # Create deployment
    print(f"\n🚢 Creating deployment: {DEPLOYMENT_NAME}")
    deployment = ManagedOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        code_configuration=CodeConfiguration(
            code=".",
            scoring_script="score.py"
        ),
        environment=env,
        instance_type="Standard_DS2_v2",  # 2 cores, 7GB RAM (fits quota)
        instance_count=1
    )
    
    ml_client.online_deployments.begin_create_or_update(deployment).result()
    print(f"✅ Deployment created: {DEPLOYMENT_NAME}")
    
    # Update endpoint to route 100% traffic to this deployment
    print("\n🔀 Routing traffic to deployment...")
    endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
    endpoint.traffic = {DEPLOYMENT_NAME: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print("✅ Traffic routed")
    
    # Get endpoint details
    endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
    print(f"\n{'='*60}")
    print("🎉 DEPLOYMENT SUCCESSFUL!")
    print(f"{'='*60}")
    print(f"Endpoint Name: {endpoint.name}")
    print(f"Scoring URI: {endpoint.scoring_uri}")
    print(f"Swagger URI: {endpoint.openapi_uri}")
    print(f"{'='*60}")
    
    # Get keys
    keys = ml_client.online_endpoints.get_keys(ENDPOINT_NAME)
    print(f"\n🔑 Authentication Keys:")
    print(f"Primary Key: {keys.primary_key}")
    print(f"Secondary Key: {keys.secondary_key}")
    print(f"{'='*60}")
    
    print("\n✅ Deployment complete! Test your endpoint with:")
    print(f"\ncurl -X POST {endpoint.scoring_uri} \\")
    print(f'  -H "Content-Type: application/json" \\')
    print(f'  -H "Authorization: Bearer {keys.primary_key}" \\')
    print(f'  -d \'{{"text": "This is amazing!"}}\'')

if __name__ == "__main__":
    main()
