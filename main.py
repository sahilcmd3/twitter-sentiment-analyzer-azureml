import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Download models from Azure Blob Storage only if not using Azure ML endpoint
if not os.getenv('AZURE_ML_ENDPOINT'):
    from download_models import download_models_from_blob
    download_models_from_blob()
else:
    print("Using Azure ML endpoint - skipping model download")

from app import create_app
import logging

# Reduce Flask logging verbosity
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    print(f"\n✅ Server running on: http://localhost:{port}\n")

    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
