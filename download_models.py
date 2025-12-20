"""
Download models from Azure Blob Storage on startup
Keeps Git repository size small while enabling full functionality
"""
import os
import urllib.request
import zipfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Azure Blob Storage URLs (public access)
MODEL_URLS = {
    'bert_sentiment.zip': os.getenv('BERT_MODEL_URL', 'https://sentimentkeliyesamaan.blob.core.windows.net/models/bert_sentiment.zip'),
    'best_model.pkl': os.getenv('TRADITIONAL_MODEL_URL', 'https://sentimentkeliyesamaan.blob.core.windows.net/models/best_model.pkl'),
    'vectorizer.pkl': os.getenv('VECTORIZER_URL', 'https://sentimentkeliyesamaan.blob.core.windows.net/models/vectorizer.pkl'),
}

def download_file(url, destination):
    """Download file from URL to destination"""
    try:
        logger.info(f"Downloading {os.path.basename(destination)}...")
        urllib.request.urlretrieve(url, destination)
        logger.info(f"✅ Downloaded {os.path.basename(destination)}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download {os.path.basename(destination)}: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """Extract ZIP file"""
    try:
        logger.info(f"Extracting {os.path.basename(zip_path)}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(zip_path)
        logger.info(f"✅ Extracted {os.path.basename(zip_path)}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to extract {os.path.basename(zip_path)}: {e}")
        return False

def download_models_from_blob():
    """
    Download models from Azure Blob Storage if not present locally
    This is called on app startup
    """
    models_dir = 'app/models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Check if models already exist
    bert_exists = os.path.exists(f'{models_dir}/bert_sentiment')
    traditional_exists = os.path.exists(f'{models_dir}/best_model.pkl')
    vectorizer_exists = os.path.exists(f'{models_dir}/vectorizer.pkl')
    
    if bert_exists and traditional_exists and vectorizer_exists:
        logger.info("✅ All models already exist locally, skipping download")
        return True
    
    logger.info("📥 Downloading models from Azure Blob Storage...")
    
    success = True
    
    # Download BERT model (ZIP)
    if not bert_exists:
        bert_zip = f'{models_dir}/bert_sentiment.zip'
        if download_file(MODEL_URLS['bert_sentiment.zip'], bert_zip):
            if not extract_zip(bert_zip, models_dir):
                success = False
        else:
            success = False
    
    # Download traditional ML model
    if not traditional_exists:
        if not download_file(MODEL_URLS['best_model.pkl'], f'{models_dir}/best_model.pkl'):
            success = False
    
    # Download vectorizer
    if not vectorizer_exists:
        if not download_file(MODEL_URLS['vectorizer.pkl'], f'{models_dir}/vectorizer.pkl'):
            success = False
    
    if success:
        logger.info("All models downloaded successfully!")
    else:
        logger.warning("Warning: Some models failed to download. App will use fallback methods.")
    
    return success

if __name__ == "__main__":
    # Test the download function
    download_models_from_blob()
