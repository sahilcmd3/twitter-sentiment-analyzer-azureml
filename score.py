import json
import logging
import os
import nltk
from download_models import download_models_from_blob
from app.core.sentiment_analyzer import SentimentAnalyzer
from app.core.text_preprocessing import TextPreprocessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
analyzer = None
preprocessor = None

def init():
    """
    Called when the service is loaded
    Download models and initialize analyzer
    """
    global analyzer, preprocessor
    
    logger.info("Initializing sentiment analysis service...")
    
    # Download NLTK data
    try:
        logger.info("Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        logger.info("NLTK data downloaded successfully")
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {e}")
    
    # Download models from Azure Blob Storage
    try:
        download_models_from_blob()
        logger.info("Models downloaded successfully")
    except Exception as e:
        logger.error(f"Failed to download models: {e}")
    
    # Initialize components
    preprocessor = TextPreprocessor()
    analyzer = SentimentAnalyzer(use_ml_model=True, model_type='auto')
    
    logger.info("Service initialized successfully!")

def run(raw_data):
    """
    Called for each prediction request
    
    Input format:
    {
        "text": "This is amazing!",
        "use_bert": true  # optional, defaults to true
    }
    
    Output format:
    {
        "sentiment": "positive",
        "confidence": 0.95,
        "polarity": 0.95,
        "method": "BERT"
    }
    """
    try:
        # Parse input
        data = json.loads(raw_data)
        text = data.get('text', '')
        
        if not text:
            return json.dumps({'error': 'Text parameter is required'})
        
        # Preprocess text
        processed_text = preprocessor.preprocess(text)
        
        # Analyze sentiment
        result = analyzer.analyze(processed_text)
        
        # Add original text to response
        result['original_text'] = text
        result['processed_text'] = processed_text
        
        return json.dumps(result)
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return json.dumps({'error': str(e)})
