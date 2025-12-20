"""
Azure ML Inference Script (score.py)

This script is the entry point for Azure ML managed endpoints.
It handles model loading and prediction requests for the deployed BERT model.

Required functions:
- init(): Called once when container starts - loads models
- run(raw_data): Called for each prediction request - returns sentiment analysis
"""

import json
import logging
import os
import nltk
from download_models import download_models_from_blob
from app.core.sentiment_analyzer import SentimentAnalyzer
from app.core.text_preprocessing import TextPreprocessor

# Setup logging for Azure ML container
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables - persist across requests for performance
analyzer = None
preprocessor = None

def init():
    """
    Initialize the sentiment analysis service.
    
    This function is called once when the Azure ML container starts.
    It performs one-time setup operations:
    1. Downloads required NLTK data (tokenizers, stopwords)
    2. Downloads ML model files from Azure Blob Storage
    3. Initializes preprocessor and analyzer instances
    
    The loaded models are stored in global variables and reused
    across all prediction requests for better performance.
    """
    global analyzer, preprocessor
    
    logger.info("Initializing sentiment analysis service...")
    
    # Download NLTK data required for text preprocessing
    try:
        logger.info("Downloading NLTK data...")
        nltk.download('punkt', quiet=True)  # Sentence tokenizer
        nltk.download('punkt_tab', quiet=True)  # Punkt tokenizer models
        nltk.download('stopwords', quiet=True)  # Common stopwords list
        nltk.download('wordnet', quiet=True)  # WordNet lexical database
        logger.info("NLTK data downloaded successfully")
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {e}")
    
    # Download ML model files from Azure Blob Storage
    # This includes BERT model weights and traditional ML models
    try:
        download_models_from_blob()
        logger.info("Models downloaded successfully")
    except Exception as e:
        logger.error(f"Failed to download models: {e}")
    
    # Initialize text preprocessor (cleans and tokenizes text)
    preprocessor = TextPreprocessor()
    
    # Initialize sentiment analyzer with automatic model selection
    # Priority: BERT > Traditional ML > TextBlob
    analyzer = SentimentAnalyzer(use_ml_model=True, model_type='auto')
    
    logger.info("Service initialized successfully!")

def run(raw_data):
    """
    Process a sentiment analysis request.
    
    This function is called for each HTTP POST request to the Azure ML endpoint.
    It handles the request/response cycle for sentiment analysis.
    
    Args:
        raw_data (str): JSON string containing the request data
        
    Input format:
        {
            "text": "This is amazing!",
            "use_bert": true  # optional, defaults to true
        }
    
    Returns:
        str: JSON string containing sentiment analysis results
        
    Output format:
        {
            "sentiment": "positive",
            "confidence": 0.95,
            "polarity": 0.95,
            "subjectivity": 0.5,
            "method": "BERT",
            "original_text": "This is amazing!",
            "processed_text": "amazing"
        }
    
    Error format:
        {
            "error": "Error message description"
        }
    """
    try:
        # Parse JSON request data
        data = json.loads(raw_data)
        text = data.get('text', '')
        
        # Validate input
        if not text:
            return json.dumps({'error': 'Text parameter is required'})
        
        # Step 1: Preprocess text (clean, tokenize, remove stopwords)
        processed_text = preprocessor.preprocess(text)
        
        # Step 2: Analyze sentiment using best available model
        result = analyzer.analyze(processed_text)
        
        # Step 3: Add original and processed text to response for debugging
        result['original_text'] = text
        result['processed_text'] = processed_text
        
        # Return JSON string (Azure ML expects string, not dict)
        return json.dumps(result)
    
    except Exception as e:
        # Log error and return error response
        logger.error(f"Error in prediction: {str(e)}")
        return json.dumps({'error': str(e)})
