from textblob import TextBlob
import pandas as pd
import pickle
import os
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Multi-model sentiment analyzer supporting BERT, traditional ML, and TextBlob.
    
    This class provides a flexible sentiment analysis system that can use different
    models based on availability and configuration:
    - BERT (DistilBERT): Deep learning model with 99.6% accuracy
    - Traditional ML: Logistic Regression, SVM, or Naive Bayes
    - TextBlob: Rule-based fallback method
    """
    
    def __init__(self, use_ml_model=True, model_type="auto"):
        """
        Initialize sentiment analyzer

        Args:
            use_ml_model (bool): If True, attempt to use trained ML models. 
                                 If False, use TextBlob as baseline
            model_type (str): Model selection strategy:
                             - 'auto': Try BERT first, then traditional ML, then TextBlob
                             - 'bert': Only use BERT model
                             - 'traditional': Use traditional ML models (SVM, Logistic Regression)
                             - 'textblob': Use rule-based TextBlob
        """
        self.use_ml_model = use_ml_model
        self.model_type = model_type
        self.model = None  # Traditional ML model (pickle file)
        self.vectorizer = None  # TF-IDF vectorizer for traditional ML
        self.bert_model = None  # Fine-tuned BERT model
        self.bert_tokenizer = None  # BERT tokenizer

        # Load models based on configuration
        if use_ml_model:
            self._load_models()

    def _load_models(self):
        """
        Load trained ML models based on model_type configuration.
        
        Loading priority:
        1. BERT model (if model_type is 'auto' or 'bert')
        2. Traditional ML model (if model_type is 'auto' or 'traditional')
        3. Falls back to TextBlob if no models are found
        """
        # Try to load BERT first if requested or auto
        if self.model_type in ["auto", "bert"]:
            loaded_bert = self._load_bert_model()
            if loaded_bert and self.model_type == "bert":
                return  # BERT-only mode, stop here

        # Load traditional ML model as fallback or primary choice
        if self.model_type in ["auto", "traditional"]:
            self._load_traditional_model()

    def _load_bert_model(self):
        """
        Load fine-tuned BERT (DistilBERT) model from disk.
        
        The BERT model is loaded from 'app/models/bert_sentiment/' directory
        which contains the model weights, tokenizer, and configuration.
        
        Returns:
            bool: True if BERT model loaded successfully, False otherwise
        """
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model_path = "app/models/bert_sentiment"

            if os.path.exists(model_path):
                logger.info("Loading BERT model...")
                # Load tokenizer for text preprocessing
                self.bert_tokenizer = AutoTokenizer.from_pretrained(model_path)
                # Load pre-trained model weights
                self.bert_model = AutoModelForSequenceClassification.from_pretrained(
                    model_path
                )
                # Set to evaluation mode (disables dropout)
                self.bert_model.eval()
                logger.info("Loaded fine-tuned BERT model")
                return True
            else:
                logger.info("Fine-tuned BERT model not found")
                return False

        except ImportError:
            logger.warning(
                "Transformers library not installed. Install with: pip install transformers torch"
            )
            return False
        except Exception as e:
            logger.warning(f"Error loading BERT model: {e}")
            return False

    def _load_traditional_model(self):
        """
        Load traditional ML model (Logistic Regression, SVM, or Naive Bayes).
        
        Loads both the trained model and its TF-IDF vectorizer from pickle files.
        These models are trained on the Sentiment140 dataset.
        
        Files loaded:
        - app/models/best_model.pkl: Trained classifier
        - app/models/vectorizer.pkl: TF-IDF vectorizer for text transformation
        """
        try:
            model_path = "app/models/best_model.pkl"
            vectorizer_path = "app/models/vectorizer.pkl"

            if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                # Load serialized model
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)

                # Load TF-IDF vectorizer
                with open(vectorizer_path, "rb") as f:
                    self.vectorizer = pickle.load(f)

                logger.info("Loaded traditional ML model")
            else:
                logger.warning(
                    "Traditional ML model not found, using TextBlob fallback"
                )
                self.use_ml_model = False

        except Exception as e:
            logger.warning(f"Error loading traditional ML model: {e}")
            logger.info("Using TextBlob fallback")
            self.use_ml_model = False

    def analyze_with_bert(self, text: str) -> Dict:
        """
        Perform sentiment analysis using BERT model.
        
        Process:
        1. Tokenize input text using BERT tokenizer
        2. Pass through BERT model to get logits
        3. Apply softmax to get probabilities
        4. Classify as positive/negative/neutral based on confidence
        
        Args:
            text (str): Preprocessed text to analyze
            
        Returns:
            dict: Contains sentiment, polarity, confidence, subjectivity, and method
                  Example: {
                      'sentiment': 'positive',
                      'polarity': 0.96,
                      'confidence': 0.96,
                      'subjectivity': 0.5,
                      'method': 'BERT'
                  }
        """
        try:
            import torch

            # Tokenize text and convert to PyTorch tensors
            inputs = self.bert_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512, padding=True
            )

            # Run inference without gradient calculation (faster)
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                logits = outputs.logits  # Raw model outputs
                # Convert logits to probabilities using softmax
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                # Get predicted class (0=negative, 1=positive)
                prediction = torch.argmax(probabilities, dim=1).item()
                # Get confidence score for predicted class
                confidence = probabilities[0][prediction].item()

            # Map prediction to sentiment label
            sentiment = "positive" if prediction == 1 else "negative"
            
            # Add neutral category for low confidence predictions
            # If model is unsure (confidence < 65%), classify as neutral
            if confidence < 0.65:
                sentiment = "neutral"
                confidence = 1.0 - max(probabilities[0]).item()
            
            # Calculate polarity score (-1 to 1)
            polarity = confidence if sentiment == "positive" else (-confidence if sentiment == "negative" else 0.0)

            return {
                "sentiment": sentiment,
                "polarity": polarity,
                "confidence": confidence,
                "subjectivity": 0.5,  # BERT doesn't provide subjectivity
                "method": "BERT",
            }
        except Exception as e:
            logger.error(f"Error in BERT analysis: {e}")
            # Fallback to traditional ML or TextBlob on error
            if self.model:
                return self.analyze_with_ml(text)
            else:
                return self.analyze_with_textblob(text)

    def analyze_with_ml(self, text: str) -> Dict:
        """
        Perform sentiment analysis using traditional ML model.
        
        Process:
        1. Transform text using TF-IDF vectorizer
        2. Predict sentiment using trained classifier
        3. Get confidence from probability estimates
        
        Args:
            text (str): Preprocessed text to analyze
            
        Returns:
            dict: Contains sentiment, polarity, confidence, subjectivity, and method
        """
        # Convert text to TF-IDF features
        vectorized = self.vectorizer.transform([text])

        # Get prediction (0=negative, 1=positive)
        prediction = self.model.predict(vectorized)[0]

        # Get confidence score if model supports probability estimates
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(vectorized)[0]
            confidence = probabilities[prediction]
        else:
            confidence = 0.8  # Default confidence for SVM (no probability estimates)

        # Map prediction to sentiment label
        sentiment = "positive" if prediction == 1 else "negative"
        # Calculate polarity score
        polarity = confidence if sentiment == "positive" else -confidence

        return {
            "sentiment": sentiment,
            "polarity": polarity,
            "confidence": confidence,
            "subjectivity": 0.5,  # ML models don't provide subjectivity scores
            "method": "ML Model",
        }

    def analyze_with_textblob(self, text: str) -> Dict:
        """
        Perform sentiment analysis using TextBlob (rule-based approach).
        
        TextBlob uses a lexicon-based approach with predefined word sentiment scores.
        It's fast but less accurate than ML models (~65% accuracy).
        
        Args:
            text (str): Preprocessed text to analyze
            
        Returns:
            dict: Contains sentiment, polarity (-1 to 1), subjectivity (0 to 1),
                  confidence, and method
        """
        blob = TextBlob(text)
        # Get polarity (-1=negative, 1=positive) and subjectivity (0=objective, 1=subjective)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # Classify sentiment with threshold of 0.1
        # This creates a "neutral zone" between -0.1 and 0.1
        if polarity > 0.1:
            sentiment = "positive"
        elif polarity < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "sentiment": sentiment,
            "polarity": polarity,
            "subjectivity": subjectivity,
            "confidence": abs(polarity),  # Use absolute polarity as confidence
            "method": "TextBlob",
        }

    def analyze(self, text: str) -> Dict:
        """
        Analyze sentiment using the best available method.
        
        Priority order:
        1. BERT (if loaded) - 99.6% accuracy
        2. Traditional ML (if loaded) - 78% accuracy
        3. TextBlob (fallback) - 65% accuracy
        
        Args:
            text (str): Preprocessed text to analyze
            
        Returns:
            dict: Sentiment analysis results with sentiment, confidence, polarity, etc.
        """
        # Use the highest accuracy model available
        if self.bert_model:
            return self.analyze_with_bert(text)
        elif self.use_ml_model and self.model:
            return self.analyze_with_ml(text)
        else:
            return self.analyze_with_textblob(text)

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sentiment for all tweets in a DataFrame.
        
        Applies sentiment analysis to each preprocessed text in the DataFrame
        and adds new columns with results.
        
        Args:
            df (pd.DataFrame): DataFrame with 'processed_text' column
            
        Returns:
            pd.DataFrame: Original DataFrame with added columns:
                         - sentiment: positive/negative/neutral
                         - polarity: -1 to 1 score
                         - confidence: 0 to 1 confidence score
                         - subjectivity: 0 to 1 subjectivity score
                         - method: Model used (BERT/ML Model/TextBlob)
        """
        # Apply analysis to each row
        results = df["processed_text"].apply(self.analyze)

        # Extract fields from result dictionaries and create new columns
        df["sentiment"] = results.apply(lambda x: x["sentiment"])
        df["polarity"] = results.apply(lambda x: x["polarity"])
        df["confidence"] = results.apply(lambda x: x["confidence"])
        df["subjectivity"] = results.apply(lambda x: x.get("subjectivity", 0.5))
        df["method"] = results.apply(lambda x: x.get("method", "Unknown"))

        return df

    def get_model_info(self) -> Dict:
        """
        Get information about currently loaded models.
        
        Returns:
            dict: Model status information including:
                 - bert_loaded: Whether BERT model is loaded
                 - traditional_ml_loaded: Whether traditional ML model is loaded
                 - using_ml: Whether ML models are being used
                 - active_model: Name of the active model (BERT/Traditional ML/TextBlob)
        """
        return {
            "bert_loaded": self.bert_model is not None,
            "traditional_ml_loaded": self.model is not None,
            "using_ml": self.use_ml_model,
            "active_model": (
                "BERT"
                if self.bert_model
                else ("Traditional ML" if self.model else "TextBlob")
            ),
        }
