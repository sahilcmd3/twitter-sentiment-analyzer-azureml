from textblob import TextBlob
import pandas as pd
import pickle
import os
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    def __init__(self, use_ml_model=True, model_type="auto"):
        """
        Initialize sentiment analyzer

        Args:
            use_ml_model: If True, use trained ML model. If False, use TextBlob
            model_type: 'auto', 'bert', 'traditional', or 'textblob'
        """
        self.use_ml_model = use_ml_model
        self.model_type = model_type
        self.model = None
        self.vectorizer = None
        self.bert_model = None
        self.bert_tokenizer = None

        if use_ml_model:
            self._load_models()

    def _load_models(self):
        """Load trained ML models (BERT and/or traditional)"""
        # Try to load BERT first if requested or auto
        if self.model_type in ["auto", "bert"]:
            loaded_bert = self._load_bert_model()
            if loaded_bert and self.model_type == "bert":
                return

        # Load traditional ML model
        if self.model_type in ["auto", "traditional"]:
            self._load_traditional_model()

    def _load_bert_model(self):
        """Load fine-tuned BERT model"""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model_path = "app/models/bert_sentiment"

            if os.path.exists(model_path):
                logger.info("Loading BERT model...")
                self.bert_tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.bert_model = AutoModelForSequenceClassification.from_pretrained(
                    model_path
                )
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
        """Load traditional ML model and vectorizer"""
        try:
            model_path = "app/models/best_model.pkl"
            vectorizer_path = "app/models/vectorizer.pkl"

            if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)

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
        """Analyze using fine-tuned BERT model"""
        try:
            import torch

            # Tokenize
            inputs = self.bert_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512, padding=True
            )

            # Predict
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item()

            sentiment = "positive" if prediction == 1 else "negative"
            polarity = confidence if sentiment == "positive" else -confidence

            return {
                "sentiment": sentiment,
                "polarity": polarity,
                "confidence": confidence,
                "subjectivity": 0.5,  # Not available from BERT
                "method": "BERT",
            }
        except Exception as e:
            logger.error(f"Error in BERT analysis: {e}")
            # Fallback to traditional ML or TextBlob
            if self.model:
                return self.analyze_with_ml(text)
            else:
                return self.analyze_with_textblob(text)

    def analyze_with_ml(self, text: str) -> Dict:
        """Analyze using trained ML model"""
        # Vectorize
        vectorized = self.vectorizer.transform([text])

        # Predict
        prediction = self.model.predict(vectorized)[0]

        # Get probability if available
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(vectorized)[0]
            confidence = probabilities[prediction]
        else:
            confidence = 0.8  # Default confidence for SVM

        sentiment = "positive" if prediction == 1 else "negative"
        polarity = confidence if sentiment == "positive" else -confidence

        return {
            "sentiment": sentiment,
            "polarity": polarity,
            "confidence": confidence,
            "subjectivity": 0.5,  # Not available from ML model
            "method": "ML Model",
        }

    def analyze_with_textblob(self, text: str) -> Dict:
        """Analyze using TextBlob (rule-based)"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # Classify sentiment
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
            "confidence": abs(polarity),
            "method": "TextBlob",
        }

    def analyze(self, text: str) -> Dict:
        """Analyze sentiment using best available method"""
        # Priority: BERT > Traditional ML > TextBlob
        if self.bert_model:
            return self.analyze_with_bert(text)
        elif self.use_ml_model and self.model:
            return self.analyze_with_ml(text)
        else:
            return self.analyze_with_textblob(text)

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze sentiment for all tweets"""
        results = df["processed_text"].apply(self.analyze)

        df["sentiment"] = results.apply(lambda x: x["sentiment"])
        df["polarity"] = results.apply(lambda x: x["polarity"])
        df["confidence"] = results.apply(lambda x: x["confidence"])
        df["subjectivity"] = results.apply(lambda x: x.get("subjectivity", 0.5))
        df["method"] = results.apply(lambda x: x.get("method", "Unknown"))

        return df

    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
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
