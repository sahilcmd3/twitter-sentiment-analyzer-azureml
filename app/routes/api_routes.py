from flask import Blueprint, request, jsonify, current_app
from app.core.twitter_collector import TwitterCollector
from app.core.text_preprocessing import TextPreprocessor
from app.core.sentiment_analyzer import SentimentAnalyzer
from app.core.visualizer import SentimentVisualizer
import pandas as pd
from datetime import datetime, timezone
import os
import requests

api_bp = Blueprint("api", __name__)

# Lazy initialization - components will be created on first use
_collector = None
_preprocessor = None
_analyzer = None
_visualizer = None


def get_collector():
    """Get or create TwitterCollector singleton"""
    global _collector
    if _collector is None:
        _collector = TwitterCollector()
    return _collector


def get_preprocessor():
    """Get or create TextPreprocessor singleton"""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = TextPreprocessor()
    return _preprocessor


def get_analyzer():
    """Get or create SentimentAnalyzer singleton"""
    global _analyzer
    if _analyzer is None:
        # Use Azure ML endpoint if available, otherwise local models
        use_azure_ml = os.getenv('AZURE_ML_ENDPOINT') is not None
        if use_azure_ml:
            current_app.logger.info("Using Azure ML endpoint for inference")
        _analyzer = SentimentAnalyzer(use_ml_model=not use_azure_ml)
    return _analyzer


def analyze_with_azure_ml(text):
    """Call Azure ML endpoint for sentiment analysis"""
    endpoint = os.getenv('AZURE_ML_ENDPOINT')
    api_key = os.getenv('AZURE_ML_API_KEY')
    
    if not endpoint or not api_key:
        return None
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {"text": text}
        
        response = requests.post(endpoint, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        current_app.logger.error(f"Azure ML endpoint error: {e}")
        return None


def get_visualizer():
    """Get or create SentimentVisualizer singleton"""
    global _visualizer
    if _visualizer is None:
        _visualizer = SentimentVisualizer()
    return _visualizer


@api_bp.route("/analyze", methods=["POST"])
def analyze():
    """
    Analyze tweets sentiment using trained ML model
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Request body is required"}), 400

        query = data.get("query", "Python programming")
        max_tweets = min(int(data.get("max_tweets", 50)), 100)

        current_app.logger.info(f"Analyzing tweets for: {query}")

        # Get singletons
        collector = get_collector()
        preprocessor = get_preprocessor()
        analyzer = get_analyzer()
        visualizer = get_visualizer()

        # Collect tweets
        tweets = collector.search_tweets(query, max_results=max_tweets)

        if not tweets:
            return jsonify({"error": "No tweets found"}), 404

        # Process tweets
        df = pd.DataFrame(tweets)
        df = preprocessor.preprocess_dataframe(df)
        
        # Use Azure ML if available, otherwise local analyzer
        if os.getenv('AZURE_ML_ENDPOINT'):
            # Analyze each tweet with Azure ML
            sentiments = []
            for text in df['processed_text']:
                result = analyze_with_azure_ml(text)
                if result and 'sentiment' in result:
                    sentiments.append({
                        'sentiment': result['sentiment'],
                        'confidence': result.get('confidence', 0.5)
                    })
                else:
                    # Fallback to TextBlob if Azure ML fails
                    sentiments.append({'sentiment': 'neutral', 'confidence': 0.5})
            
            df['sentiment'] = [s['sentiment'] for s in sentiments]
            df['confidence'] = [s['confidence'] for s in sentiments]
        else:
            # Use local analyzer
            df = analyzer.analyze_dataframe(df)

        # Generate visualizations
        sentiment_dist = visualizer.plot_sentiment_distribution(df)
        timeline = visualizer.plot_sentiment_timeline(df)
        stats = visualizer.generate_stats(df)

        # Sample tweets with more details
        sample_tweets = (
            df[["text", "sentiment", "confidence", "author_username"]]
            .head(10)
            .to_dict("records")
        )

        response = {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "using_ml_model": analyzer.use_ml_model,
            "model_method": "Trained ML Model" if analyzer.use_ml_model else "TextBlob",
            "stats": stats,
            "visualizations": {
                "sentiment_distribution": sentiment_dist,
                "timeline": timeline,
            },
            "sample_tweets": sample_tweets,
        }

        current_app.logger.info(
            f"Analyzed {len(tweets)} tweets using {response['model_method']}"
        )

        return jsonify(response), 200

    except Exception as e:
        current_app.logger.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/analyze-text", methods=["POST"])
def analyze_text():
    """
    Analyze custom text sentiment using trained ML model
    """
    try:
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "Text parameter is required"}), 400

        text = data.get("text", "")

        if not text.strip():
            return jsonify({"error": "Text cannot be empty"}), 400

        # Get singletons
        preprocessor = get_preprocessor()
        analyzer = get_analyzer()

        # Preprocess
        processed = preprocessor.preprocess(text)

        # Analyze
        result = analyzer.analyze(processed)

        response = {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "original_text": text,
            "processed_text": processed,
            "sentiment": result["sentiment"],
            "confidence": result["confidence"],
            "polarity": result.get("polarity", 0),
            "method": result.get("method", "Unknown"),
        }

        return jsonify(response), 200

    except Exception as e:
        current_app.logger.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/model-info", methods=["GET"])
def model_info():
    """Get information about the current model"""
    analyzer = get_analyzer()

    # Get comprehensive model info
    model_info_data = analyzer.get_model_info()

    model_status = {
        "bert_loaded": model_info_data["bert_loaded"],
        "traditional_ml_loaded": model_info_data["traditional_ml_loaded"],
        "using_ml_model": analyzer.use_ml_model,
        "active_model": model_info_data["active_model"],
        "method": model_info_data["active_model"],
    }

    # Check for model comparison results
    comparison_file = "data/model_comparison.csv"
    if os.path.exists(comparison_file):
        comparison_df = pd.read_csv(comparison_file)
        model_status["model_comparison"] = comparison_df.to_dict("records")

    return jsonify(model_status), 200


@api_bp.route("/twitter-status", methods=["GET"])
def twitter_status():
    """Get Twitter API connection status"""
    collector = get_collector()
    status = collector.get_api_status()
    return jsonify(status), 200


@api_bp.route("/stats", methods=["GET"])
def get_stats():
    """Get API statistics"""
    return (
        jsonify(
            {
                "status": "success",
                "message": "Twitter Sentiment Analysis API",
                "author": "sahilcmd3",
                "version": "1.0.0",
                "endpoints": {
                    "POST /api/analyze": "Analyze tweets sentiment",
                    "POST /api/analyze-text": "Analyze custom text",
                    "GET /api/model-info": "Get ML model information",
                    "GET /api/twitter-status": "Get Twitter API status",
                    "GET /api/stats": "Get API statistics",
                },
            }
        ),
        200,
    )
