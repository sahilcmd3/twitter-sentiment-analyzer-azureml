import pickle
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.text_preprocessing import TextPreprocessor


def load_model():
    """Load trained model and vectorizer"""
    try:
        with open("app/models/best_model.pkl", "rb") as f:
            model = pickle.load(f)

        with open("app/models/vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

        return model, vectorizer
    except FileNotFoundError:
        print("Models not found. Please run: python scripts/train_models.py")
        sys.exit(1)


def predict_sentiment(text, model, vectorizer, preprocessor):
    """Predict sentiment of text"""
    # Preprocess
    processed = preprocessor.preprocess(text)

    # Vectorize
    vectorized = vectorizer.transform([processed])

    # Predict
    prediction = model.predict(vectorized)[0]
    probability = (
        model.predict_proba(vectorized)[0] if hasattr(model, "predict_proba") else None
    )

    sentiment = "positive" if prediction == 1 else "negative"
    confidence = probability[prediction] if probability is not None else 0.0

    return sentiment, confidence


def main():
    """Test models with sample tweets"""

    print(
        """
    ╔══════════════════════════════════════════════════════════╗
    ║   Testing Trained Models                                 ║
    ║   Author: sahil & aryan                                  ║
    ╚══════════════════════════════════════════════════════════╝
    """
    )

    # Load model
    print("Loading trained model...")
    model, vectorizer = load_model()
    preprocessor = TextPreprocessor()
    print("Model loaded\n")

    # Test tweets
    test_tweets = [
        "I love Python programming! It's amazing! 😍",
        "This is the worst code I've ever seen. Terrible!",
        "Machine learning is fascinating and powerful",
        "I hate debugging. So frustrating and annoying.",
        "Just deployed my first app to Azure! So excited!",
        "This API documentation is useless. Makes no sense.",
        "Natural language processing is really interesting",
        "Why doesn't my code work? I'm so confused and angry.",
        "Flask makes web development so easy and fun!",
        "This error message is driving me crazy. Worst day ever.",
    ]

    print("Testing with sample tweets:\n")
    print(f"{'Tweet':<50} {'Sentiment':<12} {'Confidence':<10}")
    print("=" * 75)

    for tweet in test_tweets:
        sentiment, confidence = predict_sentiment(
            tweet, model, vectorizer, preprocessor
        )

        # Truncate long tweets
        display_tweet = tweet[:47] + "..." if len(tweet) > 50 else tweet

        # Color coding
        color = "\033[92m" if sentiment == "positive" else "\033[91m"
        reset = "\033[0m"

        print(
            f"{display_tweet:<50} {color}{sentiment.upper():<12}{reset} {confidence:.2%}"
        )

    print("\Testing complete!")

    # Interactive testing
    print("\n" + "=" * 60)
    print("Interactive Testing (type 'quit' to exit)")
    print("=" * 60)

    while True:
        user_input = input("\nEnter text to analyze: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        sentiment, confidence = predict_sentiment(
            user_input, model, vectorizer, preprocessor
        )

        color = "\033[92m" if sentiment == "positive" else "\033[91m"
        reset = "\033[0m"

        print(f"Sentiment: {color}{sentiment.upper()}{reset}")
        print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
