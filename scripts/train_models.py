import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
import time
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.text_preprocessing import TextPreprocessor


def load_sentiment140(filepath="data/sentiment140.csv", sample_size=None):
    """
    Load Sentiment140 dataset

    Dataset format:
    - Column 0: Sentiment (0=negative, 2=neutral, 4=positive)
    - Column 5: Tweet text
    """
    print("Loading Sentiment140 dataset...")

    # Load with correct column names
    column_names = ["sentiment", "id", "date", "query", "user", "text"]

    try:
        df = pd.read_csv(
            filepath,
            encoding="latin-1",
            names=column_names,
            usecols=["sentiment", "text"],
        )
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        print("Please download the dataset from:")
        print("https://www.kaggle.com/datasets/kazanova/sentiment140")
        sys.exit(1)

    print(f"Loaded {len(df)} tweets")

    # Map sentiments: 0 -> 0 (negative), 4 -> 2 (positive)
    # We'll treat this as binary for now (no neutral in Sentiment140)
    df["sentiment"] = df["sentiment"].map({0: 0, 4: 1})  # 0=negative, 1=positive

    # Sample if requested (for faster training during development)
    if sample_size:
        print(f"Sampling {sample_size} tweets...")
        df = df.sample(n=min(sample_size, len(df)), random_state=42)

    print(f"Final dataset size: {len(df)} tweets")
    print(f"Negative: {(df['sentiment'] == 0).sum()}")
    print(f"Positive: {(df['sentiment'] == 1).sum()}")

    return df


def preprocess_data(df):
    """Preprocess tweet text"""
    print("\nPreprocessing tweets...")

    preprocessor = TextPreprocessor()

    start_time = time.time()
    df["processed_text"] = df["text"].apply(preprocessor.preprocess)

    # Remove empty processed texts
    df = df[df["processed_text"].str.len() > 0]

    processing_time = time.time() - start_time
    print(f"Preprocessing complete in {processing_time:.2f} seconds")
    print(
        f"Average text length: {df['processed_text'].str.len().mean():.0f} characters"
    )

    return df


def train_model(name, model, X_train, X_test, y_train, y_test):
    """Train and evaluate a single model"""
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")

    # Training
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Prediction
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = (time.time() - start_time) / len(y_test) * 1000  # ms per sample

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["Negative", "Positive"]
    )
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nResults:")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Training Time: {train_time:.2f} seconds")
    print(f"Avg Inference Time: {inference_time:.2f} ms")

    print(f"\nClassification Report:")
    print(report)

    print(f"\nConfusion Matrix:")
    print(f"True Neg  False Pos")
    print(f"{cm[0][0]:<9} {cm[0][1]:<9}")
    print(f"False Neg True Pos")
    print(f"{cm[1][0]:<9} {cm[1][1]:<9}")

    return {
        "model_name": name,
        "accuracy": accuracy,
        "train_time": train_time,
        "inference_time_ms": inference_time,
        "model": model,
    }


def main():
    """Main training pipeline"""

    print(
        """
    ╔══════════════════════════════════════════════════════════╗
    ║   Twitter Sentiment Analysis - Model Training            ║
    ║   Author: sahil & aryan                                  ║
    ╚══════════════════════════════════════════════════════════╝
    """
    )

    # Configuration
    SAMPLE_SIZE = 100000  # Use 100k for faster training, set to None for full dataset
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # Load data
    df = load_sentiment140("data/sentiment140.csv", sample_size=SAMPLE_SIZE)

    # Preprocess
    df = preprocess_data(df)

    # Split data
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["processed_text"],
        df["sentiment"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["sentiment"],
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Vectorization
    print("\nVectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.8
    )

    start_time = time.time()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    vec_time = time.time() - start_time

    print(f"Vectorization complete in {vec_time:.2f} seconds")
    print(f"Feature dimensions: {X_train_vec.shape}")

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Naive Bayes": MultinomialNB(alpha=1.0),
        "Linear SVM": LinearSVC(random_state=RANDOM_STATE, max_iter=1000, dual=False),
    }

    # Train all models
    results = []
    best_accuracy = 0
    best_model_name = None
    best_model = None

    for name, model in models.items():
        result = train_model(name, model, X_train_vec, X_test_vec, y_train, y_test)
        results.append(result)

        if result["accuracy"] > best_accuracy:
            best_accuracy = result["accuracy"]
            best_model_name = name
            best_model = result["model"]

    # Save results
    print(f"\n{'='*60}")
    print("Saving models and results...")
    print(f"{'='*60}")

    # Create models directory
    os.makedirs("app/models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Save best model
    with open("app/models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    print(f"Best model saved: {best_model_name}")

    # Save vectorizer
    with open("app/models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print("Vectorizer saved")

    # Save all models for comparison
    for result in results:
        model_filename = (
            f"app/models/{result['model_name'].lower().replace(' ', '_')}.pkl"
        )
        with open(model_filename, "wb") as f:
            pickle.dump(result["model"], f)
        print(f"Saved: {model_filename}")

    # Save comparison results
    comparison_df = pd.DataFrame(
        [
            {
                "Model": r["model_name"],
                "Accuracy": f"{r['accuracy']:.4f}",
                "Accuracy %": f"{r['accuracy']*100:.2f}%",
                "Training Time (s)": f"{r['train_time']:.2f}",
                "Inference Time (ms)": f"{r['inference_time_ms']:.2f}",
            }
            for r in results
        ]
    )

    comparison_df.to_csv("data/model_comparison.csv", index=False)
    print("Comparison results saved: data/model_comparison.csv")

    # Final summary
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(comparison_df.to_string(index=False))

    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"{'='*60}")

    print("\Training complete! Models ready for deployment.")
    print("\nNext steps:")
    print("1. Test models: python scripts/test_models.py")
    print("2. Update app to use trained models")
    print("3. Deploy to Azure")


if __name__ == "__main__":
    main()
