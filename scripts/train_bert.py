import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import time
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.text_preprocessing import TextPreprocessor


def check_dependencies():
    """Check if required libraries are installed"""
    try:
        import torch
        import transformers

        print(f"PyTorch version: {torch.__version__}")
        print(f"Transformers version: {transformers.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("\nPlease install required packages:")
        print("pip install torch transformers sentencepiece")
        return False


def load_sentiment140(filepath="data/sentiment140.csv", sample_size=None):
    """Load Sentiment140 dataset"""
    print("Loading Sentiment140 dataset...")

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

    # Map sentiments: 0 -> 0 (negative), 4 -> 1 (positive)
    df["sentiment"] = df["sentiment"].map({0: 0, 4: 1})

    # Sample if requested
    if sample_size:
        print(f"Sampling {sample_size} tweets for training...")
        df = df.sample(n=min(sample_size, len(df)), random_state=42)

    print(f"Dataset size: {len(df)} tweets")
    print(f"Negative: {(df['sentiment'] == 0).sum()}")
    print(f"Positive: {(df['sentiment'] == 1).sum()}")

    return df


def train_bert_model(
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    model_name="distilbert-base-uncased",
    epochs=3,
    batch_size=16,
):
    """Train BERT model for sentiment analysis"""

    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        DataCollatorWithPadding,
    )
    from torch.utils.data import Dataset

    print(f"\n{'='*60}")
    print(f"Training BERT Model: {model_name}")
    print(f"{'='*60}")

    # Load tokenizer and model
    print("\nLoading pre-trained model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Create custom dataset
    class SentimentDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=128):
            self.encodings = tokenizer(
                texts.tolist(),
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            self.labels = labels.tolist()

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item["labels"] = self.labels[idx]
            return item

        def __len__(self):
            return len(self.labels)

    # Create datasets
    print("Tokenizing data...")
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir="app/models/bert_checkpoints",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="logs",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        report_to="none",
    )

    # Define metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy}

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    # Train
    print("\nStarting training...")
    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time

    # Evaluate
    print("\nEvaluating model...")
    eval_results = trainer.evaluate()

    # Save model
    print("\nSaving model...")
    model_save_path = "app/models/bert_sentiment"
    os.makedirs(model_save_path, exist_ok=True)
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    print(f"\n{'='*60}")
    print("BERT MODEL RESULTS")
    print(f"{'='*60}")
    print(f"Training Time: {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
    print(
        f"Validation Accuracy: {eval_results['eval_accuracy']:.4f} ({eval_results['eval_accuracy']*100:.2f}%)"
    )
    print(f"Validation Loss: {eval_results['eval_loss']:.4f}")
    print(f"{'='*60}")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "accuracy": eval_results["eval_accuracy"],
        "train_time": train_time,
    }


def compare_with_traditional_models(val_texts, val_labels):
    """Compare BERT with traditional models"""
    comparison_file = "data/model_comparison.csv"

    if os.path.exists(comparison_file):
        print("\nLoading previous model comparison...")
        comparison_df = pd.read_csv(comparison_file)
        print(comparison_df.to_string(index=False))
    else:
        print("\nNo previous model comparison found.")
        print("Run 'python scripts/train_models.py' first to train traditional models.")


def main():
    """Main training pipeline"""

    print(
        """
    ╔══════════════════════════════════════════════════════════╗
    ║   Twitter Sentiment Analysis - BERT Training             ║
    ║   Author: sahil & aryan                                  ║
    ╚══════════════════════════════════════════════════════════╝
    """
    )

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Configuration
    SAMPLE_SIZE = 50000  # Use 50k for BERT (computationally expensive)
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    MODEL_NAME = "distilbert-base-uncased"  # Smaller, faster version of BERT
    EPOCHS = 3
    BATCH_SIZE = 16

    print("\nConfiguration:")
    print(f"Sample Size: {SAMPLE_SIZE}")
    print(f"Test Size: {TEST_SIZE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")

    # Load data
    df = load_sentiment140("data/sentiment140.csv", sample_size=SAMPLE_SIZE)

    # Basic text cleaning (BERT handles most preprocessing)
    print("\nBasic text cleaning...")
    df["text"] = df["text"].str.replace(r"http\S+", "", regex=True)  # Remove URLs
    df["text"] = df["text"].str.replace(r"@\w+", "", regex=True)  # Remove mentions
    df = df[df["text"].str.len() > 10]  # Remove very short texts

    # Split data
    print("\nSplitting data...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"],
        df["sentiment"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["sentiment"],
    )

    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")

    # Train BERT model
    result = train_bert_model(
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        model_name=MODEL_NAME,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    # Update comparison file
    print("\nUpdating model comparison...")
    comparison_file = "data/model_comparison.csv"

    if os.path.exists(comparison_file):
        comparison_df = pd.read_csv(comparison_file)
    else:
        comparison_df = pd.DataFrame()

    # Add BERT results
    bert_result = pd.DataFrame(
        [
            {
                "Model": "BERT (DistilBERT)",
                "Accuracy": f"{result['accuracy']:.4f}",
                "Accuracy %": f"{result['accuracy']*100:.2f}%",
                "Training Time (s)": f"{result['train_time']:.2f}",
                "Inference Time (ms)": "N/A (batch dependent)",
            }
        ]
    )

    # Append or replace BERT entry
    comparison_df = comparison_df[
        ~comparison_df["Model"].str.contains("BERT", na=False)
    ]
    comparison_df = pd.concat([comparison_df, bert_result], ignore_index=True)
    comparison_df.to_csv(comparison_file, index=False)

    print("Model comparison updated!")

    # Show comparison
    compare_with_traditional_models(val_texts, val_labels)

    print(f"\n{'='*60}")
    print("BERT TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Model saved to: app/models/bert_sentiment")
    print(f"\nBERT Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
    print("\nNext steps:")
    print("1. The app will automatically use BERT model (highest priority)")
    print("2. Test the model: python main.py")
    print("3. Deploy to Azure for production use")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
