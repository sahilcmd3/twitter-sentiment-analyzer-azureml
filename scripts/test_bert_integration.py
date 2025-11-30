import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_dependencies():
    """Test if BERT dependencies are installed"""
    print("=" * 60)
    print("Testing BERT Dependencies")
    print("=" * 60)

    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("PyTorch not installed")
        print("Install: pip install torch")
        return False

    try:
        import transformers

        print(f"Transformers: {transformers.__version__}")
    except ImportError:
        print("Transformers not installed")
        print("Install: pip install transformers")
        return False

    return True


def test_sentiment_analyzer():
    """Test sentiment analyzer with different models"""
    print("\n" + "=" * 60)
    print("Testing Sentiment Analyzer")
    print("=" * 60)

    from app.core.sentiment_analyzer import SentimentAnalyzer

    test_texts = [
        "I absolutely love this product! It's amazing!",
        "This is terrible. Worst experience ever.",
        "It's okay, nothing special.",
    ]

    # Test with auto mode (will use BERT if available)
    print("\Testing with AUTO mode (prefers BERT):")
    analyzer_auto = SentimentAnalyzer(use_ml_model=True, model_type="auto")
    model_info = analyzer_auto.get_model_info()
    print(f"Active Model: {model_info['active_model']}")
    print(f"BERT Loaded: {model_info['bert_loaded']}")
    print(f"Traditional ML Loaded: {model_info['traditional_ml_loaded']}")

    print("\nTest Results:")
    for text in test_texts:
        result = analyzer_auto.analyze(text)
        print(f"Text: {text[:50]}...")
        print(
            f"→ Sentiment: {result['sentiment']} (confidence: {result['confidence']:.3f}, method: {result['method']})"
        )

    # Test with traditional ML
    print("\Testing with TRADITIONAL ML:")
    analyzer_trad = SentimentAnalyzer(use_ml_model=True, model_type="traditional")
    model_info_trad = analyzer_trad.get_model_info()
    print(f"Active Model: {model_info_trad['active_model']}")

    if analyzer_trad.model:
        result = analyzer_trad.analyze(test_texts[0])
        print(f"Sample result: {result['sentiment']} ({result['method']})")
    else:
        print("Traditional ML model not found")

    # Test with BERT specifically
    print("\Testing with BERT (if available):")
    analyzer_bert = SentimentAnalyzer(use_ml_model=True, model_type="bert")
    model_info_bert = analyzer_bert.get_model_info()

    if analyzer_bert.bert_model:
        print("BERT model loaded!")
        result = analyzer_bert.analyze(test_texts[0])
        print(
            f"Sample result: {result['sentiment']} (confidence: {result['confidence']:.3f})"
        )
    else:
        print("BERT model not found")
        print("Train BERT model: python scripts/train_bert.py")

    # Test with TextBlob
    print("\nTesting with TEXTBLOB (fallback):")
    analyzer_tb = SentimentAnalyzer(use_ml_model=False)
    result = analyzer_tb.analyze(test_texts[0])
    print(f"Result: {result['sentiment']} ({result['method']})")


def check_model_files():
    """Check which model files exist"""
    print("\n" + "=" * 60)
    print("Checking Model Files")
    print("=" * 60)

    models_to_check = [
        ("BERT Model", "app/models/bert_sentiment"),
        ("Traditional ML Model", "app/models/best_model.pkl"),
        ("Vectorizer", "app/models/vectorizer.pkl"),
        ("Model Comparison", "data/model_comparison.csv"),
    ]

    for name, path in models_to_check:
        if os.path.exists(path):
            if os.path.isdir(path):
                size = sum(
                    os.path.getsize(os.path.join(path, f))
                    for f in os.listdir(path)
                    if os.path.isfile(os.path.join(path, f))
                )
                print(f"✅ {name}: {size / (1024*1024):.1f} MB")
            else:
                size = os.path.getsize(path)
                print(f"✅ {name}: {size / (1024*1024):.1f} MB")
        else:
            print(f"❌ {name}: Not found")


def main():
    print(
        """
    ╔══════════════════════════════════════════════════════════╗
    ║   BERT Model Integration Test                            ║
    ║   Author: sahil & aryan                                  ║
    ╚══════════════════════════════════════════════════════════╝
    """
    )

    # Test dependencies
    deps_ok = test_dependencies()

    # Check model files
    check_model_files()

    # Test sentiment analyzer
    test_sentiment_analyzer()

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    if deps_ok:
        print("✅ All dependencies installed")
        print("\nNext steps:")
        print("1. To train BERT: python scripts/train_bert.py")
        print("2. To train traditional models: python scripts/train_models.py")
        print("3. To run the app: python main.py")
    else:
        print("Some dependencies missing")
        print("\nInstall missing packages:")
        print("pip install torch transformers sentencepiece")

    print("=" * 60)


if __name__ == "__main__":
    main()
