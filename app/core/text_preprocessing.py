import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK data is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)


class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))

    def clean_text(self, text: str) -> str:
        """Clean tweet text"""
        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        # Remove user mentions
        text = re.sub(r"@\w+", "", text)

        # Remove hashtags but keep the word
        text = re.sub(r"#(\w+)", r"\1", text)

        # Remove special characters
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # Remove extra spaces
        text = " ".join(text.split())

        return text

    def preprocess(self, text: str) -> str:
        """Full preprocessing pipeline"""
        cleaned = self.clean_text(text)

        # Tokenize
        tokens = word_tokenize(cleaned)

        # Remove stopwords and short tokens
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]

        return " ".join(tokens)

    def preprocess_dataframe(self, df):
        """Preprocess all tweets in dataframe"""
        df["processed_text"] = df["text"].apply(self.preprocess)
        return df
