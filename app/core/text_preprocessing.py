import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK data is available - download if missing
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)


class TextPreprocessor:
    """
    Text preprocessing class for cleaning and tokenizing tweets.
    
    This class handles all text cleaning steps required before sentiment analysis:
    - URL removal
    - User mention removal
    - Hashtag processing
    - Special character removal
    - Stopword filtering
    - Tokenization
    """
    
    def __init__(self):
        """Initialize with English stopwords from NLTK."""
        # Load English stopwords (common words like 'the', 'is', 'at', etc.)
        self.stop_words = set(stopwords.words("english"))

    def clean_text(self, text: str) -> str:
        """
        Clean tweet text by removing noise and normalizing format.
        
        Cleaning steps:
        1. Convert to lowercase for consistency
        2. Remove URLs (http://, https://, www.)
        3. Remove user mentions (@username)
        4. Extract words from hashtags (#word -> word)
        5. Remove special characters and numbers
        6. Remove extra whitespace
        
        Args:
            text (str): Raw tweet text
            
        Returns:
            str: Cleaned text ready for tokenization
            
        Example:
            Input: "Check out @user's awesome post! #Python #AI https://example.com"
            Output: "check out awesome post python ai"
        """
        # Convert to lowercase for uniformity
        text = text.lower()

        # Remove URLs - matches http://, https://, www. patterns
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        # Remove user mentions (@username) - not useful for sentiment
        text = re.sub(r"@\w+", "", text)

        # Remove hashtag symbol but keep the word (hashtag content is useful)
        text = re.sub(r"#(\w+)", r"\1", text)

        # Remove special characters, numbers, punctuation - keep only letters and spaces
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # Remove extra spaces and normalize whitespace
        text = " ".join(text.split())

        return text

    def preprocess(self, text: str) -> str:
        """
        Full preprocessing pipeline for a single text.
        
        Pipeline:
        1. Clean text (URLs, mentions, special chars)
        2. Tokenize into words
        3. Remove stopwords (common words with little sentiment value)
        4. Filter out very short words (<= 2 characters)
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            str: Preprocessed text ready for sentiment analysis
            
        Example:
            Input: "This is an amazing product! I love it! 😍"
            Output: "amazing product love"
        """
        # Step 1: Clean the text
        cleaned = self.clean_text(text)

        # Step 2: Tokenize into individual words
        tokens = word_tokenize(cleaned)

        # Step 3 & 4: Remove stopwords and short tokens
        # Keep only meaningful words longer than 2 characters
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]

        # Join tokens back into a string
        return " ".join(tokens)

    def preprocess_dataframe(self, df):
        """
        Preprocess all tweets in a DataFrame.
        
        Applies the full preprocessing pipeline to each tweet in the 'text' column
        and adds results to a new 'processed_text' column.
        
        Args:
            df (pd.DataFrame): DataFrame with 'text' column containing raw tweets
            
        Returns:
            pd.DataFrame: Original DataFrame with added 'processed_text' column
        """
        # Apply preprocessing to each tweet
        df["processed_text"] = df["text"].apply(self.preprocess)
        return df
