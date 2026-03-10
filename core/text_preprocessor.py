"""
Text Preprocessor Module
Handles text cleaning, normalization, and sentence segmentation.
"""

import re
import string


def preprocess_text(text: str) -> str:
    """Clean and normalize text for analysis."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep sentence punctuation
    text = re.sub(r'[^\w\s.,!?;:\'\"-]', ' ', text)
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    return text.strip()


def segment_sentences(text: str) -> list:
    """Split text into individual sentences using NLTK."""
    try:
        import nltk
        # Download required NLTK data if not present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)

        from nltk.tokenize import sent_tokenize
        cleaned = preprocess_text(text)
        sentences = sent_tokenize(cleaned)
        # Filter out very short sentences (< 10 chars)
        sentences = [s.strip() for s in sentences if len(s.strip()) >= 10]
        return sentences

    except Exception:
        # Fallback: simple regex sentence splitter
        return _fallback_sentence_split(text)


def _fallback_sentence_split(text: str) -> list:
    """Fallback sentence splitter using regex."""
    cleaned = preprocess_text(text)
    sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    return [s.strip() for s in sentences if len(s.strip()) >= 10]


def tokenize_words(text: str) -> list:
    """Tokenize text into words, removing stopwords."""
    try:
        import nltk
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
    except Exception:
        stop_words = set()

    words = re.findall(r'\b[a-z]+\b', text.lower())
    return [w for w in words if w not in stop_words and len(w) > 2]


def get_ngrams(tokens: list, n: int) -> list:
    """Generate n-grams from a list of tokens."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
