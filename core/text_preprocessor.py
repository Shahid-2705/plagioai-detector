"""
Text Preprocessor Module
Handles text cleaning, normalization, and sentence segmentation.

Performance fixes:
  - NLTK tokenizer + stopwords loaded ONCE at module import (not per call)
  - sent_tokenize cached tokenizer instance reused across requests
  - tokenize_words uses pre-compiled regex and frozen stopword set
"""

import re

# ── Pre-compile regex patterns (compiled once at import) ──────────────────────
_WHITESPACE_RE  = re.compile(r'\s+')
_SPECIAL_RE     = re.compile(r'[^\w\s.,!?;:\'\"-]')
_WORD_RE        = re.compile(r'\b[a-z]+\b')
_SENTENCE_RE    = re.compile(r'(?<=[.!?])\s+')

# ── Load NLTK resources once at module import ─────────────────────────────────
_sent_tokenize  = None
_stop_words     = frozenset()

def _init_nltk():
    global _sent_tokenize, _stop_words
    if _sent_tokenize is not None:
        return  # already loaded

    try:
        import nltk

        for resource, path in [
            ('punkt',     'tokenizers/punkt'),
            ('punkt_tab', 'tokenizers/punkt_tab'),
            ('stopwords', 'corpora/stopwords'),
        ]:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(resource, quiet=True)

        from nltk.tokenize import sent_tokenize as _st
        from nltk.corpus import stopwords
        _sent_tokenize = _st
        _stop_words    = frozenset(stopwords.words('english'))

    except Exception:
        _sent_tokenize = None   # will use regex fallback
        _stop_words    = frozenset()

# Eagerly initialise so the first request doesn't pay the cost
_init_nltk()


# ── Public API ─────────────────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """Clean and normalise text for analysis."""
    text = _WHITESPACE_RE.sub(' ', text)
    text = _SPECIAL_RE.sub(' ', text)
    text = (text
            .replace('\u201c', '"').replace('\u201d', '"')
            .replace('\u2018', "'").replace('\u2019', "'"))
    return text.strip()


def segment_sentences(text: str) -> list:
    """Split text into sentences, returning only those ≥ 10 chars."""
    cleaned = preprocess_text(text)

    if _sent_tokenize is not None:
        sentences = _sent_tokenize(cleaned)
    else:
        sentences = _SENTENCE_RE.split(cleaned)

    return [s.strip() for s in sentences if len(s.strip()) >= 10]


def tokenize_words(text: str) -> list:
    """Lower-case word tokens with stopwords removed."""
    words = _WORD_RE.findall(text.lower())
    return [w for w in words if w not in _stop_words and len(w) > 2]


def get_ngrams(tokens: list, n: int) -> list:
    """Generate n-grams from a token list."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]