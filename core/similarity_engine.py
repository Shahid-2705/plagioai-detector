"""
Similarity Engine Module
Combines semantic, n-gram, and Jaccard similarity metrics.
"""

import numpy as np
from core.text_preprocessor import tokenize_words, get_ngrams


class SimilarityEngine:
    """
    Computes multiple similarity metrics between sentence pairs.
    Combines:
      - Semantic (SentenceTransformer cosine similarity)
      - N-gram overlap (bigram + trigram)
      - Jaccard similarity (word overlap)
    """

    SEMANTIC_WEIGHT = 0.5
    NGRAM_WEIGHT = 0.3
    JACCARD_WEIGHT = 0.2

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Run: pip install sentence-transformers"
                )
        return self._model

    # ------------------------------------------------------------------
    # Semantic similarity
    # ------------------------------------------------------------------

    def compute_embeddings(self, sentences: list) -> np.ndarray:
        """Return a 2-D array of sentence embeddings."""
        model = self._load_model()
        return model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)

    def semantic_similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Cosine similarity between two embedding vectors."""
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(emb_a, emb_b) / (norm_a * norm_b))

    # ------------------------------------------------------------------
    # N-gram similarity
    # ------------------------------------------------------------------

    def ngram_similarity(self, text_a: str, text_b: str, n: int = 2) -> float:
        """
        Dice coefficient for n-gram overlap between two texts.
        Returns a value in [0, 1].
        """
        tokens_a = tokenize_words(text_a)
        tokens_b = tokenize_words(text_b)
        if not tokens_a or not tokens_b:
            return 0.0
        grams_a = set(get_ngrams(tokens_a, n))
        grams_b = set(get_ngrams(tokens_b, n))
        if not grams_a and not grams_b:
            return 0.0
        intersection = grams_a & grams_b
        return 2 * len(intersection) / (len(grams_a) + len(grams_b))

    def combined_ngram_similarity(self, text_a: str, text_b: str) -> float:
        """Average of bigram and trigram similarity."""
        bigram = self.ngram_similarity(text_a, text_b, n=2)
        trigram = self.ngram_similarity(text_a, text_b, n=3)
        return (bigram + trigram) / 2

    # ------------------------------------------------------------------
    # Jaccard similarity
    # ------------------------------------------------------------------

    def jaccard_similarity(self, text_a: str, text_b: str) -> float:
        """Jaccard index on word-level sets."""
        set_a = set(tokenize_words(text_a))
        set_b = set(tokenize_words(text_b))
        if not set_a and not set_b:
            return 0.0
        union = set_a | set_b
        if not union:
            return 0.0
        return len(set_a & set_b) / len(union)

    # ------------------------------------------------------------------
    # Combined score
    # ------------------------------------------------------------------

    def combined_score(
        self,
        semantic: float,
        ngram: float,
        jaccard: float,
    ) -> float:
        """Weighted combination of the three metrics."""
        score = (
            self.SEMANTIC_WEIGHT * semantic
            + self.NGRAM_WEIGHT * ngram
            + self.JACCARD_WEIGHT * jaccard
        )
        return min(max(score, 0.0), 1.0)
