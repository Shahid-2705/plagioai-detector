"""
Similarity Engine Module
Combines semantic, n-gram, and Jaccard similarity metrics.

Performance fixes:
  - Cosine similarity computed via vectorised matrix multiply (no Python loop)
  - N-gram / Jaccard tokens cached per sentence (computed once, reused N times)
  - SentenceTransformer runs on GPU when available
"""

import numpy as np
from core.text_preprocessor import tokenize_words, get_ngrams


class SimilarityEngine:
    SEMANTIC_WEIGHT = 0.5
    NGRAM_WEIGHT    = 0.3
    JACCARD_WEIGHT  = 0.2

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self._model     = None

    # ── Model loader ───────────────────────────────────────────────────────────

    def _load_model(self):
        if self._model is None:
            try:
                import torch
                from sentence_transformers import SentenceTransformer
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self._model = SentenceTransformer(self.model_name, device=device)
            except ImportError:
                raise ImportError(
                    'sentence-transformers is required. '
                    'Run: pip install sentence-transformers'
                )
        return self._model

    # ── Semantic similarity ────────────────────────────────────────────────────

    def compute_embeddings(self, sentences: list) -> np.ndarray:
        """Batch-encode all sentences and return L2-normalised embeddings."""
        model = self._load_model()
        embs  = model.encode(
            sentences,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=64,               # GPU can handle large batches
            normalize_embeddings=True,   # pre-normalise → cosine = dot product
        )
        return embs  # shape (N, D), unit vectors

    def compute_all_cosine_similarities(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Return an (N, N) cosine-similarity matrix in one matrix multiply.
        Since embeddings are L2-normalised, cosine(a,b) = dot(a,b).
        This replaces the O(N²) Python loop with a single BLAS call.
        """
        sim_matrix = embeddings @ embeddings.T           # (N, N)
        np.fill_diagonal(sim_matrix, -1.0)               # exclude self-similarity
        return sim_matrix.clip(-1.0, 1.0)

    def semantic_similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Cosine similarity between two pre-normalised vectors."""
        return float(np.dot(emb_a, emb_b))

    # ── N-gram similarity ──────────────────────────────────────────────────────

    def _token_cache(self, sentences: list) -> list:
        """Pre-tokenise all sentences once and return list of token lists."""
        return [tokenize_words(s) for s in sentences]

    def ngram_similarity_from_tokens(
        self, tokens_a: list, tokens_b: list, n: int = 2
    ) -> float:
        """Dice coefficient on n-gram sets (uses pre-tokenised input)."""
        if not tokens_a or not tokens_b:
            return 0.0
        grams_a = set(get_ngrams(tokens_a, n))
        grams_b = set(get_ngrams(tokens_b, n))
        total   = len(grams_a) + len(grams_b)
        if total == 0:
            return 0.0
        return 2 * len(grams_a & grams_b) / total

    def combined_ngram_from_tokens(self, tokens_a: list, tokens_b: list) -> float:
        """Average bigram + trigram Dice (cached tokens)."""
        return (
            self.ngram_similarity_from_tokens(tokens_a, tokens_b, 2) +
            self.ngram_similarity_from_tokens(tokens_a, tokens_b, 3)
        ) / 2

    def jaccard_from_tokens(self, tokens_a: list, tokens_b: list) -> float:
        """Jaccard on word sets (cached tokens)."""
        set_a = set(tokens_a)
        set_b = set(tokens_b)
        union = set_a | set_b
        if not union:
            return 0.0
        return len(set_a & set_b) / len(union)

    # ── Legacy single-call wrappers (kept for compatibility) ──────────────────

    def ngram_similarity(self, text_a: str, text_b: str, n: int = 2) -> float:
        return self.ngram_similarity_from_tokens(
            tokenize_words(text_a), tokenize_words(text_b), n
        )

    def combined_ngram_similarity(self, text_a: str, text_b: str) -> float:
        ta, tb = tokenize_words(text_a), tokenize_words(text_b)
        return self.combined_ngram_from_tokens(ta, tb)

    def jaccard_similarity(self, text_a: str, text_b: str) -> float:
        return self.jaccard_from_tokens(tokenize_words(text_a), tokenize_words(text_b))

    # ── Combined score ─────────────────────────────────────────────────────────

    def combined_score(self, semantic: float, ngram: float, jaccard: float) -> float:
        score = (
            self.SEMANTIC_WEIGHT * semantic +
            self.NGRAM_WEIGHT    * ngram    +
            self.JACCARD_WEIGHT  * jaccard
        )
        return float(np.clip(score, 0.0, 1.0))