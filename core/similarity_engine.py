"""
Similarity Engine
- Primary model : all-mpnet-base-v2  (~420 MB, higher accuracy)
- Fallback model: all-MiniLM-L6-v2  (~90 MB,  fast fallback)
- Cosine similarity via vectorised matrix multiply (no Python loop)
- N-gram / Jaccard tokens cached per sentence
- GPU when available
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

from core.text_preprocessor import tokenize_words, get_ngrams


# Models tried in order; first one that loads wins
_MODEL_CANDIDATES = [
    'all-mpnet-base-v2',
    'all-MiniLM-L6-v2',
]


class SimilarityEngine:
    SEMANTIC_WEIGHT = 0.5
    NGRAM_WEIGHT    = 0.3
    JACCARD_WEIGHT  = 0.2

    def __init__(self, model_name: str = None):
        self.model_name = model_name   # None = auto-select
        self._model     = None
        self._loaded_model_name = None

    # ── Model loader with fallback ─────────────────────────────────────────────

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                'sentence-transformers is required. '
                'Run: pip install sentence-transformers'
            )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        candidates = ([self.model_name] if self.model_name else []) + _MODEL_CANDIDATES

        last_err = None
        for name in candidates:
            try:
                logger.info('Loading embedding model: %s on %s', name, device)
                self._model = SentenceTransformer(name, device=device)
                self._loaded_model_name = name
                print(f'[SimilarityEngine] Loaded: {name} on {device}')
                return self._model
            except Exception as e:
                last_err = e
                logger.warning('Failed to load %s: %s — trying next', name, e)

        raise RuntimeError(
            f'Could not load any embedding model. Last error: {last_err}'
        )

    # ── Embeddings ─────────────────────────────────────────────────────────────

    def compute_embeddings(self, sentences: list) -> np.ndarray:
        """Batch-encode sentences → L2-normalised (N, D) array."""
        model = self._load_model()
        return model.encode(
            sentences,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=64,
            normalize_embeddings=True,
        )

    def compute_all_cosine_similarities(self, embeddings: np.ndarray) -> np.ndarray:
        """Full (N×N) cosine-similarity matrix via one BLAS call."""
        sim = embeddings @ embeddings.T
        np.fill_diagonal(sim, -1.0)
        return sim.clip(-1.0, 1.0)

    def semantic_similarity(self, emb_a, emb_b) -> float:
        return float(np.dot(emb_a, emb_b))

    # ── Token cache ────────────────────────────────────────────────────────────

    def _token_cache(self, sentences: list) -> list:
        return [tokenize_words(s) for s in sentences]

    # ── N-gram similarity (cached tokens) ──────────────────────────────────────

    def ngram_similarity_from_tokens(self, tokens_a, tokens_b, n=2) -> float:
        if not tokens_a or not tokens_b:
            return 0.0
        ga = set(get_ngrams(tokens_a, n))
        gb = set(get_ngrams(tokens_b, n))
        total = len(ga) + len(gb)
        return 2 * len(ga & gb) / total if total else 0.0

    def combined_ngram_from_tokens(self, tokens_a, tokens_b) -> float:
        return (self.ngram_similarity_from_tokens(tokens_a, tokens_b, 2) +
                self.ngram_similarity_from_tokens(tokens_a, tokens_b, 3)) / 2

    def jaccard_from_tokens(self, tokens_a, tokens_b) -> float:
        sa, sb = set(tokens_a), set(tokens_b)
        union = sa | sb
        return len(sa & sb) / len(union) if union else 0.0

    # ── Legacy wrappers ────────────────────────────────────────────────────────

    def ngram_similarity(self, text_a, text_b, n=2) -> float:
        return self.ngram_similarity_from_tokens(
            tokenize_words(text_a), tokenize_words(text_b), n)

    def combined_ngram_similarity(self, text_a, text_b) -> float:
        ta, tb = tokenize_words(text_a), tokenize_words(text_b)
        return self.combined_ngram_from_tokens(ta, tb)

    def jaccard_similarity(self, text_a, text_b) -> float:
        return self.jaccard_from_tokens(tokenize_words(text_a), tokenize_words(text_b))

    # ── Combined score ─────────────────────────────────────────────────────────

    def combined_score(self, semantic, ngram, jaccard) -> float:
        score = (self.SEMANTIC_WEIGHT * semantic +
                 self.NGRAM_WEIGHT    * ngram    +
                 self.JACCARD_WEIGHT  * jaccard)
        return float(np.clip(score, 0.0, 1.0))