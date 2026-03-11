"""
Plagiarism Detector Module
Orchestrates similarity analysis across all sentences.

Performance fixes:
  - Cosine similarity: single (N×N) matrix multiply instead of O(N²) Python loop
  - N-gram / Jaccard: tokens pre-computed once per sentence, reused N times
  - argmax on cosine row finds best semantic match in one numpy call
  - Overall score and risk counts computed with numpy vectorised ops
"""

import numpy as np
from core.similarity_engine import SimilarityEngine


class PlagiarismDetector:
    HIGH_RISK_THRESHOLD   = 0.70
    MEDIUM_RISK_THRESHOLD = 0.40

    def __init__(self):
        self.engine = SimilarityEngine()

    # ── Main entry point ───────────────────────────────────────────────────────

    def analyze(self, sentences: list) -> dict:
        """
        Analyse a list of sentences and return per-sentence scores
        plus an overall plagiarism score.
        """
        if len(sentences) < 2:
            return self._empty_result(sentences)

        n = len(sentences)

        # ── Step 1: batch-encode all sentences (one GPU call) ─────────────────
        embeddings = self.engine.compute_embeddings(sentences)   # (N, D) normalised

        # ── Step 2: full cosine-similarity matrix (one BLAS call) ─────────────
        sim_matrix = self.engine.compute_all_cosine_similarities(embeddings)  # (N, N)

        # ── Step 3: pre-tokenise all sentences once ───────────────────────────
        token_cache = self.engine._token_cache(sentences)

        # ── Step 4: per-sentence scoring ──────────────────────────────────────
        sentence_scores = []
        for i in range(n):
            # Best semantic match index (diagonal already set to -1)
            best_j        = int(np.argmax(sim_matrix[i]))
            best_semantic = float(sim_matrix[i, best_j])

            # N-gram + Jaccard against the best semantic match
            best_ngram   = self.engine.combined_ngram_from_tokens(
                token_cache[i], token_cache[best_j]
            )
            best_jaccard = self.engine.jaccard_from_tokens(
                token_cache[i], token_cache[best_j]
            )

            combined  = self.engine.combined_score(best_semantic, best_ngram, best_jaccard)
            risk      = self._risk_level(combined)

            sentence_scores.append({
                'sentence':            sentences[i],
                'semantic_score':      round(best_semantic, 4),
                'ngram_score':         round(best_ngram,    4),
                'jaccard_score':       round(best_jaccard,  4),
                'combined_score':      round(combined,      4),
                'risk_level':          risk,
                'most_similar_index':  best_j,
            })

        # ── Step 5: aggregate ─────────────────────────────────────────────────
        combined_arr  = np.array([s['combined_score'] for s in sentence_scores])
        overall_score = float(combined_arr.mean()) * 100
        risk_levels   = np.array([s['risk_level'] for s in sentence_scores])

        return {
            'sentence_scores':   sentence_scores,
            'overall_score':     round(overall_score, 2),
            'high_risk_count':   int((risk_levels == 'high').sum()),
            'medium_risk_count': int((risk_levels == 'medium').sum()),
            'low_risk_count':    int((risk_levels == 'low').sum()),
        }

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _risk_level(self, score: float) -> str:
        if score >= self.HIGH_RISK_THRESHOLD:
            return 'high'
        if score >= self.MEDIUM_RISK_THRESHOLD:
            return 'medium'
        return 'low'

    def _empty_result(self, sentences: list) -> dict:
        scores = [{
            'sentence':           s,
            'semantic_score':     0.0,
            'ngram_score':        0.0,
            'jaccard_score':      0.0,
            'combined_score':     0.0,
            'risk_level':         'low',
            'most_similar_index': None,
        } for s in sentences]
        return {
            'sentence_scores':   scores,
            'overall_score':     0.0,
            'high_risk_count':   0,
            'medium_risk_count': 0,
            'low_risk_count':    len(sentences),
        }