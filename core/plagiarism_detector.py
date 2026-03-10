"""
Plagiarism Detector Module
Orchestrates similarity analysis across all sentences.
"""

import numpy as np
from core.similarity_engine import SimilarityEngine


class PlagiarismDetector:
    """
    Detects plagiarism by comparing each sentence against all others
    in the corpus. Each sentence's plagiarism score is the maximum
    similarity found with any other sentence.
    """

    HIGH_RISK_THRESHOLD = 0.70
    MEDIUM_RISK_THRESHOLD = 0.40

    def __init__(self):
        self.engine = SimilarityEngine()

    def analyze(self, sentences: list) -> dict:
        """
        Analyse a list of sentences and return per-sentence scores
        plus an overall plagiarism score.

        Returns
        -------
        {
            'sentence_scores': [
                {
                    'sentence': str,
                    'semantic_score': float,
                    'ngram_score': float,
                    'jaccard_score': float,
                    'combined_score': float,
                    'risk_level': 'high' | 'medium' | 'low',
                    'most_similar_index': int | None,
                },
                ...
            ],
            'overall_score': float,   # 0-100
            'high_risk_count': int,
            'medium_risk_count': int,
            'low_risk_count': int,
        }
        """
        if len(sentences) < 2:
            return self._empty_result(sentences)

        # Compute all embeddings at once (batch is faster)
        embeddings = self.engine.compute_embeddings(sentences)

        sentence_scores = []
        for i, sentence in enumerate(sentences):
            best_semantic = 0.0
            best_ngram = 0.0
            best_jaccard = 0.0
            best_idx = None

            for j, other in enumerate(sentences):
                if i == j:
                    continue

                sem = self.engine.semantic_similarity(embeddings[i], embeddings[j])
                ng = self.engine.combined_ngram_similarity(sentence, other)
                jac = self.engine.jaccard_similarity(sentence, other)
                combined = self.engine.combined_score(sem, ng, jac)

                if combined > self.engine.combined_score(best_semantic, best_ngram, best_jaccard):
                    best_semantic = sem
                    best_ngram = ng
                    best_jaccard = jac
                    best_idx = j

            combined_score = self.engine.combined_score(best_semantic, best_ngram, best_jaccard)
            risk_level = self._risk_level(combined_score)

            sentence_scores.append({
                'sentence': sentence,
                'semantic_score': round(best_semantic, 4),
                'ngram_score': round(best_ngram, 4),
                'jaccard_score': round(best_jaccard, 4),
                'combined_score': round(combined_score, 4),
                'risk_level': risk_level,
                'most_similar_index': best_idx,
            })

        overall_score = self._calculate_overall_score(sentence_scores)
        high_risk = sum(1 for s in sentence_scores if s['risk_level'] == 'high')
        medium_risk = sum(1 for s in sentence_scores if s['risk_level'] == 'medium')
        low_risk = sum(1 for s in sentence_scores if s['risk_level'] == 'low')

        return {
            'sentence_scores': sentence_scores,
            'overall_score': round(overall_score, 2),
            'high_risk_count': high_risk,
            'medium_risk_count': medium_risk,
            'low_risk_count': low_risk,
        }

    def _risk_level(self, score: float) -> str:
        if score >= self.HIGH_RISK_THRESHOLD:
            return 'high'
        elif score >= self.MEDIUM_RISK_THRESHOLD:
            return 'medium'
        return 'low'

    def _calculate_overall_score(self, sentence_scores: list) -> float:
        """
        Overall plagiarism score as a percentage.
        Weighted average favouring high-risk sentences.
        """
        if not sentence_scores:
            return 0.0
        scores = [s['combined_score'] for s in sentence_scores]
        return float(np.mean(scores)) * 100

    def _empty_result(self, sentences: list) -> dict:
        scores = [
            {
                'sentence': s,
                'semantic_score': 0.0,
                'ngram_score': 0.0,
                'jaccard_score': 0.0,
                'combined_score': 0.0,
                'risk_level': 'low',
                'most_similar_index': None,
            }
            for s in sentences
        ]
        return {
            'sentence_scores': scores,
            'overall_score': 0.0,
            'high_risk_count': 0,
            'medium_risk_count': 0,
            'low_risk_count': len(sentences),
        }
