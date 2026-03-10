"""
Rewrite Engine Module
Uses HuggingFace Transformers (FLAN-T5) to paraphrase sentences.
"""

import torch


class RewriteEngine:
    """
    Paraphrases sentences using google/flan-t5-base via HuggingFace.
    Automatically uses GPU if CUDA is available, else CPU.
    """

    MODEL_NAME = 'google/flan-t5-base'
    MAX_INPUT_LENGTH = 512
    MAX_OUTPUT_LENGTH = 256

    def __init__(self):
        self._pipeline = None
        self.device = 0 if torch.cuda.is_available() else -1

    def _load_pipeline(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
                self._pipeline = pipeline(
                    'text2text-generation',
                    model=self.MODEL_NAME,
                    device=self.device,
                    max_length=self.MAX_OUTPUT_LENGTH,
                )
            except ImportError:
                raise ImportError(
                    "transformers and torch are required. "
                    "Run: pip install transformers torch"
                )
        return self._pipeline

    def rewrite(self, sentence: str) -> str:
        """
        Paraphrase a single sentence.
        Returns the rewritten sentence, or the original if rewriting fails.
        """
        if not sentence or len(sentence.strip()) < 5:
            return sentence

        try:
            pipe = self._load_pipeline()
            prompt = (
                f"Paraphrase the following sentence while keeping the meaning:\n{sentence}"
            )
            # Truncate if too long
            prompt = prompt[:self.MAX_INPUT_LENGTH]

            results = pipe(prompt, num_return_sequences=1, do_sample=True, temperature=0.7)
            if results and results[0].get('generated_text'):
                rewritten = results[0]['generated_text'].strip()
                # Sanity check: must be different and non-empty
                if rewritten and rewritten.lower() != sentence.lower():
                    return rewritten
        except Exception:
            pass  # Fall through to return original

        return sentence

    def rewrite_batch(self, sentences: list) -> list:
        """Rewrite a batch of sentences for efficiency."""
        results = []
        for sentence in sentences:
            results.append(self.rewrite(sentence))
        return results
