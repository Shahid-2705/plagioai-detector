"""
Rewrite Engine Module
Uses HuggingFace Transformers (FLAN-T5) to paraphrase sentences.

GPU optimisations applied:
  - Batch-rewrites all sentences in one pipeline call (no sequential warning)
  - torch_dtype=float16 on CUDA (halves VRAM, ~2x faster)
  - truncation=True prevents input-overflow warnings
  - warnings filter suppresses the dataset hint
"""

import warnings
import logging
import torch

logger = logging.getLogger(__name__)


class RewriteEngine:
    """
    Paraphrases sentences using google/flan-t5-base via HuggingFace.
    Automatically uses GPU (float16) if CUDA is available, else CPU (float32).
    """

    MODEL_NAME        = 'google/flan-t5-base'
    MAX_INPUT_TOKENS  = 256   # token limit for the prompt
    MAX_OUTPUT_TOKENS = 200   # token limit for the paraphrase
    BATCH_SIZE        = 8     # sentences processed per GPU forward pass

    def __init__(self):
        self._pipeline  = None
        self._tokenizer = None
        self.use_cuda   = torch.cuda.is_available()
        self.device     = 0 if self.use_cuda else -1
        self.dtype      = torch.float16 if self.use_cuda else torch.float32
        logger.info('RewriteEngine: device=%s dtype=%s', 
                    'cuda:0' if self.use_cuda else 'cpu', self.dtype)

    # ── Internal loader ────────────────────────────────────────────────────────

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        try:
            from transformers import pipeline, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)

            # Suppress the "use a dataset" sequential-GPU hint
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore',
                    message='.*sequentially on GPU.*',
                    category=UserWarning,
                )
                self._pipeline = pipeline(
                    'text2text-generation',
                    model=self.MODEL_NAME,
                    tokenizer=self._tokenizer,
                    device=self.device,
                    torch_dtype=self.dtype,   # float16 on GPU → faster + less VRAM
                    max_new_tokens=self.MAX_OUTPUT_TOKENS,
                    truncation=True,
                )

            logger.info('RewriteEngine pipeline loaded on %s.',
                        'cuda:0' if self.use_cuda else 'cpu')

        except ImportError:
            raise ImportError(
                'transformers and torch are required. '
                'Run: pip install transformers torch'
            )

        return self._pipeline

    # ── Public API ─────────────────────────────────────────────────────────────

    def rewrite(self, sentence: str) -> str:
        """Paraphrase a single sentence (delegates to rewrite_batch)."""
        results = self.rewrite_batch([sentence])
        return results[0] if results else sentence

    def rewrite_batch(self, sentences: list) -> list:
        """
        Batch-paraphrase a list of sentences in one GPU call.
        Returns a list of rewritten strings (same length as input).
        Falls back to the original sentence on any error.
        """
        if not sentences:
            return []

        pipe = self._load_pipeline()

        # Build prompts, keep originals for fallback
        prompts   = []
        originals = []
        for s in sentences:
            s = (s or '').strip()
            originals.append(s)
            if len(s) >= 5:
                prompts.append(
                    f'Paraphrase the following sentence while keeping the meaning:\n{s}'
                )
            else:
                prompts.append(None)   # placeholder — skip short sentences

        results = []
        valid_prompts = [p for p in prompts if p is not None]

        if not valid_prompts:
            return originals

        try:
            # Single batched call — eliminates sequential-GPU warning
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                outputs = pipe(
                    valid_prompts,
                    batch_size=self.BATCH_SIZE,
                    do_sample=True,
                    temperature=0.75,
                    num_return_sequences=1,
                    truncation=True,
                )

            # outputs is a list of lists when multiple sequences returned
            # Flatten: each item → {'generated_text': '...'}
            flat_outputs = []
            for o in outputs:
                if isinstance(o, list):
                    flat_outputs.append(o[0])
                else:
                    flat_outputs.append(o)

            # Re-align with originals (skipping None prompts)
            out_iter = iter(flat_outputs)
            for i, prompt in enumerate(prompts):
                original = originals[i]
                if prompt is None:
                    results.append(original)
                    continue
                try:
                    out  = next(out_iter)
                    text = (out.get('generated_text') or '').strip()
                    # Accept the rewrite only if meaningfully different
                    results.append(text if text and text.lower() != original.lower() else original)
                except StopIteration:
                    results.append(original)

        except Exception as exc:
            logger.warning('Batch rewrite failed (%s), returning originals.', exc)
            results = originals

        return results