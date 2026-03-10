# PlagioAI — AI Plagiarism Detector & Rewriter

A production-ready web application that detects plagiarism in uploaded documents
and rewrites flagged sentences using AI.

---

## Quick Start

### 1. Install dependencies

```bash
cd plagiarism_ai_web
pip install -r requirements.txt
```

> **GPU acceleration**: if you have an NVIDIA GPU with CUDA, replace the `torch`
> line in `requirements.txt` with the appropriate CUDA wheel from
> https://pytorch.org/get-started/locally/

### 2. Download NLTK data (one-time)

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

### 3. Run the server

```bash
python app.py
```

Open your browser at **http://localhost:5000**

---

## Workflow

```
Upload File  →  Check Plagiarism  →  Rewrite Content  →  Download Report
```

1. **Upload** — PDF, DOCX, or TXT (up to 16 MB), or paste text directly.
2. **Detect** — Each sentence is scored with three metrics:
   - Semantic similarity (SentenceTransformer `all-MiniLM-L6-v2`)
   - N-gram overlap (bigram + trigram Dice coefficient)
   - Jaccard word-overlap index
   - **Combined score** = 0.5 × semantic + 0.3 × n-gram + 0.2 × Jaccard
3. **Rewrite** — Sentences scoring ≥ 70% are paraphrased by `google/flan-t5-base`.
4. **Report** — Export a full report as PDF or DOCX.

---

## Project Structure

```
plagiarism_ai_web/
├── app.py                    # Flask application & routes
├── requirements.txt
├── README.md
├── uploads/                  # Temporary file storage
├── core/
│   ├── file_parser.py        # PDF / DOCX / TXT extraction
│   ├── text_preprocessor.py  # Cleaning, tokenisation, sentence splitting
│   ├── similarity_engine.py  # Semantic + n-gram + Jaccard metrics
│   ├── plagiarism_detector.py# Orchestration & scoring
│   ├── rewrite_engine.py     # FLAN-T5 paraphrasing
│   └── report_generator.py   # PDF & DOCX report creation
├── templates/
│   ├── index.html            # Landing page
│   └── checker.html          # Main app (4-step flow)
└── static/
    ├── css/style.css
    └── js/script.js
```

---

## Configuration

| Setting | Default | Notes |
|---------|---------|-------|
| Rewrite threshold | 70% | Sentences above this are rewritten |
| Semantic weight | 0.50 | Adjust in `similarity_engine.py` |
| N-gram weight | 0.30 | Adjust in `similarity_engine.py` |
| Jaccard weight | 0.20 | Adjust in `similarity_engine.py` |
| Model (embedding) | `all-MiniLM-L6-v2` | Can use larger models |
| Model (rewriting) | `google/flan-t5-base` | Can use `flan-t5-large` for quality |
| GPU | Auto-detected | Falls back to CPU automatically |

---

## Notes

- The first run will download model weights (~100 MB for MiniLM, ~250 MB for FLAN-T5).
- GPU acceleration is used automatically when CUDA is available.
- Large documents (>500 sentences) may take a few minutes on CPU.
"# plagiarism_detector" 
