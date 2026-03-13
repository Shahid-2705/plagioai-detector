# PlagioAI — AI Plagiarism Detector & Rewriter

PlagioAI is a full-stack AI web application that detects plagiarism in documents and rewrites flagged sentences using modern NLP models.

The system analyzes uploaded content using semantic similarity and statistical overlap metrics, then automatically paraphrases high-similarity sentences using a transformer-based language model.

---

# Features

* Upload **PDF, DOCX, or TXT** files
* Paste raw text directly for analysis
* Multi-metric plagiarism detection
* AI-powered sentence rewriting
* Interactive results visualization
* Downloadable plagiarism reports
* Export fully rewritten documents
* GPU acceleration support
* Modern responsive UI

---

# Live Architecture

The application is deployed using a modern cloud stack.

Frontend
Powered by **Vercel**

Backend API
Powered by **Render**

Database & Storage
Powered by **Supabase**

```
User
 ↓
Vercel (Frontend UI)
 ↓
Render (Flask AI API)
 ↓
Supabase (Database & Storage)
```

---

# Workflow

```
Upload Document
      ↓
Detect Plagiarism
      ↓
Rewrite Flagged Sentences
      ↓
Generate Report
```

### 1 Upload

Users can upload files or paste text directly.

Supported formats

* PDF
* DOCX
* TXT

Maximum size

```
16 MB
```

---

### 2 Detect

Each sentence is analyzed using three metrics.

**Semantic Similarity**

```
SentenceTransformer model
all-MiniLM-L6-v2
```

**N-gram Overlap**

```
Bigram + Trigram Dice coefficient
```

**Jaccard Similarity**

```
Word overlap index
```

Final plagiarism score

```
Combined Score =
0.5 × Semantic
+ 0.3 × N-gram
+ 0.2 × Jaccard
```

---

### 3 Rewrite

Sentences exceeding the plagiarism threshold are automatically paraphrased using

```
google/flan-t5-base
```

Adaptive rewriting threshold

```
< 8 words  → 40%
8-20 words → 55%
> 20 words → 70%
```

This prevents unnecessary rewriting of short technical phrases.

---

### 4 Report

Users can export results as

```
PDF report
DOCX report
Full rewritten document
```

Reports include

* Original plagiarism score
* New plagiarism score
* Sentence-level analysis
* Rewritten sentences
* Overall improvement summary

---

# Project Structure

```
plagiarism_ai_web
│
├── app.py
├── requirements.txt
├── render.yaml
├── vercel.json
├── README.md
│
├── uploads
│
├── core
│   ├── file_parser.py
│   ├── text_preprocessor.py
│   ├── similarity_engine.py
│   ├── plagiarism_detector.py
│   ├── rewrite_engine.py
│   └── report_generator.py
│
├── templates
│   ├── index.html
│   └── checker.html
│
└── static
    ├── css
    │   └── style.css
    └── js
        └── script.js
```

---

# Local Development

## 1 Install dependencies

```
pip install -r requirements.txt
```

---

## 2 Download NLTK resources

```
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---

## 3 Run the server

```
python app.py
```

Open the app in your browser

```
http://localhost:5000
```

---

# GPU Acceleration

If CUDA is available, PyTorch automatically runs models on GPU.

To enable CUDA support install the correct PyTorch build

```
https://pytorch.org/get-started/locally
```

Example

```
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

# Configuration

| Parameter         | Default          | Location             |
| ----------------- | ---------------- | -------------------- |
| Rewrite threshold | 70%              | rewrite_engine.py    |
| Semantic weight   | 0.50             | similarity_engine.py |
| N-gram weight     | 0.30             | similarity_engine.py |
| Jaccard weight    | 0.20             | similarity_engine.py |
| Embedding model   | all-MiniLM-L6-v2 | similarity_engine.py |
| Rewrite model     | flan-t5-base     | rewrite_engine.py    |

---

# Performance Notes

First startup downloads AI model weights

```
Sentence Transformer  ~100MB
FLAN-T5 model         ~250MB
```

Initial load time

```
1–2 minutes
```

Subsequent runs are significantly faster due to caching.

---

# Deployment

Frontend

```
Vercel
```

Backend

```
Render
```

Database

```
Supabase
```

---

# Future Improvements

* Vector plagiarism detection using embeddings
* Supabase pgvector semantic search
* Multi-document plagiarism comparison
* Real-time collaboration
* User authentication
* History of plagiarism reports

---

# License

MIT License

---

# Author

Mohamed Shahid
