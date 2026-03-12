"""
PlagioAI — app.py

Key changes in this revision
─────────────────────────────
  - /status now returns 'error' immediately when loading thread fails,
    not 'loading' forever.  Frontend can show the real error message.
  - _load_detector / _load_rewriter set status='error' + full traceback
    so the banner tells the user exactly what went wrong.
  - Detector falls back gracefully when the heavy model download times out
    (similarity_engine tries all-mpnet-base-v2 first, then all-MiniLM-L6-v2).
  - /detect and /rewrite return 503 with the actual error string on failure.
  - Adaptive thresholding (< 8 words: 0.40, 8-20: 0.55, > 20: 0.70).
  - POST /download_rewritten returns full rewritten doc as DOCX or TXT.
"""

import os
import re
import uuid
import tempfile
import warnings
import threading
import traceback
import logging

warnings.filterwarnings('ignore', message='.*sequentially on GPU.*', category=UserWarning)
logging.basicConfig(level=logging.INFO)

from flask import Flask, render_template, request, jsonify, send_file, session
from flask_cors import CORS
from flask_session import Session
from werkzeug.utils import secure_filename

from core.file_parser          import extract_text_from_file
from core.text_preprocessor    import segment_sentences
from core.plagiarism_detector   import PlagiarismDetector
from core.rewrite_engine        import RewriteEngine
from core.report_generator      import generate_report

# ── GPU info printout ──────────────────────────────────────────────────────────
try:
    import torch
    if torch.cuda.is_available():
        print(f'[PlagioAI] Running on: CUDA — {torch.cuda.get_device_name(0)}')
    else:
        print('[PlagioAI] Running on: CPU')
except Exception:
    print('[PlagioAI] Running on: CPU (torch unavailable)')

# ── Flask setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.urandom(24)

_BASE_DIR    = os.path.dirname(__file__)
_SESSION_DIR = os.path.join(_BASE_DIR, 'flask_sessions')
_UPLOAD_DIR  = os.path.join(_BASE_DIR, 'uploads')
os.makedirs(_SESSION_DIR, exist_ok=True)
os.makedirs(_UPLOAD_DIR,  exist_ok=True)

app.config.update(
    SESSION_TYPE            = 'filesystem',
    SESSION_FILE_DIR        = _SESSION_DIR,
    SESSION_PERMANENT       = False,
    SESSION_USE_SIGNER      = True,
    SESSION_COOKIE_SAMESITE = 'Lax',
    SESSION_COOKIE_SECURE   = False,
    UPLOAD_FOLDER           = _UPLOAD_DIR,
    MAX_CONTENT_LENGTH      = 16 * 1024 * 1024,
)

Session(app)
CORS(app, supports_credentials=True)

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

# ── Non-rewritable sentence patterns ──────────────────────────────────────────
_SKIP_PAT = re.compile(
    r'\b(figure|fig\.|table|tbl\.|references?|bibliography|doi|et\s+al\.?'
    r'|journal|proc\.|proceedings|vol\.|pp\.|isbn|issn|rrn|dept|course\s+code'
    r'|http|https|www\.)\b|10\.\d{4,}|https?://',
    re.IGNORECASE,
)
_NUMS_ONLY = re.compile(
    r'^[\d\s\.\,\:\;\-\+\%\(\)\/\=\u00b0\u03bc\u00b1\u00d7\u00f7]+$'
)


def _is_non_rewritable(s: str) -> bool:
    s = s.strip()
    return len(s) < 20 or bool(_SKIP_PAT.search(s)) or bool(_NUMS_ONLY.match(s))


def _adaptive_threshold(s: str) -> float:
    wc = len(s.split())
    if wc < 8:   return 0.40
    if wc <= 20: return 0.55
    return 0.70


# ── Model registry ─────────────────────────────────────────────────────────────
_models: dict = {
    'detector':        None,
    'rewriter':        None,
    'detector_ready':  False,
    'rewriter_ready':  False,
    'detector_status': 'loading',   # loading | ready | error
    'rewriter_status': 'loading',
    'detector_error':  '',
    'rewriter_error':  '',
}
_lock = threading.Lock()


def _load_detector():
    """Background thread: load the plagiarism detector model."""
    try:
        d = PlagiarismDetector()
        d.engine.compute_embeddings(['warmup'])        # forces model download now
        model_name = d.engine._loaded_model_name or 'embedding model'
        with _lock:
            _models['detector']        = d
            _models['detector_ready']  = True
            _models['detector_status'] = 'ready'
        print(f'[PlagioAI] Detector ready ({model_name})')
    except Exception:
        err = traceback.format_exc()
        with _lock:
            _models['detector_status'] = 'error'
            _models['detector_error']  = err.strip().splitlines()[-1]  # last line is most informative
        print(f'[PlagioAI] Detector FAILED:\n{err}')


def _load_rewriter():
    """Background thread: load the rewriter model."""
    try:
        r = RewriteEngine()
        r.rewrite('Warm-up sentence for the paraphrase model.')
        with _lock:
            _models['rewriter']        = r
            _models['rewriter_ready']  = True
            _models['rewriter_status'] = 'ready'
        print('[PlagioAI] Rewriter ready (flan-t5-large)')
    except Exception:
        err = traceback.format_exc()
        with _lock:
            _models['rewriter_status'] = 'error'
            _models['rewriter_error']  = err.strip().splitlines()[-1]
        print(f'[PlagioAI] Rewriter FAILED:\n{err}')


print('[PlagioAI] Pre-loading models in background...')
threading.Thread(target=_load_detector, daemon=True).start()
threading.Thread(target=_load_rewriter, daemon=True).start()


def get_detector():
    with _lock:
        return _models['detector']

def get_rewriter():
    with _lock:
        return _models['rewriter']

def allowed_file(fn: str) -> bool:
    return '.' in fn and fn.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Document helpers ───────────────────────────────────────────────────────────

def _build_rewritten_text(rewritten_sentences: list) -> str:
    return ' '.join(
        (s.get('rewritten') or s.get('original') or '').strip()
        for s in rewritten_sentences
        if (s.get('rewritten') or s.get('original') or '').strip()
    )


def _save_rewritten_docx(text: str, filepath: str):
    from docx import Document
    doc = Document()
    doc.add_heading('Rewritten Document', 0)
    doc.add_paragraph('Generated by PlagioAI — AI Plagiarism Detector & Rewriter')
    doc.add_paragraph('')
    for para in (text.split('\n\n') if '\n\n' in text else [text]):
        if para.strip():
            doc.add_paragraph(para.strip())
    doc.save(filepath)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/checker')
def checker():
    return render_template('checker.html')


@app.route('/status')
def model_status():
    """
    Polled every 2.5 s by the frontend.
    Returns the real status of both models including any error message.
    """
    with _lock:
        ds = _models['detector_status']
        rs = _models['rewriter_status']
        return jsonify({
            'detector_status': ds,
            'rewriter_status': rs,
            'detector_error':  _models['detector_error'],
            'rewriter_error':  _models['rewriter_error'],
            'all_ready':       ds == 'ready' and rs == 'ready',
            'ready':           ds == 'ready' and rs == 'ready',
        })



@app.route('/debug')
def debug_info():
    """Diagnostic page — shows model load status + last error in browser."""
    with _lock:
        ds   = _models['detector_status']
        rs   = _models['rewriter_status']
        derr = _models['detector_error']
        rerr = _models['rewriter_error']

    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'
    except Exception as e:
        gpu = f'torch error: {e}'

    try:
        import sentence_transformers
        st_version = sentence_transformers.__version__
    except Exception:
        st_version = 'NOT INSTALLED'

    try:
        import transformers
        tr_version = transformers.__version__
    except Exception:
        tr_version = 'NOT INSTALLED'

    lines = [
        f'detector_status : {ds}',
        f'rewriter_status : {rs}',
        f'GPU             : {gpu}',
        f'sentence-transformers: {st_version}',
        f'transformers    : {tr_version}',
        '',
        '--- Detector error (if any) ---',
        derr or '(none)',
        '',
        '--- Rewriter error (if any) ---',
        rerr or '(none)',
    ]
    return '<pre style="font-family:monospace;padding:2rem;">' + '\n'.join(lines) + '</pre>'


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type. Use PDF, DOCX, or TXT.'}), 400

    filename  = secure_filename(file.filename)
    filepath  = os.path.join(app.config['UPLOAD_FOLDER'], f'{uuid.uuid4()}_{filename}')
    file.save(filepath)

    try:
        text = extract_text_from_file(filepath)
        if not text or len(text.strip()) < 50:
            return jsonify({'error': 'Could not extract enough text from the file.'}), 400
        session['original_text'] = text
        return jsonify({
            'success':    True,
            'full_text':  text,
            'preview':    text[:500] + '...' if len(text) > 500 else text,
            'word_count': len(text.split()),
            'char_count': len(text),
            'filename':   filename,
        })
    except Exception as e:
        return jsonify({'error': f'Error processing file: {e}'}), 500


@app.route('/detect', methods=['POST'])
def detect_plagiarism():
    detector = get_detector()
    if detector is None:
        with _lock:
            status = _models['detector_status']
            err    = _models['detector_error']
        if status == 'error':
            return jsonify({'error': f'Detector failed to load: {err}'}), 503
        return jsonify({'error': 'Detector model is still loading — please wait and try again.'}), 503

    body = request.get_json(silent=True) or {}
    text = (body.get('text') or '').strip() or (session.get('original_text') or '').strip()
    if not text:
        return jsonify({'error': 'No text to analyse. Upload or paste text first.'}), 400

    try:
        sentences = segment_sentences(text)
        if len(sentences) < 2:
            return jsonify({'error': 'Need at least 2 sentences for analysis.'}), 400

        results = detector.analyze(sentences)
        session['sentences']         = sentences
        session['detection_results'] = results

        return jsonify({
            'success':           True,
            'sentences':         sentences,
            'results':           results['sentence_scores'],
            'overall_score':     results['overall_score'],
            'high_risk_count':   results['high_risk_count'],
            'medium_risk_count': results['medium_risk_count'],
        })
    except Exception as e:
        return jsonify({'error': f'Detection error: {e}'}), 500


@app.route('/rewrite', methods=['POST'])
def rewrite_content():
    rewriter = get_rewriter()
    if rewriter is None:
        with _lock:
            status = _models['rewriter_status']
            err    = _models['rewriter_error']
        if status == 'error':
            return jsonify({'error': f'Rewriter failed to load: {err}'}), 503
        return jsonify({'error': 'Rewrite model is still loading — please wait and try again.'}), 503

    body      = request.get_json(silent=True) or {}
    sentences = body.get('sentences') or session.get('sentences', [])
    if not sentences:
        return jsonify({'error': 'No sentences found. Run detection first.'}), 400

    stored      = session.get('detection_results', {})
    sent_scores = stored.get('sentence_scores', [])
    overall_in  = float(stored.get('overall_score', 0))

    if not sent_scores:
        raw         = body.get('results') or {}
        sent_scores = raw.get('sentence_scores', [])
        overall_in  = float(raw.get('overall_score', overall_in))

    try:
        scores_list = [
            float((sent_scores[i] if i < len(sent_scores) else {}).get('combined_score', 0))
            for i in range(len(sentences))
        ]

        flagged_idx   = []
        flagged_sents = []
        thresholds    = []

        for i, sentence in enumerate(sentences):
            score     = scores_list[i]
            threshold = _adaptive_threshold(sentence)
            thresholds.append(threshold)
            if score >= threshold and not _is_non_rewritable(sentence):
                flagged_idx.append(i)
                flagged_sents.append(sentence)

        batch_out   = rewriter.rewrite_batch(flagged_sents) if flagged_sents else []
        rewrite_map = dict(zip(flagged_idx, batch_out))

        rewritten_sentences = []
        for i, sentence in enumerate(sentences):
            rewritten_text     = rewrite_map.get(i, sentence)
            actually_rewritten = (
                i in rewrite_map
                and rewritten_text.strip().lower() != sentence.strip().lower()
                and len(rewritten_text.strip()) > 10
            )
            rewritten_sentences.append({
                'original':       sentence,
                'rewritten':      rewritten_text if actually_rewritten else sentence,
                'was_rewritten':  actually_rewritten,
                'original_score': scores_list[i],
                'threshold_used': thresholds[i],
                'non_rewritable': _is_non_rewritable(sentence),
            })

        new_text    = _build_rewritten_text(rewritten_sentences)
        new_sents   = segment_sentences(new_text)
        det         = get_detector()
        new_results = det.analyze(new_sents) if det and len(new_sents) >= 2 else {'overall_score': 0}
        new_score   = new_results.get('overall_score', 0)

        session['rewritten_sentences'] = rewritten_sentences
        session['new_score']           = new_score

        rewritten_count = sum(1 for s in rewritten_sentences if s['was_rewritten'])
        skipped_count   = sum(1 for s in rewritten_sentences
                              if s['non_rewritable'] and s['original_score'] >= s['threshold_used'])

        return jsonify({
            'success':             True,
            'rewritten_sentences': rewritten_sentences,
            'original_score':      overall_in,
            'new_score':           new_score,
            'rewritten_count':     rewritten_count,
            'skipped_count':       skipped_count,
            'total_sentences':     len(sentences),
            'score_reduction':     round(overall_in - new_score, 2),
        })
    except Exception as e:
        return jsonify({'error': f'Rewriting error: {e}'}), 500


@app.route('/download_rewritten', methods=['POST'])
def download_rewritten():
    body = request.get_json(silent=True) or {}
    fmt  = body.get('format', 'docx').lower()

    rewritten_sentences = session.get('rewritten_sentences', [])
    if not rewritten_sentences:
        return jsonify({'error': 'No rewritten document. Complete the rewrite step first.'}), 400

    full_text = _build_rewritten_text(rewritten_sentences)
    tmp_id    = uuid.uuid4()

    try:
        if fmt == 'docx':
            filepath = os.path.join(tempfile.gettempdir(), f'rewritten_{tmp_id}.docx')
            _save_rewritten_docx(full_text, filepath)
            name     = 'rewritten_document.docx'
            mime     = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        else:
            filepath = os.path.join(tempfile.gettempdir(), f'rewritten_{tmp_id}.txt')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('REWRITTEN DOCUMENT\nGenerated by PlagioAI\n' + '=' * 60 + '\n\n')
                f.write(full_text)
            name = 'rewritten_document.txt'
            mime = 'text/plain'

        return send_file(filepath, as_attachment=True, download_name=name, mimetype=mime)
    except Exception as e:
        return jsonify({'error': f'Download error: {e}'}), 500


@app.route('/report', methods=['POST'])
def download_report():
    body        = request.get_json(silent=True) or {}
    format_type = body.get('format', 'pdf')
    orig_score  = float(body.get('original_score', 0))

    rewritten_sentences = session.get('rewritten_sentences', [])
    new_score           = float(session.get('new_score', 0))

    if not rewritten_sentences:
        return jsonify({'error': 'No report data — complete the rewrite step first.'}), 400

    try:
        filepath = generate_report(rewritten_sentences, orig_score, new_score, format_type)
        return send_file(filepath, as_attachment=True,
                         download_name=f'plagiarism_report.{format_type}')
    except Exception as e:
        return jsonify({'error': f'Report error: {e}'}), 500


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(
        debug=True, host='0.0.0.0', port=5000,
        exclude_patterns=[
            '*site-packages*', '*torch*',
            '*transformers*',  '*sentence_transformers*',
        ],
    )