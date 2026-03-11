"""
AI Plagiarism Detector & Rewriter - Flask Application

Performance fixes:
  - Models pre-loaded in a background thread at startup
  - /status endpoint so the UI can show real loading progress
  - Server-side sessions (flask-session) — no 4 KB cookie limit
  - Reloader excludes site-packages to prevent infinite restarts
"""

import os
import uuid
import warnings
warnings.filterwarnings('ignore', message='.*sequentially on GPU.*', category=UserWarning)
import threading
from flask import Flask, render_template, request, jsonify, send_file, session
from flask_cors import CORS
from flask_session import Session
from werkzeug.utils import secure_filename
from core.file_parser import extract_text_from_file
from core.text_preprocessor import segment_sentences
from core.plagiarism_detector import PlagiarismDetector
from core.rewrite_engine import RewriteEngine
from core.report_generator import generate_report

# ── App setup ──────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = os.urandom(24)

SESSION_DIR = os.path.join(os.path.dirname(__file__), 'flask_sessions')
os.makedirs(SESSION_DIR, exist_ok=True)

app.config.update(
    SESSION_TYPE            = 'filesystem',
    SESSION_FILE_DIR        = SESSION_DIR,
    SESSION_PERMANENT       = False,
    SESSION_USE_SIGNER      = True,
    SESSION_COOKIE_SAMESITE = 'Lax',
    SESSION_COOKIE_SECURE   = False,
    UPLOAD_FOLDER           = os.path.join(os.path.dirname(__file__), 'uploads'),
    MAX_CONTENT_LENGTH      = 16 * 1024 * 1024,
)

Session(app)
CORS(app, supports_credentials=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

# ── Model state (shared across threads) ───────────────────────────────────────

_models = {
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
    try:
        with _lock:
            _models['detector_status'] = 'loading'
        d = PlagiarismDetector()
        # Warm up: encode a dummy sentence so the model weights are in memory
        d.engine.compute_embeddings(['warmup sentence'])
        with _lock:
            _models['detector']        = d
            _models['detector_ready']  = True
            _models['detector_status'] = 'ready'
        print('[PlagioAI] ✓ Detector ready')
    except Exception as e:
        with _lock:
            _models['detector_status'] = 'error'
            _models['detector_error']  = str(e)
        print(f'[PlagioAI] ✗ Detector failed: {e}')


def _load_rewriter():
    try:
        with _lock:
            _models['rewriter_status'] = 'loading'
        r = RewriteEngine()
        # Warm up: paraphrase a short sentence
        r.rewrite('This is a warm-up sentence.')
        with _lock:
            _models['rewriter']        = r
            _models['rewriter_ready']  = True
            _models['rewriter_status'] = 'ready'
        print('[PlagioAI] ✓ Rewriter ready')
    except Exception as e:
        with _lock:
            _models['rewriter_status'] = 'error'
            _models['rewriter_error']  = str(e)
        print(f'[PlagioAI] ✗ Rewriter failed: {e}')


# Start loading both models in background threads immediately at import time
print('[PlagioAI] Loading models in background…')
threading.Thread(target=_load_detector, daemon=True).start()
threading.Thread(target=_load_rewriter, daemon=True).start()


def get_detector():
    with _lock:
        return _models['detector']


def get_rewriter():
    with _lock:
        return _models['rewriter']


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/checker')
def checker():
    return render_template('checker.html')


@app.route('/status')
def status():
    """Polled by the frontend to show real model-loading progress."""
    with _lock:
        return jsonify({
            'detector_status': _models['detector_status'],
            'rewriter_status': _models['rewriter_status'],
            'detector_error':  _models['detector_error'],
            'rewriter_error':  _models['rewriter_error'],
            'all_ready':       _models['detector_ready'] and _models['rewriter_ready'],
        })


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type. Use PDF, DOCX, or TXT.'}), 400

    filename        = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    filepath        = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    try:
        text = extract_text_from_file(filepath)
        if not text or len(text.strip()) < 50:
            return jsonify({'error': 'Could not extract enough text from the file.'}), 400

        session['original_text'] = text

        preview    = text[:500] + '…' if len(text) > 500 else text
        word_count = len(text.split())
        char_count = len(text)

        return jsonify({
            'success':    True,
            'full_text':  text,
            'preview':    preview,
            'word_count': word_count,
            'char_count': char_count,
            'filename':   filename,
        })

    except Exception as e:
        return jsonify({'error': f'Error processing file: {e}'}), 500


@app.route('/detect', methods=['POST'])
def detect_plagiarism():
    detector = get_detector()
    if detector is None:
        with _lock:
            err = _models['detector_error']
        msg = f'Detector error: {err}' if err else 'Models are still loading. Please wait a moment and try again.'
        return jsonify({'error': msg}), 503

    body = request.get_json(silent=True) or {}
    text = (body.get('text') or '').strip()
    if not text:
        text = (session.get('original_text') or '').strip()
    if not text:
        return jsonify({'error': 'No text to analyse. Upload or paste text first.'}), 400

    try:
        sentences = segment_sentences(text)
        if len(sentences) < 2:
            return jsonify({'error': 'Need at least 2 sentences.'}), 400

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
            err = _models['rewriter_error']
        msg = f'Rewriter error: {err}' if err else 'Rewrite model is still loading. Please wait and try again.'
        return jsonify({'error': msg}), 503

    body      = request.get_json(silent=True) or {}
    threshold = float(body.get('threshold', 0.30))

    sentences = body.get('sentences') or session.get('sentences', [])
    if not sentences:
        return jsonify({'error': 'No sentences to rewrite. Run detection first.'}), 400

    stored      = session.get('detection_results', {})
    sent_scores = stored.get('sentence_scores', [])
    overall_in  = float(stored.get('overall_score', 0))

    if not sent_scores:
        raw         = body.get('results') or {}
        sent_scores = raw.get('sentence_scores', [])
        overall_in  = float(raw.get('overall_score', 0))

    # Adaptive threshold
    if sent_scores:
        all_scores = [float(s.get('combined_score', 0)) for s in sent_scores]
        if max(all_scores) < threshold:
            sorted_scores = sorted(all_scores, reverse=True)
            top_n         = max(1, len(sorted_scores) // 3)
            threshold     = max(sorted_scores[top_n - 1], 0.05)

    try:
        # Collect per-sentence scores
        scores_list = [
            float((sent_scores[i] if i < len(sent_scores) else {}).get('combined_score', 0))
            for i in range(len(sentences))
        ]

        # Identify which sentences need rewriting
        flagged_idx   = [i for i, sc in enumerate(scores_list) if sc >= threshold]
        flagged_sents = [sentences[i] for i in flagged_idx]

        # Single batched GPU/CPU call — no sequential-pipeline warning
        batch_results = rewriter.rewrite_batch(flagged_sents) if flagged_sents else []
        rewrite_map   = dict(zip(flagged_idx, batch_results))

        rewritten_sentences = []
        for i, sentence in enumerate(sentences):
            score = scores_list[i]
            if i in rewrite_map:
                rewritten_sentences.append({
                    'original': sentence, 'rewritten': rewrite_map[i],
                    'was_rewritten': True, 'original_score': score,
                })
            else:
                rewritten_sentences.append({
                    'original': sentence, 'rewritten': sentence,
                    'was_rewritten': False, 'original_score': score,
                })

        new_text    = ' '.join(s['rewritten'] for s in rewritten_sentences)
        new_sents   = segment_sentences(new_text)
        new_results = get_detector().analyze(new_sents)
        new_score   = new_results['overall_score']

        session['rewritten_sentences'] = rewritten_sentences
        session['new_score']           = new_score

        return jsonify({
            'success':             True,
            'rewritten_sentences': rewritten_sentences,
            'original_score':      overall_in,
            'new_score':           new_score,
            'rewritten_count':     sum(1 for s in rewritten_sentences if s['was_rewritten']),
        })

    except Exception as e:
        return jsonify({'error': f'Rewriting error: {e}'}), 500


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


if __name__ == '__main__':
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        exclude_patterns=[
            '*site-packages*',
            '*torch*',
            '*transformers*',
            '*sentence_transformers*',
        ],
    )