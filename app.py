"""
PlagioAI — Advanced Plagiarism Detector & Rewriter
app.py

Key features:
  - Adaptive thresholding per sentence length:
      < 8 words  -> 0.40
      8-20 words -> 0.55
      > 20 words -> 0.70
  - Skips non-rewritable sentences (figures, tables, DOIs, URLs, et al., etc.)
  - Upgraded models: all-mpnet-base-v2 (embeddings) + flan-t5-large (rewriting)
  - POST /download_rewritten -- full rewritten doc as TXT or DOCX
  - GPU auto-detection printed at startup
  - Server-side sessions (flask-session) -- no 4 KB cookie limit
  - Background model pre-loading + /status polling endpoint
"""

import os
import re
import uuid
import tempfile
import warnings
import threading
import logging

warnings.filterwarnings('ignore', message='.*sequentially on GPU.*', category=UserWarning)
logging.basicConfig(level=logging.INFO)

from flask import Flask, render_template, request, jsonify, send_file, session
from flask_cors import CORS
from flask_session import Session
from werkzeug.utils import secure_filename

from core.file_parser         import extract_text_from_file
from core.text_preprocessor   import segment_sentences
from core.plagiarism_detector  import PlagiarismDetector
from core.rewrite_engine       import RewriteEngine
from core.report_generator     import generate_report

# -- GPU detection printout ---------------------------------------------------
try:
    import torch
    if torch.cuda.is_available():
        print(f'[PlagioAI] Running on: CUDA -- {torch.cuda.get_device_name(0)}')
    else:
        print('[PlagioAI] Running on: CPU')
except Exception:
    print('[PlagioAI] Running on: CPU (torch not available)')

# -- Flask setup --------------------------------------------------------------
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

# -- Non-rewritable sentence patterns ----------------------------------------
_SKIP_PATTERN = re.compile(
    r'\b(figure|fig\.|table|tbl\.|references?|bibliography|doi|et\s+al\.?'
    r'|journal|proc\.|proceedings|vol\.|pp\.|isbn|issn|rrn|dept|course\s+code'
    r'|http|https|www\.)\b'
    r'|10\.\d{4,}'
    r'|https?://',
    re.IGNORECASE,
)
_NUMBERS_ONLY = re.compile(r'^[\d\s\.\,\:\;\-\+\%\(\)\/\=\u00b0\u03bc\u03b1\u03b2\u03b3\u00b1\u00d7\u00f7]+$')


def _is_non_rewritable(sentence):
    s = sentence.strip()
    if len(s) < 20:
        return True
    if _SKIP_PATTERN.search(s):
        return True
    if _NUMBERS_ONLY.match(s):
        return True
    return False


def _adaptive_threshold(sentence):
    """
    Adaptive threshold based on sentence word count:
      < 8 words  -> 0.40
      8-20 words -> 0.55
      > 20 words -> 0.70
    """
    word_count = len(sentence.split())
    if word_count < 8:
        return 0.40
    elif word_count <= 20:
        return 0.55
    else:
        return 0.70


# -- Model registry -----------------------------------------------------------
_models = {
    'detector':        None,
    'rewriter':        None,
    'detector_ready':  False,
    'rewriter_ready':  False,
    'detector_status': 'loading',
    'rewriter_status': 'loading',
    'detector_error':  '',
    'rewriter_error':  '',
}
_lock = threading.Lock()


def _load_detector():
    try:
        d = PlagiarismDetector()
        d.engine.compute_embeddings(['warmup sentence'])
        with _lock:
            _models['detector']        = d
            _models['detector_ready']  = True
            _models['detector_status'] = 'ready'
        print('[PlagioAI] Detector ready (all-mpnet-base-v2)')
    except Exception as e:
        with _lock:
            _models['detector_status'] = 'error'
            _models['detector_error']  = str(e)
        print(f'[PlagioAI] Detector failed: {e}')


def _load_rewriter():
    try:
        r = RewriteEngine()
        r.rewrite('This is a warm-up sentence for the paraphrase model.')
        with _lock:
            _models['rewriter']        = r
            _models['rewriter_ready']  = True
            _models['rewriter_status'] = 'ready'
        print('[PlagioAI] Rewriter ready (flan-t5-large)')
    except Exception as e:
        with _lock:
            _models['rewriter_status'] = 'error'
            _models['rewriter_error']  = str(e)
        print(f'[PlagioAI] Rewriter failed: {e}')


print('[PlagioAI] Pre-loading models in background...')
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


# -- Document helpers ---------------------------------------------------------

def _build_rewritten_text(rewritten_sentences):
    parts = []
    for s in rewritten_sentences:
        text = (s.get('rewritten') or s.get('original') or '').strip()
        if text:
            parts.append(text)
    return ' '.join(parts)


def _save_rewritten_docx(text, filepath):
    from docx import Document
    doc = Document()
    doc.add_heading('Rewritten Document', 0)
    doc.add_paragraph('Generated by PlagioAI -- AI Plagiarism Detector & Rewriter')
    doc.add_paragraph('')
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        paragraphs = [text]
    for para in paragraphs:
        doc.add_paragraph(para)
    doc.save(filepath)


# -- Routes -------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/checker')
def checker():
    return render_template('checker.html')


@app.route('/status')
def model_status():
    with _lock:
        return jsonify({
            'detector_status': _models['detector_status'],
            'rewriter_status': _models['rewriter_status'],
            'detector_error':  _models['detector_error'],
            'rewriter_error':  _models['rewriter_error'],
            'all_ready':       _models['detector_ready'] and _models['rewriter_ready'],
            'ready':           _models['detector_ready'] and _models['rewriter_ready'],
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
    unique_filename = f'{uuid.uuid4()}_{filename}'
    filepath        = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    try:
        text = extract_text_from_file(filepath)
        if not text or len(text.strip()) < 50:
            return jsonify({'error': 'Could not extract enough text from the file.'}), 400

        session['original_text'] = text
        preview    = text[:500] + '...' if len(text) > 500 else text
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
        msg = (f'Detector error: {err}' if err
               else 'Detector model is still loading -- please wait and try again.')
        return jsonify({'error': msg}), 503

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
            err = _models['rewriter_error']
        msg = (f'Rewriter error: {err}' if err
               else 'Rewrite model is still loading -- please wait and try again.')
        return jsonify({'error': msg}), 503

    body = request.get_json(silent=True) or {}

    # Authoritative source: server-side session
    sentences = body.get('sentences') or session.get('sentences', [])
    if not sentences:
        return jsonify({'error': 'No sentences found. Run detection first.'}), 400

    stored      = session.get('detection_results', {})
    sent_scores = stored.get('sentence_scores', [])
    overall_in  = float(stored.get('overall_score', 0))

    # Fallback to client-supplied scores if session expired
    if not sent_scores:
        raw         = body.get('results') or {}
        sent_scores = raw.get('sentence_scores', [])
        overall_in  = float(raw.get('overall_score', overall_in))

    try:
        scores_list = [
            float((sent_scores[i] if i < len(sent_scores) else {}).get('combined_score', 0))
            for i in range(len(sentences))
        ]

        # Build flagged list using adaptive thresholds + skip filter
        flagged_idx    = []
        flagged_sents  = []
        thresholds     = []

        for i, sentence in enumerate(sentences):
            score     = scores_list[i]
            threshold = _adaptive_threshold(sentence)
            thresholds.append(threshold)
            if score >= threshold and not _is_non_rewritable(sentence):
                flagged_idx.append(i)
                flagged_sents.append(sentence)

        # Single batched GPU/CPU call for all flagged sentences
        batch_out   = rewriter.rewrite_batch(flagged_sents) if flagged_sents else []
        rewrite_map = dict(zip(flagged_idx, batch_out))

        rewritten_sentences = []
        for i, sentence in enumerate(sentences):
            score          = scores_list[i]
            rewritten_text = rewrite_map.get(i, sentence)
            is_nr          = _is_non_rewritable(sentence)

            # Accept rewrite only if output is genuinely different and non-empty
            actually_rewritten = (
                i in rewrite_map
                and rewritten_text.strip().lower() != sentence.strip().lower()
                and len(rewritten_text.strip()) > 10
            )

            rewritten_sentences.append({
                'original':       sentence,
                'rewritten':      rewritten_text if actually_rewritten else sentence,
                'was_rewritten':  actually_rewritten,
                'original_score': score,
                'threshold_used': thresholds[i],
                'non_rewritable': is_nr,
            })

        # Re-score the rewritten document
        new_text    = _build_rewritten_text(rewritten_sentences)
        new_sents   = segment_sentences(new_text)
        det         = get_detector()
        new_results = det.analyze(new_sents) if (det and len(new_sents) >= 2) else {'overall_score': 0}
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
    """
    Return the fully rewritten document as a downloadable file.
    Body: { "format": "docx" | "txt" }
    """
    body = request.get_json(silent=True) or {}
    fmt  = body.get('format', 'docx').lower()

    rewritten_sentences = session.get('rewritten_sentences', [])
    if not rewritten_sentences:
        return jsonify({'error': 'No rewritten document. Complete the rewrite step first.'}), 400

    full_text = _build_rewritten_text(rewritten_sentences)

    try:
        tmp_id = uuid.uuid4()
        if fmt == 'docx':
            filepath      = os.path.join(tempfile.gettempdir(), f'rewritten_{tmp_id}.docx')
            _save_rewritten_docx(full_text, filepath)
            download_name = 'rewritten_document.docx'
            mimetype      = ('application/vnd.openxmlformats-officedocument'
                             '.wordprocessingml.document')
        else:
            filepath      = os.path.join(tempfile.gettempdir(), f'rewritten_{tmp_id}.txt')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('REWRITTEN DOCUMENT\n')
                f.write('Generated by PlagioAI\n')
                f.write('=' * 60 + '\n\n')
                f.write(full_text)
            download_name = 'rewritten_document.txt'
            mimetype      = 'text/plain'

        return send_file(
            filepath,
            as_attachment=True,
            download_name=download_name,
            mimetype=mimetype,
        )

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
        return jsonify({'error': 'No report data -- complete the rewrite step first.'}), 400

    try:
        filepath = generate_report(rewritten_sentences, orig_score, new_score, format_type)
        return send_file(
            filepath,
            as_attachment=True,
            download_name=f'plagiarism_report.{format_type}',
        )
    except Exception as e:
        return jsonify({'error': f'Report error: {e}'}), 500


# -- Entry point --------------------------------------------------------------
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