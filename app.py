"""
AI Plagiarism Detector & Rewriter - Flask Application
"""

import os
import uuid
from flask import Flask, render_template, request, jsonify, send_file, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
from core.file_parser import extract_text_from_file
from core.text_preprocessor import preprocess_text, segment_sentences
from core.plagiarism_detector import PlagiarismDetector
from core.rewrite_engine import RewriteEngine
from core.report_generator import generate_report

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False

CORS(app, supports_credentials=True)

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

_detector = None
_rewriter = None


def get_detector():
    global _detector
    if _detector is None:
        _detector = PlagiarismDetector()
    return _detector


def get_rewriter():
    global _rewriter
    if _rewriter is None:
        _rewriter = RewriteEngine()
    return _rewriter


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/checker')
def checker():
    return render_template('checker.html')


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
            'full_text':  text,       # full text returned so JS can forward it to /detect
            'preview':    preview,
            'word_count': word_count,
            'char_count': char_count,
            'filename':   filename,
        })

    except Exception as e:
        return jsonify({'error': f'Error processing file: {e}'}), 500


@app.route('/detect', methods=['POST'])
def detect_plagiarism():
    body = request.get_json(silent=True) or {}
    text = (body.get('text') or '').strip()

    # Fallback to session if the client didn't resend the full text
    if not text:
        text = (session.get('original_text') or '').strip()

    if not text:
        return jsonify({'error': 'No text to analyse. Please upload or paste text first.'}), 400

    try:
        detector  = get_detector()
        sentences = segment_sentences(text)

        if len(sentences) < 2:
            return jsonify({'error': 'Text too short — need at least 2 sentences.'}), 400

        results = detector.analyze(sentences)

        # Persist to session so /rewrite can retrieve them without the client resending
        session['sentences']         = sentences
        session['detection_results'] = results   # contains 'sentence_scores' and 'overall_score'

        return jsonify({
            'success':           True,
            'sentences':         sentences,
            # key name that the JS renderResults() reads
            'results':           results['sentence_scores'],
            'overall_score':     results['overall_score'],
            'high_risk_count':   results['high_risk_count'],
            'medium_risk_count': results['medium_risk_count'],
        })

    except Exception as e:
        return jsonify({'error': f'Detection error: {e}'}), 500


@app.route('/rewrite', methods=['POST'])
def rewrite_content():
    body      = request.get_json(silent=True) or {}
    threshold = float(body.get('threshold', 0.30))   # sensible default: rewrite ≥ 30 %

    # ── Sentences ────────────────────────────────────────────────────────────
    sentences = body.get('sentences') or session.get('sentences', [])
    if not sentences:
        return jsonify({'error': 'No sentences to rewrite. Run detection first.'}), 400

    # ── Sentence scores ───────────────────────────────────────────────────────
    # JS sends: { results: { sentence_scores: [...], overall_score: N } }
    raw          = body.get('results') or {}
    sent_scores  = raw.get('sentence_scores', [])
    overall_in   = float(raw.get('overall_score', 0))

    # Always prefer the session-stored results (authoritative, server-side)
    stored = session.get('detection_results', {})
    if not sent_scores:
        sent_scores = stored.get('sentence_scores', [])
    if not overall_in:
        overall_in = float(stored.get('overall_score', 0))

    # ── Adaptive threshold ────────────────────────────────────────────────────
    # If every sentence is below the requested threshold (common for original
    # text being compared against itself) we lower the bar to the top-30 % of
    # scores in the document, so the user always sees *something* rewritten.
    if sent_scores:
        all_scores = [float(s.get('combined_score', 0)) for s in sent_scores]
        max_score  = max(all_scores) if all_scores else 0

        if max_score < threshold:
            # Use top-30th-percentile of document scores as threshold
            sorted_scores = sorted(all_scores, reverse=True)
            top_n         = max(1, len(sorted_scores) // 3)   # top third
            threshold     = sorted_scores[top_n - 1]
            # Clamp: never rewrite everything if the doc is fully original
            threshold     = max(threshold, 0.05)

    try:
        rewriter            = get_rewriter()
        rewritten_sentences = []

        for i, sentence in enumerate(sentences):
            score_data = sent_scores[i] if i < len(sent_scores) else {}
            score      = float(score_data.get('combined_score', 0))
            should_rewrite = score >= threshold

            if should_rewrite:
                rewritten = rewriter.rewrite(sentence)
                rewritten_sentences.append({
                    'original':       sentence,
                    'rewritten':      rewritten,
                    'was_rewritten':  True,
                    'original_score': score,
                })
            else:
                rewritten_sentences.append({
                    'original':       sentence,
                    'rewritten':      sentence,
                    'was_rewritten':  False,
                    'original_score': score,
                })

        # Re-score the rewritten text
        new_text     = ' '.join(s['rewritten'] for s in rewritten_sentences)
        detector     = get_detector()
        new_sents    = segment_sentences(new_text)
        new_results  = detector.analyze(new_sents)
        new_score    = new_results['overall_score']

        session['rewritten_sentences'] = rewritten_sentences
        session['new_score']           = new_score

        rewritten_count = sum(1 for s in rewritten_sentences if s['was_rewritten'])

        return jsonify({
            'success':             True,
            'rewritten_sentences': rewritten_sentences,
            'original_score':      overall_in,
            'new_score':           new_score,
            'rewritten_count':     rewritten_count,
        })

    except Exception as e:
        return jsonify({'error': f'Rewriting error: {e}'}), 500


@app.route('/report', methods=['POST'])
def download_report():
    body         = request.get_json(silent=True) or {}
    format_type  = body.get('format', 'pdf')
    orig_score   = float(body.get('original_score', 0))

    rewritten_sentences = session.get('rewritten_sentences', [])
    new_score           = float(session.get('new_score', 0))

    if not rewritten_sentences:
        return jsonify({'error': 'No report data — complete the rewrite step first.'}), 400

    try:
        filepath = generate_report(rewritten_sentences, orig_score, new_score, format_type)
        return send_file(
            filepath,
            as_attachment=True,
            download_name=f'plagiarism_report.{format_type}',
        )
    except Exception as e:
        return jsonify({'error': f'Report error: {e}'}), 500


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)