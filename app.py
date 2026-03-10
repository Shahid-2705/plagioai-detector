"""
AI Plagiarism Detector & Rewriter - Flask Application
"""

import os
import json
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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False

# Allow all origins (needed for local dev and any reverse proxies)
CORS(app, supports_credentials=True)

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

# Initialize engines (lazy loading)
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
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not supported. Use PDF, DOCX, or TXT.'}), 400

    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    try:
        text = extract_text_from_file(filepath)
        if not text or len(text.strip()) < 50:
            return jsonify({'error': 'Could not extract sufficient text from file.'}), 400

        session['filepath'] = filepath
        session['original_text'] = text

        preview = text[:500] + '...' if len(text) > 500 else text
        word_count = len(text.split())
        char_count = len(text)

        return jsonify({
            'success': True,
            'full_text': text,          # full text sent back so JS can forward to /detect
            'preview': preview,
            'word_count': word_count,
            'char_count': char_count,
            'filename': filename
        })

    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500


@app.route('/detect', methods=['POST'])
def detect_plagiarism():
    data = request.get_json(silent=True) or {}
    text = data.get('text', '').strip()

    # Fallback to session if frontend didn't re-send full text
    if not text:
        text = session.get('original_text', '').strip()

    if not text:
        return jsonify({'error': 'No text to analyze. Please upload or paste text first.'}), 400

    try:
        detector = get_detector()
        sentences = segment_sentences(text)

        if len(sentences) < 2:
            return jsonify({'error': 'Text too short for analysis. Need at least 2 sentences.'}), 400

        results = detector.analyze(sentences)
        session['sentences'] = sentences
        session['detection_results'] = results

        return jsonify({
            'success': True,
            'sentences': sentences,
            'results': results['sentence_scores'],
            'overall_score': results['overall_score'],
            'high_risk_count': results['high_risk_count'],
            'medium_risk_count': results['medium_risk_count']
        })

    except Exception as e:
        return jsonify({'error': f'Detection error: {str(e)}'}), 500


@app.route('/rewrite', methods=['POST'])
def rewrite_content():
    data = request.get_json(silent=True) or {}
    sentences = data.get('sentences') or session.get('sentences', [])
    threshold = float(data.get('threshold', 0.70))

    # Accept sentence_scores either nested under 'results' or at top level
    raw_results = data.get('results') or {}
    sentence_scores = raw_results.get('sentence_scores', [])
    overall_score_in = float(raw_results.get('overall_score', 0))

    # Also fall back to session-stored detection results
    if not sentence_scores:
        stored = session.get('detection_results', {})
        sentence_scores = stored.get('sentence_scores', [])
        if not overall_score_in:
            overall_score_in = stored.get('overall_score', 0)

    if not sentences:
        return jsonify({'error': 'No sentences to rewrite. Run detection first.'}), 400

    try:
        rewriter = get_rewriter()

        rewritten_sentences = []
        for i, sentence in enumerate(sentences):
            score_data = sentence_scores[i] if i < len(sentence_scores) else {}
            score = float(score_data.get('combined_score', 0))

            if score >= threshold:
                rewritten = rewriter.rewrite(sentence)
                rewritten_sentences.append({
                    'original': sentence,
                    'rewritten': rewritten,
                    'was_rewritten': True,
                    'original_score': score
                })
            else:
                rewritten_sentences.append({
                    'original': sentence,
                    'rewritten': sentence,
                    'was_rewritten': False,
                    'original_score': score
                })

        # Calculate new score on rewritten text
        new_text = ' '.join([s['rewritten'] for s in rewritten_sentences])
        detector = get_detector()
        new_sentences = segment_sentences(new_text)
        new_results = detector.analyze(new_sentences)
        new_score = new_results['overall_score']

        session['rewritten_sentences'] = rewritten_sentences
        session['new_score'] = new_score

        return jsonify({
            'success': True,
            'rewritten_sentences': rewritten_sentences,
            'original_score': overall_score_in,
            'new_score': new_score,
            'rewritten_count': sum(1 for s in rewritten_sentences if s['was_rewritten'])
        })

    except Exception as e:
        return jsonify({'error': f'Rewriting error: {str(e)}'}), 500


@app.route('/report', methods=['POST'])
def download_report():
    data = request.get_json()
    format_type = data.get('format', 'pdf')
    rewritten_sentences = session.get('rewritten_sentences', [])
    original_score = data.get('original_score', 0)
    new_score = session.get('new_score', 0)

    if not rewritten_sentences:
        return jsonify({'error': 'No report data available'}), 400

    try:
        filepath = generate_report(
            rewritten_sentences, original_score, new_score, format_type
        )
        return send_file(
            filepath,
            as_attachment=True,
            download_name=f'plagiarism_report.{format_type}'
        )
    except Exception as e:
        return jsonify({'error': f'Report generation error: {str(e)}'}), 500


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
