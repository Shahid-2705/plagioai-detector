/* =============================================
   PlagioAI — Main App Script
   ============================================= */

'use strict';

// ── State ────────────────────────────────────
const state = {
    text: '',
    filename: '',
    sentences: [],
    detectionResults: null,
    rewrittenData: null,
};

// ── Helpers ──────────────────────────────────
const $ = (id) => document.getElementById(id);
const qs = (sel) => document.querySelector(sel);

function showLoading(msg = 'Processing...') {
    $('loading-msg').textContent = msg;
    $('loading-overlay').classList.remove('hidden');
}

function hideLoading() {
    $('loading-overlay').classList.add('hidden');
}

function showToast(msg, type = '') {
    const t = $('toast');
    t.textContent = msg;
    t.className = 'toast show ' + type;
    setTimeout(() => { t.className = 'toast hidden'; }, 3500);
}

function activateStep(n) {
    for (let i = 1; i <= 4; i++) {
        const el = $(`step-${i}`);
        if (!el) continue;
        el.classList.remove('active', 'done');
        if (i < n) el.classList.add('done');
        if (i === n) el.classList.add('active');
    }
}

function showPanel(name) {
    document.querySelectorAll('.panel').forEach(p => {
        p.classList.remove('active');
        p.classList.add('hidden');
    });
    const target = $(`panel-${name}`);
    if (target) {
        target.classList.remove('hidden');
        target.classList.add('active');
    }
}

function riskClass(score) {
    if (score >= 0.70) return 'high';
    if (score >= 0.40) return 'medium';
    return 'low';
}

function pct(score) {
    return Math.round(score * 100) + '%';
}

// ── Step navigation ──────────────────────────
function goToDetect() {
    showPanel('results');
    activateStep(2);
}
function goToUpload() {
    showPanel('upload');
    activateStep(1);
}
function goToRewrite() {
    showPanel('rewritten');
    activateStep(3);
}
function goToResults() {
    showPanel('results');
    activateStep(2);
}
function goToReport() {
    showPanel('report');
    activateStep(4);
}

// ── Upload zone ──────────────────────────────
function initUploadZone() {
    const zone = $('upload-zone');
    const input = $('file-input');

    zone.addEventListener('click', () => input.click());

    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('drag-over');
    });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file) handleFileSelected(file);
    });

    input.addEventListener('change', () => {
        if (input.files[0]) handleFileSelected(input.files[0]);
    });

    $('remove-file').addEventListener('click', resetFile);

    // Manual text enables button
    $('manual-text').addEventListener('input', () => {
        const val = $('manual-text').value.trim();
        if (val.length > 50) {
            state.text = val;
            state.filename = 'pasted-text.txt';
            $('btn-detect').disabled = false;
        } else {
            if (!state.filename || state.filename === 'pasted-text.txt') {
                state.text = '';
                $('btn-detect').disabled = true;
            }
        }
    });
}

async function handleFileSelected(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    const allowedExts = ['pdf', 'docx', 'txt'];
    if (!allowedExts.includes(ext)) {
        showToast('Unsupported file type. Use PDF, DOCX, or TXT.', 'error');
        return;
    }

    showLoading('Extracting text…');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/upload', { method: 'POST', body: formData, credentials: 'include' });
        const data = await res.json();
        hideLoading();

        if (!res.ok || data.error) {
            showToast(data.error || 'Upload failed', 'error');
            return;
        }

        state.text = data.full_text || data.preview;   // backend now sends full_text
        state.filename = data.filename;

        $('file-name').textContent = data.filename;
        $('file-meta').textContent = `${data.word_count.toLocaleString()} words · ${data.char_count.toLocaleString()} chars`;
        $('file-info').classList.remove('hidden');
        $('text-preview').textContent = data.preview;
        $('text-preview-wrap').classList.remove('hidden');
        $('btn-detect').disabled = false;

        showToast('File uploaded successfully!', 'success');
    } catch (err) {
        hideLoading();
        showToast('Upload error: ' + err.message, 'error');
    }
}

function resetFile() {
    state.text = '';
    state.filename = '';
    $('file-input').value = '';
    $('file-info').classList.add('hidden');
    $('text-preview-wrap').classList.add('hidden');
    $('btn-detect').disabled = true;
    $('manual-text').value = '';
}

// ── Detection ────────────────────────────────
async function runDetection() {
    const text = state.text || $('manual-text').value.trim();
    if (!text) { showToast('Please upload or paste text first.', 'error'); return; }

    showLoading('Analysing plagiarism… this may take a moment.');

    try {
        const res = await fetch('/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ text }),
        });
        const data = await res.json();
        hideLoading();

        if (!res.ok || data.error) {
            showToast(data.error || 'Detection failed', 'error');
            return;
        }

        state.sentences = data.sentences;
        state.detectionResults = data;
        renderResults(data);
        goToDetect();

    } catch (err) {
        hideLoading();
        showToast('Detection error: ' + err.message, 'error');
    }
}

function renderResults(data) {
    const score = data.overall_score;
    const pctVal = Math.round(score);

    // Animate score ring
    $('score-pct').textContent = pctVal + '%';
    const ring = $('score-ring');
    const circumference = 326.73;
    const offset = circumference - (score / 100) * circumference;
    ring.style.strokeDashoffset = offset;

    // Color ring based on score
    if (score >= 70) ring.style.stroke = 'var(--high)';
    else if (score >= 40) ring.style.stroke = 'var(--medium)';
    else ring.style.stroke = 'var(--low)';

    $('count-high').textContent   = data.high_risk_count;
    $('count-medium').textContent = data.medium_risk_count;
    $('count-low').textContent    = data.sentences.length - data.high_risk_count - data.medium_risk_count;

    // Sentence list
    const list = $('sentence-list');
    list.innerHTML = '';
    data.results.forEach((s, i) => {
        const risk = riskClass(s.combined_score);
        const div = document.createElement('div');
        div.className = `sentence-item ${risk}`;
        div.innerHTML = `
            <span class="s-index">#${i + 1}</span>
            <div class="s-content">
                <p class="s-text">${escapeHtml(s.sentence)}</p>
                <div class="s-metrics">
                    <span class="s-metric">Semantic: ${pct(s.semantic_score)}</span>
                    <span class="s-metric">N-gram: ${pct(s.ngram_score)}</span>
                    <span class="s-metric">Jaccard: ${pct(s.jaccard_score)}</span>
                </div>
            </div>
            <span class="s-score ${risk}">${pct(s.combined_score)}</span>
        `;
        list.appendChild(div);
    });
}

// ── Rewriting ────────────────────────────────
async function runRewrite() {
    showLoading('Rewriting high-risk sentences with AI… please wait.');

    try {
        const res = await fetch('/rewrite', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({
                sentences: state.sentences,
                results: {
                    sentence_scores: state.detectionResults ? state.detectionResults.results : [],
                    overall_score: state.detectionResults ? state.detectionResults.overall_score : 0,
                },
                threshold: 0.70,
            }),
        });
        const data = await res.json();
        hideLoading();

        if (!res.ok || data.error) {
            showToast(data.error || 'Rewrite failed', 'error');
            return;
        }

        state.rewrittenData = data;
        renderRewritten(data);
        goToRewrite();

    } catch (err) {
        hideLoading();
        showToast('Rewrite error: ' + err.message, 'error');
    }
}

function renderRewritten(data) {
    $('compare-before').textContent = Math.round(data.original_score) + '%';
    $('compare-after').textContent  = Math.round(data.new_score) + '%';
    $('compare-count').textContent  = data.rewritten_count + ' sentences';

    const list = $('rewrite-list');
    list.innerHTML = '';

    data.rewritten_sentences.forEach((s, i) => {
        const div = document.createElement('div');
        div.className = 'rewrite-item' + (s.was_rewritten ? ' changed' : '');
        div.innerHTML = `
            <div class="rewrite-header">
                <span>Sentence ${i + 1} · Original score: ${Math.round(s.original_score * 100)}%</span>
                <span class="rewrite-tag ${s.was_rewritten ? 'rewritten' : 'unchanged'}">
                    ${s.was_rewritten ? '✎ Rewritten' : '— Unchanged'}
                </span>
            </div>
            <div class="rewrite-body">
                <div class="rewrite-row">
                    <span class="rewrite-row-label">Original</span>
                    <p class="rewrite-row-text original">${escapeHtml(s.original)}</p>
                </div>
                ${s.was_rewritten ? `
                <div class="rewrite-row">
                    <span class="rewrite-row-label">Rewritten</span>
                    <p class="rewrite-row-text rewritten-text">${escapeHtml(s.rewritten)}</p>
                </div>` : ''}
            </div>
        `;
        list.appendChild(div);
    });
}

// ── Report ───────────────────────────────────
function renderReportSummary() {
    const d = state.rewrittenData;
    if (!d) return;
    const reduction = d.original_score - d.new_score;
    $('final-summary').textContent =
        `Original Plagiarism Score:  ${Math.round(d.original_score)}%\n` +
        `New Plagiarism Score:        ${Math.round(d.new_score)}%\n` +
        `Score Reduction:             ${reduction >= 0 ? '-' : '+'}${Math.abs(Math.round(reduction))}%\n` +
        `Sentences Rewritten:         ${d.rewritten_count}\n` +
        `Total Sentences Analysed:    ${d.rewritten_sentences.length}`;
}

async function downloadReport(format) {
    showLoading('Generating ' + format.toUpperCase() + ' report…');
    try {
        const d = state.rewrittenData;
        const res = await fetch('/report', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({
                format,
                original_score: d ? d.original_score : 0,
            }),
        });
        hideLoading();
        if (!res.ok) {
            const err = await res.json();
            showToast(err.error || 'Report generation failed', 'error');
            return;
        }
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `plagiarism_report.${format}`;
        a.click();
        URL.revokeObjectURL(url);
        showToast('Report downloaded!', 'success');
    } catch (err) {
        hideLoading();
        showToast('Download error: ' + err.message, 'error');
    }
}

// ── Escape HTML ──────────────────────────────
function escapeHtml(str) {
    const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
    return String(str).replace(/[&<>"']/g, m => map[m]);
}

// ── Wire up buttons ──────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initUploadZone();

    const btnDetect = $('btn-detect');
    if (btnDetect) btnDetect.addEventListener('click', runDetection);

    const btnRewrite = $('btn-rewrite');
    if (btnRewrite) btnRewrite.addEventListener('click', runRewrite);

    const btnReport = $('btn-report');
    if (btnReport) btnReport.addEventListener('click', () => {
        renderReportSummary();
        goToReport();
    });

    const btnBackUpload  = $('btn-back-upload');
    if (btnBackUpload)  btnBackUpload.addEventListener('click', goToUpload);

    const btnBackResults = $('btn-back-results');
    if (btnBackResults) btnBackResults.addEventListener('click', goToResults);
});
