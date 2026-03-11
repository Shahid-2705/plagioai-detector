'use strict';

/* ===========================================
   PLAGIOAI — script.js
   Full feature set:
   - Drag & drop upload
   - File remove
   - Manual text paste
   - 4-step panel navigation
   - Score ring animation
   - Sentence list rendering
   - Rewrite side-by-side view
   - Report summary + download
   - Loading overlay & toast notifications
   - credentials: 'include' on every fetch
=========================================== */

/* ── State ─────────────────────────────── */

const state = {
    text:             '',
    filename:         '',
    sentences:        [],
    detectionResults: null,
    rewrittenData:    null
};

/* ── DOM helper ─────────────────────────── */

const $ = (id) => document.getElementById(id);

/* ── Loading overlay ────────────────────── */

function showLoading(msg) {
    const overlay = $('loading-overlay');
    const label   = $('loading-msg');
    if (overlay) overlay.classList.remove('hidden');
    if (label)   label.textContent = msg || 'Processing…';
}

function hideLoading() {
    const overlay = $('loading-overlay');
    if (overlay) overlay.classList.add('hidden');
}

/* ── Toast notifications ────────────────── */

function showToast(msg, type) {
    const t = $('toast');
    if (!t) return;
    t.textContent = msg;
    t.className   = 'toast show ' + (type || '');
    clearTimeout(t._timer);
    t._timer = setTimeout(function () { t.className = 'toast hidden'; }, 3500);
}

/* ── Risk helpers ───────────────────────── */

function riskClass(score) {
    if (score >= 0.70) return 'high';
    if (score >= 0.40) return 'medium';
    return 'low';
}

function pct(score) {
    if (score == null) return '0%';
    return (score <= 1 ? Math.round(score * 100) : Math.round(score)) + '%';
}

function escapeHtml(str) {
    var map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
    return String(str || '').replace(/[&<>"']/g, function (m) { return map[m]; });
}

/* ── Step tracker ───────────────────────── */

function activateStep(n) {
    for (var i = 1; i <= 4; i++) {
        var el = $('step-' + i);
        if (!el) continue;
        el.classList.remove('active', 'done');
        if (i < n)  el.classList.add('done');
        if (i === n) el.classList.add('active');
    }
}

/* ── Panel navigation ───────────────────── */

function showPanel(name) {
    document.querySelectorAll('.panel').forEach(function (p) {
        p.classList.remove('active');
        p.classList.add('hidden');
    });
    var target = $('panel-' + name);
    if (target) {
        target.classList.remove('hidden');
        target.classList.add('active');
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
}

/* ── Upload zone (drag & drop + click) ─── */

function initUploadZone() {
    var zone  = $('upload-zone');
    var input = $('file-input');
    if (!zone || !input) return;

    /* Click to open file picker */
    zone.addEventListener('click', function () { input.click(); });

    /* Drag over */
    zone.addEventListener('dragover', function (e) {
        e.preventDefault();
        zone.classList.add('drag-over');
    });

    /* Drag leave */
    zone.addEventListener('dragleave', function (e) {
        if (!zone.contains(e.relatedTarget)) {
            zone.classList.remove('drag-over');
        }
    });

    /* Drop */
    zone.addEventListener('drop', function (e) {
        e.preventDefault();
        zone.classList.remove('drag-over');
        var file = e.dataTransfer.files[0];
        if (file) handleFileSelected(file);
    });

    /* Input change */
    input.addEventListener('change', function () {
        if (input.files.length > 0) handleFileSelected(input.files[0]);
    });

    /* Remove file button */
    var removeBtn = $('remove-file');
    if (removeBtn) removeBtn.addEventListener('click', resetFile);

    /* Manual text area — enable detect button when enough text is pasted */
    var manualArea = $('manual-text');
    if (manualArea) {
        manualArea.addEventListener('input', function () {
            var val = manualArea.value.trim();
            if (val.length > 50) {
                state.text     = val;
                state.filename = 'pasted-text.txt';
                enableDetect(true);
            } else if (!state.filename || state.filename === 'pasted-text.txt') {
                state.text = '';
                enableDetect(false);
            }
        });
    }
}

function enableDetect(on) {
    var btn = $('btn-detect');
    if (btn) btn.disabled = !on;
}

/* ── File selected handler ──────────────── */

async function handleFileSelected(file) {
    var ext     = file.name.split('.').pop().toLowerCase();
    var allowed = ['pdf', 'docx', 'txt'];

    if (!allowed.includes(ext)) {
        showToast('Unsupported file type. Use PDF, DOCX, or TXT.', 'error');
        return;
    }

    showLoading('Extracting text from file…');

    var formData = new FormData();
    formData.append('file', file);

    try {
        var res  = await fetch('/upload', {
            method: 'POST',
            body:   formData,
            credentials: 'include'
        });
        var data = await res.json();
        hideLoading();

        if (!res.ok || data.error) {
            showToast(data.error || 'Upload failed', 'error');
            return;
        }

        /* Store the FULL text so /detect receives the complete document */
        state.text     = data.full_text || data.preview || '';
        state.filename = data.filename  || file.name;

        /* Update UI */
        var nameEl = $('file-name');
        var metaEl = $('file-meta');
        var infoEl = $('file-info');
        var prevEl = $('text-preview');
        var prevWr = $('text-preview-wrap');

        if (nameEl) nameEl.textContent = data.filename;
        if (metaEl) metaEl.textContent =
            (data.word_count || 0).toLocaleString() + ' words · ' +
            (data.char_count || 0).toLocaleString() + ' chars';
        if (infoEl) infoEl.classList.remove('hidden');
        if (prevEl) prevEl.textContent = data.preview || '';
        if (prevWr) prevWr.classList.remove('hidden');

        enableDetect(true);
        showToast('File uploaded successfully!', 'success');

    } catch (err) {
        hideLoading();
        showToast('Upload error: ' + err.message, 'error');
    }
}

/* ── Reset file ─────────────────────────── */

function resetFile() {
    state.text     = '';
    state.filename = '';

    var input = $('file-input');
    if (input) input.value = '';

    ['file-info', 'text-preview-wrap'].forEach(function (id) {
        var el = $(id);
        if (el) el.classList.add('hidden');
    });

    var manual = $('manual-text');
    if (manual) manual.value = '';

    enableDetect(false);
}

/* ── Detection ──────────────────────────── */

async function runDetection() {
    var manualEl = $('manual-text');
    var manual   = manualEl ? manualEl.value.trim() : '';
    var text     = state.text || manual;

    if (!text) {
        showToast('Upload a file or paste text first.', 'error');
        return;
    }

    showLoading('Analysing plagiarism… this may take a moment.');

    try {
        var res  = await fetch('/detect', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body:    JSON.stringify({ text: text })
        });
        var data = await res.json();
        hideLoading();

        if (!res.ok || data.error) {
            showToast(data.error || 'Detection failed', 'error');
            return;
        }

        state.sentences        = data.sentences || [];
        state.detectionResults = data;

        renderResults(data);
        showPanel('results');
        activateStep(2);

    } catch (err) {
        hideLoading();
        showToast('Detection error: ' + err.message, 'error');
    }
}

/* ── Render detection results ───────────── */

function renderResults(data) {
    var score = data.overall_score || 0;

    /* Score percentage label */
    var pctEl = $('score-pct');
    if (pctEl) pctEl.textContent = Math.round(score) + '%';

    /* Animate SVG score ring */
    animateRing(score);

    /* Risk counters */
    var highEl   = $('count-high');
    var mediumEl = $('count-medium');
    var lowEl    = $('count-low');
    var totalSentences = (data.sentences || []).length;
    var highCount      = data.high_risk_count   || 0;
    var mediumCount    = data.medium_risk_count || 0;

    if (highEl)   highEl.textContent   = highCount;
    if (mediumEl) mediumEl.textContent = mediumCount;
    if (lowEl)    lowEl.textContent    = totalSentences - highCount - mediumCount;

    /* Sentence list */
    var list = $('sentence-list');
    if (!list) return;
    list.innerHTML = '';

    /* Backend returns sentence scores under key "results" */
    var scores    = data.results || data.sentence_scores || [];
    var sentences = data.sentences || [];

    scores.forEach(function (s, i) {
        var sentence = sentences[i] || s.sentence || '';
        var combined = s.combined_score || 0;
        var risk     = riskClass(combined);

        var div = document.createElement('div');
        div.className = 'sentence-item ' + risk;

        div.innerHTML =
            '<span class="s-index">#' + (i + 1) + '</span>' +
            '<div class="s-content">' +
                '<p class="s-text">' + escapeHtml(sentence) + '</p>' +
                '<div class="s-metrics">' +
                    '<span class="s-metric">Semantic: ' + pct(s.semantic_score) + '</span>' +
                    '<span class="s-metric">N-gram: '   + pct(s.ngram_score)    + '</span>' +
                    '<span class="s-metric">Jaccard: '  + pct(s.jaccard_score)  + '</span>' +
                '</div>' +
            '</div>' +
            '<span class="s-score ' + risk + '">' + pct(combined) + '</span>';

        list.appendChild(div);
    });
}

/* ── Score ring animation ───────────────── */

function animateRing(score) {
    var ring = $('score-ring');
    if (!ring) return;

    var circumference = 326.73;           /* 2π × r=52 */
    var targetOffset  = circumference - (Math.min(score, 100) / 100) * circumference;

    /* Colour by severity */
    if (score >= 70)      ring.style.stroke = 'var(--high)';
    else if (score >= 40) ring.style.stroke = 'var(--medium)';
    else                  ring.style.stroke = 'var(--low)';

    /* Start from full offset (empty ring) then animate to target */
    ring.style.transition = 'none';
    ring.style.strokeDashoffset = circumference;

    /* Force reflow so the transition actually plays */
    ring.getBoundingClientRect();

    ring.style.transition       = 'stroke-dashoffset 1.2s cubic-bezier(0.4, 0, 0.2, 1)';
    ring.style.strokeDashoffset = targetOffset;
}

/* ── Rewrite ────────────────────────────── */

async function runRewrite() {
    if (!state.detectionResults) {
        showToast('Run detection first.', 'error');
        return;
    }

    showLoading('Rewriting high-risk sentences with AI… please wait.');

    try {
        var res  = await fetch('/rewrite', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({
                sentences: state.sentences,
                results: {
                    sentence_scores: state.detectionResults.results ||
                                     state.detectionResults.sentence_scores || [],
                    overall_score:   state.detectionResults.overall_score   || 0
                },
                threshold: 0.30
            })
        });
        var data = await res.json();
        hideLoading();

        if (!res.ok || data.error) {
            showToast(data.error || 'Rewrite failed', 'error');
            return;
        }

        state.rewrittenData = data;
        renderRewritten(data);
        showPanel('rewritten');
        activateStep(3);

    } catch (err) {
        hideLoading();
        showToast('Rewrite error: ' + err.message, 'error');
    }
}

/* ── Render rewritten content ───────────── */

function renderRewritten(data) {
    /* Score comparison bar
       original_score comes from the detection overall_score stored server-side.
       If it arrives as 0, fall back to the value saved in state. */
    var origScore = data.original_score ||
                    (state.detectionResults ? state.detectionResults.overall_score : 0) || 0;
    var newScore  = data.new_score || 0;
    var rwCount   = data.rewritten_count || 0;

    var before = $('compare-before');
    var after  = $('compare-after');
    var count  = $('compare-count');
    if (before) before.textContent = Math.round(origScore) + '%';
    if (after)  after.textContent  = Math.round(newScore)  + '%';
    if (count)  count.textContent  = rwCount + ' sentences';

    /* Side-by-side list */
    var list = $('rewrite-list');
    if (!list) return;
    list.innerHTML = '';

    (data.rewritten_sentences || []).forEach(function (s, i) {
        var tagClass = s.was_rewritten ? 'rewritten' : 'unchanged';
        var tagLabel = s.was_rewritten ? '&#9998; Rewritten' : '&mdash; Unchanged';

        var div       = document.createElement('div');
        div.className = 'rewrite-item' + (s.was_rewritten ? ' changed' : '');

        var rewrittenRow = s.was_rewritten
            ? '<div class="rewrite-row">' +
                  '<span class="rewrite-row-label">Rewritten</span>' +
                  '<p class="rewrite-row-text rewritten-text">' + escapeHtml(s.rewritten) + '</p>' +
              '</div>'
            : '';

        div.innerHTML =
            '<div class="rewrite-header">' +
                '<span>Sentence ' + (i + 1) + ' &middot; ' +
                    'Original score: ' + Math.round((s.original_score || 0) * 100) + '%</span>' +
                '<span class="rewrite-tag ' + tagClass + '">' + tagLabel + '</span>' +
            '</div>' +
            '<div class="rewrite-body">' +
                '<div class="rewrite-row">' +
                    '<span class="rewrite-row-label">Original</span>' +
                    '<p class="rewrite-row-text original">' + escapeHtml(s.original) + '</p>' +
                '</div>' +
                rewrittenRow +
            '</div>';

        list.appendChild(div);
    });
}

/* ── Report ─────────────────────────────── */

function renderReportSummary() {
    var d  = state.rewrittenData;
    if (!d) return;
    var el = $('final-summary');
    if (!el) return;

    var reduction = (d.original_score || 0) - (d.new_score || 0);
    el.textContent =
        'Original Plagiarism Score:   ' + Math.round(d.original_score || 0) + '%\n' +
        'New Plagiarism Score:         ' + Math.round(d.new_score      || 0) + '%\n' +
        'Score Reduction:              ' + (reduction >= 0 ? '-' : '+') +
                                           Math.abs(Math.round(reduction)) + '%\n' +
        'Sentences Rewritten:          ' + (d.rewritten_count || 0) + '\n' +
        'Total Sentences Analysed:     ' + (d.rewritten_sentences || []).length;
}

async function downloadReport(format) {
    showLoading('Generating ' + format.toUpperCase() + ' report…');

    try {
        var d   = state.rewrittenData || {};
        var res = await fetch('/report', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ format: format, original_score: d.original_score || 0 })
        });
        hideLoading();

        if (!res.ok) {
            var err = await res.json().catch(function () { return {}; });
            showToast(err.error || 'Report generation failed', 'error');
            return;
        }

        var blob = await res.blob();
        var url  = URL.createObjectURL(blob);
        var a    = document.createElement('a');
        a.href     = url;
        a.download = 'plagiarism_report.' + format;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        showToast('Report downloaded!', 'success');

    } catch (err) {
        hideLoading();
        showToast('Download error: ' + err.message, 'error');
    }
}

/* ── Initialise ─────────────────────────── */

document.addEventListener('DOMContentLoaded', function () {

    /* Upload zone (drag-drop, file picker, manual textarea) */
    initUploadZone();

    /* Step 1 → 2: Detect */
    var btnDetect = $('btn-detect');
    if (btnDetect) btnDetect.addEventListener('click', runDetection);

    /* Step 2 → 3: Rewrite */
    var btnRewrite = $('btn-rewrite');
    if (btnRewrite) btnRewrite.addEventListener('click', runRewrite);

    /* Step 3 → 4: Report */
    var btnReport = $('btn-report');
    if (btnReport) btnReport.addEventListener('click', function () {
        renderReportSummary();
        showPanel('report');
        activateStep(4);
    });

    /* Back: results → upload */
    var btnBackUpload = $('btn-back-upload');
    if (btnBackUpload) btnBackUpload.addEventListener('click', function () {
        showPanel('upload');
        activateStep(1);
    });

    /* Back: rewritten → results */
    var btnBackResults = $('btn-back-results');
    if (btnBackResults) btnBackResults.addEventListener('click', function () {
        showPanel('results');
        activateStep(2);
    });
});