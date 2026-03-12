'use strict';

/* =====================================================
   PlagioAI — script.js  (Advanced Edition)

   Workflow:
     Upload → Detect → Rewrite High-Risk → Compare
     → Download Report  |  Download Full Document

   New in this version:
     - Hard threshold 0.70 for rewriting
     - btn-download-rewritten (TXT + DOCX)
     - Non-rewritable sentences shown with "Skipped" badge
     - Score reduction badge on compare panel
     - Richer status polling with per-model pills
===================================================== */

/* ── State ──────────────────────────────────────────── */

var state = {
    text:             '',
    filename:         '',
    sentences:        [],
    detectionResults: null,
    rewrittenData:    null
};

/* ── DOM shorthand ──────────────────────────────────── */

var $ = function (id) { return document.getElementById(id); };

/* ── Loading overlay ─────────────────────────────────── */

function showLoading(msg) {
    var overlay = $('loading-overlay');
    var label   = $('loading-msg');
    if (overlay) overlay.classList.remove('hidden');
    if (label)   label.textContent = msg || 'Processing…';
}

function hideLoading() {
    var overlay = $('loading-overlay');
    if (overlay) overlay.classList.add('hidden');
}

/* ── Toast ───────────────────────────────────────────── */

function showToast(msg, type) {
    var t = $('toast');
    if (!t) return;
    t.textContent = msg;
    t.className   = 'toast show ' + (type || '');
    clearTimeout(t._timer);
    t._timer = setTimeout(function () { t.className = 'toast hidden'; }, 4000);
}

/* ── Utilities ───────────────────────────────────────── */

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

function triggerDownload(blob, filename) {
    var url = URL.createObjectURL(blob);
    var a   = document.createElement('a');
    a.href     = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

/* ── Step tracker ────────────────────────────────────── */

function activateStep(n) {
    for (var i = 1; i <= 4; i++) {
        var el = $('step-' + i);
        if (!el) continue;
        el.classList.remove('active', 'done');
        if (i < n)   el.classList.add('done');
        if (i === n) el.classList.add('active');
    }
}

/* ── Panel navigation ────────────────────────────────── */

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

/* ── Model status polling ─────────────────────────────── */

var _modelsReady = false;

(function startPolling() {
    var STATUS_ICON = { loading: '⏳', ready: '✓', error: '✗' };

    function updatePill(id, status, label) {
        var pill = $(id);
        if (!pill) return;
        pill.textContent = STATUS_ICON[status] + ' ' + label;
        pill.className   = 'status-pill ' + status;
    }

    function poll() {
        fetch('/status', { credentials: 'include' })
            .then(function (r) { return r.json(); })
            .then(function (d) {
                updatePill('pill-detector',  d.detector_status, 'Detector');
                updatePill('pill-rewriter',  d.rewriter_status, 'Rewriter');

                var banner    = $('model-banner');
                var bannerMsg = $('model-banner-msg');

                if (d.all_ready) {
                    _modelsReady = true;
                    if (bannerMsg) bannerMsg.textContent = '✓ All models ready — GPU accelerated.';
                    if (banner)    banner.classList.add('ready');
                    setTimeout(function () {
                        if (banner) banner.classList.add('hidden');
                    }, 2000);
                } else {
                    var msgs = [];
                    if (d.detector_status === 'loading') msgs.push('Loading detector…');
                    if (d.rewriter_status === 'loading') msgs.push('Loading rewriter (flan-t5-large)…');
                    if (d.detector_status === 'error')   msgs.push('Detector error: ' + d.detector_error);
                    if (d.rewriter_status === 'error')   msgs.push('Rewriter error: ' + d.rewriter_error);
                    if (bannerMsg) bannerMsg.textContent = msgs.join('  ·  ') || 'Loading models…';
                    setTimeout(poll, 2500);
                }
            })
            .catch(function () { setTimeout(poll, 3000); });
    }

    document.addEventListener('DOMContentLoaded', function () { poll(); });
}());

/* ── Upload zone ─────────────────────────────────────── */

function initUploadZone() {
    var zone  = $('upload-zone');
    var input = $('file-input');
    if (!zone || !input) return;

    zone.addEventListener('click', function () { input.click(); });

    zone.addEventListener('dragover', function (e) {
        e.preventDefault();
        zone.classList.add('drag-over');
    });
    zone.addEventListener('dragleave', function (e) {
        if (!zone.contains(e.relatedTarget)) zone.classList.remove('drag-over');
    });
    zone.addEventListener('drop', function (e) {
        e.preventDefault();
        zone.classList.remove('drag-over');
        if (e.dataTransfer.files[0]) handleFileSelected(e.dataTransfer.files[0]);
    });
    input.addEventListener('change', function () {
        if (input.files.length > 0) handleFileSelected(input.files[0]);
    });

    var removeBtn = $('remove-file');
    if (removeBtn) removeBtn.addEventListener('click', resetFile);

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

async function handleFileSelected(file) {
    var ext = file.name.split('.').pop().toLowerCase();
    if (!['pdf', 'docx', 'txt'].includes(ext)) {
        showToast('Unsupported file type. Use PDF, DOCX, or TXT.', 'error');
        return;
    }

    showLoading('Extracting text…');
    var formData = new FormData();
    formData.append('file', file);

    try {
        var res  = await fetch('/upload', { method: 'POST', body: formData, credentials: 'include' });
        var data = await res.json();
        hideLoading();

        if (!res.ok || data.error) { showToast(data.error || 'Upload failed', 'error'); return; }

        state.text     = data.full_text || data.preview || '';
        state.filename = data.filename  || file.name;

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

function resetFile() {
    state.text = ''; state.filename = '';
    var input = $('file-input');
    if (input) input.value = '';
    ['file-info', 'text-preview-wrap'].forEach(function (id) {
        var el = $(id); if (el) el.classList.add('hidden');
    });
    var manual = $('manual-text');
    if (manual) manual.value = '';
    enableDetect(false);
}

/* ── Detection ───────────────────────────────────────── */

async function runDetection() {
    var manual = $('manual-text') ? $('manual-text').value.trim() : '';
    var text   = state.text || manual;

    if (!text) { showToast('Upload a file or paste text first.', 'error'); return; }

    showLoading(_modelsReady ? 'Analysing plagiarism…' : 'Loading detector + analysing… first run takes 30–60 s.');

    try {
        var res  = await fetch('/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ text: text })
        });
        var data = await res.json();
        hideLoading();

        if (!res.ok || data.error) { showToast(data.error || 'Detection failed', 'error'); return; }

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

/* ── Render detection results ────────────────────────── */

function renderResults(data) {
    var score = data.overall_score || 0;

    // Score ring
    var pctEl = $('score-pct');
    if (pctEl) pctEl.textContent = Math.round(score) + '%';
    animateRing(score);

    // Risk counters
    var hi  = $('count-high');
    var med = $('count-medium');
    var lo  = $('count-low');
    var total = (data.sentences || []).length;
    if (hi)  hi.textContent  = data.high_risk_count   || 0;
    if (med) med.textContent = data.medium_risk_count || 0;
    if (lo)  lo.textContent  = total - (data.high_risk_count || 0) - (data.medium_risk_count || 0);

    // Sentence list
    var list    = $('sentence-list');
    if (!list) return;
    list.innerHTML = '';

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

function animateRing(score) {
    var ring = $('score-ring');
    if (!ring) return;
    var C = 326.73;
    ring.style.stroke = score >= 70 ? 'var(--high)' : score >= 40 ? 'var(--medium)' : 'var(--low)';
    ring.style.transition = 'none';
    ring.style.strokeDashoffset = C;
    ring.getBoundingClientRect();   // force reflow
    ring.style.transition = 'stroke-dashoffset 1.2s cubic-bezier(0.4,0,0.2,1)';
    ring.style.strokeDashoffset = C - (Math.min(score, 100) / 100) * C;
}

/* ── Rewrite ─────────────────────────────────────────── */

async function runRewrite() {
    if (!state.detectionResults) { showToast('Run detection first.', 'error'); return; }

    showLoading(_modelsReady
        ? 'Rewriting high-risk sentences (≥70%) with flan-t5-large…'
        : 'Loading rewriter + rewriting… first run takes 60–120 s.');

    try {
        var res  = await fetch('/rewrite', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({
                sentences: state.sentences,
                results: {
                    sentence_scores: state.detectionResults.results ||
                                     state.detectionResults.sentence_scores || [],
                    overall_score:   state.detectionResults.overall_score || 0
                },
                threshold: 0.70   // hard threshold — only genuine high-risk
            })
        });
        var data = await res.json();
        hideLoading();

        if (!res.ok || data.error) { showToast(data.error || 'Rewrite failed', 'error'); return; }

        state.rewrittenData = data;
        renderRewritten(data);
        showPanel('rewritten');
        activateStep(3);

    } catch (err) {
        hideLoading();
        showToast('Rewrite error: ' + err.message, 'error');
    }
}

/* ── Render rewritten content ────────────────────────── */

function renderRewritten(data) {
    var origScore = data.original_score ||
                    (state.detectionResults ? state.detectionResults.overall_score : 0) || 0;
    var newScore  = data.new_score  || 0;
    var rwCount   = data.rewritten_count || 0;
    var reduction = origScore - newScore;

    if ($('compare-before')) $('compare-before').textContent = Math.round(origScore) + '%';
    if ($('compare-after'))  $('compare-after').textContent  = Math.round(newScore)  + '%';
    if ($('compare-count'))  $('compare-count').textContent  = rwCount + ' sentences';

    // Score reduction badge
    var badge = $('compare-reduction');
    if (badge) {
        badge.textContent = (reduction >= 0 ? '↓ ' : '↑ ') + Math.abs(Math.round(reduction)) + '% reduction';
        badge.className   = 'compare-reduction-badge ' + (reduction >= 0 ? 'positive' : 'negative');
    }

    var list = $('rewrite-list');
    if (!list) return;
    list.innerHTML = '';

    (data.rewritten_sentences || []).forEach(function (s, i) {
        var wasRewritten = s.was_rewritten;
        var isHighRisk   = (s.original_score || 0) >= 0.70;

        // Determine badge
        var tagClass, tagLabel;
        if (wasRewritten) {
            tagClass = 'rewritten'; tagLabel = '&#9998; Rewritten';
        } else if (isHighRisk) {
            tagClass = 'skipped';   tagLabel = '&#9888; Skipped';   // non-rewritable
        } else {
            tagClass = 'unchanged'; tagLabel = '&mdash; Unchanged';
        }

        var div = document.createElement('div');
        div.className = 'rewrite-item' + (wasRewritten ? ' changed' : '');
        div.innerHTML =
            '<div class="rewrite-header">' +
                '<span>Sentence ' + (i + 1) + ' &middot; Score: ' +
                    Math.round((s.original_score || 0) * 100) + '%</span>' +
                '<span class="rewrite-tag ' + tagClass + '">' + tagLabel + '</span>' +
            '</div>' +
            '<div class="rewrite-body">' +
                '<div class="rewrite-row">' +
                    '<span class="rewrite-row-label">Original</span>' +
                    '<p class="rewrite-row-text original">' + escapeHtml(s.original) + '</p>' +
                '</div>' +
                (wasRewritten
                    ? '<div class="rewrite-row">' +
                          '<span class="rewrite-row-label">Rewritten</span>' +
                          '<p class="rewrite-row-text rewritten-text">' + escapeHtml(s.rewritten) + '</p>' +
                      '</div>'
                    : '') +
            '</div>';

        list.appendChild(div);
    });
}

/* ── Report ──────────────────────────────────────────── */

function renderReportSummary() {
    var d  = state.rewrittenData;
    if (!d) return;
    var el = $('final-summary');
    if (!el) return;

    var reduction = (d.original_score || 0) - (d.new_score || 0);
    el.textContent =
        'Original Plagiarism Score:   ' + Math.round(d.original_score || 0) + '%\n' +
        'New Plagiarism Score:         ' + Math.round(d.new_score      || 0) + '%\n' +
        'Score Reduction:              ' + (reduction >= 0 ? '-' : '+') + Math.abs(Math.round(reduction)) + '%\n' +
        'Sentences Rewritten:          ' + (d.rewritten_count || 0) + '\n' +
        'Total Sentences Analysed:     ' + (d.rewritten_sentences || []).length + '\n' +
        'Rewrite Threshold:            ≥ 70% similarity\n' +
        'Embedding Model:              all-mpnet-base-v2\n' +
        'Rewrite Model:                google/flan-t5-large';
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
        triggerDownload(await res.blob(), 'plagiarism_report.' + format);
        showToast('Report downloaded!', 'success');
    } catch (err) {
        hideLoading();
        showToast('Download error: ' + err.message, 'error');
    }
}

/* ── Download full rewritten document ───────────────── */

async function downloadRewrittenDoc(format) {
    if (!state.rewrittenData) {
        showToast('Complete the rewrite step first.', 'error');
        return;
    }

    format = format || 'docx';
    showLoading('Preparing rewritten document (' + format.toUpperCase() + ')…');

    try {
        var res = await fetch('/download_rewritten', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ format: format })
        });
        hideLoading();

        if (!res.ok) {
            var err = await res.json().catch(function () { return {}; });
            showToast(err.error || 'Download failed', 'error');
            return;
        }

        triggerDownload(await res.blob(), 'rewritten_document.' + format);
        showToast('Rewritten document downloaded!', 'success');

    } catch (err) {
        hideLoading();
        showToast('Download error: ' + err.message, 'error');
    }
}

/* ── Init ────────────────────────────────────────────── */

document.addEventListener('DOMContentLoaded', function () {

    initUploadZone();

    var btnDetect = $('btn-detect');
    if (btnDetect) btnDetect.addEventListener('click', runDetection);

    var btnRewrite = $('btn-rewrite');
    if (btnRewrite) btnRewrite.addEventListener('click', runRewrite);

    var btnReport = $('btn-report');
    if (btnReport) btnReport.addEventListener('click', function () {
        renderReportSummary();
        showPanel('report');
        activateStep(4);
    });

    var btnDownloadRewritten = $('btn-download-rewritten');
    if (btnDownloadRewritten) {
        btnDownloadRewritten.addEventListener('click', function () {
            downloadRewrittenDoc('docx');
        });
    }

    var btnDownloadRewrittenTxt = $('btn-download-rewritten-txt');
    if (btnDownloadRewrittenTxt) {
        btnDownloadRewrittenTxt.addEventListener('click', function () {
            downloadRewrittenDoc('txt');
        });
    }

    var btnBackUpload = $('btn-back-upload');
    if (btnBackUpload) btnBackUpload.addEventListener('click', function () {
        showPanel('upload'); activateStep(1);
    });

    var btnBackResults = $('btn-back-results');
    if (btnBackResults) btnBackResults.addEventListener('click', function () {
        showPanel('results'); activateStep(2);
    });
});