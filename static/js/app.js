/**
 * app.js — Frontend interactivity for Fruit Ripeness Ordinal AI Dashboard
 * Handles: tab navigation, sample gallery, file upload, analysis results, charts
 */

'use strict';

// ─── Global State ────────────────────────────────────────────────────────────

let baselineChart = null;
let ordinalChart  = null;
let allSamples    = {};       // { stage_name: [{name, url, path}, ...] }
let activeFilter  = 'all';

const STAGE_COLORS = {
  'Unripe':        'rgba(74, 222, 128, 0.85)',
  'Slightly Ripe': 'rgba(250, 204, 21, 0.85)',
  'Ripe':          'rgba(249, 115, 22, 0.85)',
  'Overripe':      'rgba(220, 38, 38, 0.85)',
};

const STAGE_BORDER = {
  'Unripe':        'rgb(21, 128, 61)',
  'Slightly Ripe': 'rgb(161, 98, 7)',
  'Ripe':          'rgb(154, 52, 18)',
  'Overripe':      'rgb(127, 29, 29)',
};

// ─── DOM Helpers ─────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);
const qs = sel => document.querySelector(sel);
const qsa = sel => document.querySelectorAll(sel);

// ─── Tab Navigation ──────────────────────────────────────────────────────────

function initTabs() {
  qsa('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.tab;
      // Deactivate all
      qsa('.tab-btn').forEach(b => b.classList.remove('active'));
      qsa('.tab-content').forEach(s => s.classList.remove('active'));
      // Activate clicked
      btn.classList.add('active');
      const sec = $(target);
      if (sec) sec.classList.add('active');
    });
  });
}

// ─── Backend Status ───────────────────────────────────────────────────────────

async function fetchStatus() {
  try {
    const res = await fetch('/api/status');
    if (!res.ok) throw new Error('offline');
    const data = await res.json();
    const badge = $('device-badge');
    const statusTxt = $('backend-status');
    if (badge)     badge.textContent  = (data.device || 'cpu').toUpperCase();
    if (statusTxt) statusTxt.textContent = 'Backend Online';
    // Update status dot colour
    const dot = qs('.status-indicator');
    if (dot) { dot.classList.remove('offline'); dot.classList.add('online'); }
  } catch (_) {
    const statusTxt = $('backend-status');
    if (statusTxt) statusTxt.textContent = 'Backend Offline';
    const dot = qs('.status-indicator');
    if (dot) { dot.classList.remove('online'); dot.classList.add('offline'); }
  }
}

async function fetchMetrics() {
  const body = $('metrics-table-body');
  if (!body) return;
  try {
    const response = await fetch('/api/metrics');
    if (!response.ok) throw new Error('Failed to load evaluation metrics');
    const data = await response.json();
    body.innerHTML = '';
    if (!data.rows || data.rows.length === 0) {
      const row = body.insertRow();
      const cell = row.insertCell();
      cell.colSpan = 3;
      cell.textContent = 'No held-out evaluation results found yet.';
      return;
    }

    data.rows.forEach(metric => {
      const row = body.insertRow();
      row.insertCell().textContent = metric.metric;
      row.insertCell().textContent = metric.baseline
        ? formatMetric(metric.metric, Number(metric.baseline))
        : '—';
      row.insertCell().textContent = metric.ordinal
        ? formatMetric(metric.metric, Number(metric.ordinal))
        : '—';
    });
  } catch (error) {
    const row = body.insertRow();
    const cell = row.insertCell();
    cell.colSpan = 3;
    cell.textContent = error.message;
  }
}

function formatMetric(name, value) {
  if (name.includes('Accuracy')) return `${(value * 100).toFixed(2)}%`;
  return value.toFixed(4);
}

// ─── Sample Gallery ───────────────────────────────────────────────────────────

async function fetchSamples() {
  const gallery = $('sample-gallery');
  if (!gallery) return;
  gallery.innerHTML = '<div class="loading-spinner">Loading Fruits-360 samples...</div>';

  try {
    const res = await fetch('/api/samples');
    if (!res.ok) throw new Error('Failed to fetch samples');
    allSamples = await res.json();
    renderGallery(activeFilter);
  } catch (err) {
    gallery.innerHTML = `<p class="error-msg">Could not load samples: ${err.message}</p>`;
  }
}

function renderGallery(filter) {
  const gallery = $('sample-gallery');
  if (!gallery) return;
  gallery.innerHTML = '';

  const stageLabels = {
    unripe:        'Unripe',
    slightly_ripe: 'Slightly Ripe',
    ripe:          'Ripe',
    overripe:      'Overripe',
  };

  let total = 0;

  for (const [stageName, items] of Object.entries(allSamples)) {
    if (filter !== 'all' && stageName !== filter) continue;
    if (!items || items.length === 0) continue;

    // Group label
    const groupLabel = document.createElement('div');
    groupLabel.className = 'gallery-group-label';
    groupLabel.textContent = stageLabels[stageName] || stageName;
    gallery.appendChild(groupLabel);

    items.forEach(item => {
      const wrapper = document.createElement('div');
      wrapper.className = 'sample-item';
      wrapper.dataset.path  = item.path;
      wrapper.dataset.stage = stageName;
      wrapper.title = `${stageLabels[stageName] || stageName} — ${item.name}`;

      const img = document.createElement('img');
      img.src   = item.url;
      img.alt   = item.name;
      img.loading = 'lazy';
      img.onerror = () => { wrapper.style.display = 'none'; };

      const lbl = document.createElement('span');
      lbl.className = 'sample-label';
      lbl.textContent = item.name.replace(/\.[^.]+$/, '').slice(0, 12);

      wrapper.appendChild(img);
      wrapper.appendChild(lbl);
      wrapper.addEventListener('click', () => analyzeFromSample(item.path, item.url));
      gallery.appendChild(wrapper);
      total++;
    });
  }

  if (total === 0) {
    gallery.innerHTML = '<p class="error-msg">No samples found for this filter. Make sure data/ folder is populated.</p>';
  }
}

function initSampleFilters() {
  qsa('.sample-filter').forEach(btn => {
    btn.addEventListener('click', () => {
      qsa('.sample-filter').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeFilter = btn.dataset.stage;
      renderGallery(activeFilter);
    });
  });

  const refreshBtn = $('btn-refresh-samples');
  if (refreshBtn) refreshBtn.addEventListener('click', fetchSamples);
}

// ─── File Drop-zone Upload ────────────────────────────────────────────────────

function initDropZone() {
  const dropZone  = $('drop-zone');
  const fileInput = $('file-input');
  if (!dropZone || !fileInput) return;

  // Click to browse
  dropZone.addEventListener('click', () => fileInput.click());

  // File selected via input
  fileInput.addEventListener('change', () => {
    if (fileInput.files && fileInput.files[0]) {
      analyzeFromFile(fileInput.files[0]);
    }
  });

  // Drag events
  dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      analyzeFromFile(file);
    }
  });
}

// ─── Analysis: from local file ────────────────────────────────────────────────

function analyzeFromFile(file) {
  const formData = new FormData();
  formData.append('file', file);

  // Show local preview immediately
  const reader = new FileReader();
  reader.onload = e => { showLoadingState(e.target.result); };
  reader.readAsDataURL(file);

  postAnalyze(formData, false);
}

// ─── Analysis: from sample path ──────────────────────────────────────────────

function analyzeFromSample(samplePath, previewUrl) {
  showLoadingState(previewUrl);
  postAnalyze(JSON.stringify({ sample_path: samplePath }), true);
}

// ─── Core POST to /api/analyze ────────────────────────────────────────────────

async function postAnalyze(body, isJson) {
  const badge = $('inference-time-badge');
  if (badge) badge.textContent = 'Analyzing...';

  const t0 = performance.now();

  try {
    const fetchOpts = isJson
      ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body }
      : { method: 'POST', body };

    const res = await fetch('/api/analyze', fetchOpts);
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || `HTTP ${res.status}`);
    }
    const data = await res.json();
    const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
    if (badge) badge.textContent = `Inference: ${elapsed}s`;
    renderResults(data);
  } catch (err) {
    if (badge) badge.textContent = 'Error';
    showErrorInResults(err.message);
  }
}

// ─── Results Rendering ────────────────────────────────────────────────────────

function showLoadingState(previewSrc) {
  const placeholder = $('results-placeholder');
  const content     = $('results-content');
  if (placeholder) placeholder.classList.add('hidden');
  if (content)     content.classList.remove('hidden');

  const origImg = $('res-orig-img');
  if (origImg) origImg.src = previewSrc;
  const croppedImg = $('res-cropped-img');
  if (croppedImg) croppedImg.src = previewSrc;

  $('res-baseline-label').textContent = 'Analyzing...';
  $('res-baseline-conf').textContent  = '--';
  $('res-ordinal-label').textContent  = 'Analyzing...';
  $('res-ordinal-rank').textContent   = 'Rank: --';
  $('res-baseline-expl').textContent  = '';
  $('res-ordinal-expl').textContent   = '';
  $('threshold-gates').innerHTML      = '';

  clearOrdinalBar();
}

function renderResults(data) {
  const placeholder = $('results-placeholder');
  const content     = $('results-content');
  if (placeholder) placeholder.classList.add('hidden');
  if (content)     content.classList.remove('hidden');

  const b = data.baseline;
  const o = data.ordinal;

  // Original + CAM images
  $('res-orig-img').src      = data.original_image;
  $('res-cropped-img').src   = data.fruit_crop_image;
  $('res-ordinal-cam').src   = o.gradcam_image;

  const baselinePanel = $('baseline-panel');
  const baselineVisual = $('baseline-visual-col');
  const comparisonGrid = $('models-comparison-grid');
  const visualRow = qs('.visual-row');
  if (b) {
    $('res-baseline-cam').src = b.gradcam_image;
    $('res-baseline-label').textContent = `Stage: ${b.predicted_label}`;
    $('res-baseline-conf').textContent  = `${(b.confidence * 100).toFixed(1)}%`;
    $('res-baseline-expl').textContent  = b.explanation;
    renderBaselineChart(b.class_probabilities);
    baselinePanel.classList.remove('hidden');
    baselineVisual.classList.remove('hidden');
    comparisonGrid.classList.remove('ordinal-only');
    visualRow.classList.remove('ordinal-only');
  } else {
    baselinePanel.classList.add('hidden');
    baselineVisual.classList.add('hidden');
    comparisonGrid.classList.add('ordinal-only');
    visualRow.classList.add('ordinal-only');
  }

  // Ordinal panel
  $('res-ordinal-label').textContent = `Stage: ${o.predicted_label}`;
  $('res-ordinal-rank').textContent  = `Rank: ${o.predicted_index} / ${o.class_names.length - 1}`;
  $('res-ordinal-expl').textContent  = o.explanation;

  // Ordinal step bar
  renderOrdinalBar(o.predicted_index, o.class_names);

  // Threshold gates
  renderThresholdGates(o.threshold_probabilities);

  // Charts
  renderOrdinalChart(o.derived_probabilities);
}

function showErrorInResults(msg) {
  const content = $('results-content');
  if (content) {
    content.innerHTML = `<div class="error-msg" style="padding:2rem;text-align:center;">
      <strong>Error:</strong> ${msg}
    </div>`;
    content.classList.remove('hidden');
  }
  const badge = $('inference-time-badge');
  if (badge) badge.textContent = 'Error';
}

// ─── Ordinal Step Bar ─────────────────────────────────────────────────────────

function clearOrdinalBar() {
  const container = $('ordinal-progress-bar');
  if (container) container.innerHTML = '';
}

function renderOrdinalBar(rank, classNames) {
  const container = $('ordinal-progress-bar');
  if (!container) return;
  container.innerHTML = '';

  classNames.forEach((name, index) => {
    const step = document.createElement('div');
    step.className = `ordinal-step ${index < rank ? 'active' : ''} ${index === rank ? 'target' : ''}`;
    const number = document.createElement('span');
    number.textContent = String(index);
    const label = document.createElement('small');
    label.textContent = name;
    step.append(number, label);
    container.appendChild(step);

    if (index < classNames.length - 1) {
      const line = document.createElement('div');
      line.className = `ordinal-line ${index < rank ? 'active' : ''}`;
      container.appendChild(line);
    }
  });
}

// ─── Threshold Gates ─────────────────────────────────────────────────────────

function renderThresholdGates(thresholds) {
  const container = $('threshold-gates');
  if (!container) return;
  container.innerHTML = '';

  thresholds.forEach((t, idx) => {
    const row = document.createElement('div');
    row.className = `gate-row ${t.active ? 'passed' : 'blocked'}`;

    const icon  = t.active ? '✔' : '✘';
    const pct   = (t.probability * 100).toFixed(1);
    const label = t.active ? 'PASSED' : 'BLOCKED';

    row.innerHTML = `
      <span class="gate-icon">${icon}</span>
      <span class="gate-label">${t.threshold}</span>
      <span class="gate-prob">${pct}%</span>
      <span class="gate-status ${t.active ? 'passed' : 'blocked'}">${label}</span>
    `;
    container.appendChild(row);
  });
}

// ─── Chart.js Rendering ───────────────────────────────────────────────────────

function makeChartData(classProbabilities) {
  const labels = classProbabilities.map(cp => cp.label);
  const values = classProbabilities.map(cp => +(cp.probability * 100).toFixed(2));
  const bgColors     = labels.map(l => STAGE_COLORS[l]  || 'rgba(99,102,241,0.7)');
  const borderColors = labels.map(l => STAGE_BORDER[l]  || 'rgb(67,56,202)');
  return { labels, values, bgColors, borderColors };
}

function renderBaselineChart(classProbs) {
  const canvas = $('chart-baseline');
  if (!canvas) return;

  const { labels, values, bgColors, borderColors } = makeChartData(classProbs);

  if (baselineChart) { baselineChart.destroy(); baselineChart = null; }

  baselineChart = new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Softmax Probability (%)',
        data: values,
        backgroundColor: bgColors,
        borderColor: borderColors,
        borderWidth: 2,
        borderRadius: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.parsed.y.toFixed(1)}%`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: '#94a3b8', font: { size: 11 } },
          grid:  { color: 'rgba(148,163,184,0.1)' },
        },
        y: {
          beginAtZero: true,
          max: 100,
          ticks: { color: '#94a3b8', font: { size: 11 }, callback: v => `${v}%` },
          grid:  { color: 'rgba(148,163,184,0.1)' },
        },
      },
    },
  });
}

function renderOrdinalChart(derivedProbs) {
  const canvas = $('chart-ordinal');
  if (!canvas) return;

  const { labels, values, bgColors, borderColors } = makeChartData(derivedProbs);

  if (ordinalChart) { ordinalChart.destroy(); ordinalChart = null; }

  ordinalChart = new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Derived Probability Mass (%)',
        data: values,
        backgroundColor: bgColors,
        borderColor: borderColors,
        borderWidth: 2,
        borderRadius: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.parsed.y.toFixed(1)}%`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: '#94a3b8', font: { size: 11 } },
          grid:  { color: 'rgba(148,163,184,0.1)' },
        },
        y: {
          beginAtZero: true,
          max: 100,
          ticks: { color: '#94a3b8', font: { size: 11 }, callback: v => `${v}%` },
          grid:  { color: 'rgba(148,163,184,0.1)' },
        },
      },
    },
  });
}

// ─── Boot ─────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initDropZone();
  initSampleFilters();
  fetchStatus();
  fetchSamples();
  fetchMetrics();
});
