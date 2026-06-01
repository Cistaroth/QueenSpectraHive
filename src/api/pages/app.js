const dropZone     = document.getElementById('dropZone');
const fileInput    = document.getElementById('fileInput');
const fileError    = document.getElementById('fileError');
const fileErrorMsg = document.getElementById('fileErrorMsg');
const audioPanel   = document.getElementById('audioPanel');
const fileName     = document.getElementById('fileName');
const fileSize     = document.getElementById('fileSize');
const uploadBtn    = document.getElementById('uploadBtn');
const deleteBtn    = document.getElementById('deleteBtn');
const formatsHint  = document.getElementById('formatsHint');
const audioEl      = document.getElementById('audioEl');
const playBtn      = document.getElementById('playBtn');
const playIcon     = document.getElementById('playIcon');
const pauseIcon    = document.getElementById('pauseIcon');
const progressFill = document.getElementById('progressFill');
const progressWrap = document.getElementById('progressWrap');
const timeDisplay  = document.getElementById('timeDisplay');
const loadingScreen= document.getElementById('loadingScreen');
const resultScreen = document.getElementById('resultScreen');
const dataToggle   = document.getElementById('dataToggle');
const tabularPanel = document.getElementById('tabularPanel');
const tabularClose = document.getElementById('tabularClose');
const resetTabular = document.getElementById('resetTabular');

let currentFile = null;
let tabularOpen  = false;

// ── Tabular panel toggle ────────────────────────────────────────────────────
function openTabularPanel() {
  tabularPanel.style.display = 'flex';
  void tabularPanel.getBoundingClientRect(); // force reflow so transition fires
  tabularPanel.classList.add('visible');
  tabularPanel.setAttribute('aria-hidden', 'false');
  dataToggle.classList.add('active');
  tabularOpen = true;
  document.body.style.overflowY = 'auto';
}

function closeTabularPanel() {
  tabularPanel.classList.remove('visible');
  tabularPanel.setAttribute('aria-hidden', 'true');
  dataToggle.classList.remove('active');
  tabularOpen = false;
  setTimeout(() => {
    if (!tabularOpen) {
      tabularPanel.style.display = 'none';
      document.body.style.overflowY = '';
    }
  }, 380);
}

dataToggle.addEventListener('click', () => {
  if (tabularOpen) closeTabularPanel(); else openTabularPanel();
});
tabularClose.addEventListener('click', closeTabularPanel);

// Every tabular field the API requires, mapped to its DOM id, the server-side
// form field name, and a human label used in validation/error messages.
const TABULAR_FIELDS = [
  { id: 'fieldHiveTemp',        name: 'hive_temp',        label: 'Hive Temperature' },
  { id: 'fieldHiveHumidity',    name: 'hive_humidity',    label: 'Hive Humidity' },
  { id: 'fieldHivePressure',    name: 'hive_pressure',    label: 'Hive Pressure' },
  { id: 'fieldFrames',          name: 'frames',           label: 'Frames' },
  { id: 'fieldWeatherTemp',     name: 'weather_temp',     label: 'Weather Temperature' },
  { id: 'fieldWeatherHumidity', name: 'weather_humidity', label: 'Weather Humidity' },
  { id: 'fieldWeatherPressure', name: 'weather_pressure', label: 'Weather Pressure' },
  { id: 'fieldWind',            name: 'wind_speed',       label: 'Wind Speed' },
  { id: 'fieldCloud',           name: 'cloud_coverage',   label: 'Cloud Coverage' },
  { id: 'fieldDate',            name: 'date',             label: 'Date & Time' },
  { id: 'fieldDevice',          name: 'device',           label: 'Device №' },
  { id: 'fieldHive',            name: 'hive_number',      label: 'Hive №' },
];

const tabularValidation = document.getElementById('tabularValidation');

function showTabularValidation(msg) {
  tabularValidation.textContent = msg;
  tabularValidation.classList.add('visible');
}

function hideTabularValidation() {
  tabularValidation.classList.remove('visible');
}

resetTabular.addEventListener('click', () => {
  TABULAR_FIELDS.forEach(f => {
    const el = document.getElementById(f.id);
    el.value = '';
    el.classList.remove('invalid');
  });
  hideTabularValidation();
});

// Clearing a field's error as soon as the user starts correcting it.
TABULAR_FIELDS.forEach(f => {
  document.getElementById(f.id).addEventListener('input', () => {
    document.getElementById(f.id).classList.remove('invalid');
    if (!document.querySelector('.field-input.invalid')) hideTabularValidation();
  });
});

// Mark every empty required field and return the list of missing ones.
function validateTabular() {
  const missing = [];
  TABULAR_FIELDS.forEach(f => {
    const el = document.getElementById(f.id);
    if (el.value.trim() === '') {
      el.classList.add('invalid');
      missing.push(f);
    } else {
      el.classList.remove('invalid');
    }
  });
  return missing;
}

// Collect tabular fields into an object. All are required, so by the time this
// runs (after validateTabular) every field has a value.
function getTabularData() {
  const out = {};
  TABULAR_FIELDS.forEach(f => {
    const v = document.getElementById(f.id).value.trim();
    if (v !== '') out[f.name] = v;
  });
  return out;
}

// Gauge geometry — must match the <circle r="70"> in index.html
const RING_CIRC = 2 * Math.PI * 70;

function fmt(b) {
  if (b < 1024)    return b + ' B';
  if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
  return (b / 1048576).toFixed(1) + ' MB';
}

function fmtTime(s) {
  const m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return m + ':' + String(sec).padStart(2, '0');
}

function isWav(f) {
  return f.name.toLowerCase().endsWith('.wav') || f.type === 'audio/wav' || f.type === 'audio/x-wav';
}

// Accepts probability as a fraction (0–1) or a percentage (0–100); returns 0–1 or null.
function normalizeProb(p) {
  if (typeof p !== 'number' || isNaN(p)) return null;
  if (p > 1) p = p / 100;
  return Math.max(0, Math.min(1, p));
}

function showFile(file) {
  fileError.classList.remove('visible');
  if (!isWav(file)) {
    fileError.classList.add('visible');
    return;
  }
  currentFile = file;
  audioEl.src = URL.createObjectURL(file);
  fileName.textContent = file.name;
  fileSize.textContent = fmt(file.size);
  dropZone.classList.add('hidden');
  formatsHint.classList.add('hidden');
  audioPanel.classList.add('visible');
}

function resetAll() {
  audioEl.pause();
  audioEl.src = '';
  playIcon.style.display  = '';
  pauseIcon.style.display = 'none';
  progressFill.style.width = '0%';
  timeDisplay.textContent = '0:00 / 0:00';
  uploadBtn.textContent = 'Upload File';
  uploadBtn.disabled = false;
  audioPanel.classList.remove('visible');
  dropZone.classList.remove('hidden');
  formatsHint.classList.remove('hidden');
  fileError.classList.remove('visible');
  fileInput.value = '';
  currentFile = null;
}

function showLoading() {
  loadingScreen.classList.add('visible');
}

function hideLoading() {
  loadingScreen.classList.remove('visible');
}

// Animate a numeric count-up into el (rendered as a percentage).
function countUp(el, target) {
  const dur = 1100, start = performance.now();
  function tick(now) {
    const t = Math.min((now - start) / dur, 1);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = Math.round(eased * target) + '%';
    if (t < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// Drive the radial gauge + the numeric readout.
// kind is 'queen' | 'no-queen'; prob is 0–1 or null (null => readout hidden).
function setGauge(prob, kind) {
  const gaugeFill         = document.getElementById('gaugeFill');
  const confidenceReadout = document.getElementById('confidenceReadout');
  const confidencePct     = document.getElementById('confidencePct');

  gaugeFill.setAttribute('class', 'gauge-fill ' + kind);
  gaugeFill.style.strokeDasharray = RING_CIRC;

  if (prob === null) {
    // No probability available (e.g. error) — leave the ring empty, hide readout.
    gaugeFill.style.transition = 'none';
    gaugeFill.style.strokeDashoffset = RING_CIRC;
    confidenceReadout.className = 'confidence-readout hidden';
    return;
  }

  // Reset the ring to empty, then animate to the target on the next frame.
  gaugeFill.style.transition = 'none';
  gaugeFill.style.strokeDashoffset = RING_CIRC;
  void gaugeFill.getBoundingClientRect(); // force reflow so the reset "sticks"
  requestAnimationFrame(() => {
    gaugeFill.style.transition = '';
    gaugeFill.style.strokeDashoffset = RING_CIRC * (1 - prob);
  });

  confidenceReadout.className = 'confidence-readout ' + kind;
  confidencePct.textContent = '0%';
  countUp(confidencePct, Math.round(prob * 100));
}

function showResult(isQueen, probability, errorMsg) {
  hideLoading();

  const iconWrap   = document.getElementById('resultIconWrap');
  const glow       = document.getElementById('resultGlow');
  const label      = document.getElementById('resultLabel');
  const headline   = document.getElementById('resultHeadline');
  const sub        = document.getElementById('resultSub');
  const detail     = document.getElementById('resultDetail');
  const detailText = document.getElementById('resultDetailText');
  const detailIcon = document.getElementById('resultDetailIcon');

  const prob = normalizeProb(probability);

  // Remove any previously injected icon
  const oldIcon = iconWrap.querySelector('.result-bee-icon');
  if (oldIcon) oldIcon.remove();

  if (errorMsg) {
    glow.className = 'result-glow no-queen';
    label.className = 'result-label no-queen';
    label.textContent = 'Analysis Error';
    headline.className = 'result-headline no-queen';
    headline.textContent = 'Something Went Wrong';
    sub.textContent = 'The inference service encountered an issue.';
    detail.className = 'result-detail no-queen';
    detailText.textContent = errorMsg;
    detailIcon.setAttribute('stroke', '#f08060');
    setGauge(null, 'no-queen');
    iconWrap.insertAdjacentHTML('beforeend', `
      <svg class="result-bee-icon" width="60" height="60" viewBox="0 0 24 24" fill="none"
           stroke="#f08060" stroke-width="1.5" stroke-linecap="round">
        <circle cx="12" cy="12" r="10"/>
        <line x1="12" y1="8" x2="12" y2="13"/>
        <circle cx="12" cy="17" r="1" fill="#f08060"/>
      </svg>`);

  } else if (isQueen) {
    glow.className = 'result-glow queen';
    label.className = 'result-label queen';
    label.textContent = 'Detection Result';
    headline.className = 'result-headline queen';
    headline.textContent = 'Queen Bee Detected';
    sub.textContent = "The presence of a Queen Bee has been confirmed in the hive.";
    detail.className = 'result-detail';
    detailText.textContent = 'The spectral analysis of the submitted audio found characteristic queen bee frequency signatures. The hive has a queen.';
    detailIcon.setAttribute('stroke', '#e8970a');
    setGauge(prob, 'queen');
    iconWrap.insertAdjacentHTML('beforeend', `
      <svg class="result-bee-icon" style="animation:crownBob 2s ease-in-out infinite"
           width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
        <polygon points="32,8 44,26 58,14 54,44 10,44 6,14 20,26"
                 fill="#f5c842" stroke="#c47a00" stroke-width="2" stroke-linejoin="round"/>
        <circle cx="32" cy="8"  r="4" fill="#e8970a"/>
        <circle cx="58" cy="14" r="4" fill="#e8970a"/>
        <circle cx="6"  cy="14" r="4" fill="#e8970a"/>
        <rect x="10" y="44" width="44" height="6" rx="2" fill="#c47a00"/>
        <rect x="14" y="50" width="36" height="4" rx="1" fill="#e8970a" opacity="0.5"/>
      </svg>`);

  } else {
    glow.className = 'result-glow no-queen';
    label.className = 'result-label no-queen';
    label.textContent = 'Detection Result';
    headline.className = 'result-headline no-queen';
    headline.textContent = 'No Queen Detected';
    sub.textContent = 'The hive appears to be queenless at this time.';
    detail.className = 'result-detail no-queen';
    detailText.textContent = 'No queen bee frequency signatures were identified in the spectral data. The colony appears to be queenless.';
    detailIcon.setAttribute('stroke', '#f08060');
    setGauge(prob, 'no-queen');
    iconWrap.insertAdjacentHTML('beforeend', `
      <svg class="result-bee-icon" width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
        <polygon points="32,6 56,19 56,45 32,58 8,45 8,19"
                 fill="none" stroke="#f08060" stroke-width="2.5" stroke-linejoin="round" opacity="0.7"/>
        <polygon points="32,18 44,25 44,39 32,46 20,39 20,25"
                 fill="rgba(240,128,96,0.12)" stroke="#f08060" stroke-width="1.5" stroke-linejoin="round" opacity="0.5"/>
        <line x1="22" y1="22" x2="42" y2="42" stroke="#f08060" stroke-width="2.5" stroke-linecap="round" opacity="0.6"/>
        <line x1="42" y1="22" x2="22" y2="42" stroke="#f08060" stroke-width="2.5" stroke-linecap="round" opacity="0.6"/>
      </svg>`);
  }

  resultScreen.classList.add('visible');
}

// ── File input ──────────────────────────────────────────────────────────────
fileInput.addEventListener('change', () => { if (fileInput.files[0]) showFile(fileInput.files[0]); });

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) showFile(file);
});

// ── Player ──────────────────────────────────────────────────────────────────
playBtn.addEventListener('click', () => {
  if (audioEl.paused) {
    audioEl.play();
    playIcon.style.display  = 'none';
    pauseIcon.style.display = '';
  } else {
    audioEl.pause();
    playIcon.style.display  = '';
    pauseIcon.style.display = 'none';
  }
});

audioEl.addEventListener('timeupdate', () => {
  if (!audioEl.duration) return;
  progressFill.style.width = (audioEl.currentTime / audioEl.duration * 100) + '%';
  timeDisplay.textContent = fmtTime(audioEl.currentTime) + ' / ' + fmtTime(audioEl.duration);
});

audioEl.addEventListener('ended', () => {
  playIcon.style.display  = '';
  pauseIcon.style.display = 'none';
  progressFill.style.width = '0%';
  audioEl.currentTime = 0;
});

progressWrap.addEventListener('click', e => {
  if (!audioEl.duration) return;
  const rect = progressWrap.getBoundingClientRect();
  audioEl.currentTime = ((e.clientX - rect.left) / rect.width) * audioEl.duration;
});

deleteBtn.addEventListener('click', resetAll);

// ── Upload & inference ──────────────────────────────────────────────────────
// Turn a server error response into a human-readable message. FastAPI 422
// validation errors arrive as detail: [{loc, msg, ...}], which we map back to
// the field labels; other errors use detail as a plain string.
function describeServerError(status, body) {
  if (body && Array.isArray(body.detail)) {
    const labels = body.detail.map(e => {
      const key = Array.isArray(e.loc) ? e.loc[e.loc.length - 1] : null;
      if (key === 'audio_file') return 'Audio file';
      const field = TABULAR_FIELDS.find(f => f.name === key);
      return field ? field.label : (key || 'a field');
    });
    return 'The server rejected some required data: ' + [...new Set(labels)].join(', ') + '.';
  }
  if (body && typeof body.detail === 'string') return body.detail;
  return `Server error ${status}`;
}

async function safeJson(res) {
  try { return await res.json(); } catch (_) { return null; }
}

uploadBtn.addEventListener('click', async () => {
  if (!currentFile) return;

  // Block the request if any required hive-data field is empty — open the
  // panel, highlight what's missing, and let the user fill it in.
  const missing = validateTabular();
  if (missing.length) {
    openTabularPanel();
    showTabularValidation(
      `All ${TABULAR_FIELDS.length} hive-data fields are required. ` +
      `Please fill in: ${missing.map(m => m.label).join(', ')}.`
    );
    return;
  }
  hideTabularValidation();

  audioEl.pause();
  playIcon.style.display  = '';
  pauseIcon.style.display = 'none';
  uploadBtn.disabled = true;
  showLoading();

  const formData = new FormData();
  const wavFile = new File([currentFile], currentFile.name, { type: 'audio/wav' });
  formData.append('audio_file', wavFile);

  const tabular = getTabularData();
  for (const [key, val] of Object.entries(tabular)) {
    formData.append(key, val);
  }

  try {
    const res = await fetch('/spectral-inference', {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      showResult(false, null, describeServerError(res.status, await safeJson(res)));
      return;
    }

    const data = await res.json();
    showResult(data.queen_presence === true, data.probability, null);

  } catch (err) {
    showResult(false, null, 'Could not reach the inference service. Please check your connection and try again.');
  }
});

// ── Reset from result screen ────────────────────────────────────────────────
document.getElementById('btnAgain').addEventListener('click', () => {
  resultScreen.classList.remove('visible');
  resetAll();
});