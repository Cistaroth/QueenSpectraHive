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

let currentFile = null;

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
uploadBtn.addEventListener('click', async () => {
  if (!currentFile) return;
  audioEl.pause();
  playIcon.style.display  = '';
  pauseIcon.style.display = 'none';
  uploadBtn.disabled = true;
  showLoading();

  const formData = new FormData();
  formData.append('file', currentFile);

  try {
    const res = await fetch('/spectral-inference', {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      let detail = `Server error ${res.status}`;
      try { const j = await res.json(); detail = j.detail || detail; } catch (_) {}
      showResult(false, null, detail);
      return;
    }

    const data = await res.json();
    showResult(data.boolean === true, data.probability, null);

  } catch (err) {
    showResult(false, null, 'Could not reach the inference service. Please check your connection and try again.');
  }
});

// ── Reset from result screen ────────────────────────────────────────────────
document.getElementById('btnAgain').addEventListener('click', () => {
  resultScreen.classList.remove('visible');
  resetAll();
});