const QUALITY = {
  minFocus: 42,
  minBrightness: 45,
  maxBrightness: 215,
  minContrast: 22,
  maxMotion: 8.5,
  stableMs: 650,
  cooldownMs: 1200,
  analysisWidth: 320,
  jpegQuality: 0.95
};

const MODE_LABELS = {
  reference: "Referans Çek",
  original: "Orijinal Çek",
  copy: "Kopya Çek",
  test: "Test Et"
};

const video = document.getElementById("video");
const analysisCanvas = document.getElementById("analysisCanvas");
const captureCanvas = document.getElementById("captureCanvas");
const refInfo = document.getElementById("refInfo");
const modeBadge = document.getElementById("modeBadge");
const readyOverlay = document.getElementById("readyOverlay");
const focusVal = document.getElementById("focusVal");
const brightnessVal = document.getElementById("brightnessVal");
const contrastVal = document.getElementById("contrastVal");
const motionVal = document.getElementById("motionVal");
const qualityMessage = document.getElementById("qualityMessage");
const autoState = document.getElementById("autoState");
const statusBox = document.getElementById("statusBox");
const resultBox = document.getElementById("resultBox");
const stageButtons = Array.from(document.querySelectorAll(".stage-btn"));

let currentMode = null;
let stream = null;
let rafId = null;
let previousGray = null;
let readySince = null;
let isUploading = false;
let lastCaptureAt = 0;
let lastMetrics = null;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function setOverlay(text, cls) {
  readyOverlay.textContent = text;
  readyOverlay.className = `ready-overlay ${cls || ""}`.trim();
}

function setStatus(data) {
  let cls = "review";
  if (data.final_user_status === "ORIGINAL_APPROVED" || data.final_user_status === "REFERENCE_SAVED") cls = "ok";
  if (data.final_user_status === "COPY_RISK_REJECTED") cls = "bad";

  statusBox.className = `status-box ${cls}`;
  statusBox.innerHTML = `${data.final_user_status || "OK"}<br>${data.final_user_message || "Kaydedildi."}`;
  resultBox.textContent = JSON.stringify(data, null, 2);
}

function setError(message) {
  statusBox.className = "status-box bad";
  statusBox.textContent = message;
  resultBox.textContent = message;
}

async function loadRefs() {
  try {
    const res = await fetch("/api/refs");
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Reference load failed");
    refInfo.textContent = `Referans: ${data.reference_count}`;
  } catch (err) {
    refInfo.textContent = "Referans: 0";
  }
}

function stopCamera() {
  if (rafId) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  previousGray = null;
  readySince = null;
}

async function startCamera(mode) {
  currentMode = mode;
  previousGray = null;
  readySince = null;
  isUploading = false;
  lastCaptureAt = 0;

  stageButtons.forEach(btn => btn.classList.toggle("active", btn.dataset.mode === mode));
  modeBadge.textContent = MODE_LABELS[mode] || mode;
  statusBox.className = "status-box empty";
  statusBox.textContent = "Otomatik kalite kontrol bekleniyor.";
  resultBox.textContent = "";
  autoState.textContent = "Kamera açılıyor...";
  qualityMessage.textContent = "CDP alanını çerçeveye doldur ve telefonu sabit tut.";
  setOverlay("Kamera açılıyor", "wait");

  stopCamera();

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1920 },
        height: { ideal: 1080 }
      },
      audio: false
    });

    video.srcObject = stream;
    await video.play();
    autoState.textContent = "Manuel çekim yok. Sistem netlik yeterliyse otomatik çekecek.";
    rafId = requestAnimationFrame(analyzeLoop);
  } catch (err) {
    setOverlay("Kamera açılamadı", "bad");
    setError(`Kamera açılamadı: ${err.message || err}`);
  }
}

function computeMetrics() {
  if (!video.videoWidth || !video.videoHeight) return null;

  const aw = QUALITY.analysisWidth;
  const ah = Math.round(aw * video.videoHeight / video.videoWidth);

  analysisCanvas.width = aw;
  analysisCanvas.height = ah;

  const ctx = analysisCanvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(video, 0, 0, aw, ah);

  const imageData = ctx.getImageData(0, 0, aw, ah);
  const data = imageData.data;
  const gray = new Float32Array(aw * ah);

  let sum = 0;
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    gray[j] = y;
    sum += y;
  }

  const mean = sum / gray.length;

  let variance = 0;
  for (let i = 0; i < gray.length; i++) {
    const d = gray[i] - mean;
    variance += d * d;
  }
  variance /= gray.length;
  const contrast = Math.sqrt(variance);

  let lapSum = 0;
  let lapSqSum = 0;
  let count = 0;

  for (let y = 1; y < ah - 1; y++) {
    for (let x = 1; x < aw - 1; x++) {
      const idx = y * aw + x;
      const lap = (
        -4 * gray[idx] +
        gray[idx - 1] +
        gray[idx + 1] +
        gray[idx - aw] +
        gray[idx + aw]
      );
      lapSum += lap;
      lapSqSum += lap * lap;
      count++;
    }
  }

  const lapMean = lapSum / Math.max(1, count);
  const lapVar = lapSqSum / Math.max(1, count) - lapMean * lapMean;
  const focusScore = clamp((lapVar - 20) / 120 * 100, 0, 100);

  let motion = 0;
  if (previousGray && previousGray.length === gray.length) {
    let diff = 0;
    for (let i = 0; i < gray.length; i++) {
      diff += Math.abs(gray[i] - previousGray[i]);
    }
    motion = diff / gray.length;
  } else {
    motion = 99;
  }

  previousGray = gray;

  const passFocus = focusScore >= QUALITY.minFocus;
  const passBrightness = mean >= QUALITY.minBrightness && mean <= QUALITY.maxBrightness;
  const passContrast = contrast >= QUALITY.minContrast;
  const passMotion = motion <= QUALITY.maxMotion;

  const pass = passFocus && passBrightness && passContrast && passMotion;

  return {
    focusScore,
    lapVar,
    brightness: mean,
    contrast,
    motion,
    pass,
    passFocus,
    passBrightness,
    passContrast,
    passMotion
  };
}

function updateQualityUI(metrics) {
  focusVal.textContent = Math.round(metrics.focusScore);
  brightnessVal.textContent = Math.round(metrics.brightness);
  contrastVal.textContent = Math.round(metrics.contrast);
  motionVal.textContent = metrics.motion >= 99 ? "--" : metrics.motion.toFixed(1);

  const issues = [];
  if (!metrics.passFocus) issues.push("biraz daha netleştir / yaklaştır");
  if (!metrics.passBrightness) issues.push("ışığı düzelt");
  if (!metrics.passContrast) issues.push("CDP alanını daha düzgün kadraja al");
  if (!metrics.passMotion) issues.push("daha sabit tut");

  if (issues.length === 0) {
    qualityMessage.textContent = "Kalite uygun. Sabit kalırsa otomatik çekilecek.";
  } else {
    qualityMessage.textContent = "Bekleniyor: " + issues.join(", ") + ".";
  }
}

function analyzeLoop(timestamp) {
  if (!currentMode || isUploading) {
    rafId = requestAnimationFrame(analyzeLoop);
    return;
  }

  const metrics = computeMetrics();
  if (!metrics) {
    rafId = requestAnimationFrame(analyzeLoop);
    return;
  }

  lastMetrics = metrics;
  updateQualityUI(metrics);

  if (metrics.pass) {
    if (!readySince) readySince = timestamp;
    const stableFor = timestamp - readySince;
    const remaining = Math.max(0, QUALITY.stableMs - stableFor);

    if (stableFor >= QUALITY.stableMs) {
      setOverlay("Hazır — otomatik çekiliyor", "ready");
      autoState.textContent = "Kalite uygun. Fotoğraf otomatik çekiliyor ve kaydediliyor.";

      const now = Date.now();
      if (now - lastCaptureAt > QUALITY.cooldownMs) {
        lastCaptureAt = now;
        captureAndUpload();
        return;
      }
    } else {
      setOverlay(`Hazır — ${Math.ceil(remaining / 100) / 10}s sabit tut`, "ready");
      autoState.textContent = "Sistem CDP'yi otomatik çekmek için kısa stabil süre bekliyor.";
    }
  } else {
    readySince = null;
    setOverlay("Hazır değil", "wait");
    autoState.textContent = "Manuel çekim yok. Kalite yeterli olunca sistem otomatik çekecek.";
  }

  rafId = requestAnimationFrame(analyzeLoop);
}

async function captureAndUpload() {
  if (!currentMode || isUploading) return;

  isUploading = true;
  setOverlay("Kaydediliyor", "ready");

  try {
    const vw = video.videoWidth;
    const vh = video.videoHeight;

    captureCanvas.width = vw;
    captureCanvas.height = vh;
    const ctx = captureCanvas.getContext("2d");
    ctx.drawImage(video, 0, 0, vw, vh);

    const blob = await new Promise(resolve => {
      captureCanvas.toBlob(resolve, "image/jpeg", QUALITY.jpegQuality);
    });

    if (!blob) throw new Error("Capture blob oluşturulamadı.");

    const formData = new FormData();
    const filename = `${currentMode}_${Date.now()}.jpg`;
    formData.append("file", blob, filename);

    const res = await fetch(`/api/capture/${currentMode}`, {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed");

    data.client_quality = {
      focusScore: Math.round(lastMetrics.focusScore * 100) / 100,
      lapVar: Math.round(lastMetrics.lapVar * 100) / 100,
      brightness: Math.round(lastMetrics.brightness * 100) / 100,
      contrast: Math.round(lastMetrics.contrast * 100) / 100,
      motion: Math.round(lastMetrics.motion * 100) / 100
    };

    setStatus(data);
    setOverlay("Çekildi ve kaydedildi", "ready");
    autoState.textContent = "Bu aşama tamamlandı. Sonraki aşamayı seçebilirsin.";

    stopCamera();
    await loadRefs();
  } catch (err) {
    setOverlay("Hata", "bad");
    setError(`Otomatik kayıt hatası: ${err.message || err}`);
    isUploading = false;
    readySince = null;
    rafId = requestAnimationFrame(analyzeLoop);
  }
}

stageButtons.forEach(btn => {
  btn.addEventListener("click", () => {
    startCamera(btn.dataset.mode);
  });
});

window.addEventListener("beforeunload", stopCamera);
loadRefs();
