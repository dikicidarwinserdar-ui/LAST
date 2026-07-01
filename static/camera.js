const QUALITY = {
  minFocus: 38,
  minBrightness: 42,
  maxBrightness: 225,
  minContrast: 18,
  maxMotion: 10.5,
  stableMs: 520,
  cooldownMs: 1200,
  analysisWidth: 360,
  jpegQuality: 0.96,

  minBoxCoverage: 0.035,
  maxBoxCoverage: 0.96,
  maxCenterError: 0.42,
  minAspect: 0.42,
  maxAspect: 2.35,
  minInkFill: 0.006,
  maxInkFill: 0.90,
  minBoxStability: 0.35
};

const MODE_LABELS = {
  reference: "Referans Yükle",
  original: "Orijinal Baskı Yükle",
  copy: "Sahte / Kopya Baskı Yükle",
  test: "Test Et"
};

const MODE_SUBTITLES = {
  reference: "Backend marker/CDP dış alanını bulup normalize edecek. Çekimden sonra kaydet butonuyla onaylayacaksın.",
  original: "Orijinal baskı otomatik çekilecek, backend normalize edip skorlayacak.",
  copy: "Kopya/sahte baskı otomatik çekilecek, backend normalize edip skorlayacak.",
  test: "Test için full frame alınacak, backend marker/CDP crop yapıp referanslarla karşılaştıracak."
};

const video = document.getElementById("video");
const overlayCanvas = document.getElementById("overlayCanvas");
const analysisCanvas = document.getElementById("analysisCanvas");
const captureCanvas = document.getElementById("captureCanvas");

const cameraPanel = document.getElementById("cameraPanel");
const pendingPanel = document.getElementById("pendingPanel");

const modeTitle = document.getElementById("modeTitle");
const modeSubtitle = document.getElementById("modeSubtitle");
const readyOverlay = document.getElementById("readyOverlay");

const focusVal = document.getElementById("focusVal");
const brightnessVal = document.getElementById("brightnessVal");
const contrastVal = document.getElementById("contrastVal");
const motionVal = document.getElementById("motionVal");
const boxVal = document.getElementById("boxVal");

const qualityMessage = document.getElementById("qualityMessage");
const autoState = document.getElementById("autoState");

const pendingImage = document.getElementById("pendingImage");
const pendingStatus = document.getElementById("pendingStatus");
const resultBox = document.getElementById("resultBox");

const savePendingBtn = document.getElementById("savePendingBtn");
const retakeBtn = document.getElementById("retakeBtn");
const discardBtn = document.getElementById("discardBtn");

const refreshSavedBtn = document.getElementById("refreshSavedBtn");
const savedGrid = document.getElementById("savedGrid");
const healthInfo = document.getElementById("healthInfo");
const stopCameraBtn = document.getElementById("stopCameraBtn");

const navButtons = Array.from(document.querySelectorAll(".nav-btn"));
const pages = Array.from(document.querySelectorAll(".page"));
const startModeButtons = Array.from(document.querySelectorAll(".start-mode"));

let currentMode = null;
let stream = null;
let rafId = null;
let previousGray = null;
let previousBox = null;
let readySince = null;
let isUploading = false;
let lastCaptureAt = 0;
let lastMetrics = null;
let pendingData = null;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function round2(v) {
  if (typeof v !== "number" || !isFinite(v)) return 0;
  return Math.round(v * 100) / 100;
}

function statusToClassLabel(status) {
  if (status === "ORIGINAL_APPROVED") return "ORİJİNAL";
  if (status === "COPY_RISK_REJECTED") return "KOPYA / SAHTE RİSKİ";
  if (status === "RETAKE_REQUIRED") return "TEKRAR ÇEKİM";
  if (status === "REFERENCE_READY_TO_SAVE" || status === "REFERENCE_SAVED") return "REFERANS";
  if (status === "NO_REFERENCE") return "REFERANS YOK";
  return status || "-";
}

function buildDecisionHtml(data) {
  const status = data.final_user_status || "OK";
  const classLabel = statusToClassLabel(status);
  const scores = data.scores || {};
  const serverCrop = data.server_crop || {};

  const base = scores.base_score ?? "-";
  const adjusted = scores.adjusted_score ?? "-";
  const risk = scores.copy_risk_score ?? "-";
  const ssim = scores.ssim_score ?? "-";
  const iou = scores.mask_iou ?? "-";
  const edge = scores.edge_f1 ?? "-";
  const cropMethod = serverCrop.method || "-";
  const cropConfidence = serverCrop.confidence !== undefined ? Math.round(serverCrop.confidence * 100) + "%" : "-";

  return `
    <div style="font-size:18px;font-weight:900;margin-bottom:6px;">${status}</div>
    <div style="display:inline-block;padding:6px 10px;border-radius:999px;background:rgba(255,255,255,0.14);font-size:13px;font-weight:900;margin-bottom:8px;">
      SINIF: ${classLabel}
    </div>
    <div style="font-size:13px;line-height:1.4;margin-top:8px;margin-bottom:10px;">${data.final_user_message || ""}</div>
    <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:10px;">
      <div style="background:rgba(255,255,255,0.10);padding:8px;border-radius:10px;"><span style="display:block;font-size:11px;opacity:.75;">Base</span><b>${base}</b></div>
      <div style="background:rgba(255,255,255,0.10);padding:8px;border-radius:10px;"><span style="display:block;font-size:11px;opacity:.75;">Adjusted</span><b>${adjusted}</b></div>
      <div style="background:rgba(255,255,255,0.10);padding:8px;border-radius:10px;"><span style="display:block;font-size:11px;opacity:.75;">Copy Risk</span><b>${risk}</b></div>
      <div style="background:rgba(255,255,255,0.10);padding:8px;border-radius:10px;"><span style="display:block;font-size:11px;opacity:.75;">SSIM</span><b>${ssim}</b></div>
      <div style="background:rgba(255,255,255,0.10);padding:8px;border-radius:10px;"><span style="display:block;font-size:11px;opacity:.75;">Mask IoU</span><b>${iou}</b></div>
      <div style="background:rgba(255,255,255,0.10);padding:8px;border-radius:10px;"><span style="display:block;font-size:11px;opacity:.75;">Edge F1</span><b>${edge}</b></div>
      <div style="background:rgba(255,255,255,0.10);padding:8px;border-radius:10px;"><span style="display:block;font-size:11px;opacity:.75;">Crop</span><b>${cropMethod}</b></div>
      <div style="background:rgba(255,255,255,0.10);padding:8px;border-radius:10px;"><span style="display:block;font-size:11px;opacity:.75;">Crop Conf.</span><b>${cropConfidence}</b></div>
      <div style="background:rgba(255,255,255,0.10);padding:8px;border-radius:10px;"><span style="display:block;font-size:11px;opacity:.75;">Markers</span><b>${serverCrop.marker_count ?? "-"}</b></div>
    </div>
  `;
}

function setOverlay(text, cls) {
  if (!readyOverlay) return;
  readyOverlay.textContent = text;
  readyOverlay.className = `ready-overlay ${cls || ""}`.trim();
}

function showPage(pageName) {
  navButtons.forEach(btn => btn.classList.toggle("active", btn.dataset.page === pageName));
  pages.forEach(page => page.classList.toggle("active", page.id === `page-${pageName}`));
  if (pageName === "saved") loadSavedRecords();
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
  previousBox = null;
  readySince = null;
}

function setStatusBox(el, data) {
  let cls = "review";
  const status = data.final_user_status || "OK";
  if (status === "ORIGINAL_APPROVED" || status === "REFERENCE_SAVED" || status === "REFERENCE_READY_TO_SAVE") cls = "ok";
  if (status === "COPY_RISK_REJECTED") cls = "bad";
  if (status === "NO_REFERENCE" || status === "RETAKE_REQUIRED") cls = "review";
  el.className = `status-box ${cls}`;
  el.innerHTML = buildDecisionHtml(data);
}

function setError(message) {
  pendingStatus.className = "status-box bad";
  pendingStatus.textContent = message;
  resultBox.textContent = message;
}

async function loadHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    const refCount = data.reference_count ?? 0;
    const records = data.records ?? 0;
    healthInfo.textContent = `Referans: ${refCount} | Kayıt: ${records}`;
  } catch (err) {
    healthInfo.textContent = "Sistem kontrol edilemedi";
  }
}

async function startCamera(mode) {
  currentMode = mode;
  previousGray = null;
  previousBox = null;
  readySince = null;
  isUploading = false;
  lastCaptureAt = 0;
  pendingData = null;

  stopCamera();

  pendingPanel.classList.remove("active");
  cameraPanel.classList.add("active");

  modeTitle.textContent = MODE_LABELS[mode] || mode;
  modeSubtitle.textContent = MODE_SUBTITLES[mode] || "";

  autoState.textContent = "Kamera açılıyor. Manuel çekim yok; uygun koşulda full frame alınır ve backend crop yapar.";
  qualityMessage.textContent = "CDP/marker alanını mavi kareye getir. Çok dibine sokmana gerek yok.";
  setOverlay("Kamera açılıyor", "wait");

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

    autoState.textContent = "Sarı kutu sadece canlı yardımcıdır. Asıl marker/CDP crop backend OpenCV ile yapılacak.";
    rafId = requestAnimationFrame(analyzeLoop);
  } catch (err) {
    setOverlay("Kamera açılamadı", "bad");
    autoState.textContent = `Kamera açılamadı: ${err.message || err}`;
  }
}

function detectLiveInkBox(gray, aw, ah, mean, contrast) {
  const threshold = Math.min(155, Math.max(50, mean - contrast * 0.18));
  let minX = aw;
  let minY = ah;
  let maxX = 0;
  let maxY = 0;
  let blackCount = 0;

  const marginX = Math.floor(aw * 0.025);
  const marginY = Math.floor(ah * 0.025);

  for (let y = marginY; y < ah - marginY; y++) {
    for (let x = marginX; x < aw - marginX; x++) {
      const idx = y * aw + x;
      const v = gray[idx];
      if (v < threshold) {
        blackCount++;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (blackCount < aw * ah * 0.004) {
    return { candidate: false, score: 0, box: null };
  }

  const w = maxX - minX + 1;
  const h = maxY - minY + 1;
  const area = w * h;
  const coverage = area / Math.max(1, aw * ah);
  const aspect = w / Math.max(1, h);
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const centerError = Math.sqrt(Math.pow((centerX - aw / 2) / aw, 2) + Math.pow((centerY - ah / 2) / ah, 2));
  const inkFill = blackCount / Math.max(1, area);

  let boxStability = 1;
  if (previousBox) {
    const dx = Math.abs(centerX - previousBox.centerX) / aw;
    const dy = Math.abs(centerY - previousBox.centerY) / ah;
    const dw = Math.abs(w - previousBox.w) / aw;
    const dh = Math.abs(h - previousBox.h) / ah;
    const movement = dx + dy + dw + dh;
    boxStability = clamp(1 - movement * 3.0, 0, 1);
  }

  const candidate =
    coverage >= QUALITY.minBoxCoverage &&
    coverage <= QUALITY.maxBoxCoverage &&
    aspect >= QUALITY.minAspect &&
    aspect <= QUALITY.maxAspect &&
    centerError <= QUALITY.maxCenterError &&
    inkFill >= QUALITY.minInkFill &&
    inkFill <= QUALITY.maxInkFill;

  const stableCandidate = candidate && boxStability >= QUALITY.minBoxStability;

  let score = 0;
  score += coverage >= QUALITY.minBoxCoverage && coverage <= QUALITY.maxBoxCoverage ? 26 : 0;
  score += aspect >= QUALITY.minAspect && aspect <= QUALITY.maxAspect ? 22 : 0;
  score += centerError <= QUALITY.maxCenterError ? 22 : 0;
  score += inkFill >= QUALITY.minInkFill && inkFill <= QUALITY.maxInkFill ? 16 : 0;
  score += Math.round(boxStability * 14);
  score = clamp(score, 0, 100);

  previousBox = { minX, minY, maxX, maxY, w, h, centerX, centerY };

  return {
    candidate,
    stableCandidate,
    score,
    box: { minX, minY, maxX, maxY, w, h, coverage, aspect, centerError, inkFill, boxStability }
  };
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
      const lap = -4 * gray[idx] + gray[idx - 1] + gray[idx + 1] + gray[idx - aw] + gray[idx + aw];
      lapSum += lap;
      lapSqSum += lap * lap;
      count++;
    }
  }

  const lapMean = lapSum / Math.max(1, count);
  const lapVar = lapSqSum / Math.max(1, count) - lapMean * lapMean;
  const focusScore = clamp((lapVar - 18) / 118 * 100, 0, 100);

  let motion = 0;
  if (previousGray && previousGray.length === gray.length) {
    let diff = 0;
    for (let i = 0; i < gray.length; i++) diff += Math.abs(gray[i] - previousGray[i]);
    motion = diff / gray.length;
  } else {
    motion = 99;
  }
  previousGray = gray;

  const ink = detectLiveInkBox(gray, aw, ah, mean, contrast);

  const passFocus = focusScore >= QUALITY.minFocus;
  const passBrightness = mean >= QUALITY.minBrightness && mean <= QUALITY.maxBrightness;
  const passContrast = contrast >= QUALITY.minContrast;
  const passMotion = motion <= QUALITY.maxMotion;
  const passCdpCandidate = ink.stableCandidate || false;

  const pass = passFocus && passBrightness && passContrast && passMotion && passCdpCandidate;

  return {
    aw, ah,
    focusScore,
    lapVar,
    brightness: mean,
    contrast,
    motion,
    ink,
    pass,
    passFocus,
    passBrightness,
    passContrast,
    passMotion,
    passCdpCandidate
  };
}

function getVideoDisplayMapping() {
  const rect = video.getBoundingClientRect();
  const videoRatio = video.videoWidth / Math.max(1, video.videoHeight);
  const boxRatio = rect.width / Math.max(1, rect.height);
  let drawW, drawH, offsetX, offsetY;

  if (videoRatio > boxRatio) {
    drawH = rect.height;
    drawW = drawH * videoRatio;
    offsetX = (rect.width - drawW) / 2;
    offsetY = 0;
  } else {
    drawW = rect.width;
    drawH = drawW / videoRatio;
    offsetX = 0;
    offsetY = (rect.height - drawH) / 2;
  }
  return { rect, drawW, drawH, offsetX, offsetY };
}

function mapAnalysisBoxToDisplay(box, metrics) {
  const map = getVideoDisplayMapping();
  const sx = map.drawW / metrics.aw;
  const sy = map.drawH / metrics.ah;
  return {
    x: map.offsetX + box.minX * sx,
    y: map.offsetY + box.minY * sy,
    w: box.w * sx,
    h: box.h * sy
  };
}

function drawOverlay(metrics) {
  if (!overlayCanvas || !video) return;

  const rect = video.getBoundingClientRect();
  overlayCanvas.width = rect.width;
  overlayCanvas.height = rect.height;

  const ctx = overlayCanvas.getContext("2d");
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  const cw = overlayCanvas.width;
  const ch = overlayCanvas.height;
  const guideSize = Math.min(cw, ch) * 0.70;
  const gx = (cw - guideSize) / 2;
  const gy = (ch - guideSize) / 2;

  ctx.save();
  ctx.fillStyle = "rgba(0,0,0,0.16)";
  ctx.fillRect(0, 0, cw, ch);
  ctx.clearRect(gx, gy, guideSize, guideSize);

  ctx.strokeStyle = "rgba(59,130,246,0.95)";
  ctx.lineWidth = 4;
  ctx.setLineDash([14, 8]);
  ctx.strokeRect(gx, gy, guideSize, guideSize);
  ctx.setLineDash([]);

  ctx.fillStyle = "rgba(59,130,246,1)";
  ctx.font = "bold 15px Arial";
  ctx.fillText("CDP/marker alanını bu kareye getir", gx + 12, gy + 26);

  if (metrics && metrics.ink && metrics.ink.box) {
    const b = metrics.ink.box;
    const displayBox = mapAnalysisBoxToDisplay(b, metrics);
    const x = displayBox.x;
    const y = displayBox.y;
    const w = displayBox.w;
    const h = displayBox.h;

    const isReady = metrics.passCdpCandidate;
    const color = isReady ? "rgba(34,197,94,1)" : "rgba(251,191,36,1)";
    const fillColor = isReady ? "rgba(34,197,94,0.14)" : "rgba(251,191,36,0.14)";

    ctx.fillStyle = fillColor;
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = color;
    ctx.lineWidth = 5;
    ctx.strokeRect(x, y, w, h);

    ctx.fillStyle = color;
    ctx.font = "bold 16px Arial";
    ctx.fillText(isReady ? "CANLI ADAY STABIL" : "CDP/MARKER ADAYI", x + 8, Math.max(24, y - 10));

    ctx.font = "13px Arial";
    ctx.fillText(
      `Box ${Math.round(metrics.ink.score)} | Backend gerçek crop yapacak`,
      x + 8,
      Math.min(ch - 12, y + h + 20)
    );
  }

  ctx.restore();
}

function updateQualityUI(metrics) {
  focusVal.textContent = Math.round(metrics.focusScore);
  brightnessVal.textContent = Math.round(metrics.brightness);
  contrastVal.textContent = Math.round(metrics.contrast);
  motionVal.textContent = metrics.motion >= 99 ? "--" : metrics.motion.toFixed(1);
  boxVal.textContent = metrics.ink ? Math.round(metrics.ink.score) : "--";

  const issues = [];
  if (!metrics.passCdpCandidate) issues.push("CDP/marker alanını mavi kareye getir ve sabit tut");
  if (!metrics.passFocus) issues.push("biraz daha netleştir");
  if (!metrics.passBrightness) issues.push("ışığı düzelt");
  if (!metrics.passContrast) issues.push("kontrast düşük");
  if (!metrics.passMotion) issues.push("daha sabit tut");

  if (issues.length === 0) {
    qualityMessage.textContent = "Koşullar uygun. Full frame otomatik alınacak, backend marker/CDP crop yapacak.";
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
  drawOverlay(metrics);
  updateQualityUI(metrics);

  if (metrics.pass) {
    if (!readySince) readySince = timestamp;
    const stableFor = timestamp - readySince;
    const remaining = Math.max(0, QUALITY.stableMs - stableFor);

    if (stableFor >= QUALITY.stableMs) {
      setOverlay("Hazır — otomatik çekiliyor", "ready");
      autoState.textContent = "Full frame alınıyor. Backend marker/CDP dış alanını bulup normalize edecek.";
      const now = Date.now();
      if (now - lastCaptureAt > QUALITY.cooldownMs) {
        lastCaptureAt = now;
        captureAndPreview();
        return;
      }
    } else {
      setOverlay(`Hazır — ${Math.ceil(remaining / 100) / 10}s sabit tut`, "ready");
      autoState.textContent = "Kısa stabil süre bekleniyor.";
    }
  } else {
    readySince = null;
    setOverlay("Hazır değil", "wait");
    autoState.textContent = "Canlı kutu sadece yardımcıdır. Çekimden sonra asıl crop backend tarafından yapılacak.";
  }

  rafId = requestAnimationFrame(analyzeLoop);
}

async function captureAndPreview() {
  if (!currentMode || isUploading) return;
  isUploading = true;
  setOverlay("Çekiliyor", "ready");

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
    const filename = `${currentMode}_${Date.now()}_fullframe.jpg`;
    formData.append("file", blob, filename);
    formData.append("quality_json", JSON.stringify(buildQualityPayload()));

    const res = await fetch(`/api/preview/${currentMode}`, { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Preview upload failed");

    pendingData = data;
    const previewUrl = data.server_debug_image_url || data.pending_image_url;
    pendingImage.src = `${previewUrl}?t=${Date.now()}`;

    setStatusBox(pendingStatus, data);
    resultBox.textContent = JSON.stringify(data, null, 2);

    savePendingBtn.textContent = currentMode === "reference" ? "Referans olarak kaydet" : "Bu çekimi kaydet";
    savePendingBtn.disabled = false;

    cameraPanel.classList.remove("active");
    pendingPanel.classList.add("active");

    stopCamera();
    await loadHealth();
  } catch (err) {
    setOverlay("Hata", "bad");
    autoState.textContent = `Otomatik çekim hatası: ${err.message || err}`;
    isUploading = false;
    readySince = null;
    rafId = requestAnimationFrame(analyzeLoop);
  }
}

function buildQualityPayload() {
  if (!lastMetrics) return {};
  const b = lastMetrics.ink && lastMetrics.ink.box ? lastMetrics.ink.box : null;
  return {
    focusScore: round2(lastMetrics.focusScore),
    lapVar: round2(lastMetrics.lapVar),
    brightness: round2(lastMetrics.brightness),
    contrast: round2(lastMetrics.contrast),
    motion: round2(lastMetrics.motion),
    liveBoxScore: lastMetrics.ink ? round2(lastMetrics.ink.score) : 0,
    liveBoxCandidate: lastMetrics.ink ? !!lastMetrics.ink.candidate : false,
    liveBoxStableCandidate: lastMetrics.ink ? !!lastMetrics.ink.stableCandidate : false,
    liveBox: b ? {
      coverage: round2(b.coverage),
      aspect: round2(b.aspect),
      centerError: round2(b.centerError),
      inkFill: round2(b.inkFill),
      boxStability: round2(b.boxStability)
    } : null,
    pass: {
      focus: !!lastMetrics.passFocus,
      brightness: !!lastMetrics.passBrightness,
      contrast: !!lastMetrics.passContrast,
      motion: !!lastMetrics.passMotion,
      liveCandidate: !!lastMetrics.passCdpCandidate,
      all: !!lastMetrics.pass
    },
    thresholds: QUALITY,
    backendCrop: "server_opencv_marker_component_homography"
  };
}

async function savePending() {
  if (!pendingData || !pendingData.capture_id) return;
  savePendingBtn.disabled = true;
  savePendingBtn.textContent = "Kaydediliyor...";

  try {
    const res = await fetch(`/api/save/${pendingData.capture_id}`, { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Save failed");
    pendingData = data;
    setStatusBox(pendingStatus, data);
    resultBox.textContent = JSON.stringify(data, null, 2);
    savePendingBtn.textContent = "Kaydedildi";
    savePendingBtn.disabled = true;
    await loadHealth();
  } catch (err) {
    savePendingBtn.disabled = false;
    savePendingBtn.textContent = "Kaydet";
    setError(`Kayıt hatası: ${err.message || err}`);
  }
}

async function discardPending() {
  if (pendingData && pendingData.capture_id) {
    try {
      await fetch(`/api/pending/${pendingData.capture_id}`, { method: "DELETE" });
    } catch (err) {}
  }
  pendingData = null;
  pendingPanel.classList.remove("active");
}

async function retakeCurrentMode() {
  const mode = currentMode || (pendingData && pendingData.mode);
  await discardPending();
  if (mode) startCamera(mode);
}

function formatDate(ts) {
  if (!ts) return "-";
  return new Date(ts * 1000).toLocaleString("tr-TR");
}

function modeName(mode) {
  return MODE_LABELS[mode] || mode;
}

function scoreLine(label, value) {
  if (value === undefined || value === null || value === "") return "";
  return `<div class="score-line"><span>${label}</span><b>${value}</b></div>`;
}

async function loadSavedRecords() {
  savedGrid.innerHTML = "<div class='hint'>Kayıtlar yükleniyor...</div>";
  try {
    const res = await fetch("/api/saved");
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Saved records failed");
    if (!data.records || data.records.length === 0) {
      savedGrid.innerHTML = "<div class='hint'>Henüz kayıtlı CDP yok.</div>";
      return;
    }

    savedGrid.innerHTML = data.records.map(rec => {
      const scores = rec.scores || {};
      const q = rec.client_quality || {};
      const crop = rec.server_crop || {};
      const status = rec.final_user_status || "-";
      const statusClass = status === "ORIGINAL_APPROVED" || status === "REFERENCE_SAVED" ? "ok" : status === "COPY_RISK_REJECTED" ? "bad" : "review";
      const imgUrl = rec.debug_image_url || rec.image_url;

      return `
        <div class="saved-card">
          <img src="${imgUrl}?t=${Date.now()}" alt="${rec.record_id}" />
          <div class="saved-body">
            <div class="saved-title">${modeName(rec.mode)}</div>
            <div class="saved-meta">${formatDate(rec.created_at)}</div>
            <div class="status-box ${statusClass}" style="margin-top:10px;font-size:12px;">${status}<br>${rec.final_reason || ""}</div>
            ${scoreLine("Base", scores.base_score)}
            ${scoreLine("Adjusted", scores.adjusted_score)}
            ${scoreLine("Copy Risk", scores.copy_risk_score)}
            ${scoreLine("SSIM", scores.ssim_score)}
            ${scoreLine("Mask IoU", scores.mask_iou)}
            ${scoreLine("Edge F1", scores.edge_f1)}
            ${scoreLine("Focus", q.focusScore)}
            ${scoreLine("Live Box", q.liveBoxScore)}
            ${scoreLine("Server Crop", crop.method)}
          </div>
        </div>`;
    }).join("");
  } catch (err) {
    savedGrid.innerHTML = `<div class='hint'>Kayıtlar alınamadı: ${err.message || err}</div>`;
  }
}

navButtons.forEach(btn => btn.addEventListener("click", () => showPage(btn.dataset.page)));
startModeButtons.forEach(btn => btn.addEventListener("click", () => startCamera(btn.dataset.mode)));

if (stopCameraBtn) {
  stopCameraBtn.addEventListener("click", () => {
    stopCamera();
    cameraPanel.classList.remove("active");
  });
}

savePendingBtn.addEventListener("click", savePending);
retakeBtn.addEventListener("click", retakeCurrentMode);
discardBtn.addEventListener("click", discardPending);
refreshSavedBtn.addEventListener("click", loadSavedRecords);

window.addEventListener("beforeunload", stopCamera);
loadHealth();
