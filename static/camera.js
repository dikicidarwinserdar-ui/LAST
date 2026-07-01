const QUALITY = {
  minFocus: 58,
  minBrightness: 55,
  maxBrightness: 205,
  minContrast: 28,
  maxMotion: 5.5,
  stableMs: 950,
  cooldownMs: 1400,
  analysisWidth: 360,
  jpegQuality: 0.96,
  minBoxCoverage: 0.18,
  maxBoxCoverage: 0.82,
  maxCenterError: 0.19,
  minAspect: 0.72,
  maxAspect: 1.38,
  minInkFill: 0.035,
  maxInkFill: 0.72,
  minBoxStability: 0.86
};

const MODE_LABELS = {
  reference: "Referans Yükle",
  original: "Orijinal Baskı Yükle",
  copy: "Sahte / Kopya Baskı Yükle",
  test: "Test Et"
};

const MODE_SUBTITLES = {
  reference: "CDP referansı otomatik çekilecek. Sonra kaydet butonuyla onaylayacaksın.",
  original: "Orijinal baskı otomatik çekilecek. Sonra kaydet butonuyla kayıt alınacak.",
  copy: "Kopya/sahte baskı otomatik çekilecek. Sonra kaydet butonuyla kayıt alınacak.",
  test: "Test için CDP otomatik çekilecek. Sonra sonucu kaydedebilirsin."
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

function setOverlay(text, cls) {
  readyOverlay.textContent = text;
  readyOverlay.className = `ready-overlay ${cls || ""}`.trim();
}

function showPage(pageName) {
  navButtons.forEach(btn => btn.classList.toggle("active", btn.dataset.page === pageName));
  pages.forEach(page => page.classList.toggle("active", page.id === `page-${pageName}`));

  if (pageName === "saved") {
    loadSavedRecords();
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
  previousBox = null;
  readySince = null;
}

function setStatusBox(el, data) {
  let cls = "review";
  const status = data.final_user_status || "OK";

  if (
    status === "ORIGINAL_APPROVED" ||
    status === "REFERENCE_SAVED" ||
    status === "REFERENCE_READY_TO_SAVE"
  ) {
    cls = "ok";
  }

  if (status === "COPY_RISK_REJECTED") {
    cls = "bad";
  }

  if (status === "NO_REFERENCE" || status === "RETAKE_REQUIRED") {
    cls = "review";
  }

  el.className = `status-box ${cls}`;
  el.innerHTML = `${status}<br>${data.final_user_message || ""}`;
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
    healthInfo.textContent = `Referans: ${data.reference_count} | Kayıt: ${data.records}`;
  } catch (err) {
    healthInfo.textContent = "Sistem kontrol edilemedi";
  }
}

async function loadRefs() {
  try {
    const res = await fetch("/api/refs");
    const data = await res.json();
    healthInfo.textContent = `Referans: ${data.reference_count}`;
  } catch (err) {
    healthInfo.textContent = "Referans: 0";
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

  autoState.textContent = "Kamera açılıyor. Manuel çekim yok; sistem sadece uygun koşulda otomatik çeker.";
  qualityMessage.textContent = "CDP'yi kameraya göster. Sistem CDP kutusunu bulunca kare içine alacak.";
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

    autoState.textContent = "CDP algılanıyor. Netlik, ışık, hareket ve kutu stabilitesi yeterliyse otomatik çekilecek.";
    rafId = requestAnimationFrame(analyzeLoop);
  } catch (err) {
    setOverlay("Kamera açılamadı", "bad");
    autoState.textContent = `Kamera açılamadı: ${err.message || err}`;
  }
}

function detectCdpBox(gray, aw, ah, mean, contrast) {
  const threshold = Math.min(125, mean - contrast * 0.35);

  let minX = aw;
  let minY = ah;
  let maxX = 0;
  let maxY = 0;
  let blackCount = 0;

  const marginX = Math.floor(aw * 0.04);
  const marginY = Math.floor(ah * 0.04);

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

  if (blackCount < aw * ah * 0.01) {
    return {
      found: false,
      score: 0,
      reason: "cdp_yok",
      box: null
    };
  }

  const w = maxX - minX + 1;
  const h = maxY - minY + 1;
  const area = w * h;
  const coverage = area / (aw * ah);
  const aspect = w / Math.max(1, h);

  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;

  const centerError = Math.sqrt(
    Math.pow((centerX - aw / 2) / aw, 2) +
    Math.pow((centerY - ah / 2) / ah, 2)
  );

  const inkFill = blackCount / Math.max(1, area);

  let boxStability = 1;

  if (previousBox) {
    const dx = Math.abs(centerX - previousBox.centerX) / aw;
    const dy = Math.abs(centerY - previousBox.centerY) / ah;
    const dw = Math.abs(w - previousBox.w) / aw;
    const dh = Math.abs(h - previousBox.h) / ah;

    const movement = dx + dy + dw + dh;
    boxStability = clamp(1 - movement * 6, 0, 1);
  }

  const passCoverage = coverage >= QUALITY.minBoxCoverage && coverage <= QUALITY.maxBoxCoverage;
  const passAspect = aspect >= QUALITY.minAspect && aspect <= QUALITY.maxAspect;
  const passCenter = centerError <= QUALITY.maxCenterError;
  const passInk = inkFill >= QUALITY.minInkFill && inkFill <= QUALITY.maxInkFill;
  const passStability = boxStability >= QUALITY.minBoxStability;

  let score = 0;
  score += passCoverage ? 24 : 0;
  score += passAspect ? 22 : 0;
  score += passCenter ? 22 : 0;
  score += passInk ? 16 : 0;
  score += passStability ? 16 : 0;

  previousBox = {
    minX,
    minY,
    maxX,
    maxY,
    w,
    h,
    centerX,
    centerY
  };

  return {
    found: passCoverage && passAspect && passCenter && passInk && passStability,
    score,
    reason: "ok",
    box: {
      minX,
      minY,
      maxX,
      maxY,
      w,
      h,
      coverage,
      aspect,
      centerError,
      inkFill,
      boxStability
    },
    checks: {
      passCoverage,
      passAspect,
      passCenter,
      passInk,
      passStability
    }
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

      const lap =
        -4 * gray[idx] +
        gray[idx - 1] +
        gray[idx + 1] +
        gray[idx - aw] +
        gray[idx + aw];

      lapSum += lap;
      lapSqSum += lap * lap;
      count++;
    }
  }

  const lapMean = lapSum / Math.max(1, count);
  const lapVar = lapSqSum / Math.max(1, count) - lapMean * lapMean;
  const focusScore = clamp((lapVar - 25) / 135 * 100, 0, 100);

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

  const cdp = detectCdpBox(gray, aw, ah, mean, contrast);

  const passFocus = focusScore >= QUALITY.minFocus;
  const passBrightness = mean >= QUALITY.minBrightness && mean <= QUALITY.maxBrightness;
  const passContrast = contrast >= QUALITY.minContrast;
  const passMotion = motion <= QUALITY.maxMotion;
  const passCdp = cdp.found;

  const pass = passFocus && passBrightness && passContrast && passMotion && passCdp;

  return {
    aw,
    ah,
    focusScore,
    lapVar,
    brightness: mean,
    contrast,
    motion,
    cdp,
    pass,
    passFocus,
    passBrightness,
    passContrast,
    passMotion,
    passCdp
  };
}

function drawOverlay(metrics) {
  const rect = video.getBoundingClientRect();

  overlayCanvas.width = rect.width;
  overlayCanvas.height = rect.height;

  const ctx = overlayCanvas.getContext("2d");
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  const cw = overlayCanvas.width;
  const ch = overlayCanvas.height;

  const guideSize = Math.min(cw, ch) * 0.62;
  const gx = (cw - guideSize) / 2;
  const gy = (ch - guideSize) / 2;

  ctx.strokeStyle = "rgba(96, 165, 250, 0.75)";
  ctx.lineWidth = 3;
  ctx.setLineDash([10, 8]);
  ctx.strokeRect(gx, gy, guideSize, guideSize);
  ctx.setLineDash([]);

  ctx.fillStyle = "rgba(96, 165, 250, 0.95)";
  ctx.font = "bold 15px Arial";
  ctx.fillText("CDP alanını bu kareye sığdır", gx + 10, gy + 24);

  if (metrics && metrics.cdp && metrics.cdp.box) {
    const b = metrics.cdp.box;

    const sx = cw / metrics.aw;
    const sy = ch / metrics.ah;

    const x = b.minX * sx;
    const y = b.minY * sy;
    const w = b.w * sx;
    const h = b.h * sy;

    ctx.strokeStyle = metrics.cdp.found
      ? "rgba(34, 197, 94, 0.95)"
      : "rgba(251, 191, 36, 0.95)";

    ctx.lineWidth = 4;
    ctx.strokeRect(x, y, w, h);

    ctx.fillStyle = metrics.cdp.found
      ? "rgba(34, 197, 94, 0.95)"
      : "rgba(251, 191, 36, 0.95)";

    ctx.font = "bold 16px Arial";
    ctx.fillText(
      metrics.cdp.found ? "CDP ALGILANDI" : "CDP KUTUSU AYARLANIYOR",
      x + 8,
      Math.max(24, y - 8)
    );
  }
}

function updateQualityUI(metrics) {
  focusVal.textContent = Math.round(metrics.focusScore);
  brightnessVal.textContent = Math.round(metrics.brightness);
  contrastVal.textContent = Math.round(metrics.contrast);
  motionVal.textContent = metrics.motion >= 99 ? "--" : metrics.motion.toFixed(1);
  boxVal.textContent = metrics.cdp ? Math.round(metrics.cdp.score) : "--";

  const issues = [];

  if (!metrics.passCdp) {
    issues.push("CDP kutusunu merkeze al ve kareye sığdır");
  }

  if (!metrics.passFocus) {
    issues.push("biraz daha netleştir / yaklaştır");
  }

  if (!metrics.passBrightness) {
    issues.push("ışığı düzelt");
  }

  if (!metrics.passContrast) {
    issues.push("kontrast düşük, CDP alanını daha iyi kadraja al");
  }

  if (!metrics.passMotion) {
    issues.push("daha sabit tut");
  }

  if (issues.length === 0) {
    qualityMessage.textContent = "Koşullar uygun. CDP kutusu stabil kalırsa otomatik çekilecek.";
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
      autoState.textContent = "Tüm kalite koşulları geçti. Fotoğraf otomatik çekiliyor; kaydetme onayı sonra gelecek.";

      const now = Date.now();

      if (now - lastCaptureAt > QUALITY.cooldownMs) {
        lastCaptureAt = now;
        captureAndPreview();
        return;
      }
    } else {
      setOverlay(`Hazır — ${Math.ceil(remaining / 100) / 10}s sabit tut`, "ready");
      autoState.textContent = "CDP bulundu. Sistem daha güvenli çekim için stabil süre bekliyor.";
    }
  } else {
    readySince = null;
    setOverlay("Hazır değil", "wait");
    autoState.textContent = "Manuel çekim yok. CDP kutusu, netlik, ışık ve hareket koşulları uygun olunca otomatik çekilecek.";
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

    if (!blob) {
      throw new Error("Capture blob oluşturulamadı.");
    }

    const formData = new FormData();
    const filename = `${currentMode}_${Date.now()}.jpg`;

    formData.append("file", blob, filename);
    formData.append("quality_json", JSON.stringify(buildQualityPayload()));

    const res = await fetch(`/api/preview/${currentMode}`, {
      method: "POST",
      body: formData
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Preview upload failed");
    }

    pendingData = data;
    pendingImage.src = `${data.pending_image_url}?t=${Date.now()}`;

    setStatusBox(pendingStatus, data);
    resultBox.textContent = JSON.stringify(data, null, 2);

    const saveText = currentMode === "reference" ? "Referans olarak kaydet" : "Bu çekimi kaydet";

    savePendingBtn.textContent = saveText;
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

  const cdpBox = lastMetrics.cdp && lastMetrics.cdp.box ? lastMetrics.cdp.box : null;

  return {
    focusScore: round2(lastMetrics.focusScore),
    lapVar: round2(lastMetrics.lapVar),
    brightness: round2(lastMetrics.brightness),
    contrast: round2(lastMetrics.contrast),
    motion: round2(lastMetrics.motion),
    cdpBoxScore: lastMetrics.cdp ? round2(lastMetrics.cdp.score) : 0,
    cdpBox: cdpBox
      ? {
          coverage: round2(cdpBox.coverage),
          aspect: round2(cdpBox.aspect),
          centerError: round2(cdpBox.centerError),
          inkFill: round2(cdpBox.inkFill),
          boxStability: round2(cdpBox.boxStability)
        }
      : null,
    thresholds: QUALITY
  };
}

async function savePending() {
  if (!pendingData || !pendingData.capture_id) return;

  savePendingBtn.disabled = true;
  savePendingBtn.textContent = "Kaydediliyor...";

  try {
    const res = await fetch(`/api/save/${pendingData.capture_id}`, {
      method: "POST"
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Save failed");
    }

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
      await fetch(`/api/pending/${pendingData.capture_id}`, {
        method: "DELETE"
      });
    } catch (err) {}
  }

  pendingData = null;
  pendingPanel.classList.remove("active");
}

async function retakeCurrentMode() {
  const mode = currentMode || (pendingData && pendingData.mode);

  await discardPending();

  if (mode) {
    startCamera(mode);
  }
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

    if (!res.ok) {
      throw new Error(data.detail || "Saved records failed");
    }

    if (!data.records || data.records.length === 0) {
      savedGrid.innerHTML = "<div class='hint'>Henüz kayıtlı CDP yok.</div>";
      return;
    }

    savedGrid.innerHTML = data.records
      .map(rec => {
        const scores = rec.scores || {};
        const q = rec.client_quality || {};
        const status = rec.final_user_status || "-";

        const statusClass =
          status === "ORIGINAL_APPROVED" || status === "REFERENCE_SAVED"
            ? "ok"
            : status === "COPY_RISK_REJECTED"
              ? "bad"
              : "review";

        return `
          <div class="saved-card">
            <img src="${rec.image_url}?t=${Date.now()}" alt="${rec.record_id}" />
            <div class="saved-body">
              <div class="saved-title">${modeName(rec.mode)}</div>
              <div class="saved-meta">${formatDate(rec.created_at)}</div>
              <div class="status-box ${statusClass}" style="margin-top:10px;font-size:12px;">
                ${status}<br>${rec.final_reason || ""}
              </div>
              ${scoreLine("Base", scores.base_score)}
              ${scoreLine("Adjusted", scores.adjusted_score)}
              ${scoreLine("Copy Risk", scores.copy_risk_score)}
              ${scoreLine("SSIM", scores.ssim_score)}
              ${scoreLine("Mask IoU", scores.mask_iou)}
              ${scoreLine("Edge F1", scores.edge_f1)}
              ${scoreLine("Focus", q.focusScore)}
              ${scoreLine("CDP Box", q.cdpBoxScore)}
            </div>
          </div>
        `;
      })
      .join("");
  } catch (err) {
    savedGrid.innerHTML = `<div class='hint'>Kayıtlar alınamadı: ${err.message || err}</div>`;
  }
}

navButtons.forEach(btn => {
  btn.addEventListener("click", () => showPage(btn.dataset.page));
});

startModeButtons.forEach(btn => {
  btn.addEventListener("click", () => startCamera(btn.dataset.mode));
});

stopCameraBtn.addEventListener("click", () => {
  stopCamera();
  cameraPanel.classList.remove("active");
});

savePendingBtn.addEventListener("click", savePending);
retakeBtn.addEventListener("click", retakeCurrentMode);
discardBtn.addEventListener("click", discardPending);
refreshSavedBtn.addEventListener("click", loadSavedRecords);

window.addEventListener("beforeunload", stopCamera);

loadHealth();
