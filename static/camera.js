const QUALITY = {
  minFocus: 42,
  minBrightness: 48,
  maxBrightness: 218,
  minContrast: 20,
  maxMotion: 9.5,
  stableMs: 600,
  cooldownMs: 1200,
  analysisWidth: 360,
  jpegQuality: 0.96,

  minBoxCoverage: 0.055,
  maxBoxCoverage: 0.94,
  maxCenterError: 0.34,
  minAspect: 0.50,
  maxAspect: 1.85,
  minInkFill: 0.010,
  maxInkFill: 0.86,
  minBoxStability: 0.48,

  smartCropPadding: 0.22,
  smartCropOutputSize: 1200,
  minSmartCropSideRatio: 0.18,
  maxSmartCropSideRatio: 0.96
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


function statusToClassLabel(status) {
  if (status === "ORIGINAL_APPROVED") {
    return "ORİJİNAL";
  }

  if (status === "COPY_RISK_REJECTED") {
    return "KOPYA / SAHTE RİSKİ";
  }

  if (status === "RETAKE_REQUIRED") {
    return "TEKRAR ÇEKİM";
  }

  if (status === "REFERENCE_READY_TO_SAVE" || status === "REFERENCE_SAVED") {
    return "REFERANS";
  }

  if (status === "NO_REFERENCE") {
    return "REFERANS YOK";
  }

  return status || "-";
}


function buildDecisionHtml(data) {
  const status = data.final_user_status || "OK";
  const classLabel = statusToClassLabel(status);
  const scores = data.scores || {};

  const base = scores.base_score ?? "-";
  const adjusted = scores.adjusted_score ?? "-";
  const risk = scores.copy_risk_score ?? "-";
  const ssim = scores.ssim_score ?? "-";
  const iou = scores.mask_iou ?? "-";
  const edge = scores.edge_f1 ?? "-";

  return `
    <div style="font-size:18px;font-weight:900;margin-bottom:6px;">
      ${status}
    </div>

    <div style="display:inline-block;padding:6px 10px;border-radius:999px;background:rgba(255,255,255,0.14);font-size:13px;font-weight:900;margin-bottom:8px;">
      SINIF: ${classLabel}
    </div>

    <div style="font-size:13px;line-height:1.4;margin-top:8px;margin-bottom:10px;">
      ${data.final_user_message || ""}
    </div>

    <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:10px;">
      <div style="background:rgba(255,255,255,0.10);padding:8px;border-radius:10px;"><span style="display:block;font-size:11px;opacity:.75;">Base</span><b>${base}</b></div>
      <div style="background:rgba(255,255,255,0.10);padding:8px;border-radius:10px;"><span style="display:block;font-size:11px;opacity:.75;">Adjusted</span><b>${adjusted}</b></div>
      <div style="background:rgba(255,255,255,0.10);padding:8px;border-radius:10px;"><span style="display:block;font-size:11px;opacity:.75;">Copy Risk</span><b>${risk}</b></div>
      <div style="background:rgba(255,255,255,0.10);padding:8px;border-radius:10px;"><span style="display:block;font-size:11px;opacity:.75;">SSIM</span><b>${ssim}</b></div>
      <div style="background:rgba(255,255,255,0.10);padding:8px;border-radius:10px;"><span style="display:block;font-size:11px;opacity:.75;">Mask IoU</span><b>${iou}</b></div>
      <div style="background:rgba(255,255,255,0.10);padding:8px;border-radius:10px;"><span style="display:block;font-size:11px;opacity:.75;">Edge F1</span><b>${edge}</b></div>
    </div>
  `;
}


function setOverlay(text, cls) {
  if (!readyOverlay) return;

  readyOverlay.textContent = text;
  readyOverlay.className = `ready-overlay ${cls || ""}`.trim();
}


function showPage(pageName) {
  navButtons.forEach(btn => {
    btn.classList.toggle("active", btn.dataset.page === pageName);
  });

  pages.forEach(page => {
    page.classList.toggle("active", page.id === `page-${pageName}`);
  });

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

  autoState.textContent = "Kamera açılıyor. Manuel çekim yok; sistem uygun koşulda otomatik çeker.";
  qualityMessage.textContent = "CDP'yi kameraya göster. Sistem CDP / marker dış alanını canlıda bulmaya çalışacak.";
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

    autoState.textContent = "CDP aranıyor. Sarı kutu aday alanı, yeşil kutu çekime hazır alanı gösterir.";
    rafId = requestAnimationFrame(analyzeLoop);
  } catch (err) {
    setOverlay("Kamera açılamadı", "bad");
    autoState.textContent = `Kamera açılamadı: ${err.message || err}`;
  }
}


function detectCdpBox(gray, aw, ah, mean, contrast) {
  const threshold = Math.min(145, Math.max(55, mean - contrast * 0.22));

  let minX = aw;
  let minY = ah;
  let maxX = 0;
  let maxY = 0;
  let blackCount = 0;

  const marginX = Math.floor(aw * 0.035);
  const marginY = Math.floor(ah * 0.035);

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

  if (blackCount < aw * ah * 0.006) {
    return {
      found: false,
      candidate: false,
      score: 0,
      reason: "cdp_yok",
      box: null,
      checks: {}
    };
  }

  const w = maxX - minX + 1;
  const h = maxY - minY + 1;
  const area = w * h;
  const coverage = area / Math.max(1, aw * ah);
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
    boxStability = clamp(1 - movement * 4.2, 0, 1);
  }

  const passCoverage = coverage >= QUALITY.minBoxCoverage && coverage <= QUALITY.maxBoxCoverage;
  const passAspect = aspect >= QUALITY.minAspect && aspect <= QUALITY.maxAspect;
  const passCenter = centerError <= QUALITY.maxCenterError;
  const passInk = inkFill >= QUALITY.minInkFill && inkFill <= QUALITY.maxInkFill;
  const passStability = boxStability >= QUALITY.minBoxStability;

  let score = 0;

  score += passCoverage ? 24 : Math.max(0, 24 - Math.abs(coverage - 0.45) * 55);
  score += passAspect ? 22 : Math.max(0, 22 - Math.abs(aspect - 1.0) * 22);
  score += passCenter ? 22 : Math.max(0, 22 - centerError * 55);
  score += passInk ? 16 : Math.max(0, 16 - Math.abs(inkFill - 0.20) * 40);
  score += passStability ? 16 : Math.max(0, boxStability * 16);

  score = clamp(score, 0, 100);

  const candidate =
    coverage >= 0.055 &&
    coverage <= 0.96 &&
    aspect >= 0.48 &&
    aspect <= 1.95 &&
    inkFill >= 0.010 &&
    inkFill <= 0.88;

  const found = passCoverage && passAspect && passCenter && passInk && passStability;

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
    found,
    candidate,
    score,
    reason: found ? "ok" : "candidate",
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
  const focusScore = clamp((lapVar - 22) / 125 * 100, 0, 100);

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


function getVideoDisplayMapping() {
  const rect = video.getBoundingClientRect();

  const videoRatio = video.videoWidth / Math.max(1, video.videoHeight);
  const boxRatio = rect.width / Math.max(1, rect.height);

  let drawW;
  let drawH;
  let offsetX;
  let offsetY;

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

  return {
    rect,
    drawW,
    drawH,
    offsetX,
    offsetY
  };
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


function mapAnalysisBoxToVideoPixels(box, metrics) {
  const sx = video.videoWidth / metrics.aw;
  const sy = video.videoHeight / metrics.ah;

  return {
    x: box.minX * sx,
    y: box.minY * sy,
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

  const guideSize = Math.min(cw, ch) * 0.68;
  const gx = (cw - guideSize) / 2;
  const gy = (ch - guideSize) / 2;

  ctx.save();

  ctx.fillStyle = "rgba(0, 0, 0, 0.20)";
  ctx.fillRect(0, 0, cw, ch);
  ctx.clearRect(gx, gy, guideSize, guideSize);

  ctx.strokeStyle = "rgba(59, 130, 246, 0.95)";
  ctx.lineWidth = 4;
  ctx.setLineDash([14, 8]);
  ctx.strokeRect(gx, gy, guideSize, guideSize);
  ctx.setLineDash([]);

  ctx.fillStyle = "rgba(59, 130, 246, 1)";
  ctx.font = "bold 15px Arial";
  ctx.fillText("CDP / marker dış alanını bu kareye getir", gx + 12, gy + 26);

  if (metrics && metrics.cdp && metrics.cdp.box) {
    const b = metrics.cdp.box;
    const displayBox = mapAnalysisBoxToDisplay(b, metrics);

    const x = displayBox.x;
    const y = displayBox.y;
    const w = displayBox.w;
    const h = displayBox.h;

    const isReady = metrics.passCdp;
    const isCandidate = metrics.cdp.candidate;

    const color = isReady
      ? "rgba(34, 197, 94, 1)"
      : isCandidate
        ? "rgba(251, 191, 36, 1)"
        : "rgba(248, 113, 113, 1)";

    const fillColor = isReady
      ? "rgba(34, 197, 94, 0.16)"
      : isCandidate
        ? "rgba(251, 191, 36, 0.16)"
        : "rgba(248, 113, 113, 0.14)";

    ctx.fillStyle = fillColor;
    ctx.fillRect(x, y, w, h);

    ctx.strokeStyle = color;
    ctx.lineWidth = 5;
    ctx.setLineDash([]);
    ctx.strokeRect(x, y, w, h);

    const pad = Math.max(w, h) * QUALITY.smartCropPadding;
    const cx = x + w / 2;
    const cy = y + h / 2;
    const smartSide = Math.max(w, h) + pad * 2;

    const sx = Math.max(0, cx - smartSide / 2);
    const sy = Math.max(0, cy - smartSide / 2);
    const sside = Math.min(smartSide, cw, ch);

    ctx.strokeStyle = "rgba(168, 85, 247, 0.95)";
    ctx.lineWidth = 3;
    ctx.setLineDash([8, 6]);
    ctx.strokeRect(sx, sy, sside, sside);
    ctx.setLineDash([]);

    ctx.fillStyle = color;
    ctx.font = "bold 16px Arial";
    ctx.fillText(
      isReady ? "CDP ALGILANDI - ÇEKİME HAZIR" : "CDP ADAYI - KAREYE AL",
      x + 8,
      Math.max(24, y - 10)
    );

    ctx.fillStyle = "rgba(168, 85, 247, 1)";
    ctx.font = "bold 13px Arial";
    ctx.fillText(
      "Backend'e gönderilecek dijital zoom alanı",
      sx + 8,
      Math.min(ch - 12, sy + sside + 20)
    );

    ctx.fillStyle = color;
    ctx.font = "13px Arial";
    ctx.fillText(
      `Box ${Math.round(metrics.cdp.score)} | Alan ${(b.coverage * 100).toFixed(0)}% | Oran ${b.aspect.toFixed(2)}`,
      x + 8,
      Math.min(ch - 34, y + h + 20)
    );
  }

  ctx.restore();
}


function updateQualityUI(metrics) {
  focusVal.textContent = Math.round(metrics.focusScore);
  brightnessVal.textContent = Math.round(metrics.brightness);
  contrastVal.textContent = Math.round(metrics.contrast);
  motionVal.textContent = metrics.motion >= 99 ? "--" : metrics.motion.toFixed(1);
  boxVal.textContent = metrics.cdp ? Math.round(metrics.cdp.score) : "--";

  const issues = [];

  if (!metrics.passCdp) {
    if (metrics.cdp && metrics.cdp.candidate) {
      issues.push("sarı kutuyu merkeze al, biraz sabit tut");
    } else {
      issues.push("CDP / marker alanını kare içine al");
    }
  }

  if (!metrics.passFocus) {
    issues.push("biraz daha netleştir / yaklaştır");
  }

  if (!metrics.passBrightness) {
    issues.push("ışığı düzelt");
  }

  if (!metrics.passContrast) {
    issues.push("kontrast düşük");
  }

  if (!metrics.passMotion) {
    issues.push("daha sabit tut");
  }

  if (issues.length === 0) {
    qualityMessage.textContent = "Koşullar uygun. Yeşil kutu stabil kalırsa otomatik çekilecek.";
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
      autoState.textContent = "CDP bulundu. Sistem güvenli çekim için kısa stabil süre bekliyor.";
    }
  } else {
    readySince = null;
    setOverlay("Hazır değil", "wait");
    autoState.textContent = "Manuel çekim yok. CDP kutusu, netlik, ışık ve hareket koşulları uygun olunca otomatik çekilecek.";
  }

  rafId = requestAnimationFrame(analyzeLoop);
}


function buildSmartCaptureCanvas() {
  const vw = video.videoWidth;
  const vh = video.videoHeight;

  const outputSize = QUALITY.smartCropOutputSize;

  captureCanvas.width = outputSize;
  captureCanvas.height = outputSize;

  const ctx = captureCanvas.getContext("2d");

  if (!lastMetrics || !lastMetrics.cdp || !lastMetrics.cdp.box) {
    const side = Math.min(vw, vh);
    const sx = (vw - side) / 2;
    const sy = (vh - side) / 2;

    ctx.drawImage(
      video,
      sx,
      sy,
      side,
      side,
      0,
      0,
      outputSize,
      outputSize
    );

    return {
      usedSmartCrop: false,
      reason: "no_cdp_box",
      sourceX: round2(sx),
      sourceY: round2(sy),
      sourceW: round2(side),
      sourceH: round2(side),
      outputSize
    };
  }

  const b = mapAnalysisBoxToVideoPixels(lastMetrics.cdp.box, lastMetrics);

  const boxCenterX = b.x + b.w / 2;
  const boxCenterY = b.y + b.h / 2;

  let side = Math.max(b.w, b.h);
  side = side * (1 + QUALITY.smartCropPadding * 2);

  const minSide = Math.min(vw, vh) * QUALITY.minSmartCropSideRatio;
  const maxSide = Math.min(vw, vh) * QUALITY.maxSmartCropSideRatio;

  side = Math.max(minSide, Math.min(maxSide, side));

  let sx = boxCenterX - side / 2;
  let sy = boxCenterY - side / 2;

  sx = Math.max(0, Math.min(vw - side, sx));
  sy = Math.max(0, Math.min(vh - side, sy));

  ctx.drawImage(
    video,
    sx,
    sy,
    side,
    side,
    0,
    0,
    outputSize,
    outputSize
  );

  return {
    usedSmartCrop: true,
    reason: "cdp_box_smart_crop",
    sourceX: round2(sx),
    sourceY: round2(sy),
    sourceW: round2(side),
    sourceH: round2(side),
    outputSize,
    detectedBoxVideo: {
      x: round2(b.x),
      y: round2(b.y),
      w: round2(b.w),
      h: round2(b.h)
    }
  };
}


async function captureAndPreview() {
  if (!currentMode || isUploading) return;

  isUploading = true;
  setOverlay("Çekiliyor", "ready");

  try {
    const smartCropInfo = buildSmartCaptureCanvas();

    const blob = await new Promise(resolve => {
      captureCanvas.toBlob(resolve, "image/jpeg", QUALITY.jpegQuality);
    });

    if (!blob) {
      throw new Error("Capture blob oluşturulamadı.");
    }

    const formData = new FormData();
    const filename = `${currentMode}_${Date.now()}_smartcrop.jpg`;

    const qualityPayload = buildQualityPayload();
    qualityPayload.smartCrop = smartCropInfo;

    formData.append("file", blob, filename);
    formData.append("quality_json", JSON.stringify(qualityPayload));

    const res = await fetch(`/api/preview/${currentMode}`, {
      method: "POST",
      body: formData
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Preview upload failed");
    }

    pendingData = data;
    pendingData.client_quality = qualityPayload;

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
    cdpFound: lastMetrics.cdp ? !!lastMetrics.cdp.found : false,
    cdpCandidate: lastMetrics.cdp ? !!lastMetrics.cdp.candidate : false,

    cdpBox: cdpBox
      ? {
          coverage: round2(cdpBox.coverage),
          aspect: round2(cdpBox.aspect),
          centerError: round2(cdpBox.centerError),
          inkFill: round2(cdpBox.inkFill),
          boxStability: round2(cdpBox.boxStability)
        }
      : null,

    pass: {
      focus: !!lastMetrics.passFocus,
      brightness: !!lastMetrics.passBrightness,
      contrast: !!lastMetrics.passContrast,
      motion: !!lastMetrics.passMotion,
      cdp: !!lastMetrics.passCdp,
      all: !!lastMetrics.pass
    },

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
