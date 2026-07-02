const QUALITY = {
  jpegQuality: 0.96,
  liveJpegQuality: 0.72,
  liveWidth: 720,
  liveDetectIntervalMs: 360,
  stableFramesRequired: 3,
  cooldownMs: 1400,
  maxMotion: 13.0
};

const MODE_LABELS = {
  reference: "Referans Yükle",
  original: "Orijinal Baskı Yükle",
  copy: "Sahte / Kopya Baskı Yükle",
  test: "Test Et"
};

const MODE_SUBTITLES = {
  reference: "Kamera full frame alır. Backend marker/CDP alanını bulur, perspektif düzeltir ve referans crop olarak hazırlar.",
  original: "Orijinal baskı full frame alınır. Backend aynı Colab mantığıyla marker/CDP crop yapıp skorlar.",
  copy: "Kopya/sahte baskı full frame alınır. Backend marker/CDP crop yapıp skorlar.",
  test: "Test çekimi full frame alınır. Backend normalize crop üzerinden referansla karşılaştırır."
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
let liveTimer = null;
let previousSmallGray = null;
let lastLive = null;
let stableReadyCount = 0;
let isUploading = false;
let lastCaptureAt = 0;
let pendingData = null;

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
  const crop = data.server_crop || {};
  const cropConfidence = crop.confidence !== undefined ? Math.round(crop.confidence * 100) + "%" : "-";

  return `
    <div style="font-size:18px;font-weight:900;margin-bottom:6px;">${status}</div>
    <div style="display:inline-block;padding:6px 10px;border-radius:999px;background:rgba(255,255,255,0.14);font-size:13px;font-weight:900;margin-bottom:8px;">SINIF: ${classLabel}</div>
    <div style="font-size:13px;line-height:1.4;margin-top:8px;margin-bottom:10px;">${data.final_user_message || ""}</div>
    <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:10px;">
      <div style="background:rgba(255,255,255,0.10);padding:8px;border-radius:10px;"><span style="display:block;font-size:11px;opacity:.75;">Base</span><b>${scores.base_score ?? "-"}</b></div>
      <div style="background:rgba(255,255,255,0.10);padding:8px;border-radius:10px;"><span style="display:block;font-size:11px;opacity:.75;">Adjusted</span><b>${scores.adjusted_score ?? "-"}</b></div>
      <div style="background:rgba(255,255,255,0.10);padding:8px;border-radius:10px;"><span style="display:block;font-size:11px;opacity:.75;">Copy Risk</span><b>${scores.copy_risk_score ?? "-"}</b></div>
      <div style="background:rgba(255,255,255,0.10);padding:8px;border-radius:10px;"><span style="display:block;font-size:11px;opacity:.75;">SSIM</span><b>${scores.ssim_score ?? "-"}</b></div>
      <div style="background:rgba(255,255,255,0.10);padding:8px;border-radius:10px;"><span style="display:block;font-size:11px;opacity:.75;">Mask IoU</span><b>${scores.mask_iou ?? "-"}</b></div>
      <div style="background:rgba(255,255,255,0.10);padding:8px;border-radius:10px;"><span style="display:block;font-size:11px;opacity:.75;">Edge F1</span><b>${scores.edge_f1 ?? "-"}</b></div>
      <div style="background:rgba(255,255,255,0.10);padding:8px;border-radius:10px;"><span style="display:block;font-size:11px;opacity:.75;">Crop</span><b>${crop.method || "-"}</b></div>
      <div style="background:rgba(255,255,255,0.10);padding:8px;border-radius:10px;"><span style="display:block;font-size:11px;opacity:.75;">Crop Conf.</span><b>${cropConfidence}</b></div>
      <div style="background:rgba(255,255,255,0.10);padding:8px;border-radius:10px;"><span style="display:block;font-size:11px;opacity:.75;">Markers</span><b>${crop.marker_count ?? "-"}</b></div>
    </div>`;
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
  if (rafId) cancelAnimationFrame(rafId);
  rafId = null;
  if (liveTimer) clearInterval(liveTimer);
  liveTimer = null;
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  previousSmallGray = null;
  lastLive = null;
  stableReadyCount = 0;
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
    healthInfo.textContent = `Referans: ${data.reference_count ?? 0} | Kayıt: ${data.records ?? 0}`;
  } catch (err) {
    healthInfo.textContent = "Sistem kontrol edilemedi";
  }
}

async function startCamera(mode) {
  currentMode = mode;
  pendingData = null;
  isUploading = false;
  lastCaptureAt = 0;
  stableReadyCount = 0;
  lastLive = null;
  previousSmallGray = null;

  stopCamera();
  pendingPanel.classList.remove("active");
  cameraPanel.classList.add("active");

  modeTitle.textContent = MODE_LABELS[mode] || mode;
  modeSubtitle.textContent = MODE_SUBTITLES[mode] || "";
  autoState.textContent = "Kamera açılıyor. Sistem CDP'yi backend ile canlı yakalayacak.";
  qualityMessage.textContent = "CDP/marker alanını gölge yapmadan kameraya göster. Sabit mavi ekran yok; yakalanan alan gerçek zamanlı çizilecek.";
  setOverlay("Kamera açılıyor", "wait");

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1920 },
        height: { ideal: 1080 },
        frameRate: { ideal: 30 }
      },
      audio: false
    });

    video.srcObject = stream;
    await video.play();

    await tryImproveCameraTrack();
    autoState.textContent = "Backend canlı marker/CDP tespiti başladı. Yakalanan alan yeşil poligon olarak çizilecek.";
    setOverlay("CDP aranıyor", "wait");

    rafId = requestAnimationFrame(drawLoop);
    liveTimer = setInterval(sendLiveFrameToBackend, QUALITY.liveDetectIntervalMs);
    sendLiveFrameToBackend();
  } catch (err) {
    setOverlay("Kamera açılamadı", "bad");
    autoState.textContent = `Kamera açılamadı: ${err.message || err}`;
  }
}

async function tryImproveCameraTrack() {
  try {
    const track = stream && stream.getVideoTracks ? stream.getVideoTracks()[0] : null;
    if (!track || !track.getCapabilities || !track.applyConstraints) return;
    const caps = track.getCapabilities();
    const advanced = [];
    if (caps.focusMode && caps.focusMode.includes("continuous")) advanced.push({ focusMode: "continuous" });
    if (caps.exposureMode && caps.exposureMode.includes("continuous")) advanced.push({ exposureMode: "continuous" });
    if (caps.whiteBalanceMode && caps.whiteBalanceMode.includes("continuous")) advanced.push({ whiteBalanceMode: "continuous" });
    if (advanced.length) await track.applyConstraints({ advanced });
  } catch (err) {
    // iOS/Safari may ignore advanced camera constraints. System continues normally.
  }
}

function computeClientMotion() {
  if (!video.videoWidth || !video.videoHeight) return 99;
  const w = 160;
  const h = Math.round(w * video.videoHeight / video.videoWidth);
  analysisCanvas.width = w;
  analysisCanvas.height = h;
  const ctx = analysisCanvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(video, 0, 0, w, h);
  const data = ctx.getImageData(0, 0, w, h).data;
  const gray = new Uint8Array(w * h);
  for (let i = 0, j = 0; i < data.length; i += 4, j++) gray[j] = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
  let motion = 99;
  if (previousSmallGray && previousSmallGray.length === gray.length) {
    let diff = 0;
    for (let i = 0; i < gray.length; i++) diff += Math.abs(gray[i] - previousSmallGray[i]);
    motion = diff / gray.length;
  }
  previousSmallGray = gray;
  return motion;
}

async function sendLiveFrameToBackend() {
  if (!currentMode || isUploading || !video.videoWidth || !video.videoHeight) return;
  try {
    const liveW = QUALITY.liveWidth;
    const liveH = Math.round(liveW * video.videoHeight / video.videoWidth);
    analysisCanvas.width = liveW;
    analysisCanvas.height = liveH;
    const ctx = analysisCanvas.getContext("2d");
    ctx.drawImage(video, 0, 0, liveW, liveH);
    const blob = await new Promise(resolve => analysisCanvas.toBlob(resolve, "image/jpeg", QUALITY.liveJpegQuality));
    if (!blob) return;
    const fd = new FormData();
    fd.append("file", blob, `live_${Date.now()}.jpg`);
    const res = await fetch("/api/live-detect", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "live detect failed");

    const motion = computeClientMotion();
    data.client_motion = motion;
    data.client_motion_ok = motion <= QUALITY.maxMotion;
    data.really_ready = !!data.ready && data.client_motion_ok;
    lastLive = data;

    if (data.really_ready) stableReadyCount += 1;
    else stableReadyCount = 0;

    updateQualityUI(data);

    if (stableReadyCount >= QUALITY.stableFramesRequired) {
      const now = Date.now();
      if (now - lastCaptureAt > QUALITY.cooldownMs) {
        lastCaptureAt = now;
        captureAndPreview();
      }
    }
  } catch (err) {
    lastLive = { error: err.message || String(err), ready: false, found: false };
    stableReadyCount = 0;
    updateQualityUI(lastLive);
  }
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

function mapPointToDisplay(point, frame) {
  const map = getVideoDisplayMapping();
  return {
    x: map.offsetX + (point[0] / frame.w) * map.drawW,
    y: map.offsetY + (point[1] / frame.h) * map.drawH
  };
}

function drawLoop() {
  drawOverlay(lastLive);
  rafId = requestAnimationFrame(drawLoop);
}

function drawOverlay(live) {
  if (!overlayCanvas || !video) return;
  const rect = video.getBoundingClientRect();
  overlayCanvas.width = rect.width;
  overlayCanvas.height = rect.height;
  const ctx = overlayCanvas.getContext("2d");
  const cw = overlayCanvas.width;
  const ch = overlayCanvas.height;
  ctx.clearRect(0, 0, cw, ch);

  ctx.save();
  ctx.fillStyle = "rgba(0,0,0,0.10)";
  ctx.fillRect(0, 0, cw, ch);

  if (!live || !live.found || !live.quad || !live.frame) {
    ctx.fillStyle = "rgba(251,191,36,0.95)";
    ctx.font = "bold 16px Arial";
    ctx.fillText("CDP/marker aranıyor — gölge yapmadan göster", 18, 34);
    ctx.restore();
    return;
  }

  const pts = live.quad.map(p => mapPointToDisplay(p, live.frame));
  const ready = !!live.really_ready;
  const color = ready ? "rgba(34,197,94,1)" : "rgba(251,191,36,1)";
  const fill = ready ? "rgba(34,197,94,0.16)" : "rgba(251,191,36,0.16)";

  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.strokeStyle = color;
  ctx.lineWidth = 5;
  ctx.stroke();

  pts.forEach((p, i) => {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.fillStyle = "#03101c";
    ctx.font = "bold 11px Arial";
    ctx.fillText(String(i + 1), p.x - 3, p.y + 4);
  });

  ctx.fillStyle = color;
  ctx.font = "bold 16px Arial";
  ctx.fillText(ready ? "YAKALANDI — NETLEŞTİ — ÇEKİLİYOR" : "YAKALANDI — SABİT TUT / NETLEŞİYOR", 18, 34);
  ctx.font = "13px Arial";
  ctx.fillText(`Backend: ${live.method || "-"} | Conf: ${Math.round((live.confidence || 0) * 100)}% | Markers: ${live.marker_count ?? 0}`, 18, 58);
  ctx.restore();
}

function updateQualityUI(live) {
  if (!live) return;
  focusVal.textContent = live.focus !== undefined ? Math.round(live.focus) : "--";
  brightnessVal.textContent = live.brightness !== undefined ? Math.round(live.brightness) : "--";
  contrastVal.textContent = live.contrast !== undefined ? Math.round(live.contrast) : "--";
  motionVal.textContent = live.client_motion !== undefined ? round2(live.client_motion) : "--";
  boxVal.textContent = live.confidence !== undefined ? Math.round(live.confidence * 100) : "--";

  if (live.error) {
    qualityMessage.textContent = `Canlı tespit hatası: ${live.error}`;
    setOverlay("Canlı tespit hatası", "bad");
    return;
  }

  if (live.really_ready) {
    qualityMessage.textContent = `CDP yakalandı. Stabil kare: ${stableReadyCount}/${QUALITY.stableFramesRequired}.`;
    setOverlay("Hazır", "ready");
    autoState.textContent = "Backend gerçek marker/CDP alanını yakaladı. Full frame alınıp normalize crop üretilecek.";
    return;
  }

  const issues = [];
  if (!live.found) issues.push("CDP/marker henüz yakalanmadı");
  if (live.checks && !live.checks.focus) issues.push("netlik düşük");
  if (live.checks && !live.checks.brightness) issues.push("ışık uygun değil");
  if (live.checks && !live.checks.contrast) issues.push("kontrast düşük");
  if (!live.client_motion_ok) issues.push("telefonu daha sabit tut");
  if (live.checks && !live.checks.geometry) issues.push("CDP/marker alanını biraz merkeze al");

  qualityMessage.textContent = issues.length ? "Bekleniyor: " + issues.join(", ") + "." : "Sabit tut.";
  setOverlay(live.found ? "Yakalandı — netleşiyor" : "CDP aranıyor", "wait");
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

    const blob = await new Promise(resolve => captureCanvas.toBlob(resolve, "image/jpeg", QUALITY.jpegQuality));
    if (!blob) throw new Error("Capture blob oluşturulamadı.");

    const fd = new FormData();
    fd.append("file", blob, `${currentMode}_${Date.now()}_fullframe_backend_live.jpg`);
    fd.append("quality_json", JSON.stringify(buildQualityPayload()));

    const res = await fetch(`/api/preview/${currentMode}`, { method: "POST", body: fd });
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
    stableReadyCount = 0;
  }
}

function buildQualityPayload() {
  return {
    liveDetection: lastLive || null,
    thresholds: QUALITY,
    capture: "full_frame",
    crop: "backend_server_opencv_same_pipeline"
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
    try { await fetch(`/api/pending/${pendingData.capture_id}`, { method: "DELETE" }); } catch (err) {}
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
            ${scoreLine("Server Crop", crop.method)}
            ${scoreLine("Crop Conf.", crop.confidence)}
          </div>
        </div>`;
    }).join("");
  } catch (err) {
    savedGrid.innerHTML = `<div class='hint'>Kayıtlar alınamadı: ${err.message || err}</div>`;
  }
}

navButtons.forEach(btn => btn.addEventListener("click", () => showPage(btn.dataset.page)));
startModeButtons.forEach(btn => btn.addEventListener("click", () => startCamera(btn.dataset.mode)));
if (stopCameraBtn) stopCameraBtn.addEventListener("click", () => { stopCamera(); cameraPanel.classList.remove("active"); });
savePendingBtn.addEventListener("click", savePending);
retakeBtn.addEventListener("click", retakeCurrentMode);
discardBtn.addEventListener("click", discardPending);
refreshSavedBtn.addEventListener("click", loadSavedRecords);
window.addEventListener("beforeunload", stopCamera);
loadHealth();
