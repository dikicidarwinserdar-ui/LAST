import os
import json
import math
import time
import uuid
import shutil
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

APP_TITLE = "CDP Verify Camera Render"
APP_VERSION = "5.0.0-calibrated-quality-gate"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_DIR = os.path.join(BASE_DIR, "data")
REFERENCE_DIR = os.path.join(DATA_DIR, "references")
PENDING_DIR = os.path.join(DATA_DIR, "pending")
CAPTURE_DIR = os.path.join(DATA_DIR, "captures")
RECORD_DIR = os.path.join(DATA_DIR, "records")
DEBUG_DIR = os.path.join(DATA_DIR, "debug")

for d in [DATA_DIR, REFERENCE_DIR, PENDING_DIR, CAPTURE_DIR, RECORD_DIR, DEBUG_DIR]:
    os.makedirs(d, exist_ok=True)

COMPARE_SIZE = 768
INNER_BORDER_RATIO = 0.045
NORMALIZED_SIZE = 1200
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VALID_MODES = {"reference", "original", "copy", "test"}


# ============================================================
# CAMERA QUALITY CONFIG — calibrated from existing 38 Colab images
# Source analysis: /content/drive/MyDrive/cdp/_outputs/camera_overlay_threshold_analysis
# Logic:
# - ACCEPT => frontend may auto-capture after stable frames
# - REVIEW => show guidance, do not capture yet
# - REJECT => no capture; image quality/geometry is outside safe range
# ============================================================

CAMERA_QUALITY_CONFIG = {
    "accept": {
        "marker_count_min": 12,
        "white_position_required": "bottom_4",
        "mean_reproj_error_max": 1.75,
        "max_reproj_error_max": 5.00,
        "marker_size_cv_max": 0.12,
        "blur_score_min": 200.0,
        "cdp_black_ratio_min": 0.25,
        "cdp_black_ratio_max": 0.40,
        "raw_brightness_min": 135.0,
        "raw_brightness_max": 155.0,
        "raw_contrast_min": 37.0,
        "raw_glare_ratio_max": 0.003,
        "raw_dark_ratio_max": 0.04,
        "raw_shadow_score_max": 0.36,
        "raw_aspect_min": 1.35,
        "raw_aspect_max": 1.60,
    },
    "review": {
        "marker_count_min": 12,
        "white_position_required": "bottom_4",
        "mean_reproj_error_max": 6.0,
        "max_reproj_error_max": 14.0,
        "marker_size_cv_max": 0.18,
        "blur_score_min": 120.0,
        "cdp_black_ratio_min": 0.20,
        "cdp_black_ratio_max": 0.45,
        "raw_brightness_min": 120.0,
        "raw_brightness_max": 175.0,
        "raw_contrast_min": 30.0,
        "raw_glare_ratio_max": 0.02,
        "raw_dark_ratio_max": 0.10,
        "raw_shadow_score_max": 0.45,
        "raw_aspect_min": 1.15,
        "raw_aspect_max": 1.85,
    },
    "auto_capture": {
        "stable_accept_frames": 3,
        "min_quality_gate_score": 82,
        "review_quality_gate_score": 65,
    },
}

REFERENCE_CACHE: List[Dict[str, Any]] = []

try:
    from skimage.metrics import structural_similarity as skimage_ssim
    HAVE_SSIM = True
except Exception:
    HAVE_SSIM = False

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def now_ms() -> int:
    return int(time.time() * 1000)


def new_id(prefix: str) -> str:
    return f"{prefix}_{now_ms()}_{uuid.uuid4().hex[:10]}"


def is_image_file(name: str) -> bool:
    return os.path.splitext(name.lower())[1] in ALLOWED_EXTENSIONS


def read_image_from_bytes(data: bytes) -> np.ndarray:
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"Image too large. Max size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image could not be decoded.")
    return img


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        y = float(x)
        if math.isnan(y) or math.isinf(y):
            return default
        return y
    except Exception:
        return default


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return 0.0
    return obj


def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, ensure_ascii=False, indent=2)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pct_drop_from_ref(target_value: float, ref_value: float) -> float:
    target_value = safe_float(target_value)
    ref_value = safe_float(ref_value)
    if ref_value <= 1e-8:
        return 0.0
    return float(100.0 * (ref_value - target_value) / ref_value)


def pct_gain_from_ref(target_value: float, ref_value: float) -> float:
    target_value = safe_float(target_value)
    ref_value = safe_float(ref_value)
    if ref_value <= 1e-8:
        return 0.0
    return float(100.0 * (target_value - ref_value) / ref_value)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = float(np.std(a) * np.std(b))
    if denom < 1e-8:
        return 0.0
    corr = float(np.mean(a * b) / denom)
    return max(-1.0, min(1.0, corr))


def binary_f1(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    total = a.sum() + b.sum()
    if total == 0:
        return 100.0
    return float(200.0 * inter / total)


def binary_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 100.0
    return float(100.0 * inter / union)


def absdiff_similarity(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)))
    sim = 100.0 * (1.0 - diff)
    return float(max(0.0, min(100.0, sim)))


def grid_similarity(mask_a: np.ndarray, mask_b: np.ndarray, grid: int = 32) -> float:
    h, w = mask_a.shape
    cell_h = h // grid
    cell_w = w // grid
    vals_a = []
    vals_b = []
    for gy in range(grid):
        for gx in range(grid):
            y1 = gy * cell_h
            x1 = gx * cell_w
            y2 = h if gy == grid - 1 else (gy + 1) * cell_h
            x2 = w if gx == grid - 1 else (gx + 1) * cell_w
            vals_a.append(float(np.mean(mask_a[y1:y2, x1:x2])))
            vals_b.append(float(np.mean(mask_b[y1:y2, x1:x2])))
    vals_a = np.array(vals_a, dtype=np.float32)
    vals_b = np.array(vals_b, dtype=np.float32)
    mad = float(np.mean(np.abs(vals_a - vals_b)))
    sim = 100.0 * (1.0 - mad)
    return float(max(0.0, min(100.0, sim)))


def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def expand_quad_about_center(quad: np.ndarray, scale: float) -> np.ndarray:
    quad = np.asarray(quad, dtype=np.float32)
    c = np.mean(quad, axis=0)
    return c + (quad - c) * scale


def warp_quad_to_square(img: np.ndarray, quad: np.ndarray, size: int = NORMALIZED_SIZE) -> np.ndarray:
    src = order_points_clockwise(quad)
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, m, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return warped



# ============================================================
# CALIBRATED CAMERA QUALITY HELPERS
# ============================================================

def compute_raw_image_metrics_for_gate(img_bgr: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur_laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx * gx + gy * gy))
    glare_ratio = float(np.mean(gray >= 245))
    dark_ratio = float(np.mean(gray <= 35))
    bg = cv2.GaussianBlur(gray, (0, 0), 45)
    bg_mean = float(np.mean(bg))
    bg_std = float(np.std(bg))
    shadow_score = float(bg_std / max(bg_mean, 1.0))
    return {
        "raw_brightness": brightness,
        "raw_contrast": contrast,
        "raw_blur_laplacian": blur_laplacian,
        "raw_tenengrad": tenengrad,
        "raw_glare_ratio": glare_ratio,
        "raw_dark_ratio": dark_ratio,
        "raw_shadow_score": shadow_score,
    }


def make_black_mask_for_gate(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, (0, 0), 45)
    norm = cv2.divide(gray, bg, scale=255)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(norm)
    _, black_mask = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return gray, norm, black_mask


def compute_crop_quality_metrics_for_gate(crop_bgr: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    _, _, black_mask = make_black_mask_for_gate(crop_bgr)
    cdp_black_ratio = float(np.mean(black_mask > 0))
    crop_brightness = float(np.mean(gray))
    crop_contrast = float(np.std(gray))
    return {
        "blur_score": blur_score,
        "cdp_black_ratio": cdp_black_ratio,
        "crop_brightness": crop_brightness,
        "crop_contrast": crop_contrast,
    }


def gate_score_higher_is_better(value: float, accept_min: float, review_min: float) -> float:
    value = safe_float(value, 0.0)
    if value >= accept_min:
        return 100.0
    if value <= review_min:
        return 0.0
    return 100.0 * (value - review_min) / max(accept_min - review_min, 1e-9)


def gate_score_lower_is_better(value: float, accept_max: float, review_max: float) -> float:
    value = safe_float(value, review_max)
    if value <= accept_max:
        return 100.0
    if value >= review_max:
        return 0.0
    return 100.0 * (review_max - value) / max(review_max - accept_max, 1e-9)


def gate_score_range_is_better(value: float, accept_min: float, accept_max: float, review_min: float, review_max: float) -> float:
    value = safe_float(value, 0.0)
    if accept_min <= value <= accept_max:
        return 100.0
    if value < review_min or value > review_max:
        return 0.0
    if value < accept_min:
        return 100.0 * (value - review_min) / max(accept_min - review_min, 1e-9)
    return 100.0 * (review_max - value) / max(review_max - accept_max, 1e-9)


def compute_quality_gate_score(metrics: Dict[str, Any]) -> float:
    a = CAMERA_QUALITY_CONFIG["accept"]
    r = CAMERA_QUALITY_CONFIG["review"]
    score = (
        0.20 * gate_score_higher_is_better(metrics.get("blur_score"), a["blur_score_min"], r["blur_score_min"]) +
        0.12 * gate_score_range_is_better(metrics.get("raw_brightness"), a["raw_brightness_min"], a["raw_brightness_max"], r["raw_brightness_min"], r["raw_brightness_max"]) +
        0.08 * gate_score_higher_is_better(metrics.get("raw_contrast"), a["raw_contrast_min"], r["raw_contrast_min"]) +
        0.16 * gate_score_lower_is_better(metrics.get("mean_reproj_error"), a["mean_reproj_error_max"], r["mean_reproj_error_max"]) +
        0.10 * gate_score_lower_is_better(metrics.get("max_reproj_error"), a["max_reproj_error_max"], r["max_reproj_error_max"]) +
        0.10 * gate_score_lower_is_better(metrics.get("marker_size_cv"), a["marker_size_cv_max"], r["marker_size_cv_max"]) +
        0.12 * gate_score_range_is_better(metrics.get("cdp_black_ratio"), a["cdp_black_ratio_min"], a["cdp_black_ratio_max"], r["cdp_black_ratio_min"], r["cdp_black_ratio_max"]) +
        0.04 * gate_score_lower_is_better(metrics.get("raw_glare_ratio"), a["raw_glare_ratio_max"], r["raw_glare_ratio_max"]) +
        0.04 * gate_score_lower_is_better(metrics.get("raw_shadow_score"), a["raw_shadow_score_max"], r["raw_shadow_score_max"]) +
        0.04 * (100.0 if int(metrics.get("marker_count", 0) or 0) >= a["marker_count_min"] else 0.0)
    )
    return round(float(score), 3)


def camera_quality_decision(metrics: Dict[str, Any]) -> Dict[str, Any]:
    a = CAMERA_QUALITY_CONFIG["accept"]
    r = CAMERA_QUALITY_CONFIG["review"]
    hard = []

    if int(metrics.get("marker_count", 0) or 0) < r["marker_count_min"]:
        hard.append("marker_count_low")
    if safe_float(metrics.get("mean_reproj_error"), 999.0) > r["mean_reproj_error_max"]:
        hard.append("mean_reproj_too_high")
    if safe_float(metrics.get("max_reproj_error"), 999.0) > r["max_reproj_error_max"]:
        hard.append("max_reproj_too_high")
    if safe_float(metrics.get("marker_size_cv"), 999.0) > r["marker_size_cv_max"]:
        hard.append("marker_size_cv_too_high")
    if safe_float(metrics.get("blur_score"), 0.0) < r["blur_score_min"]:
        hard.append("blur_too_low")
    cbr = safe_float(metrics.get("cdp_black_ratio"), -1.0)
    if not (r["cdp_black_ratio_min"] <= cbr <= r["cdp_black_ratio_max"]):
        hard.append("cdp_black_ratio_out_of_range")
    if not (r["raw_brightness_min"] <= safe_float(metrics.get("raw_brightness"), 0.0) <= r["raw_brightness_max"]):
        hard.append("brightness_out_of_range")
    if safe_float(metrics.get("raw_contrast"), 0.0) < r["raw_contrast_min"]:
        hard.append("contrast_too_low")
    if safe_float(metrics.get("raw_shadow_score"), 999.0) > r["raw_shadow_score_max"]:
        hard.append("shadow_too_high")
    if safe_float(metrics.get("raw_glare_ratio"), 999.0) > r["raw_glare_ratio_max"]:
        hard.append("glare_too_high")

    gate_score = compute_quality_gate_score(metrics)

    if hard:
        return {"status": "REJECT", "ready": False, "quality_gate_score": gate_score, "reasons": hard}

    review_reasons = []
    if safe_float(metrics.get("mean_reproj_error"), 999.0) > a["mean_reproj_error_max"]:
        review_reasons.append("mean_reproj_review")
    if safe_float(metrics.get("max_reproj_error"), 999.0) > a["max_reproj_error_max"]:
        review_reasons.append("max_reproj_review")
    if safe_float(metrics.get("marker_size_cv"), 999.0) > a["marker_size_cv_max"]:
        review_reasons.append("marker_size_cv_review")
    if safe_float(metrics.get("blur_score"), 0.0) < a["blur_score_min"]:
        review_reasons.append("focus_review")
    if not (a["cdp_black_ratio_min"] <= cbr <= a["cdp_black_ratio_max"]):
        review_reasons.append("cdp_black_ratio_review")
    if not (a["raw_brightness_min"] <= safe_float(metrics.get("raw_brightness"), 0.0) <= a["raw_brightness_max"]):
        review_reasons.append("brightness_review")
    if safe_float(metrics.get("raw_contrast"), 0.0) < a["raw_contrast_min"]:
        review_reasons.append("contrast_review")
    if safe_float(metrics.get("raw_shadow_score"), 999.0) > a["raw_shadow_score_max"]:
        review_reasons.append("shadow_review")
    if safe_float(metrics.get("raw_glare_ratio"), 999.0) > a["raw_glare_ratio_max"]:
        review_reasons.append("glare_review")

    if review_reasons:
        return {"status": "REVIEW", "ready": False, "quality_gate_score": gate_score, "reasons": review_reasons}

    return {"status": "ACCEPT", "ready": True, "quality_gate_score": gate_score, "reasons": ["OK"]}


def user_message_from_gate(decision: Dict[str, Any]) -> str:
    status = decision.get("status")
    reasons = decision.get("reasons", [])
    if status == "ACCEPT":
        return "CDP/marker alanı yakalandı. Kalite uygun; otomatik çekim yapılabilir."
    mapping = {
        "marker_count_low": "marker/CDP alanı tam yakalanmadı",
        "mean_reproj_too_high": "açı/perspektif çok bozuk",
        "max_reproj_too_high": "marker hizalaması bozuk",
        "marker_size_cv_too_high": "marker boyutları tutarsız",
        "blur_too_low": "netlik düşük",
        "focus_review": "netlik biraz daha iyi olmalı",
        "cdp_black_ratio_out_of_range": "CDP siyah yoğunluğu beklenen aralıkta değil",
        "cdp_black_ratio_review": "CDP yoğunluğu sınırda",
        "brightness_out_of_range": "ışık seviyesi uygun değil",
        "brightness_review": "ışık seviyesi sınırda",
        "contrast_too_low": "kontrast düşük",
        "contrast_review": "kontrast sınırda",
        "shadow_too_high": "gölge fazla",
        "shadow_review": "gölge sınırda",
        "glare_too_high": "parlama fazla",
        "glare_review": "parlama sınırda",
    }
    readable = [mapping.get(r, r) for r in reasons]
    return "Bekleniyor: " + ", ".join(readable) + "."

def make_debug_image(img: np.ndarray, quad: Optional[np.ndarray], info: Dict[str, Any], mask_small: Optional[np.ndarray] = None) -> np.ndarray:
    debug = img.copy()
    h, w = debug.shape[:2]
    if quad is not None:
        q = np.asarray(quad, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(debug, [q], True, (0, 255, 0), max(3, w // 350))
        for i, p in enumerate(quad.astype(int)):
            cv2.circle(debug, tuple(p), max(5, w // 220), (0, 0, 255), -1)
            cv2.putText(debug, str(i), tuple(p + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    text_lines = [
        f"method: {info.get('method', '-')}",
        f"confidence: {info.get('confidence', 0)}",
        f"markers: {info.get('marker_count', 0)}",
        f"crop_status: {info.get('crop_status', '-')}",
    ]
    y = 34
    for t in text_lines:
        cv2.putText(debug, t, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(debug, t, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2, cv2.LINE_AA)
        y += 34
    return debug


def build_dark_mask_for_detection(img: np.ndarray) -> Tuple[np.ndarray, float]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    otsu_t, mask_otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dyn_t = max(35, min(170, float(np.mean(g) - 0.38 * np.std(g))))
    mask_dyn = (g < dyn_t).astype(np.uint8) * 255
    mask = cv2.bitwise_or(mask_otsu, mask_dyn)
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=1)
    return mask, float(otsu_t)


def detect_marker_outer_quad(img: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, Any], Optional[np.ndarray]]:
    h0, w0 = img.shape[:2]
    max_dim = max(w0, h0)
    scale = 1.0
    work = img
    if max_dim > 1500:
        scale = 1500.0 / max_dim
        work = cv2.resize(img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)

    h, w = work.shape[:2]
    mask, otsu_t = build_dark_mask_for_detection(work)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    marker_centers = []
    marker_boxes = []
    img_area = h * w

    for i in range(1, num_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < max(14, img_area * 0.000025):
            continue
        if area > img_area * 0.12:
            continue
        aspect = ww / max(1, hh)
        fill = area / max(1, ww * hh)
        if 0.35 <= aspect <= 2.85 and fill >= 0.18:
            cx, cy = centroids[i]
            if 0.02 * w <= cx <= 0.98 * w and 0.02 * h <= cy <= 0.98 * h:
                marker_centers.append([float(cx), float(cy)])
                marker_boxes.append([x, y, ww, hh, area, aspect, fill])

    info: Dict[str, Any] = {
        "method": "none",
        "confidence": 0.0,
        "marker_count": len(marker_centers),
        "otsu_threshold": otsu_t,
        "crop_status": "not_found",
    }

    if len(marker_centers) >= 5:
        centers = np.array(marker_centers, dtype=np.float32)
        rect = cv2.minAreaRect(centers)
        box = cv2.boxPoints(rect).astype(np.float32)
        box = expand_quad_about_center(box, 1.34)
        box = box / scale
        box[:, 0] = np.clip(box[:, 0], 0, w0 - 1)
        box[:, 1] = np.clip(box[:, 1], 0, h0 - 1)
        rw, rh = rect[1]
        rect_area = max(1.0, float(rw * rh))
        coverage = rect_area / max(1.0, img_area)
        info.update({
            "method": "marker_component_min_area_rect",
            "confidence": float(min(1.0, 0.42 + len(marker_centers) / 18.0)),
            "marker_rect_coverage": float(coverage),
            "crop_status": "ok",
        })
        return box, info, mask

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < img_area * 0.015 or area > img_area * 0.94:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.035 * peri, True)
        x, y, ww, hh = cv2.boundingRect(c)
        aspect = ww / max(1, hh)
        if 0.45 <= aspect <= 2.2:
            candidates.append((area, approx, c, (x, y, ww, hh)))

    if candidates:
        candidates.sort(key=lambda t: t[0], reverse=True)
        area, approx, contour, rect_xywh = candidates[0]
        if len(approx) == 4:
            box = approx.reshape(4, 2).astype(np.float32)
            method = "largest_quad_contour"
        else:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect).astype(np.float32)
            method = "largest_contour_min_area_rect"
        box = expand_quad_about_center(box, 1.16)
        box = box / scale
        box[:, 0] = np.clip(box[:, 0], 0, w0 - 1)
        box[:, 1] = np.clip(box[:, 1], 0, h0 - 1)
        info.update({"method": method, "confidence": 0.62, "contour_area_ratio": float(area / img_area), "crop_status": "ok"})
        return box, info, mask

    ys, xs = np.where(mask > 0)
    if len(xs) > img_area * 0.006:
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        side = max(x2 - x1, y2 - y1) * 1.35
        x1 = max(0.0, cx - side / 2)
        y1 = max(0.0, cy - side / 2)
        x2 = min(float(w - 1), cx + side / 2)
        y2 = min(float(h - 1), cy + side / 2)
        box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32) / scale
        info.update({"method": "dark_pixel_square_bbox_fallback", "confidence": 0.38, "crop_status": "fallback"})
        return box, info, mask

    return None, info, mask


def normalize_cdp_from_full_frame(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    quad, info, mask = detect_marker_outer_quad(img)
    if quad is not None:
        crop = warp_quad_to_square(img, quad, NORMALIZED_SIZE)
        debug = make_debug_image(img, quad, info, mask)
        return crop, debug, info

    h, w = img.shape[:2]
    side = min(h, w)
    x = (w - side) // 2
    y = (h - side) // 2
    crop = img[y:y + side, x:x + side]
    crop = cv2.resize(crop, (NORMALIZED_SIZE, NORMALIZED_SIZE), interpolation=cv2.INTER_CUBIC)
    info = {"method": "center_square_last_fallback", "confidence": 0.0, "marker_count": 0, "crop_status": "fallback_no_detection"}
    quad = np.array([[x, y], [x + side, y], [x + side, y + side], [x, y + side]], dtype=np.float32)
    debug = make_debug_image(img, quad, info, mask)
    return crop, debug, info


def preprocess_cdp_image(img: np.ndarray) -> Dict[str, Any]:
    img = cv2.resize(img, (COMPARE_SIZE, COMPARE_SIZE), interpolation=cv2.INTER_CUBIC)
    border = int(COMPARE_SIZE * INNER_BORDER_RATIO)
    if border > 0:
        img = img[border:-border, border:-border]
    img = cv2.resize(img, (COMPARE_SIZE, COMPARE_SIZE), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, (0, 0), 35)
    norm = cv2.divide(gray, bg, scale=255)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(norm)
    norm_blur = cv2.GaussianBlur(norm, (3, 3), 0)
    _, binary_inv = cv2.threshold(norm_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = (binary_inv > 0).astype(np.uint8)
    edges = cv2.Canny(norm_blur, 60, 160)
    edges_bin = (edges > 0).astype(np.uint8)

    lap = cv2.Laplacian(norm, cv2.CV_64F)
    lap_var = float(lap.var())
    gx = cv2.Sobel(norm, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(norm, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    tenengrad = float(np.mean(grad_mag ** 2))

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(binary, kernel, iterations=1)
    boundary = ((dilated - eroded) > 0)
    if np.sum(boundary) > 0:
        edge_acutance = float(np.mean(grad_mag[boundary]))
        boundary_pixels = norm[boundary]
        transition_softness = float(np.mean((boundary_pixels > 70) & (boundary_pixels < 185)))
        boundary_contrast_std = float(np.std(boundary_pixels))
    else:
        edge_acutance = 0.0
        transition_softness = 0.0
        boundary_contrast_std = 0.0

    black_ratio = float(np.mean(binary > 0))
    white_ratio = float(1.0 - black_ratio)
    black_white_ratio = float(black_ratio / max(white_ratio, 1e-6))

    thresholds = np.arange(55, 205, 5)
    black_curve = []
    for t in thresholds:
        mask_t = (norm < t).astype(np.uint8)
        black_curve.append(float(np.mean(mask_t)))
    black_curve = np.array(black_curve, dtype=np.float32)
    black_auc = float(np.mean(black_curve))
    black_curve_slope = float(black_curve[-1] - black_curve[0])

    blur_small = cv2.GaussianBlur(norm, (0, 0), 0.8)
    blur_large = cv2.GaussianBlur(norm, (0, 0), 2.4)
    dog = cv2.absdiff(blur_small, blur_large)
    highfreq_energy = float(np.mean(dog))
    highfreq_p90 = float(np.percentile(dog, 90))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= 2:
            areas.append(area)
    if len(areas) == 0:
        component_count = 0
        component_area_mean = 0.0
        component_area_p90 = 0.0
        component_area_max = 0.0
    else:
        areas = np.array(areas, dtype=np.float32)
        component_count = int(len(areas))
        component_area_mean = float(np.mean(areas))
        component_area_p90 = float(np.percentile(areas, 90))
        component_area_max = float(np.max(areas))

    return {
        "bgr": img,
        "gray": gray,
        "norm": norm,
        "norm_float": norm.astype(np.float32) / 255.0,
        "binary": binary,
        "edges": edges_bin,
        "lap_var": lap_var,
        "tenengrad": tenengrad,
        "edge_acutance": edge_acutance,
        "black_ratio": black_ratio,
        "white_ratio": white_ratio,
        "black_white_ratio": black_white_ratio,
        "transition_softness": transition_softness,
        "boundary_contrast_std": boundary_contrast_std,
        "black_curve": black_curve,
        "black_auc": black_auc,
        "black_curve_slope": black_curve_slope,
        "highfreq_energy": highfreq_energy,
        "highfreq_p90": highfreq_p90,
        "component_count": component_count,
        "component_area_mean": component_area_mean,
        "component_area_p90": component_area_p90,
        "component_area_max": component_area_max,
    }


def score_pair(target: Dict[str, Any], ref: Dict[str, Any]) -> Dict[str, Any]:
    norm_t = target["norm_float"]
    norm_r = ref["norm_float"]
    mask_t = target["binary"]
    mask_r = ref["binary"]
    edge_t = target["edges"]
    edge_r = ref["edges"]

    if HAVE_SSIM:
        ssim_raw = skimage_ssim(norm_t, norm_r, data_range=1.0)
        ssim_score = float(max(0.0, min(100.0, ssim_raw * 100.0)))
    else:
        ssim_score = absdiff_similarity(norm_t, norm_r)

    corr = safe_corr(norm_t, norm_r)
    corr_score = float((corr + 1.0) * 50.0)
    absdiff_score = absdiff_similarity(norm_t, norm_r)
    mask_f1 = binary_f1(mask_t, mask_r)
    mask_iou = binary_iou(mask_t, mask_r)
    edge_f1 = binary_f1(edge_t, edge_r)
    grid_score = grid_similarity(mask_t, mask_r, grid=32)

    base_score = (
        0.24 * ssim_score +
        0.20 * corr_score +
        0.16 * absdiff_score +
        0.18 * mask_f1 +
        0.10 * mask_iou +
        0.08 * grid_score +
        0.04 * edge_f1
    )
    base_score = float(max(0.0, min(100.0, base_score)))

    lap_drop_pct = pct_drop_from_ref(target["lap_var"], ref["lap_var"])
    tenengrad_drop_pct = pct_drop_from_ref(target["tenengrad"], ref["tenengrad"])
    edge_acutance_drop_pct = pct_drop_from_ref(target["edge_acutance"], ref["edge_acutance"])

    black_gain_abs = float(target["black_ratio"] - ref["black_ratio"])
    black_gain_pct = pct_gain_from_ref(target["black_ratio"], ref["black_ratio"])
    transition_gain_abs = float(target["transition_softness"] - ref["transition_softness"])
    transition_gain_pct = pct_gain_from_ref(target["transition_softness"], ref["transition_softness"])

    black_gain_component = np.clip(black_gain_abs / 0.045, 0, 1) * 100.0
    sharp_drop_component = np.clip(lap_drop_pct / 45.0, 0, 1) * 100.0
    tenengrad_drop_component = np.clip(tenengrad_drop_pct / 35.0, 0, 1) * 100.0
    edge_drop_component = np.clip(edge_acutance_drop_pct / 30.0, 0, 1) * 100.0
    transition_component = np.clip(transition_gain_pct / 35.0, 0, 1) * 100.0

    copy_risk_score = (
        0.32 * black_gain_component +
        0.18 * sharp_drop_component +
        0.20 * tenengrad_drop_component +
        0.20 * edge_drop_component +
        0.10 * transition_component
    )
    copy_risk_score = float(max(0.0, min(100.0, copy_risk_score)))
    adjusted_score = base_score - (0.20 * copy_risk_score)
    adjusted_score = float(max(0.0, min(100.0, adjusted_score)))

    return {
        "base_score": base_score,
        "adjusted_score": adjusted_score,
        "copy_risk_score": copy_risk_score,
        "ssim_score": ssim_score,
        "corr_score": corr_score,
        "absdiff_score": absdiff_score,
        "mask_f1": mask_f1,
        "mask_iou": mask_iou,
        "grid_score": grid_score,
        "edge_f1": edge_f1,
        "black_gain_abs": black_gain_abs,
        "black_gain_pct": black_gain_pct,
        "lap_drop_pct": lap_drop_pct,
        "tenengrad_drop_pct": tenengrad_drop_pct,
        "edge_acutance_drop_pct": edge_acutance_drop_pct,
        "transition_gain_abs": transition_gain_abs,
        "transition_gain_pct": transition_gain_pct,
    }


def final_decision(row: Dict[str, Any]) -> Dict[str, Any]:
    base = safe_float(row.get("base_score"))
    adjusted = safe_float(row.get("adjusted_score"))
    risk = safe_float(row.get("copy_risk_score"))
    ssim = safe_float(row.get("ssim_score"))
    corr = safe_float(row.get("corr_score"))
    mask_iou = safe_float(row.get("mask_iou"))
    edge_f1 = safe_float(row.get("edge_f1"))
    black_gain = safe_float(row.get("black_gain_abs"))
    lap_drop = safe_float(row.get("lap_drop_pct"))
    ten_drop = safe_float(row.get("tenengrad_drop_pct"))
    edge_drop = safe_float(row.get("edge_acutance_drop_pct"))

    if base >= 86.0 and adjusted >= 80.0 and risk < 30.0 and ssim >= 74.0 and corr >= 92.0 and mask_iou >= 85.0 and edge_f1 >= 48.0:
        return {"final_user_status": "ORIGINAL_APPROVED", "final_user_message": "Ürün doğrulandı. CDP deseni orijinal referans ile güçlü şekilde eşleşiyor.", "final_reason": "very_strong_original"}
    if base >= 78.0 and adjusted >= 72.0 and risk < 46.0 and ssim >= 58.0 and corr >= 87.0 and mask_iou >= 76.0 and edge_f1 >= 36.0:
        return {"final_user_status": "ORIGINAL_APPROVED", "final_user_message": "Ürün doğrulandı. CDP deseni orijinal referans ile güçlü şekilde eşleşiyor.", "final_reason": "strong_original"}
    if base >= 80.0 and adjusted >= 68.0 and black_gain < 0.0 and ssim >= 62.0 and corr >= 88.0 and mask_iou >= 78.0 and edge_f1 >= 38.0 and risk < 60.0:
        return {"final_user_status": "ORIGINAL_APPROVED", "final_user_message": "Ürün doğrulandı. CDP deseni orijinal referans ile güçlü şekilde eşleşiyor.", "final_reason": "strong_original_negative_black_gain"}

    if base < 69.5 and ssim < 42.0 and edge_f1 < 32.0:
        return {"final_user_status": "COPY_RISK_REJECTED", "final_user_message": "Yüksek kopya/sahte riski tespit edildi. CDP deseni orijinal referans ile yeterli eşleşmiyor.", "final_reason": f"very_low_similarity:base={base:.2f},ssim={ssim:.2f},edge={edge_f1:.2f}"}
    if base < 72.5 and ssim < 50.0 and edge_f1 < 35.0 and mask_iou < 73.0:
        return {"final_user_status": "COPY_RISK_REJECTED", "final_user_message": "Yüksek kopya/sahte riski tespit edildi. CDP mikro detayları orijinal referanstan belirgin şekilde farklı.", "final_reason": f"low_micro_similarity:base={base:.2f},ssim={ssim:.2f},iou={mask_iou:.2f},edge={edge_f1:.2f}"}
    if black_gain > 0.030 and edge_drop > 12.0 and ssim < 55.0 and adjusted < 76.0:
        return {"final_user_status": "COPY_RISK_REJECTED", "final_user_message": "Yüksek kopya/sahte riski tespit edildi. CDP üzerinde siyah alan artışı ve kenar yumuşaması var.", "final_reason": f"black_gain_plus_edge_softening:black_gain={black_gain:.4f},edge_drop={edge_drop:.2f}"}
    if black_gain > 0.025 and ten_drop > 18.0 and ssim < 55.0 and adjusted < 76.0:
        return {"final_user_status": "COPY_RISK_REJECTED", "final_user_message": "Yüksek kopya/sahte riski tespit edildi. CDP üzerinde siyah alan artışı ve detay kaybı var.", "final_reason": f"black_gain_plus_tenengrad_drop:black_gain={black_gain:.4f},ten_drop={ten_drop:.2f}"}
    if adjusted < 66.0 and risk > 55.0 and ssim < 55.0 and edge_f1 < 36.0 and black_gain > -0.010:
        return {"final_user_status": "COPY_RISK_REJECTED", "final_user_message": "Yüksek kopya/sahte riski tespit edildi. CDP mikro detayları orijinal referanstan belirgin şekilde farklı.", "final_reason": f"high_risk_weak_micro:risk={risk:.2f},adjusted={adjusted:.2f}"}
    if adjusted < 62.0 and risk > 60.0 and ssim < 52.0 and edge_drop > 24.0 and ten_drop > 35.0:
        return {"final_user_status": "COPY_RISK_REJECTED", "final_user_message": "Yüksek kopya/sahte riski tespit edildi. CDP detay keskinliği orijinal referansa göre belirgin düşük.", "final_reason": f"sharpness_only_copy_signal:risk={risk:.2f},adjusted={adjusted:.2f}"}
    if black_gain < 0 and (lap_drop > 35.0 or ten_drop > 30.0 or edge_drop > 14.0):
        return {"final_user_status": "RETAKE_REQUIRED", "final_user_message": "Fotoğraf netliği düşük görünüyor. Lütfen CDP alanını daha net ve sabit şekilde tekrar çekin.", "final_reason": "low_sharpness_retake"}
    if base < 78.0 or ssim < 58.0 or mask_iou < 76.0 or edge_f1 < 36.0:
        return {"final_user_status": "RETAKE_REQUIRED", "final_user_message": "CDP eşleşmesi sınırda kaldı. Lütfen daha iyi ışıkta, CDP tam kadrajda olacak şekilde tekrar çekin.", "final_reason": "borderline_similarity_retake"}
    return {"final_user_status": "RETAKE_REQUIRED", "final_user_message": "Görüntü kalitesi veya eşleşme skoru sınırda. Lütfen tekrar çekin.", "final_reason": "generic_review"}


def compact_scores(best: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not best:
        return {}
    keys = ["base_score", "adjusted_score", "copy_risk_score", "ssim_score", "corr_score", "mask_f1", "mask_iou", "grid_score", "edge_f1", "black_gain_abs", "black_gain_pct", "lap_drop_pct", "tenengrad_drop_pct", "edge_acutance_drop_pct"]
    return {k: round(safe_float(best[k]), 6 if k == "black_gain_abs" else 4) for k in keys if k in best}


def load_references() -> List[Dict[str, Any]]:
    refs = []
    os.makedirs(REFERENCE_DIR, exist_ok=True)
    for name in sorted(os.listdir(REFERENCE_DIR)):
        if not is_image_file(name):
            continue
        path = os.path.join(REFERENCE_DIR, name)
        img = cv2.imread(path)
        if img is None:
            continue
        prep = preprocess_cdp_image(img)
        image_id = os.path.splitext(name)[0]
        parts = name.split("__")
        if len(parts) >= 2:
            image_id = parts[1]
        refs.append({"ref_file": name, "ref_image_id": image_id, "ref_path": path, "prep": prep})
    return refs


def ensure_references_loaded() -> None:
    global REFERENCE_CACHE
    if len(REFERENCE_CACHE) == 0:
        REFERENCE_CACHE = load_references()


def score_against_references(target_prep: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ensure_references_loaded()
    if len(REFERENCE_CACHE) == 0:
        return None
    pair_rows = []
    for ref in REFERENCE_CACHE:
        s = score_pair(target_prep, ref["prep"])
        row = {"ref_file": ref["ref_file"], "ref_image_id": ref["ref_image_id"]}
        row.update(s)
        pair_rows.append(row)
    best = sorted(pair_rows, key=lambda x: x["base_score"], reverse=True)[0]
    top_refs = sorted(pair_rows, key=lambda x: x["base_score"], reverse=True)[:5]
    decision = final_decision(best)
    return {"best": best, "top_refs": top_refs, "decision": decision}


def list_saved_records() -> List[Dict[str, Any]]:
    records = []
    if not os.path.exists(RECORD_DIR):
        return records
    for name in os.listdir(RECORD_DIR):
        if not name.endswith(".json"):
            continue
        path = os.path.join(RECORD_DIR, name)
        try:
            records.append(read_json(path))
        except Exception:
            pass
    records.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    return records


@app.on_event("startup")
def startup_event():
    global REFERENCE_CACHE
    REFERENCE_CACHE = load_references()
    print(f"Loaded references: {len(REFERENCE_CACHE)}")


@app.get("/", response_class=HTMLResponse)
def index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h1>CDP Verify</h1><p>static/index.html not found.</p>")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/health")
def health():
    return {"ok": True, "app": APP_TITLE, "version": APP_VERSION, "reference_count": len(REFERENCE_CACHE), "records": len(list_saved_records()), "time": time.time()}


@app.get("/api/refs")
def api_refs():
    ensure_references_loaded()
    return {"reference_count": len(REFERENCE_CACHE), "references": [{"ref_file": r["ref_file"], "ref_image_id": r["ref_image_id"]} for r in REFERENCE_CACHE]}


@app.post("/api/reload-refs")
def api_reload_refs():
    global REFERENCE_CACHE
    REFERENCE_CACHE = load_references()
    return {"reference_count": len(REFERENCE_CACHE), "message": "References reloaded."}



@app.post("/api/live-detect")
async def live_detect(file: UploadFile = File(...)):
    """Calibrated live detector.

    Frontend sends low/mid-res live frames here. This endpoint uses the
    same backend crop/quality gate philosophy as the Colab pipeline and
    returns ACCEPT / REVIEW / REJECT. The frontend only draws the returned
    polygon and waits for ACCEPT over stable frames.
    """
    if not is_image_file(file.filename or "frame.jpg"):
        raise HTTPException(status_code=400, detail="Unsupported image format")

    data = await file.read()
    try:
        img = read_image_from_bytes(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    h, w = img.shape[:2]

    raw_metrics = compute_raw_image_metrics_for_gate(img)
    quad, detect_info, _mask = detect_marker_outer_quad(img)

    found = quad is not None and safe_float(detect_info.get("confidence"), 0.0) >= 0.38

    ordered_quad = order_points_clockwise(quad).tolist() if quad is not None else None
    bbox = None
    coverage = 0.0
    raw_aspect = 0.0
    center_error = 1.0

    if quad is not None:
        xs = quad[:, 0]
        ys = quad[:, 1]
        x1, x2 = float(np.min(xs)), float(np.max(xs))
        y1, y2 = float(np.min(ys)), float(np.max(ys))
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        bbox = {"x": x1, "y": y1, "w": bw, "h": bh}
        coverage = float((bw * bh) / max(1.0, w * h))
        raw_aspect = float(bw / bh)
        cx, cy = x1 + bw / 2.0, y1 + bh / 2.0
        center_error = float(math.sqrt(((cx - w / 2) / w) ** 2 + ((cy - h / 2) / h) ** 2))

    crop_quality = {}
    crop_error = ""

    if found:
        try:
            normalized_img, _debug_img, crop_info = normalize_cdp_from_full_frame(img)
            crop_quality = compute_crop_quality_metrics_for_gate(normalized_img)
        except Exception as e:
            crop_info = dict(detect_info)
            crop_error = str(e)
    else:
        crop_info = dict(detect_info)
        crop_error = "marker_not_found"

    # Existing Render detector is a live proxy; when exact 12-marker Colab
    # homography fields are unavailable, we use calibrated proxy values.
    confidence = safe_float(detect_info.get("confidence"), 0.0)
    marker_count = int(detect_info.get("marker_count", 0) or 0)

    metrics = {
        **raw_metrics,
        **crop_quality,
        "marker_count": marker_count,
        "white_position": "bottom_4" if found else "NONE",
        "mean_reproj_error": float(max(0.0, (1.0 - confidence) * 3.0)),
        "max_reproj_error": float(max(0.0, (1.0 - confidence) * 8.0)),
        "marker_size_cv": float(max(0.06, min(0.18, 0.18 - confidence * 0.10))),
        "raw_aspect": raw_aspect,
        "coverage": coverage,
        "center_error": center_error,
    }

    if not found:
        decision = {"status": "REJECT", "ready": False, "quality_gate_score": 0.0, "reasons": ["marker_count_low"]}
    else:
        decision = camera_quality_decision(metrics)

    ready = bool(decision.get("ready"))

    checks = {
        "marker": bool(found),
        "quality_gate": decision.get("status"),
        "brightness": CAMERA_QUALITY_CONFIG["review"]["raw_brightness_min"] <= raw_metrics["raw_brightness"] <= CAMERA_QUALITY_CONFIG["review"]["raw_brightness_max"],
        "contrast": raw_metrics["raw_contrast"] >= CAMERA_QUALITY_CONFIG["review"]["raw_contrast_min"],
        "focus": safe_float(metrics.get("blur_score"), 0.0) >= CAMERA_QUALITY_CONFIG["review"]["blur_score_min"],
        "geometry": bool(found),
    }

    guide = {
        "ready": ready,
        "found": bool(found),
        "quad": ordered_quad,
        "bbox": bbox,
        "method": detect_info.get("method"),
        "confidence": confidence,
        "marker_count": marker_count,
        "coverage": coverage,
        "aspect": raw_aspect,
        "center_error": center_error,
        "focus": safe_float(metrics.get("blur_score"), 0.0),
        "raw_focus": raw_metrics["raw_blur_laplacian"],
        "brightness": raw_metrics["raw_brightness"],
        "contrast": raw_metrics["raw_contrast"],
        "cdp_black_ratio": safe_float(metrics.get("cdp_black_ratio"), 0.0),
        "raw_shadow_score": raw_metrics["raw_shadow_score"],
        "raw_glare_ratio": raw_metrics["raw_glare_ratio"],
        "mean_reproj_error": metrics["mean_reproj_error"],
        "max_reproj_error": metrics["max_reproj_error"],
        "marker_size_cv": metrics["marker_size_cv"],
        "gate_status": decision.get("status"),
        "quality_gate_score": decision.get("quality_gate_score"),
        "gate_reasons": decision.get("reasons", []),
        "checks": checks,
        "message": user_message_from_gate(decision),
        "frame": {"w": w, "h": h},
        "server_crop": {**crop_info, **crop_quality, "crop_error": crop_error},
        "thresholds": CAMERA_QUALITY_CONFIG,
    }
    return JSONResponse(json_safe(guide))


@app.post("/api/preview/{mode}")
async def api_preview_capture(mode: str, file: UploadFile = File(...), quality_json: str = Form(default="{}")):
    if mode not in VALID_MODES:
        raise HTTPException(status_code=400, detail="Invalid mode.")
    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")
    try:
        full_img = read_image_from_bytes(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        client_quality = json.loads(quality_json) if quality_json else {}
    except Exception:
        client_quality = {}

    normalized_img, debug_img, crop_info = normalize_cdp_from_full_frame(full_img)
    try:
        crop_info.update(compute_raw_image_metrics_for_gate(full_img))
        crop_info.update(compute_crop_quality_metrics_for_gate(normalized_img))
        crop_info["quality_decision"] = camera_quality_decision({
            **crop_info,
            "marker_count": int(crop_info.get("marker_count", 0) or 0),
            "white_position": crop_info.get("white_position", "bottom_4"),
            "mean_reproj_error": safe_float(crop_info.get("mean_reproj_error"), max(0.0, (1.0 - safe_float(crop_info.get("confidence"), 0.0)) * 3.0)),
            "max_reproj_error": safe_float(crop_info.get("max_reproj_error"), max(0.0, (1.0 - safe_float(crop_info.get("confidence"), 0.0)) * 8.0)),
            "marker_size_cv": safe_float(crop_info.get("marker_size_cv"), max(0.06, min(0.18, 0.18 - safe_float(crop_info.get("confidence"), 0.0) * 0.10))),
            "raw_aspect": safe_float(crop_info.get("raw_aspect"), 1.46),
        })
    except Exception as e:
        crop_info["quality_metric_error"] = str(e)

    capture_id = new_id(mode)
    pending_full_path = os.path.join(PENDING_DIR, f"{capture_id}_full.jpg")
    pending_crop_path = os.path.join(PENDING_DIR, f"{capture_id}.jpg")
    pending_debug_path = os.path.join(PENDING_DIR, f"{capture_id}_debug.jpg")
    pending_json_path = os.path.join(PENDING_DIR, f"{capture_id}.json")

    cv2.imwrite(pending_full_path, full_img, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    cv2.imwrite(pending_crop_path, normalized_img, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    cv2.imwrite(pending_debug_path, debug_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    target_prep = preprocess_cdp_image(normalized_img)

    if mode == "reference":
        decision = {"final_user_status": "REFERENCE_READY_TO_SAVE", "final_user_message": "Referans otomatik çekildi ve backend marker/CDP crop yaptı. Kontrol edip kaydedebilirsin.", "final_reason": "reference_preview_backend_crop"}
        best = None
        top_refs = []
    else:
        scored = score_against_references(target_prep)
        if scored is None:
            decision = {"final_user_status": "NO_REFERENCE", "final_user_message": "Henüz kayıtlı referans yok. Önce referans çekip kaydet.", "final_reason": "missing_reference"}
            best = None
            top_refs = []
        else:
            best = scored["best"]
            top_refs = scored["top_refs"]
            decision = scored["decision"]

    response = {
        "capture_id": capture_id,
        "mode": mode,
        "input_file": file.filename,
        "saved": False,
        "pending_image_url": f"/api/pending/{capture_id}/image",
        "pending_crop_url": f"/api/pending/{capture_id}/image",
        "pending_full_image_url": f"/api/pending/{capture_id}/full",
        "server_debug_image_url": f"/api/pending/{capture_id}/debug",
        "reference_count": len(REFERENCE_CACHE),
        "final_user_status": decision["final_user_status"],
        "final_user_message": decision["final_user_message"],
        "final_reason": decision["final_reason"],
        "best_ref_image_id": best.get("ref_image_id") if best else None,
        "best_ref_file": best.get("ref_file") if best else None,
        "scores": compact_scores(best),
        "client_quality": client_quality,
        "server_crop": crop_info,
        "top_refs": [{"ref_image_id": r["ref_image_id"], "ref_file": r["ref_file"], "base_score": round(safe_float(r.get("base_score")), 4), "adjusted_score": round(safe_float(r.get("adjusted_score")), 4), "copy_risk_score": round(safe_float(r.get("copy_risk_score")), 4), "ssim_score": round(safe_float(r.get("ssim_score")), 4), "mask_iou": round(safe_float(r.get("mask_iou")), 4), "edge_f1": round(safe_float(r.get("edge_f1")), 4)} for r in top_refs],
        "created_at": time.time(),
    }
    write_json(pending_json_path, response)
    return JSONResponse(json_safe(response))


@app.post("/api/save/{capture_id}")
def api_save_capture(capture_id: str):
    global REFERENCE_CACHE
    pending_crop_path = os.path.join(PENDING_DIR, f"{capture_id}.jpg")
    pending_full_path = os.path.join(PENDING_DIR, f"{capture_id}_full.jpg")
    pending_debug_path = os.path.join(PENDING_DIR, f"{capture_id}_debug.jpg")
    pending_json_path = os.path.join(PENDING_DIR, f"{capture_id}.json")
    if not os.path.exists(pending_crop_path) or not os.path.exists(pending_json_path):
        raise HTTPException(status_code=404, detail="Pending capture not found.")
    meta = read_json(pending_json_path)
    mode = meta.get("mode")
    if mode not in VALID_MODES:
        raise HTTPException(status_code=400, detail="Invalid pending mode.")

    record_id = new_id(f"rec_{mode}")
    if mode == "reference":
        final_dir = REFERENCE_DIR
        final_name = f"reference__{record_id}.jpg"
    else:
        final_dir = os.path.join(CAPTURE_DIR, mode)
        final_name = f"{record_id}.jpg"
    os.makedirs(final_dir, exist_ok=True)
    final_img_path = os.path.join(final_dir, final_name)
    shutil.copyfile(pending_crop_path, final_img_path)

    final_full_path = None
    final_debug_path = None
    if os.path.exists(pending_full_path):
        full_dir = os.path.join(CAPTURE_DIR, "full")
        os.makedirs(full_dir, exist_ok=True)
        final_full_path = os.path.join(full_dir, f"{record_id}_full.jpg")
        shutil.copyfile(pending_full_path, final_full_path)
    if os.path.exists(pending_debug_path):
        debug_dir = os.path.join(CAPTURE_DIR, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        final_debug_path = os.path.join(debug_dir, f"{record_id}_debug.jpg")
        shutil.copyfile(pending_debug_path, final_debug_path)

    if mode == "reference":
        meta["final_user_status"] = "REFERENCE_SAVED"
        meta["final_user_message"] = "Referans CDP normalize/crop edilerek kaydedildi. Bundan sonraki testler bu referansa göre karşılaştırılacak."
        meta["final_reason"] = "reference_saved_backend_crop"
        REFERENCE_CACHE = load_references()

    record = {
        "record_id": record_id,
        "capture_id": capture_id,
        "mode": mode,
        "created_at": time.time(),
        "image_path": final_img_path,
        "full_image_path": final_full_path,
        "debug_image_path": final_debug_path,
        "image_url": f"/api/records/{record_id}/image",
        "debug_image_url": f"/api/records/{record_id}/debug" if final_debug_path else None,
        "file_name": final_name,
        "reference_count_after_save": len(REFERENCE_CACHE),
        **meta,
        "saved": True,
        "saved_image_url": f"/api/records/{record_id}/image",
    }
    write_json(os.path.join(RECORD_DIR, f"{record_id}.json"), record)

    for p in [pending_crop_path, pending_full_path, pending_debug_path, pending_json_path]:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
    return JSONResponse(json_safe(record))


@app.delete("/api/pending/{capture_id}")
def api_delete_pending(capture_id: str):
    deleted = False
    for suffix in [".jpg", "_full.jpg", "_debug.jpg", ".json"]:
        p = os.path.join(PENDING_DIR, f"{capture_id}{suffix}")
        if os.path.exists(p):
            os.remove(p)
            deleted = True
    return {"deleted": deleted, "capture_id": capture_id}


@app.get("/api/pending/{capture_id}/image")
def api_pending_image(capture_id: str):
    path = os.path.join(PENDING_DIR, f"{capture_id}.jpg")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Pending crop image not found.")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/pending/{capture_id}/full")
def api_pending_full_image(capture_id: str):
    path = os.path.join(PENDING_DIR, f"{capture_id}_full.jpg")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Pending full image not found.")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/pending/{capture_id}/debug")
def api_pending_debug_image(capture_id: str):
    path = os.path.join(PENDING_DIR, f"{capture_id}_debug.jpg")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Pending debug image not found.")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/saved")
def api_saved_records():
    records = list_saved_records()
    compact = []
    for r in records:
        compact.append({
            "record_id": r.get("record_id"),
            "mode": r.get("mode"),
            "created_at": r.get("created_at"),
            "image_url": r.get("image_url") or r.get("saved_image_url"),
            "debug_image_url": r.get("debug_image_url"),
            "final_user_status": r.get("final_user_status"),
            "final_user_message": r.get("final_user_message"),
            "final_reason": r.get("final_reason"),
            "best_ref_image_id": r.get("best_ref_image_id"),
            "scores": r.get("scores", {}),
            "client_quality": r.get("client_quality", {}),
            "server_crop": r.get("server_crop", {}),
        })
    return {"count": len(compact), "records": compact}


@app.get("/api/records/{record_id}")
def api_get_record(record_id: str):
    path = os.path.join(RECORD_DIR, f"{record_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Record not found.")
    return read_json(path)


@app.get("/api/records/{record_id}/image")
def api_record_image(record_id: str):
    path = os.path.join(RECORD_DIR, f"{record_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Record not found.")
    rec = read_json(path)
    image_path = rec.get("image_path")
    if not image_path or not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Record image not found.")
    return FileResponse(image_path, media_type="image/jpeg")


@app.get("/api/records/{record_id}/debug")
def api_record_debug_image(record_id: str):
    path = os.path.join(RECORD_DIR, f"{record_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Record not found.")
    rec = read_json(path)
    debug_path = rec.get("debug_image_path")
    if not debug_path or not os.path.exists(debug_path):
        raise HTTPException(status_code=404, detail="Record debug image not found.")
    return FileResponse(debug_path, media_type="image/jpeg")
