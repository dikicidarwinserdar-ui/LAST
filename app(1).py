import os
import math
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


APP_TITLE = "CDP Verify Camera"
APP_VERSION = "1.1.0-camera-auto"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_DIR = os.path.join(BASE_DIR, "data")
REFERENCE_DIR = os.path.join(DATA_DIR, "references")
CAPTURE_DIR = os.path.join(DATA_DIR, "captures")

CAPTURE_MODES = {"reference", "original", "copy", "test"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

COMPARE_SIZE = 768
INNER_BORDER_RATIO = 0.045
MAX_UPLOAD_BYTES = 12 * 1024 * 1024

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


# ============================================================
# FILE / IMAGE HELPERS
# ============================================================

def ensure_dirs() -> None:
    os.makedirs(REFERENCE_DIR, exist_ok=True)
    for mode in CAPTURE_MODES:
        os.makedirs(os.path.join(CAPTURE_DIR, mode), exist_ok=True)


def is_image_file(name: str) -> bool:
    ext = os.path.splitext(name.lower())[1]
    return ext in ALLOWED_EXTENSIONS


def now_id() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")


def safe_filename(name: Optional[str], fallback: str) -> str:
    if not name:
        return fallback
    base = os.path.basename(name)
    base = base.replace(" ", "_")
    cleaned = []
    for ch in base:
        if ch.isalnum() or ch in {"_", "-", "."}:
            cleaned.append(ch)
    out = "".join(cleaned).strip(".")
    return out or fallback


def read_image_from_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image could not be decoded.")
    return img


def save_upload_image(img: np.ndarray, mode: str, original_name: Optional[str]) -> str:
    if mode not in CAPTURE_MODES:
        raise ValueError(f"Invalid capture mode: {mode}")

    stamp = now_id()
    safe_name = safe_filename(original_name, f"capture_{stamp}.jpg")
    root, ext = os.path.splitext(safe_name)
    if ext.lower() not in ALLOWED_EXTENSIONS:
        ext = ".jpg"

    out_name = f"{mode}__{stamp}__{root}.jpg"

    if mode == "reference":
        out_path = os.path.join(REFERENCE_DIR, out_name)
    else:
        out_path = os.path.join(CAPTURE_DIR, mode, out_name)

    ok = cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise ValueError("Image could not be saved.")

    return out_path


def center_square_crop(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    side = min(h, w)
    x1 = (w - side) // 2
    y1 = (h - side) // 2
    return img[y1:y1 + side, x1:x1 + side].copy()


# ============================================================
# NUMERIC HELPERS
# ============================================================

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


def nonlinear_gap_penalty(gap_booster_risk: float) -> float:
    x = (float(gap_booster_risk) - 45.0) / 8.0
    penalty = 100.0 / (1.0 + np.exp(-x))
    return float(penalty)


# ============================================================
# PREPROCESS + FEATURE EXTRACTION
# ============================================================

def preprocess_cdp_image(img: np.ndarray) -> Dict[str, Any]:
    square = center_square_crop(img)
    img = cv2.resize(square, (COMPARE_SIZE, COMPARE_SIZE), interpolation=cv2.INTER_CUBIC)

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

    _, binary_inv = cv2.threshold(
        norm_blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
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


# ============================================================
# SCORING + DECISION
# ============================================================

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
    white_loss_abs = float(ref["white_ratio"] - target["white_ratio"])
    white_loss_pct = pct_drop_from_ref(target["white_ratio"], ref["white_ratio"])
    bw_ratio_gain_pct = pct_gain_from_ref(target["black_white_ratio"], ref["black_white_ratio"])
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

    black_auc_gain = float(target["black_auc"] - ref["black_auc"])
    black_curve_l1 = float(np.mean(np.abs(target["black_curve"] - ref["black_curve"])))
    black_curve_risk = np.clip(black_auc_gain / 0.035, 0, 1) * 100.0
    black_curve_diff_risk = np.clip(black_curve_l1 / 0.045, 0, 1) * 100.0

    hf_drop_pct = pct_drop_from_ref(target["highfreq_energy"], ref["highfreq_energy"])
    hf_p90_drop_pct = pct_drop_from_ref(target["highfreq_p90"], ref["highfreq_p90"])
    hf_risk = np.clip(hf_drop_pct / 35.0, 0, 1) * 100.0
    hf_p90_risk = np.clip(hf_p90_drop_pct / 35.0, 0, 1) * 100.0

    component_count_drop_pct = pct_drop_from_ref(target["component_count"], ref["component_count"])
    component_area_gain_pct = pct_gain_from_ref(target["component_area_mean"], ref["component_area_mean"])
    component_area_p90_gain_pct = pct_gain_from_ref(target["component_area_p90"], ref["component_area_p90"])
    component_count_risk = np.clip(component_count_drop_pct / 25.0, 0, 1) * 100.0
    component_area_risk = np.clip(component_area_gain_pct / 40.0, 0, 1) * 100.0
    component_p90_risk = np.clip(component_area_p90_gain_pct / 45.0, 0, 1) * 100.0

    gap_booster_risk = (
        0.25 * black_curve_risk +
        0.15 * black_curve_diff_risk +
        0.20 * hf_risk +
        0.10 * hf_p90_risk +
        0.15 * component_count_risk +
        0.10 * component_area_risk +
        0.05 * component_p90_risk
    )
    gap_booster_risk = float(max(0.0, min(100.0, gap_booster_risk)))
    gap_nonlinear_penalty = nonlinear_gap_penalty(gap_booster_risk)

    return {
        "base_score": base_score,
        "adjusted_score": adjusted_score,
        "copy_risk_score": copy_risk_score,
        "gap_booster_risk": gap_booster_risk,
        "gap_nonlinear_penalty": gap_nonlinear_penalty,
        "ssim_score": ssim_score,
        "corr_score": corr_score,
        "absdiff_score": absdiff_score,
        "mask_f1": mask_f1,
        "mask_iou": mask_iou,
        "grid_score": grid_score,
        "edge_f1": edge_f1,
        "black_gain_abs": black_gain_abs,
        "black_gain_pct": black_gain_pct,
        "white_loss_abs": white_loss_abs,
        "white_loss_pct": white_loss_pct,
        "bw_ratio_gain_pct": bw_ratio_gain_pct,
        "lap_drop_pct": lap_drop_pct,
        "tenengrad_drop_pct": tenengrad_drop_pct,
        "edge_acutance_drop_pct": edge_acutance_drop_pct,
        "transition_gain_abs": transition_gain_abs,
        "transition_gain_pct": transition_gain_pct,
        "black_auc_gain": black_auc_gain,
        "black_curve_l1": black_curve_l1,
        "hf_drop_pct": hf_drop_pct,
        "hf_p90_drop_pct": hf_p90_drop_pct,
        "component_count_drop_pct": component_count_drop_pct,
        "component_area_gain_pct": component_area_gain_pct,
        "component_area_p90_gain_pct": component_area_p90_gain_pct,
    }


def final_decision(row: Dict[str, Any]) -> Dict[str, Any]:
    base = safe_float(row["base_score"])
    adjusted = safe_float(row["adjusted_score"])
    risk = safe_float(row["copy_risk_score"])
    ssim = safe_float(row["ssim_score"])
    corr = safe_float(row["corr_score"])
    mask_iou = safe_float(row["mask_iou"])
    edge_f1 = safe_float(row["edge_f1"])
    black_gain = safe_float(row["black_gain_abs"])
    lap_drop = safe_float(row["lap_drop_pct"])
    ten_drop = safe_float(row["tenengrad_drop_pct"])
    edge_drop = safe_float(row["edge_acutance_drop_pct"])

    very_strong_original = (
        base >= 86.0 and adjusted >= 80.0 and risk < 30.0 and
        ssim >= 74.0 and corr >= 92.0 and mask_iou >= 85.0 and edge_f1 >= 48.0
    )
    strong_original = (
        base >= 78.0 and adjusted >= 72.0 and risk < 46.0 and
        ssim >= 58.0 and corr >= 87.0 and mask_iou >= 76.0 and edge_f1 >= 36.0
    )
    strong_original_with_negative_black_gain = (
        base >= 80.0 and adjusted >= 68.0 and black_gain < 0.0 and
        ssim >= 62.0 and corr >= 88.0 and mask_iou >= 78.0 and
        edge_f1 >= 38.0 and risk < 60.0
    )

    if very_strong_original or strong_original or strong_original_with_negative_black_gain:
        return {
            "final_user_status": "ORIGINAL_APPROVED",
            "final_user_message": "Ürün doğrulandı. CDP deseni orijinal referans ile güçlü şekilde eşleşiyor.",
            "final_reason": "strong_original_match",
        }

    if base < 69.5 and ssim < 42.0 and edge_f1 < 32.0:
        return {
            "final_user_status": "COPY_RISK_REJECTED",
            "final_user_message": "Yüksek kopya/sahte riski tespit edildi. CDP deseni orijinal referans ile yeterli eşleşmiyor.",
            "final_reason": f"very_low_similarity:base={base:.2f},ssim={ssim:.2f},edge={edge_f1:.2f}",
        }

    if base < 72.5 and ssim < 50.0 and edge_f1 < 35.0 and mask_iou < 73.0:
        return {
            "final_user_status": "COPY_RISK_REJECTED",
            "final_user_message": "Yüksek kopya/sahte riski tespit edildi. CDP mikro detayları orijinal referanstan belirgin şekilde farklı.",
            "final_reason": f"low_micro_similarity:base={base:.2f},ssim={ssim:.2f},iou={mask_iou:.2f},edge={edge_f1:.2f}",
        }

    if black_gain > 0.030 and edge_drop > 12.0 and ssim < 55.0 and adjusted < 76.0:
        return {
            "final_user_status": "COPY_RISK_REJECTED",
            "final_user_message": "Yüksek kopya/sahte riski tespit edildi. CDP üzerinde siyah alan artışı ve kenar yumuşaması var.",
            "final_reason": f"black_gain_plus_edge_softening:black_gain={black_gain:.4f},edge_drop={edge_drop:.2f}",
        }

    if black_gain > 0.025 and ten_drop > 18.0 and ssim < 55.0 and adjusted < 76.0:
        return {
            "final_user_status": "COPY_RISK_REJECTED",
            "final_user_message": "Yüksek kopya/sahte riski tespit edildi. CDP üzerinde siyah alan artışı ve detay kaybı var.",
            "final_reason": f"black_gain_plus_tenengrad_drop:black_gain={black_gain:.4f},ten_drop={ten_drop:.2f}",
        }

    if adjusted < 66.0 and risk > 55.0 and ssim < 55.0 and edge_f1 < 36.0 and black_gain > -0.010:
        return {
            "final_user_status": "COPY_RISK_REJECTED",
            "final_user_message": "Yüksek kopya/sahte riski tespit edildi. CDP mikro detayları orijinal referanstan belirgin şekilde farklı.",
            "final_reason": f"high_risk_weak_micro:risk={risk:.2f},adjusted={adjusted:.2f}",
        }

    if adjusted < 62.0 and risk > 60.0 and ssim < 52.0 and edge_drop > 24.0 and ten_drop > 35.0:
        return {
            "final_user_status": "COPY_RISK_REJECTED",
            "final_user_message": "Yüksek kopya/sahte riski tespit edildi. CDP detay keskinliği orijinal referansa göre belirgin düşük.",
            "final_reason": f"sharpness_only_copy_signal:risk={risk:.2f},adjusted={adjusted:.2f}",
        }

    if black_gain < 0 and (lap_drop > 35.0 or ten_drop > 30.0 or edge_drop > 14.0):
        return {
            "final_user_status": "RETAKE_REQUIRED",
            "final_user_message": "Fotoğraf netliği düşük görünüyor. Lütfen CDP alanını daha net ve sabit şekilde tekrar çekin.",
            "final_reason": "low_sharpness_retake",
        }

    if base < 78.0 or ssim < 58.0 or mask_iou < 76.0 or edge_f1 < 36.0:
        return {
            "final_user_status": "RETAKE_REQUIRED",
            "final_user_message": "CDP eşleşmesi sınırda kaldı. Lütfen daha iyi ışıkta, CDP tam kadrajda olacak şekilde tekrar çekin.",
            "final_reason": "borderline_similarity_retake",
        }

    return {
        "final_user_status": "RETAKE_REQUIRED",
        "final_user_message": "Görüntü kalitesi veya eşleşme skoru sınırda. Lütfen tekrar çekin.",
        "final_reason": "generic_review",
    }


# ============================================================
# REFERENCES + VERIFY
# ============================================================

def load_references() -> List[Dict[str, Any]]:
    ensure_dirs()
    refs = []
    for name in sorted(os.listdir(REFERENCE_DIR)):
        if not is_image_file(name):
            continue
        path = os.path.join(REFERENCE_DIR, name)
        img = cv2.imread(path)
        if img is None:
            continue
        prep = preprocess_cdp_image(img)
        parts = name.split("__")
        image_id = parts[1] if len(parts) >= 2 else os.path.splitext(name)[0]
        refs.append({
            "ref_file": name,
            "ref_image_id": image_id,
            "ref_path": path,
            "prep": prep,
        })
    return refs


def ensure_references_loaded() -> None:
    global REFERENCE_CACHE
    if len(REFERENCE_CACHE) == 0:
        REFERENCE_CACHE = load_references()
    if len(REFERENCE_CACHE) == 0:
        raise HTTPException(
            status_code=500,
            detail="No reference crops found. First use Referans Çek or put original CDP crops into data/references/."
        )


def verify_against_references(img: np.ndarray) -> Dict[str, Any]:
    ensure_references_loaded()
    target_prep = preprocess_cdp_image(img)
    pair_rows = []

    for ref in REFERENCE_CACHE:
        s = score_pair(target_prep, ref["prep"])
        row = {
            "ref_file": ref["ref_file"],
            "ref_image_id": ref["ref_image_id"],
        }
        row.update(s)
        pair_rows.append(row)

    if len(pair_rows) == 0:
        raise HTTPException(status_code=500, detail="No reference scores produced.")

    best = sorted(pair_rows, key=lambda x: x["base_score"], reverse=True)[0]
    decision = final_decision(best)
    top_refs = sorted(pair_rows, key=lambda x: x["base_score"], reverse=True)[:5]

    return {
        "final_user_status": decision["final_user_status"],
        "final_user_message": decision["final_user_message"],
        "final_reason": decision["final_reason"],
        "best_ref_image_id": best["ref_image_id"],
        "best_ref_file": best["ref_file"],
        "scores": {
            "base_score": round(best["base_score"], 4),
            "adjusted_score": round(best["adjusted_score"], 4),
            "copy_risk_score": round(best["copy_risk_score"], 4),
            "gap_booster_risk": round(best["gap_booster_risk"], 4),
            "gap_nonlinear_penalty": round(best["gap_nonlinear_penalty"], 4),
            "ssim_score": round(best["ssim_score"], 4),
            "corr_score": round(best["corr_score"], 4),
            "mask_f1": round(best["mask_f1"], 4),
            "mask_iou": round(best["mask_iou"], 4),
            "grid_score": round(best["grid_score"], 4),
            "edge_f1": round(best["edge_f1"], 4),
            "black_gain_abs": round(best["black_gain_abs"], 6),
            "black_gain_pct": round(best["black_gain_pct"], 4),
            "lap_drop_pct": round(best["lap_drop_pct"], 4),
            "tenengrad_drop_pct": round(best["tenengrad_drop_pct"], 4),
            "edge_acutance_drop_pct": round(best["edge_acutance_drop_pct"], 4),
            "black_auc_gain": round(best["black_auc_gain"], 6),
            "black_curve_l1": round(best["black_curve_l1"], 6),
            "hf_drop_pct": round(best["hf_drop_pct"], 4),
            "component_count_drop_pct": round(best["component_count_drop_pct"], 4),
            "component_area_gain_pct": round(best["component_area_gain_pct"], 4),
        },
        "top_refs": [
            {
                "ref_image_id": r["ref_image_id"],
                "ref_file": r["ref_file"],
                "base_score": round(r["base_score"], 4),
                "adjusted_score": round(r["adjusted_score"], 4),
                "copy_risk_score": round(r["copy_risk_score"], 4),
                "gap_booster_risk": round(r["gap_booster_risk"], 4),
                "ssim_score": round(r["ssim_score"], 4),
                "mask_iou": round(r["mask_iou"], 4),
                "edge_f1": round(r["edge_f1"], 4),
            }
            for r in top_refs
        ]
    }


@app.on_event("startup")
def startup_event() -> None:
    global REFERENCE_CACHE
    ensure_dirs()
    REFERENCE_CACHE = load_references()
    print(f"Loaded references: {len(REFERENCE_CACHE)}")


# ============================================================
# ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h1>CDP Verify Camera</h1><p>static/index.html not found.</p>")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "app": APP_TITLE,
        "version": APP_VERSION,
        "reference_count": len(REFERENCE_CACHE),
        "time": time.time(),
    }


@app.get("/api/refs")
def api_refs() -> Dict[str, Any]:
    global REFERENCE_CACHE
    REFERENCE_CACHE = load_references()
    return {
        "reference_count": len(REFERENCE_CACHE),
        "references": [
            {"ref_file": r["ref_file"], "ref_image_id": r["ref_image_id"]}
            for r in REFERENCE_CACHE
        ]
    }


@app.get("/api/captures")
def api_captures() -> Dict[str, Any]:
    ensure_dirs()
    counts = {}
    files = {}
    for mode in CAPTURE_MODES:
        folder = REFERENCE_DIR if mode == "reference" else os.path.join(CAPTURE_DIR, mode)
        names = [n for n in sorted(os.listdir(folder)) if is_image_file(n)] if os.path.exists(folder) else []
        counts[mode] = len(names)
        files[mode] = names[-20:]
    return {"counts": counts, "files": files}


@app.post("/api/reload-refs")
def api_reload_refs() -> Dict[str, Any]:
    global REFERENCE_CACHE
    REFERENCE_CACHE = load_references()
    return {"reference_count": len(REFERENCE_CACHE), "message": "References reloaded."}


@app.post("/api/capture/{mode}")
async def api_capture(mode: str, file: UploadFile = File(...)) -> JSONResponse:
    global REFERENCE_CACHE

    if mode not in CAPTURE_MODES:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")

    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image too large.")

    try:
        img = read_image_from_bytes(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    saved_path = save_upload_image(img, mode, file.filename)
    saved_file = os.path.basename(saved_path)

    response: Dict[str, Any] = {
        "mode": mode,
        "saved": True,
        "saved_file": saved_file,
        "saved_path": saved_path,
        "render_storage_warning": "On Render free runtime, files saved after deploy may not persist after restart. For production, use object storage.",
    }

    if mode == "reference":
        REFERENCE_CACHE = load_references()
        response.update({
            "final_user_status": "REFERENCE_SAVED",
            "final_user_message": "Referans görüntü otomatik çekildi ve kaydedildi.",
            "reference_count": len(REFERENCE_CACHE),
        })
        return JSONResponse(response)

    verification = verify_against_references(img)
    response.update(verification)

    return JSONResponse(response)


# Backward-compatible endpoint for old crop-upload UI/tests.
@app.post("/api/verify")
async def api_verify(file: UploadFile = File(...)) -> JSONResponse:
    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image too large.")
    try:
        img = read_image_from_bytes(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    verification = verify_against_references(img)
    verification["input_file"] = file.filename
    return JSONResponse(verification)
