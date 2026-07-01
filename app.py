import os
import json
import math
import time
import uuid
import shutil
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


# ============================================================
# CONFIG
# ============================================================

APP_TITLE = "CDP Verify Camera Render"
APP_VERSION = "2.0.0"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_DIR = os.path.join(BASE_DIR, "data")
REFERENCE_DIR = os.path.join(DATA_DIR, "references")
PENDING_DIR = os.path.join(DATA_DIR, "pending")
CAPTURE_DIR = os.path.join(DATA_DIR, "captures")
RECORD_DIR = os.path.join(DATA_DIR, "records")

for d in [DATA_DIR, REFERENCE_DIR, PENDING_DIR, CAPTURE_DIR, RECORD_DIR]:
    os.makedirs(d, exist_ok=True)

COMPARE_SIZE = 768
INNER_BORDER_RATIO = 0.045
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VALID_MODES = {"reference", "original", "copy", "test"}

REFERENCE_CACHE: List[Dict[str, Any]] = []

try:
    from skimage.metrics import structural_similarity as skimage_ssim
    HAVE_SSIM = True
except Exception:
    HAVE_SSIM = False


# ============================================================
# APP
# ============================================================

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
# BASIC UTILS
# ============================================================

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
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    return obj


def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, ensure_ascii=False, indent=2)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def relative_path(path: str) -> str:
    return os.path.relpath(path, BASE_DIR)


# ============================================================
# METRICS
# ============================================================

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
# PREPROCESS
# ============================================================

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
# SCORING
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

    return {
        "base_score": base_score,
        "adjusted_score": adjusted_score,
        "copy_risk_score": copy_risk_score,
        "gap_booster_risk": gap_booster_risk,
        "gap_nonlinear_penalty": nonlinear_gap_penalty(gap_booster_risk),
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
        ssim >= 62.0 and corr >= 88.0 and mask_iou >= 78.0 and edge_f1 >= 38.0 and risk < 60.0
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
# REFERENCES AND RECORDS
# ============================================================

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


def summarize_reference_count() -> int:
    ensure_references_loaded()
    return len(REFERENCE_CACHE)


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


def compact_scores(best: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not best:
        return {}
    keys = [
        "base_score", "adjusted_score", "copy_risk_score", "gap_booster_risk", "gap_nonlinear_penalty",
        "ssim_score", "corr_score", "mask_f1", "mask_iou", "grid_score", "edge_f1",
        "black_gain_abs", "black_gain_pct", "lap_drop_pct", "tenengrad_drop_pct", "edge_acutance_drop_pct",
        "black_auc_gain", "black_curve_l1", "hf_drop_pct", "component_count_drop_pct", "component_area_gain_pct"
    ]
    rounded = {}
    for k in keys:
        if k in best:
            precision = 6 if k in ["black_gain_abs", "black_auc_gain", "black_curve_l1"] else 4
            rounded[k] = round(safe_float(best[k]), precision)
    return rounded


def list_saved_records() -> List[Dict[str, Any]]:
    records = []
    if not os.path.exists(RECORD_DIR):
        return records
    for name in os.listdir(RECORD_DIR):
        if not name.endswith(".json"):
            continue
        path = os.path.join(RECORD_DIR, name)
        try:
            rec = read_json(path)
            records.append(rec)
        except Exception:
            continue
    records.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    return records


# ============================================================
# ROUTES
# ============================================================

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
    return {
        "ok": True,
        "app": APP_TITLE,
        "version": APP_VERSION,
        "reference_count": len(REFERENCE_CACHE),
        "records": len(list_saved_records()),
        "time": time.time(),
    }


@app.get("/api/refs")
def api_refs():
    ensure_references_loaded()
    return {
        "reference_count": len(REFERENCE_CACHE),
        "references": [
            {"ref_file": r["ref_file"], "ref_image_id": r["ref_image_id"]}
            for r in REFERENCE_CACHE
        ]
    }


@app.post("/api/reload-refs")
def api_reload_refs():
    global REFERENCE_CACHE
    REFERENCE_CACHE = load_references()
    return {"reference_count": len(REFERENCE_CACHE), "message": "References reloaded."}


@app.post("/api/preview/{mode}")
async def api_preview_capture(
    mode: str,
    file: UploadFile = File(...),
    quality_json: str = Form(default="{}")
):
    if mode not in VALID_MODES:
        raise HTTPException(status_code=400, detail="Invalid mode.")

    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        img = read_image_from_bytes(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        client_quality = json.loads(quality_json) if quality_json else {}
    except Exception:
        client_quality = {}

    capture_id = new_id(mode)
    pending_img_path = os.path.join(PENDING_DIR, f"{capture_id}.jpg")
    pending_json_path = os.path.join(PENDING_DIR, f"{capture_id}.json")

    cv2.imwrite(pending_img_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 96])

    target_prep = preprocess_cdp_image(img)

    if mode == "reference":
        decision = {
            "final_user_status": "REFERENCE_READY_TO_SAVE",
            "final_user_message": "Referans otomatik çekildi. Kontrol edip kaydedebilirsin.",
            "final_reason": "reference_preview",
        }
        best = None
        top_refs = []
    else:
        scored = score_against_references(target_prep)
        if scored is None:
            decision = {
                "final_user_status": "NO_REFERENCE",
                "final_user_message": "Henüz kayıtlı referans yok. Önce referans çekip kaydet.",
                "final_reason": "missing_reference",
            }
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
        "reference_count": summarize_reference_count(),
        "final_user_status": decision["final_user_status"],
        "final_user_message": decision["final_user_message"],
        "final_reason": decision["final_reason"],
        "best_ref_image_id": best.get("ref_image_id") if best else None,
        "best_ref_file": best.get("ref_file") if best else None,
        "scores": compact_scores(best),
        "client_quality": client_quality,
        "top_refs": [
            {
                "ref_image_id": r["ref_image_id"],
                "ref_file": r["ref_file"],
                "base_score": round(safe_float(r.get("base_score")), 4),
                "adjusted_score": round(safe_float(r.get("adjusted_score")), 4),
                "copy_risk_score": round(safe_float(r.get("copy_risk_score")), 4),
                "gap_booster_risk": round(safe_float(r.get("gap_booster_risk")), 4),
                "ssim_score": round(safe_float(r.get("ssim_score")), 4),
                "mask_iou": round(safe_float(r.get("mask_iou")), 4),
                "edge_f1": round(safe_float(r.get("edge_f1")), 4),
            }
            for r in top_refs
        ],
        "created_at": time.time(),
    }

    write_json(pending_json_path, response)
    return JSONResponse(json_safe(response))


@app.post("/api/save/{capture_id}")
def api_save_capture(capture_id: str):
    global REFERENCE_CACHE

    pending_img_path = os.path.join(PENDING_DIR, f"{capture_id}.jpg")
    pending_json_path = os.path.join(PENDING_DIR, f"{capture_id}.json")

    if not os.path.exists(pending_img_path) or not os.path.exists(pending_json_path):
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
    shutil.copyfile(pending_img_path, final_img_path)

    if mode == "reference":
        meta["final_user_status"] = "REFERENCE_SAVED"
        meta["final_user_message"] = "Referans CDP kaydedildi. Bundan sonraki testler bu referansa göre karşılaştırılacak."
        meta["final_reason"] = "reference_saved"
        REFERENCE_CACHE = load_references()

    record = {
        "record_id": record_id,
        "capture_id": capture_id,
        "mode": mode,
        "created_at": time.time(),
        "image_path": final_img_path,
        "image_url": f"/api/records/{record_id}/image",
        "file_name": final_name,
        "reference_count_after_save": len(REFERENCE_CACHE),
        **meta,
        "saved": True,
        "saved_image_url": f"/api/records/{record_id}/image",
    }

    record_json_path = os.path.join(RECORD_DIR, f"{record_id}.json")
    write_json(record_json_path, record)

    try:
        os.remove(pending_img_path)
        os.remove(pending_json_path)
    except Exception:
        pass

    return JSONResponse(json_safe(record))


@app.delete("/api/pending/{capture_id}")
def api_delete_pending(capture_id: str):
    deleted = False
    for ext in [".jpg", ".json"]:
        p = os.path.join(PENDING_DIR, f"{capture_id}{ext}")
        if os.path.exists(p):
            os.remove(p)
            deleted = True
    return {"deleted": deleted, "capture_id": capture_id}


@app.get("/api/pending/{capture_id}/image")
def api_pending_image(capture_id: str):
    path = os.path.join(PENDING_DIR, f"{capture_id}.jpg")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Pending image not found.")
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
            "final_user_status": r.get("final_user_status"),
            "final_user_message": r.get("final_user_message"),
            "final_reason": r.get("final_reason"),
            "best_ref_image_id": r.get("best_ref_image_id"),
            "scores": r.get("scores", {}),
            "client_quality": r.get("client_quality", {}),
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
