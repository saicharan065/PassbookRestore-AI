"""OpenCV scan-artifact reduction. Every function takes and returns uint8 arrays."""

from __future__ import annotations

import cv2
import numpy as np

from .config import RestoreConfig


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def normalize_background(gray: np.ndarray, kernel: int = 41) -> np.ndarray:
    """Divide by an estimate of the paper background.

    A morphological close removes dark text and leaves paper, stains and illumination gradients.
    Dividing the page by that estimate flattens yellowing, stains and shadows while keeping ink.
    """
    kernel = kernel | 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se)
    bg = cv2.medianBlur(bg, kernel)
    out = cv2.divide(gray, bg, scale=255)
    return out


def denoise(gray: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return gray
    return cv2.fastNlMeansDenoising(gray, None, h=strength, templateWindowSize=7, searchWindowSize=21)


def stretch_contrast(gray: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(gray, [low_pct, high_pct])
    if hi - lo < 1:
        return gray
    out = (gray.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_clahe(gray: np.ndarray, clip: float, grid: int) -> np.ndarray:
    if clip <= 0:
        return gray
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(gray)


def unsharp(gray: np.ndarray, amount: float, sigma: float) -> np.ndarray:
    if amount <= 0:
        return gray
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.addWeighted(gray, 1 + amount, blurred, -amount, 0)


def estimate_skew(gray: np.ndarray, max_deg: float = 7.0) -> float:
    """Skew angle in degrees (positive = counter-clockwise), found by maximising the sharpness of
    the horizontal projection profile over a small angle range. Robust for ruled tables and text."""
    h, w = gray.shape
    scale = 600.0 / max(h, w)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    ink = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    sh, sw = ink.shape
    center = (sw / 2, sh / 2)

    def score(angle: float) -> float:
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(ink, m, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)
        prof = rot.sum(axis=1).astype(np.float64)
        return float(np.sum(np.diff(prof) ** 2))

    coarse = np.arange(-max_deg, max_deg + 1e-9, 0.5)
    best = max(coarse, key=score)
    fine = np.arange(best - 0.5, best + 0.5 + 1e-9, 0.1)
    best = max(fine, key=score)
    return float(best) if abs(best) >= 0.2 else 0.0


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    if angle == 0:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    border = int(np.median(img)) if img.ndim == 2 else (255, 255, 255)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def remove_rules(gray: np.ndarray, min_frac: float = 0.08) -> np.ndarray:
    """Erase long horizontal/vertical ruled lines (table borders, field boxes, cheque lines).

    Tesseract reads text inside ruled cells poorly: borders get mistaken for characters ('|', '_')
    and boxed fields are often skipped entirely. This is applied to the *OCR copy only*; the saved
    restored image keeps its lines so the document layout is preserved.
    """
    h, w = gray.shape
    ink = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 31, 12)
    mask = np.zeros_like(ink)
    for horizontal in (True, False):
        length = max(int((w if horizontal else h) * min_frac), 15)
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1) if horizontal else (1, length))
        mask = cv2.bitwise_or(mask, cv2.morphologyEx(ink, cv2.MORPH_OPEN, se))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    out = gray.copy()
    out[mask > 0] = int(np.percentile(gray, 90))     # paper level
    return out


def clean_before_upscale(gray: np.ndarray, cfg: RestoreConfig) -> np.ndarray:
    """Stage 1: remove background artefacts and noise at the original resolution."""
    out = gray
    if cfg.median_ksize >= 3:
        out = cv2.medianBlur(out, cfg.median_ksize | 1)
    if cfg.normalize_background:
        out = normalize_background(out, cfg.background_kernel)
    out = denoise(out, cfg.denoise_strength)
    return out


def enhance_after_upscale(gray: np.ndarray, cfg: RestoreConfig) -> tuple[np.ndarray, float]:
    """Stage 2: straighten, then contrast and sharpness on the upscaled page.
    Returns (image, skew_angle_applied_in_degrees)."""
    skew = estimate_skew(gray, cfg.max_deskew_deg) if cfg.deskew else 0.0
    out = stretch_contrast(gray, *cfg.stretch_percentiles)
    out = apply_clahe(out, cfg.clahe_clip, cfg.clahe_grid)
    out = unsharp(out, cfg.sharpen_amount, cfg.sharpen_sigma)
    return rotate(out, skew), skew
