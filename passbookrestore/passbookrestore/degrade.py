"""Simulate decades of wear on a clean document image (BGR uint8 in, BGR uint8 out).

Each effect mirrors a real failure mode of legacy scans: fading ink, yellowed and stained paper,
uneven illumination, blur from the scanner, sensor noise, JPEG artefacts, small tears and low
resolution. `severity` scales every effect between 0 (almost clean) and 1 (heavily damaged).
"""

from __future__ import annotations

import cv2
import numpy as np


def fade(img: np.ndarray, severity: float) -> np.ndarray:
    """Lift ink towards the paper colour and compress contrast."""
    alpha = 1.0 - 0.45 * severity          # contrast multiplier
    beta = 255 * (1 - alpha) * 0.85        # push blacks up
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def yellow_tint(img: np.ndarray, severity: float) -> np.ndarray:
    out = img.astype(np.float32)
    out[:, :, 0] *= 1 - 0.28 * severity     # B down
    out[:, :, 1] *= 1 - 0.10 * severity     # G down slightly
    return np.clip(out, 0, 255).astype(np.uint8)


def uneven_illumination(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    gx = np.linspace(rng.uniform(-1, 1), rng.uniform(-1, 1), w)[None, :]
    gy = np.linspace(rng.uniform(-1, 1), rng.uniform(-1, 1), h)[:, None]
    field = (gx + gy) / 2
    gain = 1 + 0.25 * severity * field
    return np.clip(img.astype(np.float32) * gain[:, :, None], 0, 255).astype(np.uint8)


def stains(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    layer = np.zeros((h, w), np.float32)
    for _ in range(int(2 + 5 * severity)):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(max(20, w // 25), max(40, w // 8)))
        cv2.circle(layer, (cx, cy), r, float(rng.uniform(0.3, 0.8)), -1)
    layer = cv2.GaussianBlur(layer, (0, 0), sigmaX=max(10, w // 40))
    tint = np.array([0.55, 0.75, 0.9], np.float32)       # brownish (BGR multiplier at full strength)
    mult = 1 - (1 - tint)[None, None, :] * layer[:, :, None] * (0.6 * severity)
    return np.clip(img.astype(np.float32) * mult, 0, 255).astype(np.uint8)


def tears(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Thin white-ish cracks and a torn corner. Kept small so content is degraded, not destroyed."""
    out = img.copy()
    h, w = out.shape[:2]
    for _ in range(int(1 + 2 * severity)):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        pts = [(x0, y0)]
        for _ in range(8):
            x0 = int(np.clip(x0 + rng.integers(-12, 13), 0, w - 1))
            y0 = int(np.clip(y0 + rng.integers(8, 30), 0, h - 1))
            pts.append((x0, y0))
        cv2.polylines(out, [np.array(pts, np.int32)], False, (238, 240, 240), thickness=int(rng.integers(1, 3)))
    k = int(min(h, w) * 0.04 * (0.5 + severity))
    tri = np.array([[0, 0], [k, 0], [0, k]], np.int32)
    cv2.fillPoly(out, [tri], (236, 238, 238))
    return out


def blur(img: np.ndarray, severity: float) -> np.ndarray:
    sigma = 0.4 + 1.1 * severity
    return cv2.GaussianBlur(img, (0, 0), sigma)


def noise(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    sigma = 3 + 14 * severity
    n = rng.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + n
    salt = rng.random(img.shape[:2])
    out[salt < 0.0015 * severity] = 255
    out[salt > 1 - 0.0015 * severity] = 0
    return np.clip(out, 0, 255).astype(np.uint8)


def low_resolution(img: np.ndarray, severity: float) -> np.ndarray:
    scale = 0.62 - 0.17 * severity          # 0.62 .. 0.45 of original resolution
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def jpeg(img: np.ndarray, severity: float) -> np.ndarray:
    q = int(70 - 40 * severity)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def rotate_small(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def degrade(img: np.ndarray, severity: float = 0.6, seed: int = 0, skew: bool = True) -> np.ndarray:
    """Apply the full degradation chain to a BGR image."""
    severity = float(np.clip(severity, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    out = img
    out = fade(out, severity)
    out = yellow_tint(out, severity)
    out = uneven_illumination(out, severity, rng)
    out = stains(out, severity, rng)
    if skew:
        out = rotate_small(out, float(rng.uniform(-2.0, 2.0)) * severity)
    out = low_resolution(out, severity)
    out = blur(out, severity)
    out = tears(out, severity, rng)
    out = noise(out, severity, rng)
    out = jpeg(out, severity)
    return out
