"""Quality, OCR and structure-preservation metrics."""

from __future__ import annotations

import re
from collections import Counter

import cv2
import numpy as np
from skimage.metrics import structural_similarity


# ---------------------------------------------------------------- text / OCR metrics
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def levenshtein(a: str, b: str) -> int:
    """Edit distance with a rolling row (O(len(a)*len(b)) time, O(len(b)) memory)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate on whitespace-normalised text (can exceed 1.0 for very noisy output)."""
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def word_accuracy(reference: str, hypothesis: str) -> float:
    """Fraction of reference words found in the OCR output (order independent, so a different
    reading order of table columns does not count as an error)."""
    ref = Counter(normalize_text(reference).split())
    hyp = Counter(normalize_text(hypothesis).split())
    total = sum(ref.values())
    if total == 0:
        return 1.0
    return sum(min(c, hyp[w]) for w, c in ref.items()) / total


# ---------------------------------------------------------------- image-quality metrics
def _gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _to_width(gray: np.ndarray, width: int) -> np.ndarray:
    if gray.shape[1] == width:
        return gray
    h = max(1, round(gray.shape[0] * width / gray.shape[1]))
    return cv2.resize(gray, (width, h), interpolation=cv2.INTER_AREA)


def sharpness(img: np.ndarray, width: int = 1000) -> float:
    """Variance of the Laplacian after resizing to a common width, so scans of different
    resolution are comparable."""
    g = _to_width(_gray(img), width)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def contrast(img: np.ndarray) -> float:
    """Spread between ink and paper: 95th minus 5th percentile of grey levels (0-255)."""
    p5, p95 = np.percentile(_gray(img), [5, 95])
    return float(p95 - p5)


def background_uniformity(img: np.ndarray) -> float:
    """Std-dev of the paper background (lower = flatter, fewer stains). Ink is excluded by taking
    only pixels brighter than the median."""
    g = _gray(img).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    paper = g[g > np.median(g)]
    return float(paper.std()) if paper.size else 0.0


def noise_level(img: np.ndarray) -> float:
    """Immerkaer fast noise-sigma estimate."""
    g = _gray(img).astype(np.float32)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    h, w = g.shape
    if h < 3 or w < 3:
        return 0.0
    conv = cv2.filter2D(g, -1, k)
    return float(np.sqrt(np.pi / 2) * np.abs(conv[1:-1, 1:-1]).sum() / (6 * (w - 2) * (h - 2)))


def quality_report(img: np.ndarray) -> dict:
    return {
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "sharpness": round(sharpness(img), 2),
        "contrast": round(contrast(img), 2),
        "background_std": round(background_uniformity(img), 2),
        "noise_sigma": round(noise_level(img), 2),
    }


def ssim_against_reference(img: np.ndarray, reference: np.ndarray) -> float:
    """SSIM between an image and the clean reference, both resized to the reference size."""
    ref = _gray(reference)
    g = cv2.resize(_gray(img), (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
    return float(structural_similarity(g, ref, data_range=255))


# ---------------------------------------------------------------- structure preservation
def _ink(gray: np.ndarray, width: int = 1000) -> np.ndarray:
    g = cv2.GaussianBlur(_to_width(gray, width), (3, 3), 0)
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12)


def _count_rules(ink: np.ndarray, horizontal: bool) -> int:
    """Count long horizontal or vertical ruled lines (table borders, cheque lines, field boxes)."""
    h, w = ink.shape
    length = max(int((w if horizontal else h) * 0.18), 20)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1) if horizontal else (1, length))
    opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, se)
    # merge strokes thickened by blur/noise into one line per rule
    merge = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7) if horizontal else (7, 1))
    opened = cv2.dilate(opened, merge)
    n, *_ = cv2.connectedComponentsWithStats(opened)
    return n - 1


def _count_text_rows(ink: np.ndarray) -> int:
    """Count text rows with a horizontal projection profile (rules removed first)."""
    h, w = ink.shape
    for horizontal in (True, False):
        length = max(int((w if horizontal else h) * 0.18), 20)
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1) if horizontal else (1, length))
        rules = cv2.dilate(cv2.morphologyEx(ink, cv2.MORPH_OPEN, se), np.ones((3, 3), np.uint8))
        ink = cv2.bitwise_and(ink, cv2.bitwise_not(rules))
    prof = cv2.GaussianBlur(ink.sum(axis=1).astype(np.float32).reshape(-1, 1), (1, 0), sigmaX=0, sigmaY=2).ravel()
    thresh = max(prof.max() * 0.18, 1.0)
    on = prof > thresh
    return int(np.sum(on[1:] & ~on[:-1]) + int(on[0]))


def _ratio_score(a: float, b: float) -> float:
    """1.0 when equal, falling towards 0 as the counts diverge."""
    if a == 0 and b == 0:
        return 1.0
    return min(a, b) / max(a, b)


def structure_report(candidate: np.ndarray, reference: np.ndarray) -> dict:
    """Compare layout of `candidate` against `reference` (the clean render when known, otherwise the
    original scan). Everything is measured on a common 1000 px width so resolution does not matter."""
    ci, ri = _ink(_gray(candidate)), _ink(_gray(reference))
    c_h, r_h = _count_rules(ci, True), _count_rules(ri, True)
    c_v, r_v = _count_rules(ci, False), _count_rules(ri, False)
    c_rows, r_rows = _count_text_rows(ci), _count_text_rows(ri)

    # aspect-ratio drift (cropping / stretching)
    ar_c = candidate.shape[1] / candidate.shape[0]
    ar_r = reference.shape[1] / reference.shape[0]
    aspect = _ratio_score(ar_c, ar_r)

    # ink overlap after a small dilation (tolerates 1-2 px of sub-pixel shift)
    h = min(ci.shape[0], ri.shape[0])
    a = cv2.dilate(ci[:h], np.ones((5, 5), np.uint8)) > 0
    b = cv2.dilate(ri[:h], np.ones((5, 5), np.uint8)) > 0
    inter = np.logical_and(ci[:h] > 0, b).sum()
    inter2 = np.logical_and(ri[:h] > 0, a).sum()
    ink_cov = float((inter / max((ci[:h] > 0).sum(), 1) + inter2 / max((ri[:h] > 0).sum(), 1)) / 2)

    rules = [_ratio_score(c_h, r_h), _ratio_score(c_v, r_v)]
    rows = _ratio_score(c_rows, r_rows)
    score = float(np.mean([np.mean(rules), rows, aspect, ink_cov]))
    return {
        "h_rules": c_h, "h_rules_ref": r_h,
        "v_rules": c_v, "v_rules_ref": r_v,
        "text_rows": c_rows, "text_rows_ref": r_rows,
        "rule_score": round(float(np.mean(rules)), 3),
        "row_score": round(rows, 3),
        "aspect_score": round(aspect, 3),
        "ink_overlap": round(ink_cov, 3),
        "structure_score": round(score, 3),
    }
