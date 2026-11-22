"""Tesseract OCR wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract


@dataclass
class OCRResult:
    text: str
    mean_conf: float      # mean word confidence, 0-100 (-1 if nothing was recognised)
    n_words: int


def prepare_for_ocr(img: np.ndarray, target_height: int) -> np.ndarray:
    """Tesseract works best when x-height is roughly 20-30 px. Resize so the page height is near
    `target_height`, but never shrink very small images below their size by more than needed."""
    h = img.shape[0]
    if target_height <= 0 or h == 0:
        return img
    scale = target_height / h
    if 0.8 < scale < 1.25:
        return img
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)


def run_ocr(img: np.ndarray, lang: str = "eng", psm: int = 6, target_height: int = 1600) -> OCRResult:
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = prepare_for_ocr(gray, target_height)
    config = f"--oem 1 --psm {psm}"
    data = pytesseract.image_to_data(gray, lang=lang, config=config, output_type=pytesseract.Output.DICT)
    words, confs, line_key, lines, cur = [], [], None, [], []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if line_key is not None and key != line_key:
            lines.append(" ".join(cur))
            cur = []
        line_key = key
        cur.append(word)
        words.append(word)
        c = float(data["conf"][i])
        if c >= 0:
            confs.append(c)
    if cur:
        lines.append(" ".join(cur))
    return OCRResult("\n".join(lines), float(np.mean(confs)) if confs else -1.0, len(words))
