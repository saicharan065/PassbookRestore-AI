"""End-to-end restoration + evaluation pipeline.

    scan -> gray -> background normalisation -> denoise -> upscale (Real-ESRGAN) -> contrast/sharpen -> deskew
         -> restored image -> Tesseract -> OCR / quality / structure metrics vs. the untouched baseline

`RestorationPipeline.run_directory()` processes a folder of scans, writes restored images, before/after
comparison sheets, a CSV of per-file metrics and a Markdown + JSON summary.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from . import metrics, preprocess
from .config import IMAGE_EXTENSIONS, RestoreConfig
from .ocr import run_ocr
from .upscale import OpenCVUpscaler, get_upscaler

log = logging.getLogger("passbookrestore")


@dataclass
class RestorationResult:
    restored: np.ndarray
    timings: dict = field(default_factory=dict)
    skew_deg: float = 0.0
    upscaler: str = ""


def read_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not read image: {path}")
    return img


class RestorationPipeline:
    def __init__(self, config: RestoreConfig | None = None):
        self.cfg = config or RestoreConfig()
        self._upscaler = None

    # -------------------------------------------------------------- restoration
    @property
    def upscaler(self):
        if self._upscaler is None:
            self._upscaler = get_upscaler(self.cfg)
            log.info("upscaler: %s", self._upscaler.name)
        return self._upscaler

    def _upscale(self, gray: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        if cfg.scale == 1:
            return gray
        up = self.upscaler
        if max(gray.shape) > cfg.max_input_side and up.name.startswith("real-esrgan"):
            log.info("input larger than max_input_side; using the OpenCV upscaler for this image")
            up = OpenCVUpscaler(cfg.scale)
        out = up.upscale(gray)
        return preprocess.to_gray(out)

    def restore(self, img: np.ndarray) -> RestorationResult:
        cfg = self.cfg
        t = {}
        t0 = time.perf_counter()
        gray = preprocess.to_gray(img)
        gray = preprocess.clean_before_upscale(gray, cfg)
        t["clean"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        gray = self._upscale(gray)
        t["upscale"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        gray, skew = preprocess.enhance_after_upscale(gray, cfg)
        t["enhance"] = time.perf_counter() - t0
        name = self.upscaler.name if cfg.scale != 1 else "none"
        return RestorationResult(gray, t, skew, name)

    # -------------------------------------------------------------- evaluation
    def ocr_view(self, img: np.ndarray) -> np.ndarray:
        """The copy of a page that is sent to Tesseract (ruled lines removed when configured)."""
        gray = preprocess.to_gray(img)
        return preprocess.remove_rules(gray) if self.cfg.ocr_remove_rules else gray

    def _ocr(self, img: np.ndarray):
        c = self.cfg
        return run_ocr(img, c.ocr_lang, c.ocr_psm, c.ocr_target_height)

    def evaluate(self, original: np.ndarray, restored: np.ndarray, truth: str | None = None,
                 clean: np.ndarray | None = None) -> dict:
        """Metrics for one document. `truth` (ground-truth text) and `clean` (clean reference image)
        are optional; with them the report gains CER, word accuracy, SSIM and a structure score
        against the true layout."""
        base_ocr = self._ocr(original)                 # baseline: Tesseract on the raw scan, nothing else
        rest_ocr = self._ocr(self.ocr_view(restored))
        row: dict = {
            "baseline_conf": round(base_ocr.mean_conf, 1),
            "restored_conf": round(rest_ocr.mean_conf, 1),
            "baseline_words": base_ocr.n_words,
            "restored_words": rest_ocr.n_words,
        }
        for k, v in metrics.quality_report(original).items():
            row[f"orig_{k}"] = v
        for k, v in metrics.quality_report(restored).items():
            row[f"rest_{k}"] = v
        if truth is not None:
            row["baseline_cer"] = round(metrics.cer(truth, base_ocr.text), 4)
            row["restored_cer"] = round(metrics.cer(truth, rest_ocr.text), 4)
            row["baseline_word_acc"] = round(metrics.word_accuracy(truth, base_ocr.text), 4)
            row["restored_word_acc"] = round(metrics.word_accuracy(truth, rest_ocr.text), 4)
        ref = clean if clean is not None else original
        row["structure_reference"] = "clean" if clean is not None else "original_scan"
        row.update({f"struct_{k}": v for k, v in metrics.structure_report(restored, ref).items()})
        if clean is not None:
            row["baseline_struct_score"] = metrics.structure_report(original, clean)["structure_score"]
            row["baseline_ssim"] = round(metrics.ssim_against_reference(original, clean), 4)
            row["restored_ssim"] = round(metrics.ssim_against_reference(restored, clean), 4)
        row["_baseline_text"] = base_ocr.text
        row["_restored_text"] = rest_ocr.text
        return row

    def ablation(self, original: np.ndarray, truth: str) -> dict[str, dict]:
        """OCR accuracy of each processing variant, to show which stage actually helps."""
        out = {}
        # (name, image builder, whether the OCR copy has ruled lines removed)
        variants = {
            "1. baseline (raw scan)": (lambda: original, False),
            "2. rule removal only": (lambda: original, True),
            "3. OpenCV cleanup, no upscale": (lambda: RestorationPipeline(replace(self.cfg, scale=1)).restore(original).restored, True),
            "4. upscale only": (lambda: self._upscale(preprocess.to_gray(original)), True),
            "5. full pipeline": (lambda: self.restore(original).restored, True),
        }
        for name, (fn, strip) in variants.items():
            img = fn()
            res = self._ocr(self.ocr_view(img) if strip and self.cfg.ocr_remove_rules else preprocess.to_gray(img))
            out[name] = {
                "cer": round(metrics.cer(truth, res.text), 4),
                "word_acc": round(metrics.word_accuracy(truth, res.text), 4),
                "conf": round(res.mean_conf, 1),
            }
        return out

    # -------------------------------------------------------------- batch
    def run_directory(self, input_dir: str | Path, output_dir: str | Path,
                      truth_dir: str | Path | None = None, clean_dir: str | Path | None = None,
                      make_comparisons: bool = True) -> dict:
        input_dir, output_dir = Path(input_dir), Path(output_dir)
        (output_dir / "restored").mkdir(parents=True, exist_ok=True)
        if make_comparisons:
            (output_dir / "comparisons").mkdir(exist_ok=True)
        (output_dir / "ocr_text").mkdir(exist_ok=True)

        files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not files:
            raise FileNotFoundError(f"no images found in {input_dir}")

        rows, failures = [], []
        for path in files:
            try:
                t0 = time.perf_counter()
                original = read_image(path)
                result = self.restore(original)
                truth = None
                if truth_dir and (Path(truth_dir) / f"{path.stem}.txt").is_file():
                    truth = (Path(truth_dir) / f"{path.stem}.txt").read_text(encoding="utf-8")
                clean = None
                if clean_dir and (Path(clean_dir) / path.name).is_file():
                    clean = read_image(Path(clean_dir) / path.name)
                row = self.evaluate(original, result.restored, truth, clean)
                (output_dir / "ocr_text" / f"{path.stem}.baseline.txt").write_text(row.pop("_baseline_text"), encoding="utf-8")
                (output_dir / "ocr_text" / f"{path.stem}.restored.txt").write_text(row.pop("_restored_text"), encoding="utf-8")
                cv2.imwrite(str(output_dir / "restored" / f"{path.stem}.png"), result.restored)
                if make_comparisons:
                    cv2.imwrite(str(output_dir / "comparisons" / f"{path.stem}_before_after.png"),
                                before_after(original, result.restored))
                row = {"file": path.name, "doc_type": guess_type(path.name), "skew_deg": round(result.skew_deg, 2),
                       "upscaler": result.upscaler, "seconds": round(time.perf_counter() - t0, 2), **row}
                rows.append(row)
                log.info("restored %s", path.name)
            except Exception as exc:  # one bad scan must not stop the batch
                log.exception("failed on %s", path.name)
                failures.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})

        summary = summarize(rows)
        summary["failures"] = failures
        summary["config"] = self.cfg.to_dict()
        write_csv(rows, output_dir / "metrics.csv")
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (output_dir / "report.md").write_text(render_report(rows, summary), encoding="utf-8")
        return summary


# ------------------------------------------------------------------ helpers
def guess_type(name: str) -> str:
    low = name.lower()
    for t in ("passbook", "cheque", "kyc"):
        if t in low:
            return t
    return "other"


def before_after(original: np.ndarray, restored: np.ndarray, height: int = 900) -> np.ndarray:
    def fit(img):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        s = height / img.shape[0]
        return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)

    a, b = fit(original), fit(restored)
    pad = np.full((height, 20, 3), 255, np.uint8)
    sheet = np.hstack([a, pad, b])
    banner = np.full((44, sheet.shape[1], 3), 245, np.uint8)
    cv2.putText(banner, "BEFORE (scan)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
    cv2.putText(banner, "AFTER (restored)", (a.shape[1] + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
    return np.vstack([banner, sheet])


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _mean(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return round(float(np.mean(vals)), 4) if vals else None


def summarize(rows: list[dict]) -> dict:
    keys = ["baseline_cer", "restored_cer", "baseline_word_acc", "restored_word_acc",
            "baseline_conf", "restored_conf", "orig_sharpness", "rest_sharpness",
            "orig_contrast", "rest_contrast", "orig_background_std", "rest_background_std",
            "orig_noise_sigma", "rest_noise_sigma", "baseline_ssim", "restored_ssim",
            "baseline_struct_score", "struct_structure_score", "seconds"]
    out = {"n_documents": len(rows), "means": {k: _mean(rows, k) for k in keys}, "by_type": {}}
    for t in sorted({r["doc_type"] for r in rows}):
        sub = [r for r in rows if r["doc_type"] == t]
        out["by_type"][t] = {"n": len(sub), **{k: _mean(sub, k) for k in
                             ("baseline_cer", "restored_cer", "baseline_word_acc", "restored_word_acc", "struct_structure_score")}}
    cers = [(r["baseline_cer"], r["restored_cer"]) for r in rows if "restored_cer" in r]
    if cers:
        out["docs_improved"] = int(sum(1 for b, a in cers if a < b))
        out["docs_with_truth"] = len(cers)
    return out


def _fmt(v):
    return "n/a" if v is None else f"{v}"


def render_report(rows: list[dict], s: dict) -> str:
    m = s["means"]
    lines = ["# PassbookRestore AI: Evaluation Report", "",
             f"Documents processed: **{s['n_documents']}**  |  failures: **{len(s.get('failures', []))}**", ""]
    if m.get("baseline_cer") is not None:
        lines += ["## OCR accuracy (Tesseract, vs. ground truth)", "",
                  "| Metric | Baseline (raw scan) | Restored |", "|---|---|---|",
                  f"| Character error rate (lower is better) | {_fmt(m['baseline_cer'])} | {_fmt(m['restored_cer'])} |",
                  f"| Word accuracy (higher is better) | {_fmt(m['baseline_word_acc'])} | {_fmt(m['restored_word_acc'])} |",
                  f"| Mean OCR confidence | {_fmt(m['baseline_conf'])} | {_fmt(m['restored_conf'])} |", "",
                  f"CER improved on **{s.get('docs_improved', 0)} of {s.get('docs_with_truth', 0)}** documents.", ""]
    lines += ["## Image quality", "", "| Metric | Original | Restored |", "|---|---|---|",
              f"| Sharpness (Laplacian variance @1000px) | {_fmt(m['orig_sharpness'])} | {_fmt(m['rest_sharpness'])} |",
              f"| Contrast (P95-P5) | {_fmt(m['orig_contrast'])} | {_fmt(m['rest_contrast'])} |",
              f"| Background std (lower = cleaner paper) | {_fmt(m['orig_background_std'])} | {_fmt(m['rest_background_std'])} |",
              f"| Noise sigma | {_fmt(m['orig_noise_sigma'])} | {_fmt(m['rest_noise_sigma'])} |"]
    if m.get("baseline_ssim") is not None:
        lines.append(f"| SSIM vs clean reference | {_fmt(m['baseline_ssim'])} | {_fmt(m['restored_ssim'])} |")
    lines.append("")
    if m.get("struct_structure_score") is not None:
        lines += ["## Structure preservation", "",
                  "Score combines ruled-line counts, text-row counts, aspect ratio and ink overlap against the reference layout (1.0 = identical).", ""]
        if m.get("baseline_struct_score") is not None:
            lines.append(f"Mean structure score: scan **{_fmt(m['baseline_struct_score'])}** -> restored **{_fmt(m['struct_structure_score'])}**.")
        else:
            lines.append(f"Mean structure score vs original scan: **{_fmt(m['struct_structure_score'])}**.")
        lines.append("")
    if s["by_type"]:
        lines += ["## By document type", "", "| Type | n | CER base | CER restored | Word acc base | Word acc restored | Structure |",
                  "|---|---|---|---|---|---|---|"]
        for t, d in s["by_type"].items():
            lines.append(f"| {t} | {d['n']} | {_fmt(d['baseline_cer'])} | {_fmt(d['restored_cer'])} | "
                         f"{_fmt(d['baseline_word_acc'])} | {_fmt(d['restored_word_acc'])} | {_fmt(d['struct_structure_score'])} |")
        lines.append("")
    lines += ["## Per-document results", "", "| File | Type | CER base | CER restored | Struct | Upscaler | Time (s) |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['file']} | {r['doc_type']} | {_fmt(r.get('baseline_cer'))} | {_fmt(r.get('restored_cer'))} | "
                     f"{_fmt(r.get('struct_structure_score'))} | {r['upscaler']} | {r['seconds']} |")
    if s.get("failures"):
        lines += ["", "## Failures", ""] + [f"- {f['file']}: {f['error']}" for f in s["failures"]]
    return "\n".join(lines) + "\n"
