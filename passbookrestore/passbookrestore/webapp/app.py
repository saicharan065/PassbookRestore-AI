"""Flask web app: upload a legacy banking scan, restore it, and inspect OCR + quality metrics."""

from __future__ import annotations

import base64
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_file

from .. import metrics
from ..config import IMAGE_EXTENSIONS, PROJECT_ROOT, RestoreConfig
from ..pipeline import RestorationPipeline, before_after
from .store import RunStore

SAMPLE_DIRS = [PROJECT_ROOT / "data_holdout", PROJECT_ROOT / "data"]
MAX_UPLOAD_MB = 15
JOB_TTL_SECONDS = 3600


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def _png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("could not encode image")
    return buf.tobytes()


def _thumb_b64(img: np.ndarray, max_side: int = 1600) -> str:
    s = max_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _sample_index() -> dict[str, dict]:
    """name -> {raw, truth, clean} for every bundled sample scan."""
    out: dict[str, dict] = {}
    for root in SAMPLE_DIRS:
        raw = root / "raw"
        if not raw.is_dir():
            continue
        for p in sorted(raw.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            key = p.name
            if key in out:
                continue
            truth, clean = root / "ground_truth" / f"{p.stem}.txt", root / "clean" / p.name
            out[key] = {"raw": p, "truth": truth if truth.is_file() else None,
                        "clean": clean if clean.is_file() else None}
    return out


def create_app(store: RunStore | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
    work = Path(tempfile.mkdtemp(prefix="passbookrestore_web_"))
    jobs: dict[str, dict] = {}
    lock = threading.Lock()
    pipelines: dict[tuple, RestorationPipeline] = {}
    runs = store if store is not None else RunStore()

    def get_pipeline(cfg: RestoreConfig) -> RestorationPipeline:
        key = (cfg.upscaler, cfg.scale, cfg.denoise_strength, cfg.sharpen_amount, cfg.deskew,
               cfg.normalize_background, cfg.ocr_lang)
        with lock:
            if key not in pipelines:
                pipelines[key] = RestorationPipeline(cfg)
            return pipelines[key]

    def purge_old() -> None:
        now = time.time()
        for jid in [j for j, v in jobs.items() if now - v["t"] > JOB_TTL_SECONDS]:
            shutil.rmtree(work / jid, ignore_errors=True)
            jobs.pop(jid, None)

    @app.get("/")
    def index():
        return render_template("index.html", max_mb=MAX_UPLOAD_MB)

    @app.get("/api/status")
    def status():
        from ..upscale import get_upscaler
        try:
            esr = get_upscaler(RestoreConfig(upscaler="realesrgan")).name
            esr_ok, esr_msg = True, esr
        except Exception as exc:  # torch / weights not installed
            esr_ok, esr_msg = False, f"{type(exc).__name__}: {exc}"
        try:
            import pytesseract
            tess = str(pytesseract.get_tesseract_version())
        except Exception as exc:
            tess = None
        return jsonify({"mongodb": runs.enabled, "mongodb_error": runs.error, "tesseract": tess, "realesrgan_available": esr_ok, "realesrgan_detail": esr_msg})

    @app.get("/api/samples")
    def samples():
        return jsonify([{"name": k, "has_truth": v["truth"] is not None} for k, v in _sample_index().items()])

    @app.get("/api/sample-image/<name>")
    def sample_image(name: str):
        s = _sample_index().get(name)
        if not s:
            abort(404)
        return send_file(s["raw"])

    @app.post("/api/restore")
    def restore():
        purge_old()
        form = request.form
        sample_name = form.get("sample")
        truth: str | None = None
        clean = None
        if sample_name:
            s = _sample_index().get(sample_name)
            if not s:
                return jsonify(error="unknown sample"), 404
            data = np.frombuffer(s["raw"].read_bytes(), np.uint8)
            filename = sample_name
            if s["truth"]:
                truth = s["truth"].read_text(encoding="utf-8")
            if s["clean"]:
                clean = cv2.imread(str(s["clean"]), cv2.IMREAD_COLOR)
        else:
            f = request.files.get("file")
            if not f or not f.filename:
                return jsonify(error="Choose a scan to upload."), 400
            if Path(f.filename).suffix.lower() not in IMAGE_EXTENSIONS:
                return jsonify(error="Unsupported file type. Use PNG, JPG, TIFF or BMP."), 400
            data = np.frombuffer(f.read(), np.uint8)
            filename = Path(f.filename).name
            if form.get("truth", "").strip():
                truth = form["truth"]

        original = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if original is None:
            return jsonify(error="That file could not be read as an image."), 400

        try:
            cfg = RestoreConfig(
                upscaler=form.get("upscaler", "auto"),
                scale=float(form.get("scale", 2)),
                denoise_strength=float(form.get("denoise", 8)),
                sharpen_amount=float(form.get("sharpen", 0.8)),
                deskew=form.get("deskew", "true") == "true",
                normalize_background=form.get("normalize", "true") == "true",
            )
            if cfg.scale not in (1.0, 1.5, 2.0, 3.0, 4.0) or cfg.upscaler not in ("auto", "opencv", "realesrgan"):
                return jsonify(error="Invalid settings."), 400
            pipe = get_pipeline(cfg)
            t0 = time.perf_counter()
            result = pipe.restore(original)
            row = pipe.evaluate(original, result.restored, truth, clean)
        except Exception as exc:
            return jsonify(error=f"Restoration failed: {type(exc).__name__}: {exc}"), 500
        elapsed = time.perf_counter() - t0

        base_text, rest_text = row.pop("_baseline_text"), row.pop("_restored_text")
        jid = uuid.uuid4().hex[:12]
        d = work / jid
        d.mkdir()
        stem = Path(filename).stem
        (d / "restored.png").write_bytes(_png_bytes(result.restored))
        (d / "comparison.png").write_bytes(_png_bytes(before_after(original, result.restored)))
        (d / "ocr_restored.txt").write_text(rest_text, encoding="utf-8")
        with lock:
            jobs[jid] = {"t": time.time(), "stem": stem}

        payload = _jsonable({
            "id": jid, "filename": filename, "seconds": round(elapsed, 2),
            "upscaler": result.upscaler, "skew_deg": round(result.skew_deg, 2),
            "before": _thumb_b64(original), "after": _thumb_b64(result.restored),
            "baseline_text": base_text, "restored_text": rest_text,
            "has_truth": truth is not None, "truth": truth,
            "metrics": row,
        })
        saved = runs.save({k: v for k, v in payload.items() if k != "id"} | {
            "settings": cfg.to_dict(), "source": "sample" if sample_name else "upload"})
        payload["run_id"] = saved
        return jsonify(payload)

    @app.get("/api/history")
    def history():
        return jsonify({"enabled": runs.enabled, "error": runs.error, "runs": runs.list()})

    @app.get("/api/history/<run_id>")
    def history_item(run_id: str):
        d = runs.get(run_id)
        if d is None:
            abort(404)
        return jsonify(_jsonable(d))

    @app.delete("/api/history/<run_id>")
    def history_delete(run_id: str):
        return jsonify(deleted=runs.delete(run_id))

    @app.get("/api/download/<jid>/<kind>")
    def download(jid: str, kind: str):
        if jid not in jobs or kind not in ("restored", "comparison", "text"):
            abort(404)
        stem = jobs[jid]["stem"]
        if kind == "text":
            return send_file(work / jid / "ocr_restored.txt", as_attachment=True, download_name=f"{stem}_ocr.txt")
        name = "restored.png" if kind == "restored" else "comparison.png"
        return send_file(work / jid / name, as_attachment=True, download_name=f"{stem}_{kind}.png")

    return app
