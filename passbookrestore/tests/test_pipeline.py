import shutil

import cv2
import numpy as np
import pytest

from passbookrestore import RestoreConfig, RestorationPipeline, metrics, preprocess
from passbookrestore.cli import main
from passbookrestore.degrade import degrade
from passbookrestore.synthetic import DOC_TYPES, make_document, to_array
from passbookrestore.upscale import OpenCVUpscaler, get_upscaler

needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")


def test_synthetic_documents_are_deterministic_and_have_truth():
    for t in DOC_TYPES:
        a, b = make_document(t, 7), make_document(t, 7)
        assert a.text == b.text and a.text.strip()
        assert np.array_equal(to_array(a.image), to_array(b.image))


def test_degradation_is_seeded_and_reduces_resolution():
    clean = to_array(make_document("cheque", 1).image)
    d1, d2 = degrade(clean, 0.6, seed=3), degrade(clean, 0.6, seed=3)
    assert np.array_equal(d1, d2)
    assert d1.shape[1] < clean.shape[1]
    assert metrics.contrast(d1) < metrics.contrast(clean)


def test_upscaler_auto_falls_back_without_weights(tmp_path):
    cfg = RestoreConfig(upscaler="auto", weights_path=str(tmp_path / "missing.pth"), scale=2.0)
    up = get_upscaler(cfg)
    assert isinstance(up, OpenCVUpscaler)
    assert up.upscale(np.zeros((10, 20), np.uint8)).shape == (20, 40)


def test_upscaler_realesrgan_mode_raises_without_weights(tmp_path):
    cfg = RestoreConfig(upscaler="realesrgan", weights_path=str(tmp_path / "missing.pth"))
    with pytest.raises(Exception):
        get_upscaler(cfg)


def test_deskew_recovers_known_angle():
    doc = to_array(make_document("passbook", 2).image)
    gray = preprocess.to_gray(doc)
    tilted = preprocess.rotate(gray, 3.0)
    est = preprocess.estimate_skew(tilted)
    assert abs(est + 3.0) <= 0.6          # correction angle is the negative of the tilt


def test_remove_rules_keeps_text_and_drops_lines():
    doc = to_array(make_document("passbook", 2).image)
    gray = preprocess.to_gray(doc)
    stripped = preprocess.remove_rules(gray)
    before = metrics.structure_report(gray, gray)
    after = metrics.structure_report(stripped, gray)
    assert after["h_rules"] < before["h_rules"]
    assert (stripped < 128).sum() > 0.5 * (gray < 128).sum() * 0.5   # text is still there


def test_restore_output_shape_and_type():
    cfg = RestoreConfig(upscaler="opencv", scale=2.0)
    img = degrade(to_array(make_document("kyc", 3).image), 0.5, seed=3)
    out = RestorationPipeline(cfg).restore(img)
    assert out.restored.dtype == np.uint8 and out.restored.ndim == 2
    assert out.restored.shape[0] >= 2 * img.shape[0] - 2


@needs_tesseract
def test_restoration_improves_ocr_on_small_batch(tmp_path):
    """End-to-end: generate -> restore -> evaluate. Restoration must lower mean CER and preserve layout."""
    data = tmp_path / "data"
    assert main(["generate", "--out", str(data), "--per-type", "2", "--seed", "500",
                 "--min-severity", "0.3", "--max-severity", "0.6"]) == 0
    out = tmp_path / "out"
    assert main(["restore", str(data / "raw"), "--out", str(out), "--truth", str(data / "ground_truth"),
                 "--clean", str(data / "clean"), "--upscaler", "opencv"]) == 0
    import json
    s = json.loads((out / "summary.json").read_text())
    assert s["n_documents"] == 6 and not s["failures"]
    assert s["means"]["restored_cer"] < s["means"]["baseline_cer"]
    assert s["means"]["struct_structure_score"] > 0.85
    assert (out / "report.md").is_file() and (out / "metrics.csv").is_file()
    assert len(list((out / "restored").glob("*.png"))) == 6


@needs_tesseract
def test_batch_survives_a_corrupt_file(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    cv2.imwrite(str(raw / "kyc_ok.png"), degrade(to_array(make_document("kyc", 9).image), 0.4, seed=9))
    (raw / "passbook_bad.png").write_bytes(b"not an image")
    summary = RestorationPipeline(RestoreConfig(upscaler="opencv")).run_directory(raw, tmp_path / "out")
    assert summary["n_documents"] == 1
    assert len(summary["failures"]) == 1 and summary["failures"][0]["file"] == "passbook_bad.png"
