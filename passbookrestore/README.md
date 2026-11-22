# PassbookRestore AI: Legacy Banking Document Restorer

Restores scanned legacy passbooks, cheques and KYC forms (faded, stained, noisy, blurred, skewed, low-resolution)
and measures whether restoration actually improves Tesseract OCR while preserving table and form structure.

```
scan -> median (dust) -> background normalisation -> NLM denoise -> upscale (Real-ESRGAN | Lanczos fallback)
     -> contrast stretch -> unsharp -> deskew -> restored image
                                                   |-> remove ruled lines (OCR copy only) -> Tesseract
```

## Quick start

```bash
pip install -r requirements.txt          # + Tesseract engine (see requirements.txt)
python -m passbookrestore generate --out data --per-type 6            # synthetic degraded dataset + ground truth
python -m passbookrestore restore data/raw --out outputs \
       --truth data/ground_truth --clean data/clean                   # batch restore + evaluate
python -m passbookrestore ablation data/raw --truth data/ground_truth # which stage helps?
python -m pytest                                                      # 14 tests
```

Restore a single scan: `python -m passbookrestore restore scan.jpg --out outputs`

## Web app

```bash
python -m passbookrestore.webapp          # opens http://localhost:5000
python -m passbookrestore.webapp --port 8080 --no-browser
```
Upload a scan (or pick a bundled sample), choose the upscaler, scale, denoise and sharpen settings, and get a
before/after slider, Tesseract text from the raw and restored scan, OCR confidence, image-quality metrics and a
layout check. Paste ground-truth text (or use a sample) to also get character error rate and word accuracy.
Restored image, comparison sheet and OCR text can be downloaded. Everything runs locally; uploads are kept in a
temporary folder for one hour.

### Run history with MongoDB (optional)
The web app saves every run (settings, metrics, OCR text, small before/after previews) to MongoDB and lists them
under "Run history" with View and Delete. Install MongoDB Community Server (or use a free Atlas cluster) and run:
```bash
set MONGODB_URI=mongodb://localhost:27017      # PowerShell: $env:MONGODB_URI="mongodb://localhost:27017"
set MONGODB_DB=passbookrestore                 # optional, this is the default
python -m passbookrestore.webapp
```
Collection: `passbookrestore.runs`. Without MongoDB the app still works; history is just switched off.

### Enabling Real-ESRGAN
```bash
pip install torch torchvision basicsr realesrgan
python scripts/download_weights.py                       # RealESRGAN_x4plus.pth into models/
python -m passbookrestore restore data/raw --upscaler realesrgan
```
`--upscaler auto` (default) uses Real-ESRGAN when it can be loaded and otherwise logs why and falls back to OpenCV.
Known issue: recent torchvision removed `functional_tensor`, which `basicsr` imports. If you hit that error, change
that import in `basicsr/data/degradations.py` to `torchvision.transforms.functional`.

## Project layout

| Path | Purpose | Rubric item |
|---|---|---|
| `passbookrestore/synthetic.py`, `degrade.py` | Synthetic passbook / cheque / KYC generator with exact ground truth, and a degradation model (fade, yellowing, stains, uneven light, tears, blur, noise, JPEG, low-res, skew) | Data preparation |
| `passbookrestore/upscale.py` | Real-ESRGAN x4plus wrapper (tiled, CPU-safe) + OpenCV fallback | AI restoration |
| `passbookrestore/preprocess.py` | OpenCV artefact reduction, deskew, ruled-line removal | AI restoration, restoration quality |
| `passbookrestore/ocr.py`, `metrics.py` | Tesseract wrapper; CER, word accuracy, sharpness, contrast, noise, SSIM, structure score | OCR and structure |
| `passbookrestore/pipeline.py`, `cli.py` | Batch pipeline, per-file error isolation, CSV/JSON/Markdown reports, before/after sheets | Pipeline and testing |
| `tests/` | 14 pytest tests including an end-to-end generate -> restore -> evaluate test | Pipeline and testing |
| `results/` | Report, metrics CSV, ablation table, example before/after images | Documentation and demo |

## Results (held-out set: 24 documents, seeds never used for tuning, OpenCV upscaler)

| Metric | Raw scan | Restored |
|---|---|---|
| Character error rate (lower is better) | 0.793 | **0.243** |
| Word accuracy | 0.301 | **0.598** |
| Mean OCR confidence | 48.9 | **65.5** |
| Contrast (P95-P5) | 70.9 | 158.2 |
| Background std (lower = cleaner) | 3.56 | 2.37 |
| SSIM vs clean reference | 0.617 | 0.772 |
| Structure score (1.0 = identical layout) | 0.831 | **0.966** |

CER improved on all 24 of 24 documents. Full numbers: `results/report.md`, `results/metrics.csv`.

### Ablation (which stage earns its place)

| Variant | CER | Word acc |
|---|---|---|
| 1. Baseline, raw scan | 0.793 | 0.301 |
| 2. Ruled-line removal only | 0.517 | 0.520 |
| 3. OpenCV cleanup, no upscale | 0.225 | 0.610 |
| 4. Upscale only (Lanczos) | 0.716 | 0.438 |
| 5. Full pipeline | 0.243 | 0.598 |

## Findings worth putting in your report

1. **Ruled lines are the biggest OCR obstacle on banking documents.** On the *clean* synthetic passbook, Tesseract
   returned garbage rows, and on the KYC form it skipped every boxed field. Removing long horizontal/vertical lines
   from the OCR copy raised clean-image word accuracy to about 0.95. The saved restored image keeps its lines.
2. **Median filtering (dust specks) was the largest single restoration gain**: CER 0.68 -> 0.37 in the tuning sweep.
3. **CLAHE hurt OCR** (it amplified residual speckle), so it is off by default (`clahe_clip=0`). Enable it only if you
   re-check on your data.
4. **Lanczos upscaling adds nothing for OCR** (rows 3 vs 5), because OCR input is resized to a fixed height anyway.
   Whether Real-ESRGAN improves OCR is an open question to test; its main value is visual quality for archiving.
5. Defaults were tuned on a 18-document set (`seed 42`); the table above is on a separate 24-document held-out set.

## Limitations (be upfront about these in your demo)

- **Real-ESRGAN was not run in the environment where this was built** (PyTorch could not be installed there).
  The Real-ESRGAN code follows the official `RealESRGANer` API but is **untested**; the numbers above use the
  OpenCV fallback. Run the weights script and `ablation --upscaler realesrgan` on your machine, and report both.
- All data is **synthetic**. Real legacy scans (handwriting, stamps, faded dot-matrix printing, Indic scripts) will
  behave differently, and results will be lower. Add a few real scans (with personal data removed) to `data/` and use
  `restore` without `--truth` to get confidence and quality metrics.
- Real-ESRGAN is a generative model and can hallucinate strokes on very degraded digits. For compliance archiving,
  keep the original scan alongside the restored copy and do not treat restored images as the legal original.
- The worst documents (severity about 0.8) still reach CER around 0.7: restoration cannot recover text that is gone.
- English-only OCR (`--lang` via `RestoreConfig.ocr_lang`; install extra Tesseract language packs for others).

## Suggested 4-week plan for the team of 4

| Week | Work | Owner idea |
|---|---|---|
| 1 | Dataset: generate synthetic set, add real scans, inspect degradations (`data/`, `manifest.json`) | Data |
| 2 | Real-ESRGAN + OpenCV setup, sample processing, tune `RestoreConfig` | Restoration |
| 3 | OCR + structure evaluation, ablation, report | Evaluation |
| 4 | Pipeline hardening, tests, demo, README | Pipeline / docs |
