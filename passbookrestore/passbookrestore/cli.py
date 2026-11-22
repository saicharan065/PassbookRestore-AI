"""Command line interface:  python -m passbookrestore <command> ..."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2

from .config import RestoreConfig
from .degrade import degrade
from .pipeline import RestorationPipeline, read_image
from .synthetic import DOC_TYPES, make_document, save_ground_truth, to_array


def cmd_generate(args) -> int:
    out = Path(args.out)
    for sub in ("clean", "raw", "ground_truth"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    manifest = []
    idx = 0
    for doc_type in DOC_TYPES:
        for i in range(args.per_type):
            seed = args.seed + idx
            doc = make_document(doc_type, seed)
            name = f"{doc_type}_{i + 1:02d}"
            clean = to_array(doc.image)
            # spread severity across the set so results show behaviour from mild to heavy damage
            severity = args.min_severity + (args.max_severity - args.min_severity) * (i / max(args.per_type - 1, 1))
            cv2.imwrite(str(out / "clean" / f"{name}.png"), clean)
            cv2.imwrite(str(out / "raw" / f"{name}.png"), degrade(clean, severity, seed))
            save_ground_truth(doc, out / "ground_truth" / f"{name}.txt")
            manifest.append({"name": name, "type": doc_type, "severity": round(severity, 2), "seed": seed, **doc.meta})
            idx += 1
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"generated {len(manifest)} documents in {out}")
    return 0


def _config_from(args) -> RestoreConfig:
    cfg = RestoreConfig(scale=args.scale, upscaler=args.upscaler)
    if getattr(args, "weights", None):
        cfg.weights_path = args.weights
    return cfg


def cmd_restore(args) -> int:
    pipe = RestorationPipeline(_config_from(args))
    src = Path(args.input)
    if src.is_dir():
        summary = pipe.run_directory(src, args.out, args.truth, args.clean)
        print(json.dumps(summary["means"], indent=2))
        print(f"report: {Path(args.out) / 'report.md'}")
        return 1 if summary["failures"] else 0
    result = pipe.restore(read_image(src))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{src.stem}_restored.png"
    cv2.imwrite(str(dest), result.restored)
    print(f"wrote {dest}  (upscaler: {result.upscaler})")
    return 0


def cmd_ablation(args) -> int:
    pipe = RestorationPipeline(_config_from(args))
    raw, truth = Path(args.input), Path(args.truth)
    totals: dict[str, list] = {}
    for path in sorted(raw.glob("*.png")):
        t = truth / f"{path.stem}.txt"
        if not t.is_file():
            continue
        res = pipe.ablation(read_image(path), t.read_text(encoding="utf-8"))
        for name, m in res.items():
            totals.setdefault(name, []).append(m)
    lines = ["| Variant | Mean CER | Mean word acc | Mean conf |", "|---|---|---|---|"]
    for name, ms in totals.items():
        n = len(ms)
        lines.append(f"| {name} | {sum(m['cer'] for m in ms)/n:.4f} | {sum(m['word_acc'] for m in ms)/n:.4f} | {sum(m['conf'] for m in ms)/n:.1f} |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("# Ablation\n\n" + text + "\n", encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="passbookrestore", description="Restore legacy banking document scans and evaluate OCR readiness.")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="create a synthetic degraded dataset with ground truth")
    g.add_argument("--out", default="data")
    g.add_argument("--per-type", type=int, default=6)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--min-severity", type=float, default=0.3)
    g.add_argument("--max-severity", type=float, default=0.9)
    g.set_defaults(func=cmd_generate)

    def common(sp):
        sp.add_argument("--scale", type=float, default=2.0)
        sp.add_argument("--upscaler", choices=["auto", "realesrgan", "opencv"], default="auto")
        sp.add_argument("--weights", default=None, help="path to RealESRGAN_x4plus.pth")

    r = sub.add_parser("restore", help="restore one image or a whole folder")
    r.add_argument("input", help="image file or folder of scans")
    r.add_argument("--out", default="outputs")
    r.add_argument("--truth", default=None, help="folder of <name>.txt ground-truth transcripts")
    r.add_argument("--clean", default=None, help="folder of clean reference images with matching names")
    common(r)
    r.set_defaults(func=cmd_restore)

    a = sub.add_parser("ablation", help="compare OCR accuracy of each processing variant")
    a.add_argument("input", help="folder of degraded scans")
    a.add_argument("--truth", required=True)
    a.add_argument("--out", default=None)
    common(a)
    a.set_defaults(func=cmd_ablation)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
