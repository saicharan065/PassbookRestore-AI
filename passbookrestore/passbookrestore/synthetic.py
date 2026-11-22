"""Generate clean synthetic banking documents together with their ground-truth text.

Real legacy scans cannot be shipped with a student project (they contain personal data),
so the dataset is built from synthetic passbook pages, cheques and KYC forms. Every document
is rendered at full quality first (the "clean reference"), then degraded in degrade.py.
Because the exact text is known, OCR accuracy can be measured as CER / word accuracy.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "DejaVuSans.ttf",
    "arial.ttf",  # Windows (Pillow searches C:\\Windows\\Fonts)
    "Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS
    "/Library/Fonts/Arial.ttf",
]
BOLD_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "DejaVuSans-Bold.ttf",
    "arialbd.ttf",  # Windows
    "Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",  # macOS
    "/Library/Fonts/Arial Bold.ttf",
]

FIRST = ["Ravi", "Anita", "Suresh", "Lakshmi", "Kiran", "Meena", "Arjun", "Priya", "Venkat", "Sunita"]
LAST = ["Kumar", "Reddy", "Sharma", "Rao", "Patel", "Nair", "Singh", "Iyer", "Das", "Gupta"]
BANKS = ["State Union Bank", "National Savings Bank", "Bharat Co-operative Bank", "Southern Trust Bank"]
CITIES = ["Hyderabad", "Guntur", "Vijayawada", "Chennai", "Pune", "Nagpur", "Kolkata", "Jaipur"]
PARTICULARS = ["CASH DEP", "ATM WDL", "NEFT CR", "CHQ CLG", "INT CR", "UPI DR", "SALARY", "RENT PMT", "EMI DR"]
DOC_TYPES = ("passbook", "cheque", "kyc")


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (BOLD_CANDIDATES if bold else FONT_CANDIDATES):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    import warnings
    warnings.warn("No TrueType font found; synthetic documents will use Pillow's thin default font.")
    return ImageFont.load_default(size=size)


@dataclass
class SyntheticDoc:
    image: Image.Image                       # clean RGB render
    doc_type: str
    text_lines: list[str]                    # ground truth in reading order
    meta: dict = field(default_factory=dict)  # e.g. number of table rows, ruled lines

    @property
    def text(self) -> str:
        return "\n".join(self.text_lines)


def _name(rng: random.Random) -> str:
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def _acct(rng: random.Random) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(11))


def _date(rng: random.Random) -> str:
    return f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(1988, 2004)}"


def _amount(rng: random.Random, high: int = 50000) -> str:
    return f"{rng.randint(100, high):,}.00"


def make_passbook(rng: random.Random, rows: int = 12) -> SyntheticDoc:
    w, h = 1100, 140 + 60 * (rows + 2)
    img = Image.new("RGB", (w, h), (252, 250, 244))
    d = ImageDraw.Draw(img)
    f, fb, fs = load_font(26), load_font(34, True), load_font(24)
    bank, holder, acct, city = rng.choice(BANKS), _name(rng), _acct(rng), rng.choice(CITIES)
    lines: list[str] = []

    d.text((40, 24), bank.upper(), font=fb, fill=(20, 20, 20))
    lines.append(bank.upper())
    head = f"Account No {acct}   Name {holder.upper()}   Branch {city}"
    d.text((40, 76), head, font=fs, fill=(30, 30, 30))
    lines.append(head)

    cols = [40, 235, 505, 695, 880, w - 40]
    titles = ["Date", "Particulars", "Withdrawal", "Deposit", "Balance"]
    top = 130
    row_h = 60
    for i, t in enumerate(titles):
        d.text((cols[i] + 10, top + 14), t, font=fs, fill=(10, 10, 10))
    lines.append(" ".join(titles))
    balance = rng.randint(5000, 90000)
    for r in range(rows):
        y = top + row_h * (r + 1)
        date = _date(rng)
        part = rng.choice(PARTICULARS)
        amt = rng.randint(200, 20000)
        if rng.random() < 0.5:
            wd, dep, balance = f"{amt:,}.00", "", balance - amt
        else:
            wd, dep, balance = "", f"{amt:,}.00", balance + amt
        balance = max(balance, 0)
        cells = [date, part, wd, dep, f"{balance:,}.00"]
        for i, c in enumerate(cells):
            d.text((cols[i] + 10, y + 14), c, font=f, fill=(15, 15, 15))
        lines.append(" ".join(c for c in cells if c))
    bottom = top + row_h * (rows + 1)
    for r in range(rows + 2):                       # horizontal ruled lines
        y = top + row_h * r
        d.line([(cols[0], y), (cols[-1], y)], fill=(60, 60, 60), width=2)
    for x in cols:                                   # vertical column lines
        d.line([(x, top), (x, bottom)], fill=(60, 60, 60), width=2)
    return SyntheticDoc(img, "passbook", lines, {"table_rows": rows, "h_lines": rows + 2, "v_lines": len(cols)})


def make_cheque(rng: random.Random) -> SyntheticDoc:
    w, h = 1200, 520
    img = Image.new("RGB", (w, h), (246, 250, 252))
    d = ImageDraw.Draw(img)
    f, fb, fs = load_font(28), load_font(36, True), load_font(24)
    bank, payee, city = rng.choice(BANKS), _name(rng), rng.choice(CITIES)
    acct, date = _acct(rng), _date(rng)
    amt = rng.randint(1000, 90000)
    lines = []

    d.rectangle([12, 12, w - 12, h - 12], outline=(70, 70, 70), width=3)
    d.text((40, 30), bank.upper(), font=fb, fill=(15, 15, 15))
    lines.append(bank.upper())
    d.text((40, 84), f"{city} Main Branch", font=fs, fill=(40, 40, 40))
    lines.append(f"{city} Main Branch")
    d.text((820, 40), f"Date {date}", font=f, fill=(15, 15, 15))
    lines[0] += f" Date {date}"          # same visual row as the bank name
    d.text((40, 170), f"Pay {payee.upper()}", font=f, fill=(15, 15, 15))
    lines.append(f"Pay {payee.upper()}")
    d.line([(100, 205), (900, 205)], fill=(90, 90, 90), width=2)
    words = f"Rupees {amt} only"
    d.text((40, 235), words, font=f, fill=(15, 15, 15))
    lines.append(f"{words} {amt:,}.00")   # the amount box sits on the same visual row
    d.line([(40, 270), (900, 270)], fill=(90, 90, 90), width=2)
    d.rectangle([930, 215, 1160, 275], outline=(60, 60, 60), width=2)
    d.text((945, 228), f"{amt:,}.00", font=f, fill=(15, 15, 15))
    d.text((40, 360), f"A/C No {acct}", font=f, fill=(15, 15, 15))
    lines.append(f"A/C No {acct}")
    d.line([(820, 400), (1150, 400)], fill=(90, 90, 90), width=2)
    d.text((880, 410), "Authorised Signatory", font=fs, fill=(40, 40, 40))
    lines.append("Authorised Signatory")
    return SyntheticDoc(img, "cheque", lines, {"h_lines": 3, "boxes": 2})


def make_kyc(rng: random.Random) -> SyntheticDoc:
    w, h = 1000, 900
    img = Image.new("RGB", (w, h), (250, 249, 246))
    d = ImageDraw.Draw(img)
    f, fb = load_font(28), load_font(36, True)
    bank = rng.choice(BANKS)
    lines = []
    title = f"{bank.upper()} KYC FORM"
    d.text((40, 30), title, font=fb, fill=(15, 15, 15))
    lines.append(title)
    holder = _name(rng)
    fields = [
        ("Name", holder.upper()),
        ("Date of Birth", _date(rng)),
        ("Father Name", _name(rng).upper()),
        ("Address", f"{rng.randint(1, 99)}-{rng.randint(1, 99)} MG Road {rng.choice(CITIES)}"),
        ("Mobile", "9" + "".join(str(rng.randint(0, 9)) for _ in range(9))),
        ("PAN", "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(5))
                + "".join(str(rng.randint(0, 9)) for _ in range(4)) + rng.choice("ABCDEFGHJK")),
        ("Account No", _acct(rng)),
    ]
    y = 130
    for label, value in fields:
        d.text((50, y), f"{label}", font=f, fill=(30, 30, 30))
        d.rectangle([330, y - 8, w - 40, y + 46], outline=(70, 70, 70), width=2)
        d.text((345, y), value, font=f, fill=(10, 10, 10))
        lines.append(f"{label} {value}")
        y += 90
    d.text((50, y + 30), "Signature of Applicant", font=f, fill=(30, 30, 30))
    d.line([(420, y + 62), (w - 60, y + 62)], fill=(90, 90, 90), width=2)
    lines.append("Signature of Applicant")
    return SyntheticDoc(img, "kyc", lines, {"fields": len(fields), "boxes": len(fields)})


def make_document(doc_type: str, seed: int) -> SyntheticDoc:
    rng = random.Random(seed)
    if doc_type == "passbook":
        return make_passbook(rng, rows=rng.randint(9, 14))
    if doc_type == "cheque":
        return make_cheque(rng)
    if doc_type == "kyc":
        return make_kyc(rng)
    raise ValueError(f"unknown doc_type {doc_type!r}; expected one of {DOC_TYPES}")


def to_array(img: Image.Image) -> np.ndarray:
    """PIL RGB -> OpenCV BGR uint8."""
    return np.asarray(img.convert("RGB"))[:, :, ::-1].copy()


def save_ground_truth(doc: SyntheticDoc, path: Path) -> None:
    Path(path).write_text(doc.text + "\n", encoding="utf-8")
