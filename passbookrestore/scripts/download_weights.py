"""Download the official Real-ESRGAN x4plus weights (~64 MB) into models/.

    python scripts/download_weights.py
"""

import sys
import urllib.request
from pathlib import Path

URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
DEST = Path(__file__).resolve().parent.parent / "models" / "RealESRGAN_x4plus.pth"


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.is_file() and DEST.stat().st_size > 60_000_000:
        print(f"already present: {DEST}")
        return 0
    print(f"downloading {URL}")
    try:
        urllib.request.urlretrieve(URL, DEST)
    except Exception as exc:
        print(f"download failed: {exc}\nDownload the file manually from the URL above and place it at {DEST}")
        return 1
    print(f"saved {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
