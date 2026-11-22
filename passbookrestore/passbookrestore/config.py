"""Central configuration for the restoration pipeline."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "RealESRGAN_x4plus.pth"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class RestoreConfig:
    # --- upscaling -------------------------------------------------------
    scale: float = 2.0                    # final output scale relative to the input
    upscaler: str = "auto"                # "auto" | "realesrgan" | "opencv"
    weights_path: str = str(DEFAULT_WEIGHTS)
    tile: int = 256                       # Real-ESRGAN tile size (0 = no tiling); keeps CPU memory low
    max_input_side: int = 1400            # inputs larger than this skip the neural upscaler

    # --- OpenCV artifact reduction ---------------------------------------
    normalize_background: bool = True     # remove stains / uneven illumination / yellowing
    background_kernel: int = 41           # odd; should be larger than a character stroke
    median_ksize: int = 3                 # removes salt-and-pepper dust specks (0 disables)
    denoise_strength: float = 8.0         # fastNlMeansDenoising h (0 disables)
    clahe_clip: float = 0.0               # local contrast; OFF by default: it amplified speckle and raised CER in our sweep
    clahe_grid: int = 8
    deskew: bool = True
    max_deskew_deg: float = 7.0
    sharpen_amount: float = 0.8           # unsharp mask strength; 0 disables
    sharpen_sigma: float = 1.2
    stretch_percentiles: tuple = (1.0, 99.0)  # contrast stretch after cleaning

    # --- OCR ---------------------------------------------------------------
    ocr_lang: str = "eng"
    ocr_psm: int = 4                      # single column of variable-size text: best reading order on forms/tables
    ocr_target_height: int = 1600         # OCR input is resized so its height is near this
    ocr_remove_rules: bool = True         # strip ruled lines from the OCR copy of the restored page

    def to_dict(self) -> dict:
        return asdict(self)
