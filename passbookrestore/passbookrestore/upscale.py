"""Super-resolution backends.

RealESRGANUpscaler  : Real-ESRGAN x4plus (needs torch, basicsr, realesrgan and the weights file).
OpenCVUpscaler      : Lanczos upscale; always available, used as a fallback and as an ablation baseline.

`get_upscaler("auto")` picks Real-ESRGAN when it can be loaded, otherwise falls back to OpenCV and
logs why, so the pipeline keeps working on machines without a GPU stack.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .config import RestoreConfig

log = logging.getLogger("passbookrestore.upscale")


def _patch_torchvision_for_basicsr() -> None:
    """basicsr (a Real-ESRGAN dependency) imports torchvision.transforms.functional_tensor, which newer
    torchvision releases removed. Provide it as an alias of torchvision.transforms.functional."""
    import sys
    import types

    name = "torchvision.transforms.functional_tensor"
    if name in sys.modules:
        return
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
        return
    except ImportError:
        pass
    from torchvision.transforms import functional as F

    shim = types.ModuleType(name)
    shim.__dict__.update({k: v for k, v in vars(F).items() if not k.startswith("__")})
    sys.modules[name] = shim


class OpenCVUpscaler:
    name = "opencv-lanczos"

    def __init__(self, scale: float = 2.0):
        self.scale = scale

    def upscale(self, img: np.ndarray) -> np.ndarray:
        if self.scale == 1:
            return img
        return cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LANCZOS4)


class RealESRGANUpscaler:
    name = "real-esrgan-x4plus"

    def __init__(self, scale: float = 2.0, weights: str | Path = "", tile: int = 256, device: str | None = None):
        import torch  # noqa: F401  (import errors are handled by get_upscaler)
        _patch_torchvision_for_basicsr()
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(
                f"Real-ESRGAN weights not found at {weights}. Run: python scripts/download_weights.py"
            )
        self.scale = scale
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        self._sr = RealESRGANer(
            scale=4,
            model_path=str(weights),
            model=model,
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=False,                      # fp32 so it also runs on CPU
            device=device,
        )

    def upscale(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:                    # the network expects 3 channels
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        out, _ = self._sr.enhance(img, outscale=self.scale)
        return out


def get_upscaler(cfg: RestoreConfig):
    mode = cfg.upscaler.lower()
    if mode == "opencv":
        return OpenCVUpscaler(cfg.scale)
    try:
        return RealESRGANUpscaler(cfg.scale, cfg.weights_path, cfg.tile)
    except Exception as exc:  # ImportError, FileNotFoundError, torch/CUDA problems
        if mode == "realesrgan":
            raise
        log.warning("Real-ESRGAN unavailable (%s: %s); falling back to OpenCV Lanczos.", type(exc).__name__, exc)
        return OpenCVUpscaler(cfg.scale)