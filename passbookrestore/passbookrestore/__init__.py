"""PassbookRestore AI: restoration and OCR-readiness evaluation for legacy banking scans."""

from .config import RestoreConfig
from .pipeline import RestorationPipeline

__all__ = ["RestoreConfig", "RestorationPipeline"]
__version__ = "1.0.0"
