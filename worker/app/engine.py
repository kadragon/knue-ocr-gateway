"""OCR engine wrapper.

PaddleOCR is initialized once at import time to avoid reloading models on
every request. To swap engines (e.g. RapidOCR for a lighter CPU footprint),
replace the implementation of `ocr_image` and `_load_engine` while keeping
this module's public signature stable.

Detector note: PaddleOCR's lang="korean" preset pairs the Korean recognizer
with the generic "multilingual" (ml) detector, which was empirically found to
miss whole lines of dense Hangul text (verified: a 7-character university
name went completely undetected, while the "ch" detector found it reliably).
So we keep the Korean recognizer but swap in the "ch" detector, which is
trained on far more CJK data and detects Hangul lines correctly. Combining
this det/rec mix with the angle classifier (use_angle_cls=True) silently
breaks detection (zero boxes found on otherwise-detectable text), so the
classifier must stay disabled.
"""

import logging
import os
from functools import lru_cache

logger = logging.getLogger("ocr-worker.engine")

_PADDLEOCR_HOME = os.path.expanduser(os.environ.get("PADDLEOCR_HOME", "~/.paddleocr"))
_CH_DET_MODEL_DIR = os.path.join(_PADDLEOCR_HOME, "whl", "det", "ch", "ch_PP-OCRv4_det_infer")


def _ensure_ch_detector_downloaded() -> None:
    """Bootstrap the 'ch' detector into the shared model cache if missing."""
    if os.path.isdir(_CH_DET_MODEL_DIR):
        return
    from paddleocr import PaddleOCR

    logger.info("Downloading 'ch' detector (better Hangul box recall than the default)...")
    PaddleOCR(lang="ch", use_gpu=False, use_angle_cls=False, show_log=False)


@lru_cache(maxsize=1)
def _load_engine():
    from paddleocr import PaddleOCR

    _ensure_ch_detector_downloaded()

    logger.info("Loading PaddleOCR (rec=korean, det=ch, cpu)...")
    engine = PaddleOCR(
        lang="korean",
        det_model_dir=_CH_DET_MODEL_DIR,
        use_gpu=False,
        use_angle_cls=False,
        show_log=False,
    )
    logger.info("PaddleOCR ready.")
    return engine


def warmup() -> None:
    """Force model load at startup instead of on first request."""
    _load_engine()


def ocr_image(image_bytes: bytes) -> str:
    """Run OCR on a single image and return its text, lines joined by newline."""
    import numpy as np
    import cv2

    engine = _load_engine()
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    result = engine.ocr(img, cls=False)
    if not result or result[0] is None:
        return ""

    lines = [line[1][0] for line in result[0]]
    return "\n".join(lines)
