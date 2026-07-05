"""OCR engine wrapper.

PaddleOCR is initialized once at import time to avoid reloading models on
every request. To swap engines (e.g. RapidOCR for a lighter CPU footprint),
replace the implementation of `ocr_array` and `_load_engine` while keeping
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

Concurrency note: PaddlePaddle CPU predictors are not thread-safe — two
threads calling .ocr() on the same instance can segfault or corrupt results.
All inference is serialized behind _INFER_LOCK; each single inference then
gets the whole CPU (cpu_threads below), which beats oversubscribed parallel
calls where N inferences each spawn per-core OMP thread pools.
"""

import logging
import os
import threading
from functools import lru_cache

logger = logging.getLogger("ocr-worker.engine")

_PADDLEOCR_HOME = os.path.expanduser(os.environ.get("PADDLEOCR_HOME", "~/.paddleocr"))
_CH_DET_MODEL_DIR = os.path.join(_PADDLEOCR_HOME, "whl", "det", "ch", "ch_PP-OCRv4_det_infer")

_CPU_THREADS = int(os.environ.get("OCR_CPU_THREADS", os.cpu_count() or 4))

# Detection knobs, tuned with the eval harness (see eval/run_eval.py and
# docs/worker-tuning.md). unclip 2.0 (PaddleOCR default 1.5) stops Hangul
# jongseong clipping at box edges: on the synthetic Korean set it cut
# hangul-only CER 0.0141 -> 0.0092 and ran ~18% faster (fewer, merged rec
# crops). Raising det_limit_side_len was rejected: no accuracy gain, slower,
# and OOM on dense pages within the 4g container cap.
_DET_LIMIT_SIDE_LEN = int(os.environ.get("OCR_DET_LIMIT_SIDE_LEN", "960"))
_DET_DB_UNCLIP_RATIO = float(os.environ.get("OCR_DET_UNCLIP_RATIO", "2.0"))
_DET_DB_BOX_THRESH = float(os.environ.get("OCR_DET_BOX_THRESH", "0.6"))

_INFER_LOCK = threading.Lock()


def _ensure_ch_detector_downloaded() -> None:
    """Bootstrap the 'ch' detector into the shared model cache if missing.

    This also downloads the 'ch' recognizer/classifier, which the real
    engine below never uses (it keeps the Korean recognizer). Fetching only
    the detector would need reaching into paddleocr's internal model-URL
    registry, which crashed repeatedly under manual testing in this
    environment; the extra one-time, few-MB download is a safer trade.
    """
    if os.path.isdir(_CH_DET_MODEL_DIR):
        return
    from paddleocr import PaddleOCR

    logger.info("Downloading 'ch' detector (better Hangul box recall than the default)...")
    PaddleOCR(lang="ch", use_gpu=False, use_angle_cls=False, show_log=False)


@lru_cache(maxsize=1)
def _load_engine():
    from paddleocr import PaddleOCR

    _ensure_ch_detector_downloaded()

    logger.info(
        "Loading PaddleOCR (rec=korean, det=ch, cpu_threads=%d, mkldnn, "
        "det_limit=%d, unclip=%.2f, box_thresh=%.2f)...",
        _CPU_THREADS, _DET_LIMIT_SIDE_LEN, _DET_DB_UNCLIP_RATIO, _DET_DB_BOX_THRESH,
    )
    engine = PaddleOCR(
        lang="korean",
        det_model_dir=_CH_DET_MODEL_DIR,
        use_gpu=False,
        use_angle_cls=False,
        enable_mkldnn=True,
        cpu_threads=_CPU_THREADS,
        det_limit_side_len=_DET_LIMIT_SIDE_LEN,
        det_db_unclip_ratio=_DET_DB_UNCLIP_RATIO,
        det_db_box_thresh=_DET_DB_BOX_THRESH,
        show_log=False,
    )
    logger.info("PaddleOCR ready.")
    return engine


def warmup() -> None:
    """Force model load and lazy backend init (oneDNN graph compilation etc.)
    at startup instead of on the first real request."""
    import numpy as np

    engine = _load_engine()
    dummy = np.full((64, 256, 3), 255, dtype=np.uint8)
    with _INFER_LOCK:
        engine.ocr(dummy, cls=False)
    logger.info("Warmup inference done.")


def ocr_array(img) -> str:
    """Run OCR on a BGR ndarray and return its text, lines joined by newline."""
    engine = _load_engine()
    with _INFER_LOCK:
        result = engine.ocr(img, cls=False)
    if not result or result[0] is None:
        return ""

    lines = [line[1][0] for line in result[0]]
    return "\n".join(lines)


def ocr_image(image_bytes: bytes) -> str:
    """Run OCR on an encoded image (PNG/JPEG/...) and return its text."""
    import numpy as np
    import cv2

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return ocr_array(img)
