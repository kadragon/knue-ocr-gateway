"""PDF text extraction.

Digital PDFs already carry a text layer — PyMuPDF extracts it directly and
skips OCR entirely (fast, exact). Scanned PDFs have no usable text layer, so
each page is rasterized and sent through the OCR engine instead.
"""

import os
import time
from typing import List, Optional, Tuple

import fitz  # PyMuPDF

from . import engine

# 300 DPI is overkill for OCR here: PaddleOCR's detector downscales its input
# to det_limit_side_len=960 anyway, and the recognizer crops keep plenty of
# resolution at 200. Dropping to 200 cuts rendered pixels ~2.25x.
RENDER_DPI = int(os.environ.get("RENDER_DPI", "200"))
# A page with more than this many characters of native text is considered
# "digital" and OCR is skipped for it. A scanned page can still carry a short
# text layer (e.g. a scanner-stamped watermark or page label), so this must
# stay well above a handful of characters or those pages get misclassified
# as digital and lose their actual (image-only) content.
MIN_TEXT_LEN_TO_SKIP_OCR = 50
# PDF pages can declare arbitrary dimensions (spec allows up to 200in per
# side); rendering one at RENDER_DPI without a cap can allocate a
# multi-gigabyte pixmap from a tiny file. Cap the rendered area and scale
# the effective DPI down for oversized pages instead of refusing them.
MAX_RENDER_PIXELS = 4000 * 4000
MAX_PAGES = 200
# Whole-document processing budget. Must stay below the gateway's
# REQUEST_TIMEOUT_SECONDS (120 by default): once the gateway gives up, no one
# is reading the answer, so continuing to OCR remaining pages only burns CPU
# that queued requests need. Checked between pages; a heavily scanned PDF
# fails with a deadline error instead of silently outliving its caller.
DEADLINE_SECONDS = float(os.environ.get("OCR_DEADLINE_SECONDS", "110"))


class DeadlineExceeded(Exception):
    """Raised when PDF processing exceeds its time budget."""


def _render_matrix(page: "fitz.Page") -> "fitz.Matrix":
    zoom = RENDER_DPI / 72
    width, height = page.rect.width * zoom, page.rect.height * zoom
    area = width * height
    if area > MAX_RENDER_PIXELS:
        zoom *= (MAX_RENDER_PIXELS / area) ** 0.5
    return fitz.Matrix(zoom, zoom)


def _page_to_bgr(page: "fitz.Page"):
    """Rasterize a page straight into a BGR ndarray.

    Bypasses the PNG encode -> cv2.imdecode round trip, which costs hundreds
    of ms and two extra full-frame copies per multi-megapixel page.
    """
    import numpy as np
    import cv2

    pix = page.get_pixmap(matrix=_render_matrix(page))  # RGB, no alpha
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def extract(pdf_bytes: bytes, deadline_seconds: Optional[float] = None) -> List[Tuple[int, str]]:
    """Return a list of (page_number, text) tuples, 1-indexed.

    Raises DeadlineExceeded when processing time exceeds deadline_seconds
    (default DEADLINE_SECONDS)."""
    if deadline_seconds is None:
        deadline_seconds = DEADLINE_SECONDS
    start = time.monotonic()

    pages: List[Tuple[int, str]] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if doc.page_count > MAX_PAGES:
            raise ValueError(f"PDF has {doc.page_count} pages, exceeds limit of {MAX_PAGES}")
        for i, page in enumerate(doc, start=1):
            if time.monotonic() - start > deadline_seconds:
                raise DeadlineExceeded(
                    f"Processing exceeded {deadline_seconds:.0f}s budget at page {i}/{doc.page_count}"
                )
            text = page.get_text().strip()
            if len(text) >= MIN_TEXT_LEN_TO_SKIP_OCR:
                pages.append((i, text))
                continue

            ocr_text = engine.ocr_array(_page_to_bgr(page))
            pages.append((i, ocr_text))
    finally:
        doc.close()
    return pages
