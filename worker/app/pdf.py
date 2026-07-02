"""PDF text extraction.

Digital PDFs already carry a text layer — PyMuPDF extracts it directly and
skips OCR entirely (fast, exact). Scanned PDFs have no usable text layer, so
each page is rasterized and sent through the OCR engine instead.
"""

from typing import List, Tuple

import fitz  # PyMuPDF

from . import engine

RENDER_DPI = 300
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


def _render_matrix(page: "fitz.Page") -> "fitz.Matrix":
    zoom = RENDER_DPI / 72
    width, height = page.rect.width * zoom, page.rect.height * zoom
    area = width * height
    if area > MAX_RENDER_PIXELS:
        zoom *= (MAX_RENDER_PIXELS / area) ** 0.5
    return fitz.Matrix(zoom, zoom)


def extract(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    """Return a list of (page_number, text) tuples, 1-indexed."""
    pages: List[Tuple[int, str]] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if doc.page_count > MAX_PAGES:
            raise ValueError(f"PDF has {doc.page_count} pages, exceeds limit of {MAX_PAGES}")
        for i, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if len(text) >= MIN_TEXT_LEN_TO_SKIP_OCR:
                pages.append((i, text))
                continue

            pix = page.get_pixmap(matrix=_render_matrix(page))
            img_bytes = pix.tobytes("png")
            ocr_text = engine.ocr_image(img_bytes)
            pages.append((i, ocr_text))
    finally:
        doc.close()
    return pages
