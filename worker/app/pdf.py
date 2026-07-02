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
# "digital" and OCR is skipped for it.
MIN_TEXT_LEN_TO_SKIP_OCR = 20


def extract(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    """Return a list of (page_number, text) tuples, 1-indexed."""
    pages: List[Tuple[int, str]] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        zoom = RENDER_DPI / 72
        matrix = fitz.Matrix(zoom, zoom)
        for i, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if len(text) >= MIN_TEXT_LEN_TO_SKIP_OCR:
                pages.append((i, text))
                continue

            pix = page.get_pixmap(matrix=matrix)
            img_bytes = pix.tobytes("png")
            ocr_text = engine.ocr_image(img_bytes)
            pages.append((i, ocr_text))
    finally:
        doc.close()
    return pages
