import fitz
import pytest

from app import pdf


def make_pdf(pages):
    """Build a PDF; each entry is the text to stamp on that page ('' = blank)."""
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


DIGITAL_TEXT = "This page carries a real digital text layer, long enough to skip OCR entirely."


def test_digital_page_skips_ocr(monkeypatch):
    def boom(_img):
        raise AssertionError("OCR must not run for digital pages")

    monkeypatch.setattr(pdf.engine, "ocr_array", boom)
    pages = pdf.extract(make_pdf([DIGITAL_TEXT]))
    assert len(pages) == 1
    assert pages[0][0] == 1
    assert "digital text layer" in pages[0][1]


def test_blank_page_goes_through_ocr(monkeypatch):
    captured = {}

    def fake_ocr(img):
        captured["shape"] = img.shape
        return "OCR RESULT"

    monkeypatch.setattr(pdf.engine, "ocr_array", fake_ocr)
    pages = pdf.extract(make_pdf([""]))
    assert pages == [(1, "OCR RESULT")]
    h, w, c = captured["shape"]
    assert c == 3  # BGR ndarray fed directly, no PNG round trip
    assert h > 100 and w > 100


def test_mixed_pages(monkeypatch):
    monkeypatch.setattr(pdf.engine, "ocr_array", lambda img: "SCANNED")
    pages = pdf.extract(make_pdf([DIGITAL_TEXT, ""]))
    assert pages[0][1].startswith("This page")
    assert pages[1] == (2, "SCANNED")


def test_page_limit(monkeypatch):
    monkeypatch.setattr(pdf, "MAX_PAGES", 2)
    with pytest.raises(ValueError, match="exceeds limit"):
        pdf.extract(make_pdf(["", "", ""]))


def test_deadline_exceeded(monkeypatch):
    monkeypatch.setattr(pdf.engine, "ocr_array", lambda img: "X")
    with pytest.raises(pdf.DeadlineExceeded):
        pdf.extract(make_pdf([""]), deadline_seconds=-1)


def test_render_matrix_caps_oversized_pages():
    doc = fitz.open()
    # 200x200 inches, the PDF spec maximum
    page = doc.new_page(width=200 * 72, height=200 * 72)
    m = pdf._render_matrix(page)
    rendered = (page.rect.width * m.a) * (page.rect.height * m.d)
    assert rendered <= pdf.MAX_RENDER_PIXELS * 1.01
    doc.close()
