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
    assert pages[0].page == 1
    assert "digital text layer" in pages[0].text


def test_blank_page_goes_through_ocr(monkeypatch):
    captured = {}

    def fake_ocr(img):
        captured["shape"] = img.shape
        return "OCR RESULT"

    monkeypatch.setattr(pdf.engine, "ocr_array", fake_ocr)
    pages = pdf.extract(make_pdf([""]))
    assert pages == [pdf.PageResult(1, "OCR RESULT", "ocr")]
    h, w, c = captured["shape"]
    assert c == 3  # BGR ndarray fed directly, no PNG round trip
    assert h > 100 and w > 100


def test_mixed_pages(monkeypatch):
    monkeypatch.setattr(pdf.engine, "ocr_array", lambda img: "SCANNED")
    pages = pdf.extract(make_pdf([DIGITAL_TEXT, ""]))
    assert pages[0].text.startswith("This page")
    assert pages[1] == pdf.PageResult(2, "SCANNED", "ocr")


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


def test_mixed_pdf_review_metadata(monkeypatch):
    monkeypatch.setattr(pdf, "REVIEW_MIN_CHARS", 20)
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), DIGITAL_TEXT)
    scan = fitz.open(stream=make_pdf([DIGITAL_TEXT]), filetype="pdf")
    page = doc.new_page()
    page.insert_image(page.rect, pixmap=scan[0].get_pixmap())
    doc.new_page()
    data = doc.tobytes()
    doc.close()
    scan.close()
    results = iter(["스캔 본문입니다", ""])
    monkeypatch.setattr(pdf.engine, "ocr_array", lambda img: next(results))
    pages = pdf.extract(data)
    assert [p.method for p in pages] == ["text_layer", "ocr", "ocr"]
    assert [p.char_count for p in pages] == [len(DIGITAL_TEXT), 8, 0]
    assert [p.review_reasons for p in pages] == [[], ["short_text"], ["empty_text"]]
    assert [p.page for p in pages if p.review_reasons] == [2, 3]


@pytest.mark.parametrize("text, reasons", [
    ("", ["empty_text"]), (" \n\t", ["empty_text"]),
    ("한 글", ["short_text"]), ("가나다라", []), ("가나다라마", []),
])
def test_review_threshold(monkeypatch, text, reasons):
    monkeypatch.setattr(pdf, "REVIEW_MIN_CHARS", 4)
    page = pdf.PageResult(1, text, "ocr")
    assert page.char_count == len(text)
    assert page.review_reasons == reasons
    assert page.text == text


def test_zero_threshold_keeps_empty_review(monkeypatch):
    monkeypatch.setattr(pdf, "REVIEW_MIN_CHARS", 0)
    assert pdf.PageResult(1, "x", "ocr").review_reasons == []
    assert pdf.PageResult(2, "", "ocr").review_reasons == ["empty_text"]


def test_review_count_includes_line_separators(monkeypatch):
    monkeypatch.setattr(pdf, "REVIEW_MIN_CHARS", 20)
    text = "\n".join("abcdefghijk")
    page = pdf.PageResult(1, text, "ocr")
    assert page.char_count == 21
    assert page.review_reasons == []
    assert page.text == text
