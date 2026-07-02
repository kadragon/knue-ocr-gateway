import pytest
from fastapi.testclient import TestClient

from app import engine, main, pdf


@pytest.fixture
def client(monkeypatch):
    # Warmup would load PaddleOCR models; tests never run real inference.
    monkeypatch.setattr(engine, "warmup", lambda: None)
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_extension():
    assert main._extension("a.PDF") == "pdf"
    assert main._extension("a.b.jpg") == "jpg"
    assert main._extension("noext") == ""
    assert main._extension(None) == ""


def test_empty_file_rejected(client):
    r = client.post("/ocr", files={"file": ("a.pdf", b"", "application/pdf")})
    assert r.status_code == 400


def test_unsupported_type_rejected(client):
    r = client.post("/ocr", files={"file": ("a.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 400


def test_image_path(client, monkeypatch):
    monkeypatch.setattr(engine, "ocr_image", lambda data: "HELLO")
    r = client.post("/ocr", files={"file": ("a.png", b"fakepng", "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "HELLO"
    assert body["pages"] == [{"page": 1, "text": "HELLO"}]


def test_pdf_path(client, monkeypatch):
    monkeypatch.setattr(pdf, "extract", lambda data: [(1, "p1"), (2, "p2")])
    r = client.post("/ocr", files={"file": ("a.pdf", b"%PDF", "application/pdf")})
    assert r.status_code == 200
    assert r.json()["text"] == "p1\n\np2"


def test_pdf_deadline_maps_to_504(client, monkeypatch):
    def slow(data):
        raise pdf.DeadlineExceeded("budget blown")

    monkeypatch.setattr(pdf, "extract", slow)
    r = client.post("/ocr", files={"file": ("a.pdf", b"%PDF", "application/pdf")})
    assert r.status_code == 504


def test_processing_failure_maps_to_422(client, monkeypatch):
    def boom(data):
        raise RuntimeError("corrupt")

    monkeypatch.setattr(pdf, "extract", boom)
    r = client.post("/ocr", files={"file": ("a.pdf", b"%PDF", "application/pdf")})
    assert r.status_code == 422


def test_capacity_cap_returns_503(client, monkeypatch):
    monkeypatch.setattr(main, "_active", main.MAX_CONCURRENCY)
    r = client.post("/ocr", files={"file": ("a.pdf", b"%PDF", "application/pdf")})
    assert r.status_code == 503
    # cap check must not leak the slot it never took
    assert main._active == main.MAX_CONCURRENCY


def test_capacity_released_after_request(client, monkeypatch):
    monkeypatch.setattr(pdf, "extract", lambda data: [(1, "x")])
    client.post("/ocr", files={"file": ("a.pdf", b"%PDF", "application/pdf")})
    assert main._active == 0
