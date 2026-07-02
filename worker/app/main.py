import asyncio
import logging
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from . import engine, pdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ocr-worker")

IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "tiff", "tif", "bmp", "webp"}

app = FastAPI(title="knue-ocr-worker")


@app.on_event("startup")
def _startup() -> None:
    engine.warmup()


@app.get("/health")
def health():
    return {"status": "ok"}


def _extension(filename: Optional[str]) -> str:
    if not filename or "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    ext = _extension(file.filename)

    try:
        if ext == "pdf" or file.content_type == "application/pdf":
            # pdf.extract/engine.ocr_image are synchronous and CPU-bound
            # (PyMuPDF rendering, PaddleOCR inference); running them inline
            # would block the single event loop for the whole request,
            # including /health checks from other clients.
            pages = await asyncio.to_thread(pdf.extract, data)
        elif ext in IMAGE_EXTENSIONS or (file.content_type or "").startswith("image/"):
            text = await asyncio.to_thread(engine.ocr_image, data)
            pages = [(1, text)]
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext or file.content_type}")
    except HTTPException:
        raise
    except Exception:  # decoding/parsing failures -> explicit 422, not a bare 500
        logger.exception("OCR processing failed for %s", file.filename)
        raise HTTPException(status_code=422, detail="Failed to process file") from None

    full_text = "\n\n".join(text for _, text in pages)
    return JSONResponse(
        {
            "filename": file.filename,
            "pages": [{"page": n, "text": t} for n, t in pages],
            "text": full_text,
        }
    )
