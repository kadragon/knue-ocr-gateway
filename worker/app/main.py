import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from . import engine, pdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ocr-worker")

IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "tiff", "tif", "bmp", "webp"}

# Last line of defense: the gateway's semaphore releases when its client
# times out, but the worker thread it spawned keeps running (threads are not
# cancellable), so under sustained timeouts the gateway cap alone does not
# bound work queued here. Slightly above the gateway's MAX_CONCURRENCY so
# normal operation never trips it.
MAX_CONCURRENCY = int(os.environ.get("WORKER_MAX_CONCURRENCY", "8"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.warmup()
    yield


app = FastAPI(title="knue-ocr-worker", lifespan=lifespan)

# Single event loop and no await between check and increment -> no race.
_active = 0


@app.get("/health")
def health():
    return {"status": "ok"}


def _extension(filename: Optional[str]) -> str:
    if not filename or "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    global _active
    if _active >= MAX_CONCURRENCY:
        raise HTTPException(status_code=503, detail="Worker at capacity, retry later")
    _active += 1
    try:
        return await _process(file)
    finally:
        _active -= 1


async def _process(file: UploadFile):
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
    except pdf.DeadlineExceeded as e:
        logger.warning("OCR deadline exceeded for %s: %s", file.filename, e)
        raise HTTPException(status_code=504, detail="Processing deadline exceeded") from None
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
