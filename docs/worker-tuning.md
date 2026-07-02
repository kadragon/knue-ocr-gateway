# OCR worker concurrency & performance model

## Serialized inference (do not parallelize)

PaddlePaddle CPU predictors are **not thread-safe**: concurrent `.ocr()` calls
on one instance can segfault or corrupt results. `engine.py` serializes all
inference behind `_INFER_LOCK`, and each single inference uses the whole CPU
(`cpu_threads=OCR_CPU_THREADS`, mkldnn enabled).

Design consequence: throughput comes from *one fast inference at a time*, not
parallel inferences. Running N inferences in parallel would spawn N per-core
OMP thread pools (thrashing) and multiply peak memory. Do not "fix" the lock
by removing it; if more throughput is needed, scale to multiple worker
containers (each with its own model copy).

## Layered limits (keep the ordering)

| layer | limit | why |
|---|---|---|
| Caddy `max_size` | 21MB | gateway MAX_FILE_MB + multipart framing headroom |
| gateway `MAX_CONCURRENCY` | 4 | bounds concurrent forwards |
| worker `WORKER_MAX_CONCURRENCY` | 8 | last-resort cap: gateway slot frees on client timeout but the worker thread is not cancellable, so abandoned jobs would otherwise pile up unbounded |
| worker `OCR_DEADLINE_SECONDS` | 110 | **must stay < gateway `REQUEST_TIMEOUT_SECONDS` (120)** — checked between PDF pages so abandoned jobs stop burning CPU |
| `stop_grace_period` | 130s | > REQUEST_TIMEOUT_SECONDS so graceful shutdown can drain in-flight requests before SIGKILL |

## Rendering pipeline

Scanned pages go pixmap → numpy ndarray → OCR directly (`pdf._page_to_bgr`).
The old pixmap → PNG encode → `cv2.imdecode` path cost hundreds of ms and two
full-frame copies per page — do not reintroduce it.

`RENDER_DPI=200` (not 300): PaddleOCR's detector downscales to
`det_limit_side_len=960` anyway; 300 DPI only adds render/convert cost.

## Engine gotchas (also in engine.py docstring)

- Korean recognizer + **"ch" detector** (the default "ml" detector misses
  dense Hangul lines).
- `use_angle_cls=True` with this det/rec mix silently breaks detection (zero
  boxes) — the classifier must stay disabled.
- Tests mock inference; a real-inference smoke test requires model download
  and runs only in the container.
