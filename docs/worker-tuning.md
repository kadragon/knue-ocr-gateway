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

## Detection knobs & the eval harness

`engine.py` exposes three env-tunable detection knobs:
`OCR_DET_LIMIT_SIDE_LEN` (960), `OCR_DET_UNCLIP_RATIO` (2.0),
`OCR_DET_BOX_THRESH` (0.6). Defaults come from a sweep with the eval harness
in `worker/eval/` (synthetic Korean pages, 3 font sizes x 4 scan-like
degradations, 200 DPI):

- **unclip 2.0 beats the PaddleOCR default 1.5 on every metric**: hangul-only
  CER 0.0141 -> 0.0092, worst-case 0.0317 -> 0.0168, ~18% faster (looser
  boxes stop jongseong clipping and merge fragments into fewer rec crops).
  2.2 measured the same (plateau); 2.0 kept for line-merge safety margin on
  real layouts the synthetic set can't represent.
- **Raising det_limit_side_len (1280/1600/1920) was rejected**: zero accuracy
  gain (recognizer crops from the original-resolution image, so det input
  size doesn't bound rec quality), 15-40% slower, and dense pages OOM — which
  would also blow the 4g prod container cap.
- **Error decomposition** (baseline, whitespace-collapsed CER 0.25): ~15pp is
  dropped spaces, ~9pp dropped punctuation (`(`, `,`, `.`, `①`->`0`), and
  only ~1.4% actual character errors. Both are korean-recognizer behaviors
  detection knobs cannot fix; the real lever there is a PaddleOCR 3.x /
  PP-OCRv5 migration (`korean_PP-OCRv5_mobile_rec`).

Harness usage (host generates samples; sweep runs per-config in containers):

```bash
(cd worker && .venv/bin/python -m eval.generate_samples)   # needs a Korean TTF
docker build -t knue-ocr-eval ./worker
docker run --rm -v $PWD/worker/eval:/app/eval \
  -v knue-ocr-gateway_paddleocr-models:/root/.paddleocr \
  knue-ocr-eval python -m eval.run_eval --config '{"det_limit_side_len":960,...}' --filter body10
(cd worker && .venv/bin/python -m eval.analyze)            # merge + rank
```

Real scanned pages with hand-made ground truth can be dropped into
`eval/samples/` as `<name>.png` + `<name>.txt` pairs.

Eval-run gotchas (all learned the hard way):
- **Native arm64 works if `enable_mkldnn=False`** (eval auto-detects); the
  amd64-emulation route is ~10x slower, OOM-prone, and can wedge the Docker
  daemon. Prod stays amd64 + mkldnn.
- **Importing paddleocr hijacks logging config** and silences this repo's
  loggers — eval scripts print() instead.
- **paddle leaks memory per inference** (worse at larger det inputs): shard
  long runs across processes with `--filter`, one container per chunk.

## Undeclared paddle dependency: setuptools

paddle 2.6.2 does `import setuptools` at import time but does not declare it,
and uv-managed venvs ship without setuptools — so a fresh image build dies at
worker startup with `ModuleNotFoundError: No module named 'setuptools'`.
Pinned in the `engine` extra (`pyproject.toml`). pytest cannot catch this
class of break (tests skip the `engine` extra); only a real container run
does — see the verification section below.

## Verifying dependency bumps locally (paddle/pillow/opencv)

Dependencies live in `pyproject.toml`: core deps + a `dev` group (test-only)
+ an `engine` extra (paddle). `uv sync` installs core + dev **without** the
`engine` extra, so `pytest` cannot catch a break in the OCR runtime deps. To
verify a bump (e.g. a pillow major), build the worker image (which runs
`uv sync --extra engine`) and run a real inference — but **build/run
`--platform linux/amd64`**. On Apple-silicon (arm64) Docker, paddlepaddle 2.6.2 lacks the
oneDNN backend and `_load_engine()` dies with `AnalysisConfig object has no
attribute set_mkldnn_cache_capacity` (engine sets `enable_mkldnn=True`); prod
is amd64 (CI `ubuntu-latest`) where this works. Smoke test: `engine.warmup()`
then `engine.ocr_image(cv2.imencode('.png', <text-image>))` — expect the text
back. pillow is transitive-only (paddleocr); no worker code imports PIL.
