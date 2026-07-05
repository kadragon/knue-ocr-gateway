"""Detection-knob sweep for Korean OCR accuracy (runs inside the worker image).

Staged sweep over PaddleOCR detection parameters against the sample pairs in
samples/ (<name>.png + <name>.txt ground truth): first det_limit_side_len,
then det_db_unclip_ratio and det_db_box_thresh at the stage-1 winner. Each
config gets a fresh PaddleOCR instance (params are init-time only).

Metric is CER = levenshtein(pred, truth) / len(truth) on whitespace-collapsed
text, so line-ordering differences don't count as errors. Latency is wall
time per page — meaningful relatively; absolute numbers are inflated under
amd64 emulation on Apple silicon.

Usage (repo root):
    docker build --platform linux/amd64 -t knue-ocr-eval ./worker
    docker run --rm --platform linux/amd64 \
      -v ./worker/eval:/app/eval \
      -v knue-ocr-gateway_paddleocr-models:/root/.paddleocr \
      knue-ocr-eval python -m eval.run_eval
"""

import argparse
import glob
import json
import logging
import os
import re
import time
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("ocr-eval")

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")

BASELINE = {"det_limit_side_len": 960, "det_db_unclip_ratio": 1.5, "det_db_box_thresh": 0.6}
LIMIT_CANDIDATES = [960, 1280, 1600, 1920]
UNCLIP_CANDIDATES = [1.8, 2.0]
BOX_THRESH_CANDIDATES = [0.5]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def cer(pred: str, truth: str) -> float:
    pred, truth = _normalize(pred), _normalize(truth)
    return _levenshtein(pred, truth) / max(len(truth), 1)


def cer_nospace(pred: str, truth: str) -> float:
    """CER ignoring all whitespace. The Korean recognizer drops/moves spaces
    a lot (~20% of ground-truth chars are spaces), which drowns out detection
    improvements in plain CER — this isolates glyph-level accuracy, so rank
    detection knobs by this metric."""
    pred, truth = re.sub(r"\s+", "", pred), re.sub(r"\s+", "", truth)
    return _levenshtein(pred, truth) / max(len(truth), 1)


def _load_samples() -> List[Tuple[str, "object", str]]:
    import cv2

    samples = []
    for png in sorted(glob.glob(os.path.join(SAMPLES_DIR, "*.png"))):
        name = os.path.splitext(os.path.basename(png))[0]
        txt = os.path.join(SAMPLES_DIR, f"{name}.txt")
        if not os.path.isfile(txt):
            logger.warning("Skipping %s: no ground-truth .txt", name)
            continue
        with open(txt, encoding="utf-8") as f:
            samples.append((name, cv2.imread(png), f.read()))
    if not samples:
        raise SystemExit(f"No samples in {SAMPLES_DIR}; run generate_samples.py first")
    return samples


def _make_engine(config: Dict):
    import platform

    from paddleocr import PaddleOCR

    from app.engine import _CH_DET_MODEL_DIR, _CPU_THREADS, _ensure_ch_detector_downloaded

    _ensure_ch_detector_downloaded()
    # arm64 paddle builds lack oneDNN (set_mkldnn_cache_capacity missing) and
    # crash with enable_mkldnn=True. mkldnn only affects speed, not CER
    # ranking, so native-arm64 eval runs disable it instead of emulating amd64
    # (QEMU is ~10x slower and OOM-prone). Prod stays amd64 + mkldnn.
    mkldnn = platform.machine() not in ("arm64", "aarch64")
    return PaddleOCR(
        lang="korean",
        det_model_dir=_CH_DET_MODEL_DIR,
        use_gpu=False,
        use_angle_cls=False,
        enable_mkldnn=mkldnn,
        cpu_threads=_CPU_THREADS,
        show_log=False,
        **config,
    )


def _run_config(config: Dict, samples) -> Dict:
    engine = _make_engine(config)
    per_sample = {}
    latencies = []
    for name, img, truth in samples:
        start = time.monotonic()
        result = engine.ocr(img, cls=False)
        latencies.append(time.monotonic() - start)
        pred = "" if not result or result[0] is None else "\n".join(
            line[1][0] for line in result[0]
        )
        per_sample[name] = {
            "cer": cer(pred, truth),
            "cer_nospace": cer_nospace(pred, truth),
            "pred": pred,
        }
        # print, not logging: importing paddleocr reconfigures logging and
        # silences this module's logger.
        print(
            f"  {name}: CER {per_sample[name]['cer']:.4f} "
            f"nospace {per_sample[name]['cer_nospace']:.4f} ({latencies[-1]:.1f}s)",
            flush=True,
        )
    cers = [s["cer"] for s in per_sample.values()]
    nospace = [s["cer_nospace"] for s in per_sample.values()]
    return {
        "config": config,
        "mean_cer": sum(cers) / len(cers),
        "mean_cer_nospace": sum(nospace) / len(nospace),
        "worst_cer_nospace": max(nospace),
        "mean_latency_s": sum(latencies) / len(latencies),
        "per_sample": per_sample,
    }


def _label(config: Dict) -> str:
    return (
        f"limit={config['det_limit_side_len']} "
        f"unclip={config['det_db_unclip_ratio']} "
        f"box={config['det_db_box_thresh']}"
    )


def main() -> None:
    # --config '{"det_limit_side_len":960,...}' runs one config and writes
    # results-<slug>.json — lets the host fan configs out across parallel
    # containers instead of a 1h+ serial staged sweep under amd64 emulation.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="single-config JSON; omit for the staged sweep")
    # paddle leaks memory per inference (worse at larger det inputs); a
    # 12-sample run at det_limit>=1280 gets OOM-killed. --filter shards a
    # config across short-lived processes; merge the chunk JSONs offline.
    parser.add_argument("--filter", default="", help="only samples whose name contains this")
    args = parser.parse_args()

    samples = _load_samples()
    if args.filter:
        samples = [s for s in samples if args.filter in s[0]]
        if not samples:
            raise SystemExit(f"--filter {args.filter!r} matched no samples in {SAMPLES_DIR}")
    logger.info("Loaded %d samples", len(samples))

    if args.config:
        config = json.loads(args.config)
        r = _run_config(config, samples)
        slug = _label(config).replace(" ", "_").replace("=", "")
        if args.filter:
            slug += f"_{args.filter}"
        out = os.path.join(os.path.dirname(__file__), f"results-{slug}.json")
        with open(out, "w") as f:
            json.dump(r, f, indent=2, ensure_ascii=False)
        print(
            f"{_label(config)}: CER {r['mean_cer']:.4f}  "
            f"nospace {r['mean_cer_nospace']:.4f}  "
            f"{r['mean_latency_s']:.1f}s/page -> {out}",
            flush=True,
        )
        return

    # The staged sweep runs every det_limit_side_len candidate in this one
    # process; paddle's per-inference memory leak reliably OOM-kills it once
    # det_limit_side_len >= 1280 is reached (see docs/worker-tuning.md). Fail
    # fast with guidance instead of a bare OOM kill mid-sweep.
    if max(LIMIT_CANDIDATES) >= 1280:
        raise SystemExit(
            "Staged sweep (no --config) OOMs at det_limit_side_len >= 1280. "
            "Run each config in its own container instead: "
            "--config '{\"det_limit_side_len\":...}' --filter <chunk>"
        )

    results = []
    # Rank by glyph accuracy (space-insensitive), tie-break by latency.
    rank = lambda r: (r["mean_cer_nospace"], r["mean_latency_s"])  # noqa: E731

    def run(config: Dict) -> Dict:
        print(f"Running {_label(config)} ...", flush=True)
        r = _run_config(config, samples)
        print(
            f"  mean CER {r['mean_cer']:.4f}  nospace {r['mean_cer_nospace']:.4f}  "
            f"{r['mean_latency_s']:.1f}s/page",
            flush=True,
        )
        results.append(r)
        return r

    # Stage 1: detector input size (biggest lever).
    stage1 = [run({**BASELINE, "det_limit_side_len": v}) for v in LIMIT_CANDIDATES]
    best = min(stage1, key=rank)

    # Stage 2: box unclip ratio at the stage-1 winner.
    stage2 = [
        run({**best["config"], "det_db_unclip_ratio": v}) for v in UNCLIP_CANDIDATES
    ]
    best = min(stage2 + [best], key=rank)

    # Stage 3: box threshold at the stage-2 winner.
    stage3 = [
        run({**best["config"], "det_db_box_thresh": v}) for v in BOX_THRESH_CANDIDATES
    ]
    best = min(stage3 + [best], key=rank)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n| config | mean CER | nospace CER | worst nospace | s/page |", flush=True)
    print("|---|---|---|---|---|", flush=True)
    for r in sorted(results, key=rank):
        marker = " **best**" if r is best else ""
        print(
            f"| {_label(r['config'])}{marker} | {r['mean_cer']:.4f} "
            f"| {r['mean_cer_nospace']:.4f} | {r['worst_cer_nospace']:.4f} "
            f"| {r['mean_latency_s']:.1f} |",
            flush=True,
        )
    print(f"\nBest: {_label(best['config'])} (results saved to {RESULTS_PATH})", flush=True)


if __name__ == "__main__":
    main()
