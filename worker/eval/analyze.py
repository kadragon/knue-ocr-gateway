"""Merge chunked sweep results and rank configs (runs on the host).

Reads every results-*.json produced by run_eval.py --config (chunked or
whole-set), merges chunks of the same config, and prints a ranking table
with three metrics:

- cer:         plain CER (whitespace-collapsed)
- nospace:     CER ignoring all whitespace (glyph + punctuation accuracy)
- hangul_only: CER on Hangul/alphanumeric only — strips punctuation from
  both sides, isolating "did we read the actual characters" from the
  recognizer's known punctuation dropping.

Usage: .venv/bin/python -m eval.analyze
"""

import glob
import json
import os
import re

from .run_eval import _label, _levenshtein, cer, cer_nospace

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")

_KEEP = re.compile(r"[^0-9A-Za-z가-힣]+")


def cer_hangul_only(pred: str, truth: str) -> float:
    pred, truth = _KEEP.sub("", pred), _KEEP.sub("", truth)
    return _levenshtein(pred, truth) / max(len(truth), 1)


def main() -> None:
    truths = {}
    for txt in glob.glob(os.path.join(SAMPLES_DIR, "*.txt")):
        with open(txt, encoding="utf-8") as f:
            truths[os.path.splitext(os.path.basename(txt))[0]] = f.read()

    merged = {}  # label -> {sample: entry, "_latencies": [...]}
    for path in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "results-*.json"))):
        with open(path) as f:
            r = json.load(f)
        label = _label(r["config"])
        bucket = merged.setdefault(label, {"samples": {}, "latencies": []})
        bucket["samples"].update(r["per_sample"])
        bucket["latencies"].append(r["mean_latency_s"])

    rows = []
    for label, bucket in merged.items():
        entries = bucket["samples"]
        missing = set(truths) - set(entries)
        if missing:
            print(f"WARN {label}: missing samples {sorted(missing)}")
        stale = set(entries) - set(truths)
        if stale:
            print(f"WARN {label}: skipping stale samples no longer in {SAMPLES_DIR}: {sorted(stale)}")
        cers, nospaces, hanguls = [], [], []
        for name, entry in entries.items():
            if name in stale:
                continue
            pred, truth = entry["pred"], truths[name]
            cers.append(cer(pred, truth))
            nospaces.append(cer_nospace(pred, truth))
            hanguls.append(cer_hangul_only(pred, truth))
        n = len(cers)
        if n == 0:
            continue
        rows.append({
            "label": label,
            "n": n,
            "cer": sum(cers) / n,
            "nospace": sum(nospaces) / n,
            "hangul": sum(hanguls) / n,
            "worst_hangul": max(hanguls),
            "latency": sum(bucket["latencies"]) / len(bucket["latencies"]),
        })

    rows.sort(key=lambda r: (r["hangul"], r["latency"]))
    print("\n| config | n | CER | nospace | hangul-only | worst hangul | s/page |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['label']} | {r['n']} | {r['cer']:.4f} | {r['nospace']:.4f} "
            f"| {r['hangul']:.4f} | {r['worst_hangul']:.4f} | {r['latency']:.1f} |"
        )


if __name__ == "__main__":
    main()
