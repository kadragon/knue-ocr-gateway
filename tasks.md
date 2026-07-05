# Tasks

## Deferred from PR #11 review (2026-07-05)

Out of scope for this PR — low severity, `worker/eval/` is dev-only tooling (not shipped/production code).

- [ ] `worker/eval/analyze.py`: latency merge averages per-chunk `mean_latency_s` unweighted; biased "s/page" column when chunk sizes differ (e.g. `--filter dense9pt` vs a full layout). Fix: weight by per-chunk sample count.
- [ ] `worker/eval/run_eval.py:_load_samples`: `cv2.imread` return value not checked for `None`; would silently produce garbage `img=None` on unreadable/unicode-path files. Edge case doesn't apply to prod (eval runs in linux containers with ASCII sample names) — lower priority.
- [ ] `worker/eval/generate_samples.py`: `EVAL_FONT_PATH` default hardcodes macOS AppleGothic path. Already overridable via env var and documented in the module docstring; only relevant if eval harness gets run outside macOS.
