# AGENTS.md

Index of `docs/`. Read the linked file before touching its area.

| file | summary |
|---|---|
| [docs/tls-caddy.md](docs/tls-caddy.md) | Caddy TLS setup, cert renewal gotcha (PEM concat needs newline separator), API_KEY requirement |
| [docs/worker-tuning.md](docs/worker-tuning.md) | Serialized PaddleOCR inference (not thread-safe — keep the lock), layered concurrency/timeout limits, DPI/ndarray pipeline rationale, det knobs + eval harness (`worker/eval/`; unclip 2.0 tuned, det_limit raise rejected — OOM), uv deps (`engine` extra = paddle; setuptools pin is load-bearing — paddle imports it undeclared; tests skip the extra), verifying dep bumps needs `--platform linux/amd64` (arm64 lacks paddle oneDNN; native arm64 OK with mkldnn off) |
| [docs/gateway-lint.md](docs/gateway-lint.md) | golangci-lint (v2) config for the Go gateway; enabled linters + why revive unused-parameter is off + strict errcheck convention |
