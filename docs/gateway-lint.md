# Gateway static analysis

`gateway/.golangci.yml` configures [golangci-lint](https://golangci-lint.run)
(v2 schema). CI runs it via `golangci/golangci-lint-action`; run locally with
`golangci-lint run ./...` from `gateway/`.

Enabled: the v2 defaults (errcheck, govet, ineffassign, staticcheck, unused)
plus `bodyclose`, `misspell`, `unconvert`.

`revive`'s `unused-parameter` is intentionally **not** enabled — it flags the
unused `*http.Request` in fixed `http.HandlerFunc` signatures, which is noise.

errcheck is strict here: ignored errors must be explicit (`_ = x.Close()`,
`_, _ = w.Write(...)`) rather than bare calls.
