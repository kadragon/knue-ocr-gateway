# TLS (Caddy)

`caddy` service terminates TLS for `ocr-gateway.knue.ac.kr` using KNUE's GlobalSign wildcard cert (`*.knue.ac.kr`, expires 2026-09-28) and reverse-proxies to `gateway` over the internal Docker network. `gateway` is no longer published to the host — only reachable through `caddy`.

## Cert files (gitignored, not in repo)

`caddy/certs/`:
- `fullchain.pem` — leaf cert + intermediate, in that order. No root cert needed (clients already trust it).
- `privkey.pem` — unencrypted PEM key.

## Renewing / rebuilding fullchain.pem

KNUE's cert bundle zip ships each PEM file without a trailing newline. Concatenating with plain `cat leaf.crt chain.crt > fullchain.pem` glues `-----END CERTIFICATE----------BEGIN CERTIFICATE-----` onto one line and breaks parsing. Insert a newline between files:

```bash
{ cat leaf.crt; printf '\n'; cat chain.crt; } > fullchain.pem
openssl verify -no_check_time -CAfile CA_GLOBALSIGN.crt -untrusted chain.crt fullchain.pem  # expect OK
```

Then swap in the new `privkey.pem` and `docker compose restart caddy`.

## Testing before DNS is registered

`ocr-gateway.knue.ac.kr` needs an internal DNS A record (or client-side `/etc/hosts` entry) pointing at the docker host. To test without either:

```bash
curl --resolve ocr-gateway.knue.ac.kr:443:127.0.0.1 --cacert caddy/certs/fullchain.pem https://ocr-gateway.knue.ac.kr/health
```

## API_KEY

`docker-compose.yml` requires `API_KEY` via `.env` (compose fails to start without it — see `${API_KEY:?...}` in the `gateway` service). Generate with `openssl rand -hex 24`.
