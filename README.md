# knue-ocr-gateway

PDF/이미지를 업로드하면 한국어 OCR 텍스트를 반환하는 셀프호스트 서비스.

## 구조

- **caddy**: TLS 종단(`:443`), 사내 도메인 인증서로 HTTPS 제공, gateway로 리버스 프록시.
- **gateway** (Go, 내부 전용): 업로드 검증(크기·확장자·API 키), 동시성 제한, 워커로 프록시.
- **ocr-worker** (Python/FastAPI, 내부 전용): PaddleOCR(한국어) + PyMuPDF. 디지털 PDF는 텍스트레이어를 그대로 추출하고, 스캔 PDF/이미지만 OCR을 거칩니다.

## 실행

사전 준비:

1. `.env` 생성 (`.env.example` 참고): `API_KEY=<임의의 긴 문자열>`
2. `caddy/certs/`에 인증서 배치:
   - `fullchain.pem` — 서버 인증서 + 중간(intermediate) 인증서 순서로 이어붙인 파일 (root는 불필요)
   - `privkey.pem` — 개인키 (암호화 안 된 PEM)
3. 사내 DNS 또는 클라이언트 hosts 파일에 `ocr-gateway.knue.ac.kr` → 이 서버 IP 등록 (`caddy/Caddyfile`에서 도메인 변경 가능)

```bash
docker compose up --build -d
docker compose ps        # ocr-worker/gateway 모두 healthy 표시 (caddy는 healthcheck 없음)
curl -f https://ocr-gateway.knue.ac.kr/health
```

`gateway`는 더 이상 호스트에 포트를 노출하지 않습니다(`caddy`를 통해서만 접근). `API_KEY`가 없으면 compose 기동 자체가 실패합니다.

## 사용

```bash
curl -H "X-API-Key: $API_KEY" -F file=@sample.pdf https://ocr-gateway.knue.ac.kr/ocr
curl -H "X-API-Key: $API_KEY" -F file=@sample.png https://ocr-gateway.knue.ac.kr/ocr
```

응답:

```json
{
  "filename": "sample.pdf",
  "pages": [{"page": 1, "text": "..."}],
  "text": "전체 텍스트"
}
```

## 환경 변수 (gateway)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `MAX_FILE_MB` | 20 | 업로드 최대 크기 |
| `MAX_CONCURRENCY` | CPU 코어 수 | 워커로 동시에 전달할 최대 요청 수 |
| `API_KEY` | (빈 값) | `X-API-Key` 헤더 검증에 사용. `docker-compose.yml`은 `.env`의 `API_KEY`를 필수로 요구함 |
| `ALLOW_UNAUTHENTICATED` | (없음) | `API_KEY`가 비어있을 때 무인증 기동을 명시적으로 허용. `true`가 아니면 `API_KEY` 없이는 기동 거부됨 (compose에서는 사용하지 않음) |
| `REQUEST_TIMEOUT_SECONDS` | 120 | 워커 응답 대기 시간 |

## 환경 변수 (ocr-worker)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `OCR_DEADLINE_SECONDS` | 110 | PDF 처리 시간 예산. 게이트웨이 `REQUEST_TIMEOUT_SECONDS`보다 작게 유지 (초과 시 유기 작업이 CPU만 소모) |
| `WORKER_MAX_CONCURRENCY` | 8 | 워커 자체 동시 요청 상한 (초과 시 503) |
| `OCR_CPU_THREADS` | CPU 코어 수 | 추론 1건이 사용하는 스레드 수 |
| `RENDER_DPI` | 200 | 스캔 페이지 렌더링 DPI |

추론은 단일 PaddleOCR 인스턴스에서 락으로 직렬화됩니다 (Paddle predictor는 thread-safe 아님 — 상세는 `docs/worker-tuning.md`).

첫 실행 시 PaddleOCR 모델을 다운로드합니다(`paddleocr-models` 볼륨에 캐시되어 재기동 시 재사용).

## 개발

```bash
cd gateway && go test ./...
cd worker && uv venv --python 3.11 .venv && uv pip install -p .venv/bin/python -r requirements-dev.txt && .venv/bin/python -m pytest tests/
```

테스트는 PaddleOCR 모델 없이 동작합니다(추론은 mock). CI는 push/PR마다 두 스위트를 실행합니다.
