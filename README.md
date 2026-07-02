# knue-ocr-gateway

PDF/이미지를 업로드하면 한국어 OCR 텍스트를 반환하는 셀프호스트 서비스.

## 구조

- **gateway** (Go): 외부 진입점(`:8080`). 업로드 검증(크기·확장자·API 키), 동시성 제한, 워커로 프록시.
- **ocr-worker** (Python/FastAPI, 내부 전용): PaddleOCR(한국어) + PyMuPDF. 디지털 PDF는 텍스트레이어를 그대로 추출하고, 스캔 PDF/이미지만 OCR을 거칩니다.

## 실행

```bash
docker compose up --build -d
docker compose ps        # 둘 다 healthy 확인
curl -f localhost:8090/health
```

## 사용

```bash
curl -F file=@sample.pdf localhost:8090/ocr
curl -F file=@sample.png localhost:8090/ocr
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
| `API_KEY` | (빈 값) | 설정 시 `X-API-Key` 헤더 필요 |
| `ALLOW_UNAUTHENTICATED` | (없음) | `API_KEY`가 비어있을 때 무인증 기동을 명시적으로 허용. `true`가 아니면 `API_KEY` 없이는 기동 거부됨 |
| `REQUEST_TIMEOUT_SECONDS` | 120 | 워커 응답 대기 시간 |

첫 실행 시 PaddleOCR 모델을 다운로드합니다(`paddleocr-models` 볼륨에 캐시되어 재기동 시 재사용).
