"""Generate synthetic Korean OCR eval samples (runs on the host, no paddle).

Renders known Korean text onto A4 pages at the production RENDER_DPI, then
applies scan-like degradations. Each sample is a PNG plus a .txt ground-truth
sidecar; run_eval.py consumes the pairs. Real scanned pages with hand-made
ground truth can be dropped into samples/ alongside the synthetic ones —
the runner treats every <name>.png/<name>.txt pair the same way.

Synthetic caveat: knob rankings transfer to real scans, absolute CER does not.

Usage (host):
    .venv/bin/python -m eval.generate_samples

Font: needs a Korean TTF. Defaults to macOS AppleGothic; override with
EVAL_FONT_PATH.
"""

import os
from typing import List, Tuple

import cv2
import fitz
import numpy as np

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")
FONT_PATH = os.environ.get(
    "EVAL_FONT_PATH", "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
)
RENDER_DPI = 200  # keep in sync with app.pdf.RENDER_DPI

# Mixed formal/administrative Korean with numbers, dates, and English —
# the register of the documents this gateway actually processes.
PARAGRAPHS = [
    "한국교원대학교는 1984년에 설립된 국립 교원 양성 대학으로, 충청북도 청주시 흥덕구에 위치하고 있다. "
    "유아교육과 초등교육, 중등교육 전 과정의 교원을 양성하는 종합교원양성대학이며, "
    "대학원 과정을 포함하여 약 5,000명의 학생이 재학 중이다.",
    "제12조(수업일수) ① 수업일수는 매 학년도 30주 이상으로 하되, 학기당 15주 이상을 원칙으로 한다. "
    "② 천재지변 또는 그 밖의 교육과정 운영상 부득이한 사유로 수업일수를 충족할 수 없는 경우에는 "
    "학칙이 정하는 바에 따라 2주 이내에서 감축할 수 있다.",
    "2026학년도 1학기 수강신청은 2026년 2월 9일(월) 10:00부터 2월 13일(금) 17:00까지 "
    "학사정보시스템(https://portal.knue.ac.kr)에서 진행한다. 재학생은 학년별 지정 시간에, "
    "복학생 및 편입생은 2월 12일(목)부터 신청할 수 있으며, 정정 기간은 개강 후 1주간이다.",
    "본 연구는 초등학교 3학년 학생 127명을 대상으로 12주간 진행되었다. 실험집단(n=64)과 "
    "통제집단(n=63)의 사전-사후 검사 결과를 비교한 결과, 실험집단의 평균 점수가 "
    "통계적으로 유의하게 향상되었다(t=3.42, p<.01). 효과크기는 Cohen's d=0.61로 중간 수준이었다.",
    "붙임 1. 2026년도 교육실습 운영 계획(안) 1부. 2. 협력학교 명단 및 배정 인원 1부. "
    "3. 실습생 준수사항 안내문 1부. 끝. 문의: 교육실습지원센터 (043-230-3114, "
    "practicum@knue.ac.kr) 담당자 김철수 주무관.",
]

# (name, font_size_pt) — 9pt approximates dense footnote/table text, the
# hardest case for the detector's downscale.
LAYOUTS: List[Tuple[str, float]] = [
    ("body12pt", 12.0),
    ("body10pt", 10.5),
    ("dense9pt", 9.0),
]

A4_W_PT, A4_H_PT = 595, 842
MARGIN_PT = 50
LINE_SPACING = 1.5


def _render_page(font_size: float) -> Tuple[np.ndarray, str]:
    """Render all paragraphs onto one A4 page; return (BGR image, ground truth)."""
    doc = fitz.open()
    try:
        page = doc.new_page(width=A4_W_PT, height=A4_H_PT)
        rect = fitz.Rect(MARGIN_PT, MARGIN_PT, A4_W_PT - MARGIN_PT, A4_H_PT - MARGIN_PT)
        text = "\n\n".join(PARAGRAPHS)
        overflow = page.insert_textbox(
            rect,
            text,
            fontsize=font_size,
            fontfile=FONT_PATH,
            fontname="krfont",
            lineheight=LINE_SPACING,
        )
        if overflow < 0:
            raise ValueError(
                f"Text overflows page at {font_size}pt; shorten PARAGRAPHS or shrink font"
            )
        zoom = RENDER_DPI / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = np.frombuffer(pix.samples_mv, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR), text
    finally:
        doc.close()


def _degrade_blur(img: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(img, (3, 3), 0.8)


def _degrade_noisy_lowcontrast(img: np.ndarray) -> np.ndarray:
    """Faded photocopy: compressed dynamic range + gaussian noise."""
    rng = np.random.default_rng(42)
    faded = (img.astype(np.float32) * 0.55 + 100.0)
    noise = rng.normal(0, 8, img.shape).astype(np.float32)
    return np.clip(faded + noise, 0, 255).astype(np.uint8)


def _degrade_rescale(img: np.ndarray) -> np.ndarray:
    """Fax/low-res scan: downscale to ~120 DPI equivalent and back up."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w * 0.6), int(h * 0.6)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


DEGRADATIONS = [
    ("clean", lambda img: img),
    ("blur", _degrade_blur),
    ("faded", _degrade_noisy_lowcontrast),
    ("lowres", _degrade_rescale),
]


def main() -> None:
    if not os.path.isfile(FONT_PATH):
        raise SystemExit(f"Korean font not found: {FONT_PATH} (set EVAL_FONT_PATH)")
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    count = 0
    for layout_name, font_size in LAYOUTS:
        img, truth = _render_page(font_size)
        for deg_name, degrade in DEGRADATIONS:
            name = f"{layout_name}_{deg_name}"
            cv2.imwrite(os.path.join(SAMPLES_DIR, f"{name}.png"), degrade(img))
            with open(os.path.join(SAMPLES_DIR, f"{name}.txt"), "w") as f:
                f.write(truth)
            count += 1
    print(f"Wrote {count} sample pairs to {SAMPLES_DIR}")


if __name__ == "__main__":
    main()
