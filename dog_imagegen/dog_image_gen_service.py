"""
dog_image_gen_service.py

- 입력: 한국어 (breed, colors, others)
- 내부: 프롬프트에서 '영문으로 간결 번역 + 시각 정보만 유지'
- 프레이밍: 전신(머리~발끝) + 여백, '멀리서 보이는' 와이드 샷
- 비율: 가로형 4:3 (landscape)
- 후처리: 패딩 없이 'crop' 으로 4:3 강제 (중앙 크롭)
- 좌/우 위치(LEFT/RIGHT) 준수
- 출력: PNG bytes

필요:
    pip install google-genai pillow
환경:
    export GEMINI_API_KEY="YOUR_KEY"
"""

from __future__ import annotations
from typing import Optional
import os, time
from io import BytesIO
from PIL import Image
from google import genai


# ===============================
# 고정 파라미터 (여기만 바꾸면 됨)
# ===============================
ASPECT_RATIO        = "4:3"              # 가로형(4:3)
MODEL_NAME          = "gemini-2.5-flash-image"
OUT_WIDTH_PX        = 1600               # 최종 가로 해상도
SUBJECT_SCALE_PCT   = 35                 # 개가 프레임 높이에서 차지하는 비율(%), 멀리 보이게 30~40 권장
RETRIES             = 2
RETRY_BACKOFF       = 1.4                # 지수 백오프 계수


# -------------------------------
# 프롬프트 빌더
# -------------------------------
NEGATIVE = (
    "Avoid: close-up or tight framing, cropped head or missing paws, "
    "subject touching frame edges, distorted anatomy, extra limbs, "
    "mirrored/swapped body parts, text overlays, logos, artifacts, humans."
)

def build_prompt_self_filter_ko(breed: str, colors: str, others: str) -> str:
    raw = f"품종:{breed}\n색상:{colors}\n기타:{others}"
    return (
        "You are generating a single photorealistic dog image.\n"
        "The user's input is in Korean. First, translate it into concise English and keep ONLY visual attributes "
        "(breed or best guess, coat colors/patterns, size, ear/tail shape, accessories, "
        "distinctive markings, fur length/texture, eye color, and any specified LEFT/RIGHT placement). "
        "IGNORE non-visual context such as location, time, owner/people, emotions, addresses, and phone numbers.\n\n"
        f"TEXT (Korean):\n{raw}\n\n"
        "Then render exactly one dog that matches those visual attributes.\n"
        # ---- 프레이밍 강화: 멀리 + 여백 + 전신 ----
        f"Framing: full-body wide shot (head to paws) with generous top/bottom/side margins; NOT a close-up. "
        f"Subject scale: the dog should occupy about {SUBJECT_SCALE_PCT}% of the frame height (±8%). "
        "Keep a clearly visible ground plane and a subtle floor shadow; all paws fully inside the frame. "
        "Leave breathing room around ears and tail so nothing touches the frame edges.\n"
        # ---- 좌/우 위치 강제 ----
        "If LEFT or RIGHT is specified for any marking (e.g., left ear spot), strictly place it on that side "
        "and keep the opposite side clean unless stated otherwise.\n"
        "Style: photorealistic, natural proportions, clean details.\n"
        f"Canvas/Aspect: strictly use 4:3 landscape (width:height = 4:3). Keep safe margins so nothing is cropped.\n"
        "Background: neutral seamless backdrop.\n"
        f"{NEGATIVE}"
    )


# -------------------------------
# 4:3 강제 '크롭' 후처리 (중앙)
# -------------------------------
def force_aspect_ratio_crop(png_bytes: bytes, target_ratio: float = 4/3, out_width: int = OUT_WIDTH_PX) -> bytes:
    img = Image.open(BytesIO(png_bytes)).convert("RGB")
    w, h = img.size
    cur_ratio = w / h

    out_h = int(round(out_width / target_ratio))
    if abs(cur_ratio - target_ratio) < 1e-3:
        out = img.resize((out_width, out_h), Image.LANCZOS)
    elif cur_ratio > target_ratio:
        # 너무 가로로 넓음 → 좌우 중앙 크롭
        new_w = int(target_ratio * h)
        left = max(0, (w - new_w) // 2)
        box = (left, 0, left + new_w, h)
        out = img.crop(box).resize((out_width, out_h), Image.LANCZOS)
    else:
        # 너무 세로로 김 → 상하 중앙 크롭
        new_h = int(w / target_ratio)
        top = max(0, (h - new_h) // 2)
        box = (0, top, w, top + new_h)
        out = img.crop(box).resize((out_width, out_h), Image.LANCZOS)

    buf = BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


# -------------------------------
# 생성기 클래스 (간단 인터페이스)
# -------------------------------
class DogImageGenerator:
    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY 환경변수가 설정되어 있지 않습니다.")
        self.client = genai.Client(api_key=key)

    def _call_gemini(self, prompt: str) -> bytes:
        last_err = None
        for attempt in range(RETRIES + 1):
            try:
                resp = self.client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt
                )
                parts = [
                    p.inline_data.data
                    for p in resp.candidates[0].content.parts
                    if getattr(p, "inline_data", None)
                ]
                if not parts:
                    raise RuntimeError("이미지 응답이 없습니다 (inline_data 없음).")
                return parts[0]
            except Exception as e:
                last_err = e
                if attempt < RETRIES:
                    time.sleep(RETRY_BACKOFF ** attempt)
                else:
                    msg = str(last_err)
                    if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                        raise RuntimeError(
                            "Gemini 이미지 생성 쿼터가 부족합니다. 결제 연결(유료 키) 후 재시도하세요."
                        )
                    raise last_err

    def generate_from_raw_ko(self, breed: str, colors: str, others: str) -> bytes:
        """
        한국어 원문 → (프롬프트 내 번역/정제) → 생성 → 4:3 중앙 크롭 → PNG bytes
        """
        prompt = build_prompt_self_filter_ko(breed=breed, colors=colors, others=others)
        png = self._call_gemini(prompt)
        png = force_aspect_ratio_crop(png_bytes=png, target_ratio=4/3, out_width=OUT_WIDTH_PX)
        return png