# app/pipeline_embed.py
from typing import Dict, List
import numpy as np
from io import BytesIO
from PIL import Image, ImageOps

from .llm_client import normalize_to_3_sentences  # 품종/색/특징 → 문장 3개
from .yolo_crop import yolo_crop                  # YOLO 크롭 (마진 포함, DOG_CLASS_ID=16)
from .clipper import CLIPper                      # CLIP 임베딩 (이미지/텍스트 + 정규화)
from . import config

# 프로세스당 1회만 CLIP 로드
_CLIP = None
def _clip() -> CLIPper:
    global _CLIP
    if _CLIP is None:
        # ✅ CLIPper는 인자 없는 생성자! (config는 내부에서 사용)
        _CLIP = CLIPper()
    return _CLIP

def _bytes_to_pil(image_bytes: bytes) -> Image.Image:
    pil = Image.open(BytesIO(image_bytes)).convert("RGB")
    try:
        pil = ImageOps.exif_transpose(pil)
    except Exception:
        pass
    return pil

def build_embeddings(image_bytes: bytes, breed: str, colors: str, features: str) -> Dict[str, object]:
    """
    [파이프라인 #1] 이미지/텍스트 → 임베딩
    1) YOLO 크롭 → 2) LLM 정제(문장 3개) → 3) CLIP 임베딩
    반환:
      {
        "sentences": [str, str, str],
        "image_embedding": np.ndarray(512,),  # L2 정규화된 float32
        "text_embedding":  np.ndarray(512,),  # L2 정규화된 float32
      }
    """
    # ① 텍스트 정제
    sentences: List[str] = normalize_to_3_sentences(breed, colors, features)

    # ② bytes → PIL 변환 후 YOLO 크롭
    pil_raw = _bytes_to_pil(image_bytes)
    pil = yolo_crop(pil_raw)

    # ③ CLIP 임베딩 (Tensor → numpy 변환)
    clip = _clip()
    img_emb = clip.encode_image(pil).numpy()          # ✅ *_np 아님
    txt_emb = clip.encode_text_3(sentences).numpy()   # ✅ *_np 아님

    return {"sentences": sentences, "image_embedding": img_emb, "text_embedding": txt_emb}

__all__ = ["build_embeddings"]
