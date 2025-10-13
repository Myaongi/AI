from typing import Tuple, List
from PIL import Image
import io
from .llm_client import normalize_to_3_sentences
from .yolo_crop import yolo_crop
from .clipper import CLIPper
from .similarity import four_sims, weighted_score

_clip_singleton: CLIPper = None
def _clip() -> CLIPper:
    global _clip_singleton
    if _clip_singleton is None:
        _clip_singleton = CLIPper()
    return _clip_singleton

def build_embeddings(image_bytes: bytes, breed: str, colors: str, features: str):
    # 1) 텍스트 → LLM 정제(3문장)
    sents = normalize_to_3_sentences(breed, colors, features)
    # 2) 이미지 → YOLO 크롭(마진 포함)
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pil_c = yolo_crop(pil)
    # 3) CLIP 임베딩(이미지/텍스트)
    clip = _clip()
    emb_img = clip.encode_image(pil_c)
    emb_txt = clip.encode_text_3(sents)
    return sents, emb_img.numpy().tolist(), emb_txt.numpy().tolist()

def score_pair(emb_a_img, emb_a_txt, emb_b_img, emb_b_txt, weights=None):
    s4 = four_sims(emb_a_img, emb_a_txt, emb_b_img, emb_b_txt)
    score = weighted_score(s4, weights=weights)
    return s4, score
