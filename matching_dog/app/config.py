# config.py
import os

# ===== LLM 정제화 =====
LLM_BASE = os.getenv("LLM_BASE", "http://34.171.44.43:8000")
LLM_URL  = os.getenv("LLM_URL", f"{LLM_BASE}/api/v1/embed/normalize")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "20"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

# ===== YOLO 크롭 =====
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolov8s.pt")
YOLO_CONF    = float(os.getenv("YOLO_CONF", "0.30"))
YOLO_MARGIN  = float(os.getenv("YOLO_MARGIN", "0.18"))
YOLO_IMGSZ   = int(os.getenv("YOLO_IMGSZ", "512"))
DOG_CLASS_ID = int(os.getenv("DOG_CLASS_ID", "16"))  # COCO dog

# ===== CLIP (base) =====
CLIP_MODEL      = os.getenv("CLIP_MODEL", "ViT-L-14")
CLIP_PRETRAINED = os.getenv("CLIP_PRETRAINED", "datacomp_xl_s13b_b90k")

# ===== TEXT POOLING (추론용) =====
TEXT_POOLING    = os.getenv("TEXT_POOLING", "avg")  # avg|max|top1

# ===== Head-only FT 가중치 (.pt) =====
FT_WEIGHTS_URL  = os.getenv(
    "FT_WEIGHTS_URL",
    "https://github.com/eonjilim/Myaongi_AI/releases/download/untagged-c51a7ed963c83380ced0/matching_best.pt"
)
FT_WEIGHTS_PATH = os.getenv("FT_WEIGHTS_PATH", "")
FT_CACHE_DIR    = os.getenv("FT_CACHE_DIR", "/tmp/clip_ft_cache")

# ===== 유사도 가중치 =====
W_II = float(os.getenv("W_II", os.getenv("VAL_W_II", "0.20")))
W_IT = float(os.getenv("W_IT", os.getenv("VAL_W_IT", "0.00")))
W_TI = float(os.getenv("W_TI", os.getenv("VAL_W_TI", "0.00")))
W_TT = float(os.getenv("W_TT", os.getenv("VAL_W_TT", "0.80")))

# ===== 매칭 기준 임계값 =====
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.35"))
