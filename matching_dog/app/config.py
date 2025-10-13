import os

# ===== LLM 정제화 =====
LLM_BASE    = os.getenv("LLM_BASE", "http://54.180.54.51:8080")
LLM_URL     = os.getenv("LLM_URL",  f"{LLM_BASE}/api/llm-api/test1")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "20"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

# ===== YOLO 크롭 =====
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolov8s.pt")
YOLO_CONF    = float(os.getenv("YOLO_CONF", "0.30"))
YOLO_MARGIN  = float(os.getenv("YOLO_MARGIN", "0.18"))
YOLO_IMGSZ   = int(os.getenv("YOLO_IMGSZ", "512"))
DOG_CLASS_ID = int(os.getenv("DOG_CLASS_ID", "16"))  # COCO dog

# ===== CLIP =====
CLIP_MODEL      = os.getenv("CLIP_MODEL", "ViT-B-32")
CLIP_PRETRAINED = os.getenv("CLIP_PRETRAINED", "laion2b_s34b_b79k")
TEXT_POOLING    = os.getenv("TEXT_POOLING", "avg")  # avg|max|top1

# ===== 유사도 가중치 (w_ii, w_it, w_ti, w_tt) =====  * 수정 *
W_II = float(os.getenv("W_II", "0.20"))
W_IT = float(os.getenv("W_IT", "0.00"))
W_TI = float(os.getenv("W_TI", "0.00"))
W_TT = float(os.getenv("W_TT", "0.80"))

# 노출 임계값: 매칭 기준선  * 수정 *
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.35"))
