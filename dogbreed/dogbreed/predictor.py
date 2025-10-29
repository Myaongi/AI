from __future__ import annotations
from typing import Any, Dict, Optional
from pathlib import Path
import io
from PIL import Image
import torch

# --- 내부 모듈 ---
from .detector import load_yolo, detect_and_crop
from .preprocessor import preprocess
from .classifier import load_classifier, predict as clf_predict, set_entropy_threshold as clf_set_Hth
from .weights import ensure_weight  # 릴리즈 다운로드 유틸

# ====== 고정 설정 ======
ASSETS_DIR       = Path(__file__).resolve().parent.parent / "assets"

# (1) YOLO 가중치
YOLO_WEIGHTS     = "yolov8s.pt"          # 또는 str(ASSETS_DIR / "yolo_best.pt")
YOLO_WEIGHTS_URL: Optional[str] = None   # 커스텀일 때만 URL 지정

# (2) 분류기 가중치 (파일명/태그를 서로 일치시킬 것!)
CLF_WEIGHTS      = str(ASSETS_DIR / "classifier_best.pt")
CLF_WEIGHTS_URL  = "https://github.com/eonjilim/Myaongi_AI/releases/download/final/classifier_best.pt"
CLF_WEIGHTS_SHA256 = None  # 있으면 "74336ea3..." 형태로 설정

# (3) 기타
KO_MAPPING_CSV   = str(ASSETS_DIR / "ko_mapping.csv")

MODEL_NAME       = "tf_efficientnetv2_m_in21ft1k"
NUM_CLASSES      = 100

INPUT_SIZE       = 480
PAD_RATIO        = 0.16
ENTROPY_TH_INIT  = 2.0   # (이전 호환용; 현재 최종 판정엔 사용하지 않음)

_STATE: Dict[str, Any] = {"ready": False}

# ------------------------------------------------------------
# 유틸
# ------------------------------------------------------------
def _to_pil(x: Any) -> Image.Image:
    if isinstance(x, Image.Image):
        img = x
    elif isinstance(x, (bytes, bytearray, memoryview)):
        img = Image.open(io.BytesIO(x))
    elif isinstance(x, (str, Path)):
        img = Image.open(x)
    else:
        from PIL import Image as _Image
        img = _Image.fromarray(x)
    return img.convert("RGB")

def _square_pad_whole(img: Image.Image, pad_ratio: float = PAD_RATIO) -> Image.Image:
    w, h = img.size
    side = int(round(max(w, h) * (1.0 + 2.0 * pad_ratio)))
    side = max(side, 1)
    left = (side - w) // 2
    top  = (side - h) // 2
    canvas = Image.new("RGB", (side, side))
    canvas.paste(img, (left, top))
    return canvas

def set_entropy_threshold(th: float) -> None:
    # (이전 호환용) 필요 시 외부에서 호출 가능
    clf_set_Hth(float(th))

# ------------------------------------------------------------
# 1회 초기화: (가중치 보장 → 모델 로드)
# ------------------------------------------------------------
def _load_once() -> None:
    if _STATE.get("ready"):
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _STATE["device"] = device

    # 0) 분류기 가중치 확보 (없으면 릴리즈에서 다운로드)
    ensure_weight(CLF_WEIGHTS, CLF_WEIGHTS_URL, sha256=CLF_WEIGHTS_SHA256)

    # 0-추가) 커스텀 YOLO 가중치 보장
    if YOLO_WEIGHTS_URL:
        w = str(YOLO_WEIGHTS)
        if ("/" in w or "\\" in w):
            p = Path(w)
            if not p.is_absolute():
                p = (Path(__file__).resolve().parent.parent / w)
            if not p.exists():
                ensure_weight(str(p), YOLO_WEIGHTS_URL)
                YOLO_WEIGHTS_RESOLVED = str(p)
            else:
                YOLO_WEIGHTS_RESOLVED = str(p)
        else:
            YOLO_WEIGHTS_RESOLVED = YOLO_WEIGHTS
    else:
        YOLO_WEIGHTS_RESOLVED = YOLO_WEIGHTS  # 공식 이름이면 자동 다운로드

    # 1) YOLO 로드
    _STATE["yolo"] = load_yolo(
        weights=YOLO_WEIGHTS_RESOLVED,
        conf=0.30,
        iou=0.50,
        imgsz=640,
        device=device
    )

    # 2) 분류기 로드 (ko_mapping.csv만 필요)
    load_classifier(
        weight_path=CLF_WEIGHTS,
        labels_path=None,
        ko_mapping_csv=KO_MAPPING_CSV,
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        entropy_threshold=ENTROPY_TH_INIT,  # (이전 호환용)
        device=device,
        channels_last=True,
    )

    _STATE["ready"] = True

# ------------------------------------------------------------
# 공개 초기화 API (콜드스타트 제거용)
# ------------------------------------------------------------
def init(warmup: bool = False) -> None:
    """
    앱 부팅 시 한 번 호출하면 가중치 다운로드 + 모델 로드까지 미리 수행.
    warmup=True면 1회 더미 추론으로 CUDA 커널까지 예열.
    """
    _load_once()
    if warmup:
        x = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, device=_STATE["device"])
        _ = clf_predict(x, topk=1)

# ------------------------------------------------------------
# 추론
# ------------------------------------------------------------
@torch.no_grad()
def predict(image: Any, return_topk: int = 3) -> Dict[str, Any]:
    _load_once()
    device = _STATE["device"]

    img = _to_pil(image)

    det = detect_and_crop(img, pad_ratio=PAD_RATIO, pad_mode="edge", strategy="largest")
    crop = det["crop"] if det.get("crop") is not None else _square_pad_whole(img, PAD_RATIO)
    bbox = det.get("bbox")
    conf = det.get("conf")
    yolo_time = det.get("time_ms")
    W, H = (det.get("image_size") or img.size)

    x = preprocess(
        crop_img=crop,
        size=INPUT_SIZE,
        device=device,
        channels_last=True,
        add_batch=True,
    )

    out = clf_predict(x, topk=return_topk)

    result: Dict[str, Any] = {
        "prediction": {
            "decision": out["decision"], # 최종 품종 예측 결과
            "decision_type": out["decision_type"],
            "top1": out["top1"],
            "topk": out["topk"],
            "reasons": {
                "mix_rules": {
                    "trigger": (out["decision_type"] == "mixed"),
                    "H":        out["entropy"]["H"],
                    "H_norm":   out["entropy"]["H_norm"],
                    "p1":       out["entropy"]["p1"],
                    "p2":       out["entropy"]["p2"],
                    "margin":   out["entropy"]["margin"],
                    "thresholds": out["entropy"]["thresholds"]  # {H_norm_th, p1_th, margin_th}
                }
            }
        },
        "boxes": {
            "selected": bbox,
            "conf": conf,
            "detected": bbox is not None,
            "yolo_time_ms": yolo_time,
            "image_size": W if isinstance(W, (list, tuple)) else [W, H]
        },
        "meta": {
            "pad": PAD_RATIO,
            "input_size": INPUT_SIZE,
            "model": MODEL_NAME,
            "num_classes": NUM_CLASSES,
            "device": device
        }
    }
    return result
