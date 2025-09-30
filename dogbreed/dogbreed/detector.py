# detector.py (요점만: weights 기본값/경로 처리 변경)
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import math, time
from pathlib import Path  # ← 추가
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError("pip install ultralytics 로 설치하세요.") from e

_YOLO: Optional[YOLO] = None
_NAMES: Dict[int, str] = {}
_DOG_IDS: List[int] = []
_CONF = 0.30
_IOU  = 0.50
_IMGSZ = 640

def load_yolo(
    # ✅ 자동 다운로드: 파일 없으면 이름만 넘겨서 Ultralytics가 받아옴
    weights: str = "yolov8s.pt",   # ← 기존 "assets/..." 에서 변경
    conf: float = 0.30,
    iou: float  = 0.50,
    imgsz: int  = 640,
    device: Optional[str] = None,
) -> YOLO:
    """
    사전학습 YOLOv8 가중치 로드(싱글톤). weights에 경로를 주면 로컬, 이름만 주면 자동 다운로드.
    """
    global _YOLO, _NAMES, _DOG_IDS, _CONF, _IOU, _IMGSZ

    # 상대경로로 "assets/..."를 넘겨도 패키지 기준으로 보정해서 찾기
    p = Path(weights)
    if not p.is_absolute() and ("/" in weights or "\\" in weights):
        pkg_root = Path(__file__).resolve().parent.parent  # dogbreed/
        p = pkg_root / weights

    # 로컬 파일이 있으면 그걸, 없으면 이름 그대로(자동 다운로드)
    m = YOLO(str(p) if p.exists() else weights)
    if device:
        m.to(device)

    _CONF, _IOU, _IMGSZ = conf, iou, imgsz
    if hasattr(m, "overrides"):
        m.overrides["conf"] = conf
        m.overrides["iou"]  = iou

    names = getattr(m, "names", None)
    _NAMES = {int(k): v for k, v in (names.items() if isinstance(names, dict) else enumerate(names))}
    _DOG_IDS = [cid for cid, nm in _NAMES.items() if nm and "dog" in nm.lower()]

    _YOLO = m
    return m

def detect_and_crop(
    image: Image.Image,
    pad_ratio: float = 0.16,
    pad_mode: str   = "edge",   # "edge" 또는 "reflect"
    strategy: str   = "largest" # "largest" | "highest_conf"
) -> Dict[str, Any]:
    """
    입력: PIL.Image
    출력: {"crop": PIL.Image or None, "bbox":[x1,y1,x2,y2] or None, "conf": float or None,
           "time_ms": float, "image_size":[W,H]}
    """
    if _YOLO is None:
        raise RuntimeError("먼저 load_yolo()를 호출하세요.")
    if not isinstance(image, Image.Image):
        raise TypeError("PIL.Image 입력 필요")

    W, H = image.size
    t0 = time.time()
    classes_arg = _DOG_IDS if _DOG_IDS else None
    r = _YOLO.predict(image, conf=_CONF, iou=_IOU, imgsz=_IMGSZ, classes=classes_arg, verbose=False)[0]
    elapsed = (time.time() - t0) * 1000.0

    # 박스 없음
    if r.boxes is None or len(r.boxes) == 0:
        return {"crop": None, "bbox": None, "conf": None, "time_ms": elapsed, "image_size": [W, H]}

    xyxy = r.boxes.xyxy.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy().astype(float)
    cls  = r.boxes.cls.cpu().numpy().astype(int)

    # 이름 기반 dog 필터 (safety: classes_arg가 None일 수도 있음)
    if _DOG_IDS:
        mask = np.isin(cls, np.array(_DOG_IDS, dtype=int))
    else:
        # 모델이 dog만 내도록 제한 못했을 때, "dog" 문자열 포함만 허용
        mask = np.array([("dog" in _NAMES.get(int(c), "").lower()) for c in cls], dtype=bool)

    dog_boxes = [{"xyxy": xyxy[i].tolist(), "conf": float(conf[i])} for i in range(len(cls)) if mask[i]]
    if not dog_boxes:
        return {"crop": None, "bbox": None, "conf": None, "time_ms": elapsed, "image_size": [W, H]}

    # 선택 규칙
    if strategy not in ("largest", "highest_conf"): strategy = "largest"
    def area(b): 
        x1,y1,x2,y2 = b["xyxy"]; return max(0.0, x2-x1) * max(0.0, y2-y1)
    sel = max(dog_boxes, key=(lambda b: (b["conf"], area(b))) ) if strategy=="highest_conf" \
          else max(dog_boxes, key=(lambda b: (area(b), b["conf"])) )

    # pad=0.16 정사각 크롭
    crop_img = _square_pad_crop_from_xyxy(image, sel["xyxy"], pad_ratio=pad_ratio, pad_mode=pad_mode)

    return {
        "crop": crop_img,                 # ← preprocessor에서 480 리사이즈 + Normalize
        "bbox": [float(v) for v in sel["xyxy"]],
        "conf": float(sel["conf"]),
        "time_ms": elapsed,
        "image_size": [W, H],
    }

# ---------------------------
# helpers
# ---------------------------
def _square_pad_crop_from_xyxy(
    image: Image.Image,
    xyxy: List[float],
    pad_ratio: float = 0.16,
    pad_mode: str   = "edge",
) -> Image.Image:
    """
    bbox 중심을 기준으로 pad_ratio만큼 확장한 정사각 영역을 계산하고,
    원본 밖은 가장자리 복제/반사로 채워 정사각 크롭을 반환(PIL).
    """
    W, H = image.size
    x1, y1, x2, y2 = xyxy
    cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
    side   = max(x2-x1, y2-y1)
    side_p = side * (1.0 + 2.0*pad_ratio)
    S      = int(max(1, round(side_p)))

    half = side_p / 2.0
    x1p, y1p, x2p, y2p = cx-half, cy-half, cx+half, cy+half

    # 정수 좌표
    x1i, y1i, x2i, y2i = map(lambda v: int(math.floor(v)), [x1p, y1p, x2p, y2p])

    # 원본 내에서 자르는 실제 영역
    x1c, y1c = max(0, x1i), max(0, y1i)
    x2c, y2c = min(W, x2i), min(H, y2i)

    # 잘라내기
    crop = image.crop((x1c, y1c, x2c, y2c))
    arr  = np.array(crop)
    if arr.ndim == 2:  # grayscale 방어
        arr = np.expand_dims(arr, axis=-1)

    # 목표 SxS를 위한 패딩 계산 (왼/위/오/아래)
    pad_left   = int(max(0, x1c - x1i))
    pad_top    = int(max(0, y1c - y1i))
    pad_right  = int(max(0, x2i - x2c))
    pad_bottom = int(max(0, y2i - y2c))

    # rounding 보정
    h, w = arr.shape[:2]
    need_w = S - (pad_left + w + pad_right)
    need_h = S - (pad_top + h + pad_bottom)
    if need_w != 0:  pad_right  += need_w
    if need_h != 0:  pad_bottom += need_h

    mode = pad_mode if pad_mode in ("edge", "reflect") else "edge"
    pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    arr_padded = np.pad(arr, pad_width, mode=mode)

    # 혹시 모를 1~2픽셀 오차 보정
    arr_padded = arr_padded[:S, :S, :]

    return Image.fromarray(arr_padded.astype(np.uint8))
