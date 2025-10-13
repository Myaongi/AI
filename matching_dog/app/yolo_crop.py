import io, logging, contextlib
from typing import Optional, Tuple
from PIL import Image, ImageOps
import torch
from . import config

_model = None

def _load_yolo_quiet(weights: str, verbose: bool = False):
    from ultralytics import YOLO
    try:
        from ultralytics.utils import LOGGER
        prev = LOGGER.level
        LOGGER.setLevel(logging.ERROR if not verbose else logging.INFO)
    except Exception:
        prev = None
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = YOLO(weights)
        try: model.overrides["verbose"] = verbose
        except Exception: pass
    if prev is not None:
        from ultralytics.utils import LOGGER; LOGGER.setLevel(prev)
    return model

def ensure_model():
    global _model
    if _model is None:
        _model = _load_yolo_quiet(config.YOLO_WEIGHTS, verbose=False)
        try: _model.to(0 if torch.cuda.is_available() else "cpu")
        except Exception: pass
    return _model

def exif_transpose(pil: Image.Image) -> Image.Image:
    try: return ImageOps.exif_transpose(pil)
    except Exception: return pil

def detect_dog_bbox(pil: Image.Image) -> Optional[Tuple[int,int,int,int]]:
    m = ensure_model()
    res = m.predict(pil, conf=config.YOLO_CONF, imgsz=config.YOLO_IMGSZ,
                    device=0 if torch.cuda.is_available() else "cpu", verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0: return None
    import numpy as np
    xyxy = res.boxes.xyxy.cpu().numpy()
    cls  = res.boxes.cls.cpu().numpy().astype(int)
    conf = res.boxes.conf.cpu().numpy()
    mask = (cls == config.DOG_CLASS_ID) & (conf >= config.YOLO_CONF)
    if not mask.any(): return None
    areas = (xyxy[:,2]-xyxy[:,0]) * (xyxy[:,3]-xyxy[:,1])
    idxs  = [i for i in range(len(xyxy)) if mask[i]]
    best  = max(idxs, key=lambda i: areas[i])
    x1,y1,x2,y2 = xyxy[best]
    return int(x1), int(y1), int(x2), int(y2)

def crop_with_margin(pil: Image.Image, bbox: Optional[Tuple[int,int,int,int]]) -> Image.Image:
    if bbox is None: return pil
    W,H = pil.size
    x1,y1,x2,y2 = bbox
    w,h = x2-x1, y2-y1
    mx,my = int(config.YOLO_MARGIN*w), int(config.YOLO_MARGIN*h)
    X1,Y1 = max(0,x1-mx), max(0,y1-my)
    X2,Y2 = min(W,x2+mx), min(H,y2+my)
    if X2<=X1 or Y2<=Y1: return pil
    return pil.crop((X1,Y1,X2,Y2))

def yolo_crop(pil: Image.Image) -> Image.Image:
    pil = exif_transpose(pil.convert("RGB"))
    bbox = detect_dog_bbox(pil)
    return crop_with_margin(pil, bbox)
