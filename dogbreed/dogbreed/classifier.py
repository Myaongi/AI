from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple
import os, json, csv
import numpy as np
import torch
import torch.nn.functional as F
import timm
import pandas as pd

"""
classifier.py — labels.json 없이 ko_mapping.csv만으로 동작
- ko_mapping.csv에서 영어 라벨 리스트(labels)와 en→ko 매핑을 모두 생성
- 입력 텐서 -> logits -> softmax(probs) -> 엔트로피(H)
- H >= H_TH 이면 "믹스", 아니면 top1 한국어 품종명
"""

# ------------------------------------------------------------
# 내부 상태
# ------------------------------------------------------------
_STATE: Dict[str, Any] = {
    "ready": False,
    "device": None,
    "model": None,
    "labels": None,           # idx -> en (list[str])
    "ko_map": None,           # en -> ko (dict[str,str])
    "entropy_threshold": 1.6  # 자연로그 기준
}

# ------------------------------------------------------------
# ko_mapping.csv 유틸
# ------------------------------------------------------------
def _parse_ko_csv(csv_path: str) -> Tuple[List[str], Dict[str, str]]:
    """
    ko_mapping.csv에서 (labels_en 리스트, en->ko dict) 생성
    - 허용 헤더(대소문자 무시):
        * 인덱스: idx, index
        * 영어:   en, english, label, name
        * 한글:   ko, korean, ko_name, kr
    - idx가 있으면 idx 기준으로 정렬 → 분류기 출력 인덱스와 일치
      없으면 CSV 등장 순서 유지
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ko_mapping.csv not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    cols = {c.lower(): c for c in df.columns}

    en_key = cols.get("en") or cols.get("english") or cols.get("label") or cols.get("name")
    ko_key = cols.get("ko") or cols.get("korean") or cols.get("ko_name") or cols.get("kr")
    idx_key = cols.get("idx") or cols.get("index")

    if not en_key:
        raise ValueError("ko_mapping.csv must contain an English label column (en/english/label/name).")
    if idx_key:
        df = df.sort_values(idx_key)

    labels_en = df[en_key].astype(str).tolist()
    ko_map = {}
    if ko_key:
        for _, row in df.iterrows():
            en = str(row[en_key]).strip()
            ko = str(row[ko_key]).strip() if not pd.isna(row[ko_key]) else ""
            if en:
                ko_map[en] = ko or en
    else:
        # ko 열이 없어도 영문 그대로 매핑
        ko_map = {en: en for en in labels_en}

    return labels_en, ko_map

# (이전 호환: labels.json이 있으면 읽고 싶다면 아래 보조함수 유지)
def _load_labels_json(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        items = sorted(((int(k), v) for k, v in obj.items()), key=lambda x: x[0])
        return [v for _, v in items]
    raise ValueError("labels.json must be list[str] or dict[idx->str]")

def _take_topk(probs: np.ndarray, labels: List[str], k: int) -> List[Dict[str, Any]]:
    idx = np.argsort(-probs)[:k]
    out: List[Dict[str, Any]] = []
    for i in idx:
        out.append({"index": int(i), "label_en": labels[i], "prob": float(probs[i])})
    return out

# ------------------------------------------------------------
# 로더
# ------------------------------------------------------------
def load_classifier(
    weight_path: str,
    *,
    # labels_path는 선택(있으면 사용, 없으면 csv에서 생성)
    labels_path: Optional[str] = None,
    ko_mapping_csv: str,
    model_name: str = "tf_efficientnetv2_m_in21ft1k",
    num_classes: int = 105,
    entropy_threshold: float = 1.6,
    device: Optional[str] = None,
    channels_last: bool = True,
) -> None:
    """
    - ko_mapping.csv에서 en 라벨 리스트와 en→ko 매핑 생성
    - labels.json이 주어지면(옵션) 라벨 리스트로 활용 가능(디버깅/검증용)
    """
    global _STATE

    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1) 라벨/KO 매핑: csv를 기준으로 생성
    labels_from_csv, ko_map = _parse_ko_csv(ko_mapping_csv)

    # (옵션) labels.json도 들어왔으면 길이/내용 검증용으로만 사용
    labels_from_json = _load_labels_json(labels_path)
    labels = labels_from_json or labels_from_csv
    if len(labels) != num_classes:
        print(f"[classifier] WARNING: labels({len(labels)}) != num_classes({num_classes})")

    # 2) 모델 생성 & 가중치 로드
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    ckpt = torch.load(weight_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[classifier] missing keys:", missing)
    if unexpected:
        print("[classifier] unexpected keys:", unexpected)

    model.to(device).eval()
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    _STATE.update({
        "ready": True,
        "device": device,
        "model": model,
        "labels": labels,
        "ko_map": ko_map,
        "entropy_threshold": float(entropy_threshold),
    })
    print(f"[classifier] loaded on {device} | H_TH={_STATE['entropy_threshold']} | classes={len(labels)}")

def set_entropy_threshold(value: float) -> None:
    _STATE["entropy_threshold"] = float(value)

# ------------------------------------------------------------
# 추론
# ------------------------------------------------------------
@torch.no_grad()
def predict(
    x: torch.Tensor,     # [C,H,W] or [1,C,H,W]
    topk: int = 3
) -> Dict[str, Any]:
    if not _STATE["ready"]:
        raise RuntimeError("call load_classifier() first")

    model: torch.nn.Module = _STATE["model"]
    device: str            = _STATE["device"]
    labels: List[str]      = _STATE["labels"]
    ko_map: Dict[str, str] = _STATE["ko_map"]
    H_TH: float            = _STATE["entropy_threshold"]

    # 배치 차원 정리
    if x.ndim == 3:
        x = x.unsqueeze(0)
    elif x.ndim != 4:
        raise ValueError("x must be [C,H,W] or [N,C,H,W]")
    x = x.to(device, non_blocking=True)

    # 1) logits -> 2) softmax
    logits = model(x)                 # [N,K]
    probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()  # [K]

    # 3) 엔트로피 (자연로그)
    eps = 1e-9
    H = float(-(probs * np.log(probs + eps)).sum())
    mixed = H >= H_TH

    # 4) top-k & 한글 매핑
    top = _take_topk(probs, labels, k=topk)
    for t in top:
        en = t["label_en"]
        t["label_ko"] = ko_map.get(en, en)
    top1 = top[0]

    # 5) 최종 결정 문자열
    decision = "믹스" if mixed else top1["label_ko"]

    return {
        "decision": decision,
        "decision_type": "mixed" if mixed else "breed",
        "entropy": {"H": H, "threshold": H_TH},
        "top1": top1,
        "topk": top
    }

__all__ = ["load_classifier", "set_entropy_threshold", "predict"]
