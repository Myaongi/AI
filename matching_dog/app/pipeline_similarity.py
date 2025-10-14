# app/pipeline_similarity.py
from typing import Tuple, Optional, Dict
import numpy as np

from .similarity import four_sims, weighted_score  # (I-I, I-T, T-I, T-T) + 가중합
from . import config

def score_pair(
    emb_img_a: np.ndarray, emb_txt_a: np.ndarray,
    emb_img_b: np.ndarray, emb_txt_b: np.ndarray,
    weights: Optional[Tuple[float, float, float, float]] = None,
) -> Dict[str, object]:
    """
    [파이프라인 #2] 게시물 A vs B (1:1)
      - 4유형(I-I, I-T, T-I, T-T) 코사인 유사도 계산
      - 가중합으로 최종 score 계산 (기본값: config.W_II ... W_TT)
    반환:
      {
        "s_ii": float, "s_it": float, "s_ti": float, "s_tt": float,
        "score": float, "threshold": float, "pass": bool
      }
    """
    s_ii, s_it, s_ti, s_tt = four_sims(emb_img_a, emb_txt_a, emb_img_b, emb_txt_b)
    score = weighted_score((s_ii, s_it, s_ti, s_tt), weights=weights)
    return {
        "s_ii": float(s_ii),
        "s_it": float(s_it),
        "s_ti": float(s_ti),
        "s_tt": float(s_tt),
        "score": float(score),   # 최종 유사도
        "threshold": float(config.SIM_THRESHOLD),   # 임계값
        "pass": bool(score >= config.SIM_THRESHOLD),   # 임계값 기준으로 사용자에게 매칭 결과 보여주기
    }

__all__ = ["score_pair"]
