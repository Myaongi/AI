import torch
import torch.nn.functional as F
from typing import Tuple
from . import config

def _to_t(x):
    if isinstance(x, torch.Tensor): return x
    return torch.tensor(x, dtype=torch.float32)

def cosine(a, b) -> float:
    a = F.normalize(_to_t(a), dim=-1)
    b = F.normalize(_to_t(b), dim=-1)
    return float((a @ b).item())

def four_sims(emb_a_image, emb_a_text, emb_b_image, emb_b_text):
    s_ii = cosine(emb_a_image, emb_b_image)
    s_it = cosine(emb_a_image, emb_b_text)
    s_ti = cosine(emb_a_text,  emb_b_image)
    s_tt = cosine(emb_a_text,  emb_b_text)
    return s_ii, s_it, s_ti, s_tt

def weighted_score(s4: Tuple[float,float,float,float], weights=None) -> float:
    if weights is None:
        w = (config.W_II, config.W_IT, config.W_TI, config.W_TT)
    else:
        w = weights
    s = sum(w)
    w = tuple(x/s for x in w)
    return float(w[0]*s4[0] + w[1]*s4[1] + w[2]*s4[2] + w[3]*s4[3])
