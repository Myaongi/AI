from pydantic import BaseModel, Field, validator
from typing import List, Optional, Tuple

class TextNormalizeIn(BaseModel):
    breed: str = ""
    colors: str = ""
    features: str = ""

class TextNormalizeOut(BaseModel):
    sentences: List[str] = Field(default_factory=list)

class EmbeddingOut(BaseModel):
    sentences: List[str]
    image: List[float]
    text: List[float]

class PairScoreIn(BaseModel):
    emb_a_image: List[float]
    emb_a_text:  List[float]
    emb_b_image: List[float]
    emb_b_text:  List[float]
    weights: Optional[Tuple[float, float, float, float]] = None
    @validator("weights")
    def _sum_to_one(cls, v):
        if v is None: return v
        s = sum(v)
        if s <= 0: raise ValueError("weights sum must be > 0")
        return tuple(x/s for x in v)

class PairScoreOut(BaseModel):
    s_ii: float
    s_it: float
    s_ti: float
    s_tt: float
    score: float
