from fastapi import FastAPI, UploadFile, File, Form
from .schemas import TextNormalizeIn, TextNormalizeOut, EmbeddingOut, PairScoreIn, PairScoreOut
from .llm_client import normalize_to_3_sentences
from .service import build_embeddings, score_pair
from . import config

app = FastAPI(title="LostDog Matching — Zero-shot CLIP + YOLO")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/normalize", response_model=TextNormalizeOut)
def normalize_endpoint(payload: TextNormalizeIn):
    sents = normalize_to_3_sentences(payload.breed, payload.colors, payload.features)
    return {"sentences": sents}

@app.post("/embed", response_model=EmbeddingOut)
async def embed_endpoint(
    image: UploadFile = File(...),
    breed: str = Form(""),
    colors: str = Form(""),
    features: str = Form("")
):
    image_bytes = await image.read()
    sents, emb_img, emb_txt = build_embeddings(image_bytes, breed, colors, features)
    return {"sentences": sents, "image": emb_img, "text": emb_txt}

@app.post("/score", response_model=PairScoreOut)
def score_endpoint(payload: PairScoreIn):
    s4, score = score_pair(payload.emb_a_image, payload.emb_a_text,
                           payload.emb_b_image, payload.emb_b_text,
                           weights=payload.weights)
    return {"s_ii": s4[0], "s_it": s4[1], "s_ti": s4[2], "s_tt": s4[3], "score": score}

@app.post("/pass-threshold")
def pass_threshold(score: float):
    return {"pass": (score >= config.SIM_THRESHOLD), "threshold": config.SIM_THRESHOLD}
