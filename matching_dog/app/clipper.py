# clipper.py
from typing import List, Optional
import os
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import open_clip
from . import config

# -------------------- ProjMLP (학습 코드와 동일) --------------------
class ProjMLP(nn.Module):
    """잔차 + LayerNorm + Dropout로 안정화된 head (head-only FT)"""
    def __init__(self, d: int, p_drop: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.act = nn.ReLU(True)
        self.fc2 = nn.Linear(d, d)
        self.ln  = nn.LayerNorm(d)
        self.dp  = nn.Dropout(p_drop)
        self.alpha = nn.Parameter(torch.tensor(1.0))  # residual scale

    def forward(self, x):
        h = self.fc2(self.act(self.fc1(x)))
        x = x + self.alpha * self.dp(h)
        return F.normalize(self.ln(x), dim=-1)

# -------------------- 유틸: FT 가중치 로딩 --------------------
def _maybe_download(url: str, cache_dir: str) -> Optional[str]:
    if not url:
        return None
    try:
        cache_dir = cache_dir or "/tmp/clip_ft_cache"
        pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
        fname = os.path.join(cache_dir, os.path.basename(url))
        if not os.path.exists(fname):
            # torch 허브 유틸 사용 (requests 대비 의존성 적음)
            torch.hub.download_url_to_file(url, fname)
        return fname if os.path.exists(fname) else None
    except Exception:
        return None

def _load_ft_state_dict(path: str, device: str):
    state = torch.load(path, map_location=device)
    # 학습 코드에서 저장한 키:
    # - "proj_img": state_dict
    # - "proj_txt": state_dict
    # - "base": {"model":..., "pretrained":...}  # 참조용
    # - "log_tau_xx": float  (추론에서 미사용)
    return state

# -------------------- 본체 --------------------
class CLIPper:
    def __init__(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED
        )
        self.tok = open_clip.get_tokenizer(config.CLIP_MODEL)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval().to(self.device)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        # d 차원 결정 (텍스트/이미지 투영 차원 동일)
        if hasattr(self.model, "text_projection"):
            d = self.model.text_projection.shape[1]
        else:
            # 일부 CLIP 변형 호환
            d = self.model.ln_final.weight.shape[0]

        # 학습된 헤드 준비 (없으면 None → zero-shot 경로)
        self.proj_img: Optional[ProjMLP] = None
        self.proj_txt: Optional[ProjMLP] = None

        # 1) 우선 순위: FT_WEIGHTS_PATH → 2) FT_WEIGHTS_URL(download)
        pt_path = None
        if config.FT_WEIGHTS_PATH and os.path.exists(config.FT_WEIGHTS_PATH):
            pt_path = config.FT_WEIGHTS_PATH
        elif config.FT_WEIGHTS_URL:
            pt_path = _maybe_download(config.FT_WEIGHTS_URL, config.FT_CACHE_DIR)

        if pt_path:
            try:
                state = _load_ft_state_dict(pt_path, self.device)
                self.proj_img = ProjMLP(d).to(self.device)
                self.proj_txt = ProjMLP(d).to(self.device)
                self.proj_img.load_state_dict(state["proj_img"])
                self.proj_txt.load_state_dict(state["proj_txt"])
                self.proj_img.eval()
                self.proj_txt.eval()
                # 참고: state["base"]로 모델/프리트레인 불일치 체크 가능(여기선 경고만 가능)
                base = state.get("base", {})
                bm, bp = base.get("model"), base.get("pretrained")
                if bm and bm != config.CLIP_MODEL:
                    print(f"[CLIPper] ⚠️ FT base model mismatch: ckpt={bm}, runtime={config.CLIP_MODEL}")
                if bp and bp != config.CLIP_PRETRAINED:
                    print(f"[CLIPper] ⚠️ FT pretrained tag mismatch: ckpt={bp}, runtime={config.CLIP_PRETRAINED}")
                print(f"[CLIPper] ✅ Loaded FT heads from: {pt_path}")
            except Exception as e:
                # 실패 시 zero-shot로 fallback
                self.proj_img = None
                self.proj_txt = None
                print(f"[CLIPper] ⚠️ Failed to load FT heads ({pt_path}): {e}\n"
                      f"         → fallback to zero-shot embeddings")

    # ---- 내부 공통: base 인코딩 ----
    @torch.no_grad()
    def _encode_image_base(self, pil: Image.Image) -> torch.Tensor:
        x = self.preprocess(pil).unsqueeze(0).to(self.device)
        z = self.model.encode_image(x)
        return F.normalize(z, dim=-1).squeeze(0)

    @torch.no_grad()
    def _encode_texts_base(self, sentences: List[str]) -> torch.Tensor:
        tok = self.tok(sentences).to(self.device)
        zt = self.model.encode_text(tok)  # (N, d)
        return F.normalize(zt, dim=-1)

    # ---- 공개 API ----
    @torch.no_grad()
    def encode_image(self, pil: Image.Image) -> torch.Tensor:
        z = self._encode_image_base(pil)  # (d,)
        if self.proj_img is not None:
            z = self.proj_img(z.unsqueeze(0)).squeeze(0)
        return F.normalize(z, dim=-1).float().cpu()

    @torch.no_grad()
    def encode_text_3(self, sentences: List[str]) -> torch.Tensor:
        texts = [s for s in sentences if isinstance(s, str) and s.strip()] or ["a dog."]
        zt = self._encode_texts_base(texts)  # (N, d)

        # 추론 풀링: avg / max / top1(이미지 없으니 avg 권장)
        if config.TEXT_POOLING == "max" and len(texts) > 1:
            pooled = torch.max(zt, dim=0).values
        elif config.TEXT_POOLING == "top1" and len(texts) > 1:
            # 이미지가 없으니 top1은 의미가 약함 → avg fallback
            pooled = zt.mean(dim=0)
        else:
            pooled = zt.mean(dim=0)

        if self.proj_txt is not None:
            pooled = self.proj_txt(pooled.unsqueeze(0)).squeeze(0)

        return F.normalize(pooled, dim=-1).float().cpu()
