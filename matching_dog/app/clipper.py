from typing import List
import torch
import torch.nn.functional as F
from PIL import Image
import open_clip
from . import config

class CLIPper:
    def __init__(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED
        )
        self.tok = open_clip.get_tokenizer(config.CLIP_MODEL)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval().to(self.device)
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass

    @torch.no_grad()
    def encode_image(self, pil: Image.Image) -> torch.Tensor:
        x = self.preprocess(pil).unsqueeze(0).to(self.device)
        z = self.model.encode_image(x)
        return F.normalize(z, dim=-1).squeeze(0).float().cpu()

    @torch.no_grad()
    def encode_text_3(self, sentences: List[str]) -> torch.Tensor:
        texts = [s for s in sentences if s.strip()] or ["a dog."]
        tok = self.tok(texts).to(self.device)
        zt = F.normalize(self.model.encode_text(tok), dim=-1)
        if config.TEXT_POOLING == "max" and len(texts) > 1:
            pooled = torch.max(zt, dim=0).values
        else:  # avg / top1(이미지 없으니 avg로)
            pooled = zt.mean(dim=0)
        return F.normalize(pooled, dim=-1).float().cpu()
