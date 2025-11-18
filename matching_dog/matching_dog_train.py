# 매칭 서비스 모델 학습 코드
import os, json, math, random
from typing import List
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import open_clip

# -------------------- Config --------------------
CSV_PATH   = "/workspace/proj_gc/dataset/interim/petfinder/image_level_sentences_llm.csv"
OUT_DIR    = "/workspace/proj_gc/dataset/interim/petfinder/clip_ft_L14_dcxl"
os.makedirs(OUT_DIR, exist_ok=True)

CLIP_MODEL      = "ViT-L-14"
CLIP_PRETRAINED = "datacomp_xl_s13b_b90k"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

SPLIT_PATH = os.path.join(OUT_DIR, "split.json")
SPLIT_SEED = 42
R_TRAIN, R_VAL, R_TEST = 0.70, 0.15, 0.15

# Train
EPOCHS         = 20
LR             = 3e-5
WD             = 1e-4
BATCH_P        = 32
BATCH_K        = 2
GRAD_ACC       = 1
FP16           = True
EARLY_STOP     = 4
WARMUP_EPOCHS  = 2
TAU_FREEZE_EPOCHS = 0

# Loss weights (TT 낮게 시작 + 스케줄)
ALPHA_II=1.0; ALPHA_IT=1.0; ALPHA_TI=1.0; ALPHA_TT=0.3
TT_SCHEDULE = {1:0.30, 2:0.30, 3:0.25, 4:0.20}  # 이후엔 마지막 값 유지

# Eval
TEXT_POOLING = "gated"
GRID_STEP    = 0.1
TOPK_EVAL    = 50
EXCLUDE_SELF = True

# Loader
NUM_WORKERS = max(2, (os.cpu_count() or 8)//4)
PIN_MEMORY  = True
PERSISTENT  = False
PREFETCH    = 2
IMG_CHUNK   = 64
TXT_CHUNK   = 128

# -------------------- utils --------------------
def seed_all(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(SPLIT_SEED)
try: torch.set_float32_matmul_precision("high")
except: pass

def exif_transpose(im: Image.Image):
    try: return ImageOps.exif_transpose(im)
    except Exception: return im

@torch.no_grad()
def cosim(A: torch.Tensor, B: torch.Tensor)->torch.Tensor:
    A = F.normalize(A, dim=-1); B = F.normalize(B, dim=-1)
    return A @ B.T

# -------------------- Dataset --------------------
class PetImgText(Dataset):
    def __init__(self, csv_path:str):
        df = pd.read_csv(csv_path)
        assert "PetID" in df, "CSV must contain PetID"
        path_col = "image_path_cropped" if "image_path_cropped" in df.columns else "image_path"
        assert path_col in df.columns, "CSV must contain image_path(_cropped)"
        df = df[df[path_col].apply(lambda p: isinstance(p,str) and os.path.exists(p))].copy()
        for c in ["sentence1","sentence2","sentence3"]:
            if c not in df.columns: df[c] = ""
            df[c] = df[c].astype(str)
        df["PetID"] = df["PetID"].astype(str)
        cnt = df.groupby("PetID")[path_col].count()
        keep = set(cnt[cnt>=2].index)
        df = df[df["PetID"].isin(keep)].reset_index(drop=True)

        self.df = df
        self.path_col = path_col
        self.pid = df["PetID"].tolist()
        self.paths = df[path_col].tolist()
        self.sents3 = list(zip(df["sentence1"], df["sentence2"], df["sentence3"]))
        self.pid2idx = {}
        for i,p in enumerate(self.pid):
            self.pid2idx.setdefault(p, []).append(i)

    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        pid  = self.pid[i]
        path = self.paths[i]
        s1,s2,s3 = self.sents3[i]
        pil = exif_transpose(Image.open(path).convert("RGB"))
        texts = [t for t in [s1,s2,s3] if isinstance(t,str) and t.strip()] or ["a dog."]
        return {"pid": pid, "pil": pil, "texts": texts}

# -------------------- Split --------------------
def make_or_load_split(ds: PetImgText):
    if os.path.exists(SPLIT_PATH):
        with open(SPLIT_PATH, "r") as f: groups = json.load(f)
        print(f"✅ Using existing split: {SPLIT_PATH}")
    else:
        valid_pids = [p for p, idxs in ds.pid2idx.items() if len(idxs)>=2]
        rng = random.Random(SPLIT_SEED); rng.shuffle(valid_pids)
        n=len(valid_pids); n_tr=int(n*R_TRAIN); n_va=int(n*R_VAL)
        groups = {"train_pids": valid_pids[:n_tr],
                  "val_pids": valid_pids[n_tr:n_tr+n_va],
                  "test_pids": valid_pids[n_tr+n_va:]}
        with open(SPLIT_PATH,"w") as f: json.dump(groups, f, indent=2)
        print(f"🆕 Made split: {SPLIT_PATH}")

    tr, va, te = [], [], []
    tag = {}
    for p in groups["train_pids"]: tag[p]="train"
    for p in groups["val_pids"]:   tag[p]="val"
    for p in groups["test_pids"]:  tag[p]="test"
    for i,p in enumerate(ds.pid):
        s = tag.get(p, None)
        if   s=="train": tr.append(i)
        elif s=="val":   va.append(i)
        elif s=="test":  te.append(i)
    print(f"[Split] train={len(tr)}, val={len(va)}, test={len(te)}")
    return tr, va, te

# -------------------- Batch builder --------------------
class PKSampler(Sampler):
    def __init__(self, dataset:PetImgText, indices:List[int], P:int, K:int, iters:int, seed=42):
        self.indices=indices; self.ds=dataset; self.P=P; self.K=K; self.iters=iters
        self.pid2idx = defaultdict(list)
        for gi in self.indices:
            self.pid2idx[dataset.pid[gi]].append(gi)
        self.pids = list(self.pid2idx.keys())
        self.rng = random.Random(seed)
    def __iter__(self):
        for _ in range(self.iters):
            chosen = self.rng.sample(self.pids, k=min(self.P, len(self.pids)))
            batch=[]
            for p in chosen:
                idxs=self.pid2idx[p]
                if len(idxs)>=self.K: batch.extend(self.rng.sample(idxs, self.K))
                else: batch.extend([self.rng.choice(idxs) for _ in range(self.K)])
            yield batch
    def __len__(self): return self.iters

def collate_multi(batch_list):
    pids=[]; pils=[]; texts_flat=[]; map_idx=[]
    for i,b in enumerate(batch_list):
        pids.append(b["pid"]); pils.append(b["pil"])
        t=[t for t in b["texts"] if t.strip()] or ["a dog."]
        texts_flat.extend(t); map_idx.extend([i]*len(t))
    txt_to_sample = torch.tensor(map_idx, dtype=torch.long)
    return {"pid":pids, "pil":pils, "texts_flat":texts_flat,
            "txt_to_sample":txt_to_sample}

# -------------------- Model (head-only) --------------------
class ProjMLP(nn.Module):
    """잔차 + LayerNorm + Dropout로 안정화된 head"""
    def __init__(self, d:int, p_drop:float=0.1):
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

class GatedTextPool(nn.Module):
    def __init__(self, d:int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d, d//2), nn.ReLU(True), nn.Linear(d//2, 1))
    def forward(self, zt: torch.Tensor, txt_to_sample: torch.Tensor, zi_base: torch.Tensor=None):
        N = int(txt_to_sample.max().item()) + 1 if txt_to_sample.numel()>0 else 0
        d = zt.size(1)
        out = torch.zeros((N, d), device=zt.device)
        for i in range(N):
            idx = (txt_to_sample==i).nonzero(as_tuple=True)[0]
            if idx.numel()==0: continue
            z = zt[idx]
            g = torch.sigmoid(self.gate(z))
            w = g / (g.sum()+1e-6)
            out[i] = F.normalize((w*z).sum(0), dim=-1)
        return out

class CLIPHeadOnly(nn.Module):
    def __init__(self, model_name, pretrained, device, img_chunk=64, txt_chunk=128):
        super().__init__()
        self.model, self.preprocess_train, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tok = open_clip.get_tokenizer(model_name)
        self.model.eval().to(device)
        for p in self.model.parameters(): p.requires_grad_(False)
        d = (self.model.text_projection.shape[1]
             if hasattr(self.model, "text_projection")
             else self.model.ln_final.weight.shape[0])
        self.proj_img = ProjMLP(d, p_drop=0.1)
        self.proj_txt = ProjMLP(d, p_drop=0.1)
        # per-channel temperatures
        self.log_tau_ii = nn.Parameter(torch.tensor(0.0))
        self.log_tau_it = nn.Parameter(torch.tensor(0.0))
        self.log_tau_ti = nn.Parameter(torch.tensor(0.0))
        self.log_tau_tt = nn.Parameter(torch.tensor(0.0))
        self.txt_pool = GatedTextPool(d)
        self.img_chunk=img_chunk; self.txt_chunk=txt_chunk

    @torch.no_grad()
    def encode_image_batched(self, pil_list:List[Image.Image], device):
        feats=[]
        for i in range(0,len(pil_list), self.img_chunk):
            xs = torch.stack([self.preprocess(p) for p in pil_list[i:i+self.img_chunk]])\
                    .to(device, non_blocking=True)
            z  = F.normalize(self.model.encode_image(xs), dim=-1)
            feats.append(z)
        return torch.cat(feats,0)

    @torch.no_grad()
    def encode_texts_batched(self, texts:List[str], device):
        feats=[]
        for i in range(0,len(texts), self.txt_chunk):
            tok = self.tok(texts[i:i+self.txt_chunk]).to(device, non_blocking=True)
            z   = F.normalize(self.model.encode_text(tok), dim=-1)
            feats.append(z)
        return torch.cat(feats,0)

    def forward(self, batch, device):
        zi_base = self.encode_image_batched(batch["pil"], device)
        zt_base = self.encode_texts_batched(batch["texts_flat"], device)
        zi = self.proj_img(zi_base); zt = self.proj_txt(zt_base)
        # τ clamp로 드리프트 가드
        tau_ii = torch.clamp(torch.exp(self.log_tau_ii), 0.03, 0.5)
        tau_it = torch.clamp(torch.exp(self.log_tau_it), 0.03, 0.5)
        tau_ti = torch.clamp(torch.exp(self.log_tau_ti), 0.03, 0.5)
        tau_tt = torch.clamp(torch.exp(self.log_tau_tt), 0.03, 0.5)
        return F.normalize(zi,dim=-1), F.normalize(zt,dim=-1), zi_base, zt_base, batch["txt_to_sample"].to(device), (tau_ii, tau_it, tau_ti, tau_tt)

# -------------------- Losses --------------------
def multi_pos_nce(sim: torch.Tensor, pos_mask: torch.Tensor, tau: torch.Tensor):
    S = sim / tau.clamp_min(1e-6)
    logZ = torch.logsumexp(S, dim=1, keepdim=True)
    pos_scores = S[pos_mask]
    if pos_scores.numel()==0:
        return torch.tensor(0.0, device=sim.device)
    row_ids = torch.arange(S.size(0), device=S.device).unsqueeze(1).expand_as(pos_mask)[pos_mask]
    num_pos = torch.bincount(row_ids, minlength=S.size(0)).clamp_min(1)
    sum_logprob = torch.zeros(S.size(0), device=S.device)
    sum_logprob.index_add_(0, row_ids, pos_scores - logZ.squeeze(1).index_select(0, row_ids))
    return -(sum_logprob / num_pos).mean()

def contrastive_losses_multi_text(zi, zt, pid_list, txt_to_sample, taus):
    """개별 채널 손실 반환(외부에서 TT 가중치 스케줄링)."""
    tau_ii, tau_it, tau_ti, tau_tt = taus
    device = zi.device
    N, M = zi.size(0), zt.size(0)

    pid_np = np.array(pid_list)
    same_img = torch.from_numpy(pid_np[:, None] == pid_np[None, :]).to(device)
    same_img.fill_diagonal_(False)
    L_ii = multi_pos_nce(zi @ zi.T, same_img, tau_ii)

    pid_per_text = [pid_list[int(i)] for i in txt_to_sample.tolist()]
    txt_groups = defaultdict(list)
    for j, p in enumerate(pid_per_text): txt_groups[p].append(j)

    same_txt = torch.zeros((M, M), dtype=torch.bool, device=device)
    for idxs in txt_groups.values():
        if len(idxs) > 1:
            idx = torch.tensor(idxs, device=device, dtype=torch.long)
            same_txt[idx.unsqueeze(1), idx.unsqueeze(0)] = True
            same_txt[idx, idx] = False
    L_tt = multi_pos_nce(zt @ zt.T, same_txt, tau_tt)

    pos_it = torch.zeros((N, M), dtype=torch.bool, device=device)
    pid_to_textidx = {p: torch.tensor(idxs, device=device, dtype=torch.long)
                      for p, idxs in txt_groups.items()}
    for i in range(N):
        idx = pid_to_textidx.get(pid_list[i], None)
        if idx is not None and idx.numel()>0: pos_it[i, idx] = True

    L_it = multi_pos_nce(zi @ zt.T, pos_it,  tau_it)
    L_ti = multi_pos_nce(zt @ zi.T, pos_it.T, tau_ti)

    logs = {"L_ii": float(L_ii), "L_it": float(L_it), "L_ti": float(L_ti), "L_tt": float(L_tt),
            "tau_ii": float(tau_ii.detach().cpu()), "tau_it": float(tau_it.detach().cpu()),
            "tau_ti": float(tau_ti.detach().cpu()), "tau_tt": float(tau_tt.detach().cpu())}
    return (L_ii, L_it, L_ti, L_tt), logs

# -------------------- Eval helpers --------------------
def pool_texts_per_sample_avgmax_top1(zt_base: torch.Tensor, txt_to_sample: torch.Tensor,
                                      zi_base: torch.Tensor=None, mode:str="avg")->torch.Tensor:
    device = zt_base.device
    N = int(txt_to_sample.max().item()) + 1 if txt_to_sample.numel()>0 else 0
    d = zt_base.size(1)
    out = torch.zeros((N, d), device=device)
    if mode == "avg":
        counts = torch.bincount(txt_to_sample, minlength=N).clamp_min(1).unsqueeze(1)
        out.index_add_(0, txt_to_sample, zt_base); out = F.normalize(out / counts, dim=-1)
    elif mode == "max":
        for i in range(N):
            idx = (txt_to_sample==i).nonzero(as_tuple=True)[0]
            if idx.numel(): out[i] = F.normalize(torch.max(zt_base[idx], 0).values, dim=-1)
    elif mode == "top1" and zi_base is not None:
        for i in range(N):
            idx = (txt_to_sample==i).nonzero(as_tuple=True)[0]
            if idx.numel()==0: continue
            sims = (zi_base[i].unsqueeze(0) @ zt_base[idx].T).squeeze(0)
            out[i] = F.normalize(zt_base[idx[torch.argmax(sims)]], dim=-1)
    else:
        counts = torch.bincount(txt_to_sample, minlength=N).clamp_min(1).unsqueeze(1)
        out.index_add_(0, txt_to_sample, zt_base); out = F.normalize(out / counts, dim=-1)
    return out

def embed_split_eval(indices: List[int], ds: PetImgText, model: CLIPHeadOnly, device, text_pooling="gated"):
    pids, pils, texts, t2s = [], [], [], []
    for si, i in enumerate(indices):
        r = ds[i]; pids.append(r["pid"]); pils.append(r["pil"])
        tt=[t for t in r["texts"] if t.strip()] or ["a dog."]
        texts.extend(tt); t2s.extend([si]*len(tt))
    txt_to_sample = torch.tensor(t2s, dtype=torch.long, device=device)
    zi_base = model.encode_image_batched(pils, device)
    zt_base = model.encode_texts_batched(texts, device)
    zi = F.normalize(model.proj_img(zi_base), dim=-1).cpu()
    zt = F.normalize(model.proj_txt(zt_base), dim=-1)
    if text_pooling == "gated":
        zt_pooled = model.txt_pool(zt, txt_to_sample, zi_base=zi_base).cpu()
    else:
        zt_pooled = pool_texts_per_sample_avgmax_top1(zt, txt_to_sample, zi_base=zi_base, mode=text_pooling).cpu()
    return pids, zi, zt_pooled

def grid_weights(step=0.1):
    vals = np.arange(0.0, 1.0+1e-9, step)
    for w1 in vals:
        for w2 in vals:
            for w3 in vals:
                s = w1+w2+w3
                if s<=1.0+1e-9:
                    yield (float(w1),float(w2),float(w3),float(1.0-s))

def eval_same_pid_retrieval(S: torch.Tensor, ids: List[int], pid_list: List[str],
                            topk:int=50, exclude_self:bool=True):
    ranks=[]
    for i in ids:
        scores=S[i].clone()
        positives={j for j,p in enumerate(pid_list) if p==pid_list[i] and j!=i}
        if exclude_self: scores[i]=-1e9
        if not positives: continue
        order=torch.argsort(scores, descending=True).cpu().tolist()
        best=min(order.index(j)+1 for j in positives)
        ranks.append(best)
    if not ranks: return {"R@1":0,"R@5":0,"R@10":0,"MRR":0,"nDCG@10":0,"N":0}
    ranks=np.array(ranks)
    r1  = float(np.mean(ranks==1))
    r5  = float(np.mean(ranks<=5))
    r10 = float(np.mean(ranks<=10))
    mrr = float(np.mean(1.0/ranks))
    ndcg10 = np.mean([1.0 if r==1 else (1.0/np.log2(r) if 1<r<=10 else 0.0) for r in ranks])
    return {"R@1":r1,"R@5":r5,"R@10":r10,"MRR":mrr,"nDCG@10":float(ndcg10),"N":int(len(ranks))}

def search_best_weights(I:torch.Tensor, T:torch.Tensor, ids:List[int], pids:List[str], step=0.1):
    S_ii = cosim(I,I); S_it = cosim(I,T); S_ti = cosim(T,I); S_tt = cosim(T,T)
    best_key=None; best=None
    for w_ii,w_it,w_ti,w_tt in grid_weights(step):
        S = w_ii*S_ii + w_it*S_it + w_ti*S_ti + w_tt*S_tt
        m = eval_same_pid_retrieval(S, ids, pids, topk=TOPK_EVAL, exclude_self=EXCLUDE_SELF)
        key=(m["R@1"], m["MRR"], m["R@5"])
        if (best_key is None) or (key>best_key):
            best_key=key; best=((w_ii,w_it,w_ti,w_tt), m)
    return best

# -------------------- Pipeline --------------------
ds = PetImgText(CSV_PATH)
train_ids, val_ids, test_ids = make_or_load_split(ds)

iters_per_epoch = max(1, math.ceil(len(train_ids)/(BATCH_P*BATCH_K)))
sampler = PKSampler(ds, train_ids, P=BATCH_P, K=BATCH_K, iters=iters_per_epoch, seed=SPLIT_SEED)
loader  = DataLoader(ds, batch_sampler=sampler, num_workers=NUM_WORKERS, collate_fn=collate_multi,
                     pin_memory=PIN_MEMORY, persistent_workers=(NUM_WORKERS>0 and PERSISTENT), prefetch_factor=PREFETCH)

model = CLIPHeadOnly(CLIP_MODEL, CLIP_PRETRAINED, DEVICE, img_chunk=IMG_CHUNK, txt_chunk=TXT_CHUNK).to(DEVICE)

# Optimizer: head WD=WD, τ group no-WD & smaller LR
opt = torch.optim.AdamW([
    {"params": model.proj_img.parameters(), "weight_decay": WD},
    {"params": model.proj_txt.parameters(), "weight_decay": WD},
    {"params": [model.log_tau_ii, model.log_tau_it, model.log_tau_ti, model.log_tau_tt],
     "lr": LR * 0.25, "weight_decay": 0.0}
], lr=LR, weight_decay=0.0)

# Warmup + Cosine (epoch-level)
def get_epoch_lr(base_lr, epoch):
    if epoch <= WARMUP_EPOCHS:
        return base_lr * epoch / max(1, WARMUP_EPOCHS)
    t = (epoch - WARMUP_EPOCHS) / max(1, (EPOCHS - WARMUP_EPOCHS))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))

scaler = torch.amp.GradScaler("cuda", enabled=FP16)

best_key=None; stale=0
best_meta_path = os.path.join(OUT_DIR, "best_meta.json")

def current_alpha_tt(epoch:int)->float:
    if len(TT_SCHEDULE)==0: return ALPHA_TT
    last_v = ALPHA_TT
    for e in sorted(TT_SCHEDULE.keys()):
        if epoch >= e: last_v = TT_SCHEDULE[e]
    return last_v

for epoch in range(1, EPOCHS+1):
    # τ freeze early (optional)
    if TAU_FREEZE_EPOCHS > 0:
        req_grad = epoch > TAU_FREEZE_EPOCHS
        for p in [model.log_tau_ii, model.log_tau_it, model.log_tau_ti, model.log_tau_tt]:
            p.requires_grad_(req_grad)

    # set LR per epoch (warmup+cosine)
    epoch_lr = get_epoch_lr(LR, epoch)
    for i,g in enumerate(opt.param_groups):
        if i < 2:  g['lr'] = epoch_lr         # heads
        else:      g['lr'] = epoch_lr * 0.25  # taus

    alpha_tt = current_alpha_tt(epoch)

    model.train()
    pbar = tqdm(loader, desc=f"Train E{epoch}/{EPOCHS} (lr={epoch_lr:.2e}, alpha_tt={alpha_tt:.2f})")
    loss_avg=0; cnt=0
    for step, batch in enumerate(pbar, 1):
        with torch.amp.autocast("cuda", enabled=FP16):
            # ✅ single forward
            zi, zt, zi_base, zt_base, txt_to_sample, taus = model(batch, DEVICE)
            (L_ii, L_it, L_ti, L_tt), logs = contrastive_losses_multi_text(
                zi, zt, batch["pid"], txt_to_sample, taus
            )
            loss = ALPHA_II*L_ii + ALPHA_IT*L_it + ALPHA_TI*L_ti + alpha_tt*L_tt

        scaler.scale(loss/GRAD_ACC).backward()
        if step % GRAD_ACC == 0:
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        cnt+=1; loss_avg += float(loss.item())
        pbar.set_postfix(loss=f"{loss_avg/cnt:.4f}",
                         alpha_tt=f"{alpha_tt:.2f}",
                         tau_ii=f"{logs['tau_ii']:.3f}", tau_it=f"{logs['tau_it']:.3f}",
                         tau_ti=f"{logs['tau_ti']:.3f}", tau_tt=f"{logs['tau_tt']:.3f}")

    # ---- Validation
    model.eval()
    val_pids, val_I, val_T = embed_split_eval(val_ids, ds, model, DEVICE, text_pooling=TEXT_POOLING)
    (w_val, m_val) = search_best_weights(val_I, val_T, list(range(len(val_ids))), val_pids, step=GRID_STEP)
    print(f"[VAL] E{epoch} best w: {w_val}, metrics: {m_val}")

    key=(m_val["R@1"], m_val["MRR"], m_val["R@5"])
    if (best_key is None) or (key>best_key):
        best_key=key; stale=0
        ckpt = {
            "proj_img": model.proj_img.state_dict(),
            "proj_txt": model.proj_txt.state_dict(),
            "log_tau_ii": float(model.log_tau_ii.detach().cpu().item()),
            "log_tau_it": float(model.log_tau_it.detach().cpu().item()),
            "log_tau_ti": float(model.log_tau_ti.detach().cpu().item()),
            "log_tau_tt": float(model.log_tau_tt.detach().cpu().item()),
            "base": {"model": CLIP_MODEL, "pretrained": CLIP_PRETRAINED},
            "text_pooling": TEXT_POOLING,
            "val_best_weights": {"w_ii":w_val[0], "w_it":w_val[1], "w_ti":w_val[2], "w_tt":w_val[3]},
            "val_metrics": m_val, "epoch": epoch,
            "alpha_tt": alpha_tt
        }
        pt_path = os.path.join(OUT_DIR, f"epoch{epoch:03d}.pt")
        torch.save(ckpt, pt_path)
        with open(best_meta_path, "w") as f:
            json.dump({"path": pt_path, "val_metrics": m_val, "epoch": epoch, "alpha_tt": alpha_tt}, f, indent=2)
        print(f"[BEST] Updated → {pt_path}")
    else:
        stale += 1
        if stale >= EARLY_STOP:
            print(f"[EarlyStop] Stop at E{epoch} (no improve {EARLY_STOP} epochs).")
            break

# ---- Load best & TEST
with open(best_meta_path, "r") as f: meta = json.load(f)
state = torch.load(meta["path"], map_location=DEVICE)
model.proj_img.load_state_dict(state["proj_img"])
model.proj_txt.load_state_dict(state["proj_txt"])
model.log_tau_ii.data = torch.tensor(state["log_tau_ii"], device=DEVICE)
model.log_tau_it.data = torch.tensor(state["log_tau_it"], device=DEVICE)
model.log_tau_ti.data = torch.tensor(state["log_tau_ti"], device=DEVICE)
model.log_tau_tt.data = torch.tensor(state["log_tau_tt"], device=DEVICE)

def eval_with_weights(I,T,pids,w):
    S = w[0]*(I@I.T) + w[1]*(I@T.T) + w[2]*(T@I.T) + w[3]*(T@T.T)
    ids = list(range(len(pids)))
    return eval_same_pid_retrieval(S, ids, pids, topk=TOPK_EVAL, exclude_self=EXCLUDE_SELF)

train_pids, train_I, train_T = embed_split_eval(train_ids, ds, model, DEVICE, text_pooling=TEXT_POOLING)
val_pids,   val_I,   val_T   = embed_split_eval(val_ids,   ds, model, DEVICE, text_pooling=TEXT_POOLING)
test_pids,  test_I,  test_T  = embed_split_eval(test_ids,  ds, model, DEVICE, text_pooling=TEXT_POOLING)

w_best = (state["val_best_weights"]["w_ii"], state["val_best_weights"]["w_it"],
          state["val_best_weights"]["w_ti"], state["val_best_weights"]["w_tt"])

m_train = eval_with_weights(train_I, train_T, train_pids, w_best)
m_val   = eval_with_weights(val_I,   val_T,   val_pids,   w_best)
m_test  = eval_with_weights(test_I,  test_T,  test_pids,  w_best)

print("\n[Best Weights @VAL]", w_best)
print("[Train]", m_train)
print("[Val]  ", m_val)
print("[Test] ", m_test)

with open(os.path.join(OUT_DIR, "summary_results.json"), "w") as f:
    json.dump({
        "best_pt": meta["path"],
        "best_weights": {"w_ii":w_best[0], "w_it":w_best[1], "w_ti":w_best[2], "w_tt":w_best[3]},
        "train_metrics": m_train, "val_metrics": m_val, "test_metrics": m_test,
        "config": {
            "model": CLIP_MODEL, "pretrained": CLIP_PRETRAINED,
            "TEXT_POOLING": TEXT_POOLING, "GRID_STEP": GRID_STEP,
            "TOPK_EVAL": TOPK_EVAL, "EXCLUDE_SELF": EXCLUDE_SELF,
            "ratios": [R_TRAIN, R_VAL, R_TEST],
            "BATCH_P": BATCH_P, "BATCH_K": BATCH_K,
            "LR": LR, "WD": WD, "WARMUP_EPOCHS": WARMUP_EPOCHS,
            "TT_SCHEDULE": TT_SCHEDULE
        }
    }, f, indent=2)

print("\n[Done] Outputs saved to:", OUT_DIR)
