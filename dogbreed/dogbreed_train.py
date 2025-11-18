# 품종 분류 모델 학습 코드
import os, json, time, random, warnings, math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# progressing_bar
from IPython.display import display, HTML
import matplotlib.pyplot as plt

import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV2

# ------------------------ 기본 세팅 ------------------------
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ========== 경로 ==========
SPLIT_ROOT          = Path("/workspace/proj_gc/dataset/split_105")
TRAIN_VARIANTS_ROOT = Path("/workspace/proj_gc/dataset/processed_105/train_variants")
VARIANT_NAME        = "crops_pad16_jit10"
TRAIN_ROOT          = TRAIN_VARIANTS_ROOT / VARIANT_NAME

VAL_ROOT            = Path("/workspace/proj_gc/dataset/processed_105/eval_pad16_jit0_val")
TEST_ROOT           = Path("/workspace/proj_gc/dataset/processed_105/eval_pad16_jit0_test")

EXP_ROOT            = Path("/workspace/proj_gc/exp_final/v2m_480_pad16_jit10_onebar_unfreeze")
EXP_ROOT.mkdir(parents=True, exist_ok=True)

assert (SPLIT_ROOT/"train.csv").exists() and (SPLIT_ROOT/"val.csv").exists() and (SPLIT_ROOT/"test.csv").exists()
for p in [TRAIN_ROOT, VAL_ROOT, TEST_ROOT]:
    assert p.exists(), f"경로 없음: {p}"

# ========== 설정 ==========
MODEL_NAME   = "tf_efficientnetv2_m_in21ft1k"
IMG_SIZE     = 480
EPOCHS       = 50
BATCH_SIZE   = 24

BASE_LR      = 1e-4
BACKBONE_LR_FACTOR = 0.20   # 0.25 → 0.20
WEIGHT_DECAY = 0.08
LABEL_SMOOTH = 0.10

USE_MIXCUT   = True
MIXUP_ALPHA  = 0.4
CUTMIX_ALPHA = 1.0
MIXCUT_PROB  = 0.8
MIXCUT_OFF_LAST = 5          # 테일 5ep는 기본적으로 OFF

# --- MixOff/ES 가드 하이퍼 ---
ADAPTIVE_NOMIX_ON_PATIENCE = 3   # 인내심이 이 값 이하면 즉시 OFF
NOMIX_AT_FRAC = 0.40             # 전체의 70% 시점부터 OFF
MIN_NO_MIX_EPOCHS = 2            # OFF 이후 최소 non-mix epoch 보장

USE_EMA      = True
EMA_DECAY    = 0.9997
AMP          = True
CHANNELS_LAST= True

NUM_WORKERS  = 8
PIN_MEMORY   = True

# ---------------- 점진적 언프리즈 & EarlyStopping ----------------
WARMUP_EPOCHS = 5
UNFREEZE_STAGE1_AT = WARMUP_EPOCHS + 1  # blocks.6
UNFREEZE_STAGE2_AT = None
UNFREEZE_STAGE3_AT = None

ES_PATIENCE   = 7
ES_MIN_DELTA  = 1e-4

# reproducibility
SEED = 42
def set_all_seeds(s:int):
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_all_seeds(SEED)

# ------------------------ 유틸 ------------------------
def ensure_exist_or_filter(csv_path: Path, img_root: Path):
    df = pd.read_csv(csv_path)
    keep = [(img_root / p).exists() for p in df["rel_path"]]
    miss = int((~pd.Series(keep)).sum())
    if miss:
        print(f"[WARN] {img_root} : {csv_path.name} 기준 누락 {miss}개 → 필터링")
        df = df.loc[keep].reset_index(drop=True)
    else:
        print(f"[OK] {img_root} : {csv_path.name} 전부 존재")
    return df

def fmt_t(sec: int):
    m, s = divmod(int(sec), 60); h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

def move_batch_to_device(batch, device, channels_last=False):
    x, y = batch
    x = x.to(device, non_blocking=True)
    if channels_last: x = x.to(memory_format=torch.channels_last)
    y = torch.as_tensor(y, device=device)
    return x, y

def count_trainable_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ------------------------ 정사각 변환 ------------------------
class SquarePadResize(torch.nn.Module):
    def __init__(self, out_size, interpolation=InterpolationMode.BICUBIC, fill=0):
        super().__init__()
        self.out_size = out_size
        self.interp = interpolation
        self.fill = fill
    def __call__(self, img):
        w, h = img.size
        s = max(w, h)
        pad_l = (s - w) // 2
        pad_t = (s - h) // 2
        pad_r = s - w - pad_l
        pad_b = s - h - pad_t
        img = ImageOps.expand(img, border=(pad_l, pad_t, pad_r, pad_b), fill=self.fill)
        try:
            img = TF.resize(img, [self.out_size, self.out_size], interpolation=self.interp, antialias=True)
        except TypeError:
            img = TF.resize(img, [self.out_size, self.out_size], interpolation=self.interp)
        return img

# ------------------------ ToTensor ------------------------
class ToTensorNoNumpy(torch.nn.Module):
    def __call__(self, img: Image.Image):
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size; c = 3
        buf = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        tensor = buf.view(h, w, c).permute(2,0,1).contiguous().float().div(255.0)
        return tensor

# ------------------------ Dataset ------------------------
class CSVDataset(Dataset):
    def __init__(self, csv_df: pd.DataFrame, img_root: Path, class_to_idx: dict, transform=None):
        self.paths  = csv_df["rel_path"].tolist()
        labels_series = csv_df["label"].map(class_to_idx)
        if labels_series.isna().any():
            missing = csv_df.loc[labels_series.isna(), "label"].unique().tolist()
            raise ValueError(f"Unknown labels in CSV: {missing}")
        self.labels = labels_series.astype(int).tolist()
        self.root   = Path(img_root)
        self.t = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.root / self.paths[i]).convert("RGB")
        img = self.t(img) if self.t else img
        return img, self.labels[i]

# ------------------------ Transform ------------------------
mean, std = [0.485,0.456,0.406], [0.229,0.224,0.225]
sq = SquarePadResize(IMG_SIZE)
to_tensor_safe = ToTensorNoNumpy()

train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.05,0.05,0.05,0.03),
    sq,
    to_tensor_safe,
    transforms.Normalize(mean, std),
    transforms.RandomErasing(p=0.20),
])
eval_tf = transforms.Compose([
    sq,
    to_tensor_safe,
    transforms.Normalize(mean, std),
])

# ------------------------ Data ------------------------
train_df = ensure_exist_or_filter(SPLIT_ROOT/"train.csv", TRAIN_ROOT)
val_df   = ensure_exist_or_filter(SPLIT_ROOT/"val.csv",   VAL_ROOT)
test_df  = ensure_exist_or_filter(SPLIT_ROOT/"test.csv",  TEST_ROOT)

all_labels = pd.concat([train_df["label"], val_df["label"], test_df["label"]], axis=0)
classes = sorted(all_labels.unique()); num_classes = len(classes)
class_to_idx = {c:i for i,c in enumerate(classes)}
print(f"[INFO] classes={num_classes}")

train_ds = CSVDataset(train_df, TRAIN_ROOT, class_to_idx, transform=train_tf)
val_ds   = CSVDataset(val_df,   VAL_ROOT,   class_to_idx, transform=eval_tf)
test_ds  = CSVDataset(test_df,  TEST_ROOT,  class_to_idx, transform=eval_tf)
train_clean_ds = CSVDataset(train_df, TRAIN_ROOT, class_to_idx, transform=eval_tf)

# ★ 균형 데이터셋: Sampler 제거, shuffle=True 사용
train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)
train_clean_loader = DataLoader(
    train_clean_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)

# ------------------------ Model ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model(
    MODEL_NAME, pretrained=True, num_classes=num_classes,
    drop_rate=0.3, drop_path_rate=0.3
).to(device)
if CHANNELS_LAST: model = model.to(memory_format=torch.channels_last)

# 0) 헤드만 학습
for n,p in model.named_parameters():
    p.requires_grad = ("classifier" in n or "head" in n or "conv_head" in n or "bn2" in n)

def set_trainable_by_keys(m, keys, accumulative=True):
    if not accumulative:
        for p in m.parameters(): p.requires_grad = False
    hit = 0
    for n, p in m.named_parameters():
        if any(k in n for k in keys):
            p.requires_grad = True
            hit += p.numel()
    return hit

# 단계별 키(누적 적용)
STAGE1_KEYS = ("blocks.6", "conv_head", "bn2", "classifier", "head")
STAGE2_KEYS = ("blocks.5",)
STAGE3_KEYS = ("blocks.4",)

def make_optimizer():
    head, bb = [], []
    for n,p in model.named_parameters():
        if not p.requires_grad: continue
        if any(k in n for k in ('classifier','head','conv_head','bn2')):
            head.append(p)
        else:
            bb.append(p)
    return torch.optim.AdamW(
        [{"params": bb,  "lr": BASE_LR * BACKBONE_LR_FACTOR},
         {"params": head,"lr": BASE_LR}],
        weight_decay=WEIGHT_DECAY
    )

optimizer = make_optimizer()
scaler = torch.cuda.amp.GradScaler(enabled=AMP)
ema = ModelEmaV2(model, decay=EMA_DECAY, device=device) if USE_EMA else None
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2, threshold=1e-4, min_lr=1e-6
)
mixup_fn = Mixup(mixup_alpha=MIXUP_ALPHA, cutmix_alpha=CUTMIX_ALPHA, prob=MIXCUT_PROB,
                 label_smoothing=0.0, num_classes=num_classes, mode='batch') if USE_MIXCUT else None

def pick_criterion(train_mode: bool, use_mix: bool):
    if not train_mode: return nn.CrossEntropyLoss()
    if use_mix:        return SoftTargetCrossEntropy()
    return nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

# ------------------------ 한 줄 진행바 (epoch당 1개) ----------------
def _render_epoch_bar_line(ep, total_epochs, i, total_steps, avg_loss, t0):
    pct = 0 if total_steps == 0 else int(i * 100 / total_steps)
    elapsed = time.time() - t0
    left = max(0, total_steps - i)
    eta = "--:--" if i == 0 else fmt_t(int(elapsed / max(1,i) * left))
    bar_len = 28
    fill = int(bar_len * pct / 100)
    bar = "█" * fill + "░" * (bar_len - fill)
    return f"[Train] Epoch {ep}/{total_epochs}: {pct:3d}% | {i}/{total_steps} | {bar} | loss {avg_loss:.4f} | ETA {eta}"

def _epoch_bar_start():
    return display(HTML("<pre style='margin:0'></pre>"), display_id=True)

def _epoch_bar_update(handle, line: str):
    handle.update(HTML(f"<pre style='margin:0'>{line}</pre>"))

# ------------------------ Train / Eval ------------------------
def train_one_epoch(ep, loader, model_ref, device, opt, mix=None, scaler=None):
    model_ref.train(True)
    crit = pick_criterion(True, mix is not None)
    total, loss_sum = 0, 0.0
    Y, P = [], []

    handle = _epoch_bar_start()
    t0 = time.time()
    last_tick = 0.0

    for i, batch in enumerate(loader, 1):
        x, y = move_batch_to_device(batch, device, channels_last=CHANNELS_LAST)
        if (mix is not None) and (x.size(0) % 2 == 1):
            x = x[:-1]; y = y[:-1]

        y_in = y
        if mix is not None:
            x, y_in = mix(x, y)
            if y_in.ndim == 1:
                y_in = F.one_hot(y_in, num_classes=num_classes).float()

        with torch.cuda.amp.autocast(enabled=AMP):
            logits = model_ref(x)
            loss = crit(logits, y_in if mix is not None else y)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if ema is not None: ema.update(model_ref)

        bs = x.size(0); loss_sum += loss.item()*bs; total += bs
        Y.extend((y if y.ndim==1 else y.argmax(1)).detach().cpu().tolist())
        P.extend(logits.argmax(1).detach().cpu().tolist())

        now = time.time()
        if (now - last_tick) > 0.25 or i == len(loader):
            avg_loss = loss_sum / max(1, total)
            line = _render_epoch_bar_line(ep, EPOCHS, i, len(loader), avg_loss, t0)
            _epoch_bar_update(handle, line)
            last_tick = now

    final_line = _render_epoch_bar_line(ep, EPOCHS, len(loader), len(loader), loss_sum/max(1,total), t0)
    print(final_line)

    acc = accuracy_score(Y,P); f1 = f1_score(Y,P, average="macro")
    return loss_sum/max(1,total), acc, f1

@torch.no_grad()
def evaluate(loader, model_ref, device):
    model_ref.eval()
    crit = nn.CrossEntropyLoss()
    total, loss_sum, top5_correct = 0, 0.0, 0
    Y, P = [], []
    for batch in loader:
        x, y = move_batch_to_device(batch, device, channels_last=CHANNELS_LAST)
        with torch.cuda.amp.autocast(enabled=AMP):
            logits = model_ref(x)
            loss = crit(logits, y)
        bs = x.size(0); loss_sum += loss.item()*bs; total += bs
        pred = logits.argmax(1)
        Y.extend(y.detach().cpu().tolist()); P.extend(pred.detach().cpu().tolist())
        top5_correct += (logits.topk(5,1).indices == y.view(-1,1)).any(1).sum().item()
    acc = accuracy_score(Y,P); f1 = f1_score(Y,P, average="macro"); top5 = top5_correct/max(1,total)
    return loss_sum/max(1,total), acc, f1, top5

# ------------------------ 학습 루프 ------------------------
history = []
best = {"val_f1": -1.0, "path": EXP_ROOT/"best.pt"}
pat_left = ES_PATIENCE

# MixOff 상태 변수
nomix_epoch_start = None
mix_forced_off = False

print(f"[info] variant={VARIANT_NAME}  [img={IMG_SIZE}]")
print(f"[info] device={device}, model={MODEL_NAME}, epochs={EPOCHS}\n")

for ep in range(1, EPOCHS+1):

    # === 점진적 언프리즈 스케줄 ===
    if ep == UNFREEZE_STAGE1_AT:
        set_trainable_by_keys(model, STAGE1_KEYS, accumulative=True)
        optimizer = make_optimizer()
        print(f"→ Unfreeze stage1 {STAGE1_KEYS}; trainable params: {count_trainable_params(model):,}")

    # === MixUp/CutMix OFF 트리거 (테일/진행률/인내심) ===
    tail_trigger = (ep > EPOCHS - MIXCUT_OFF_LAST)
    frac_trigger = (ep >= int(math.ceil(EPOCHS * NOMIX_AT_FRAC)))
    pat_trigger  = (pat_left <= ADAPTIVE_NOMIX_ON_PATIENCE)

    if (tail_trigger or frac_trigger or pat_trigger) and not mix_forced_off:
        mix_forced_off = True
        nomix_epoch_start = ep
        print(f"→ MixCut OFF at ep={ep} (tail={tail_trigger}, frac={frac_trigger}, patience={pat_trigger})")

    use_mix = (None if mix_forced_off else mixup_fn)

    # 1) Train — 막대 1개
    tr_loss, tr_acc, tr_f1 = train_one_epoch(ep, train_loader, model, device, optimizer, mix=use_mix, scaler=scaler)

    # 2) 평가 모델(EMA 우선)
    eval_model = ema.module if (ema is not None) else model

    # 3) Val / Train(clean) — 막대 없이 요약만
    trc_loss, trc_acc, trc_f1, _ = evaluate(train_clean_loader, eval_model, device)
    va_loss,  va_acc,  va_f1,  va_top5 = evaluate(val_loader, eval_model, device)

    # 4) 스케줄러/로그/베스트 저장/조기종료
    scheduler.step(va_f1)
    history.append({
        "epoch": ep, "lr": optimizer.param_groups[0]["lr"],
        "train_loss": tr_loss, "train_acc": tr_acc, "train_f1": tr_f1,
        "val_loss": va_loss, "val_acc": va_acc, "val_f1": va_f1, "val_top5": va_top5,
        "train_clean_loss": trc_loss, "train_clean_acc": trc_acc, "train_clean_f1": trc_f1
    })

    print(f"\nEpoch {ep}/{EPOCHS}")
    print(f"  train(clean): acc {trc_acc:0.4f} f1 {trc_f1:0.4f}")
    print(f"  train       : loss {tr_loss:0.4f} acc {tr_acc:0.4f} f1 {tr_f1:0.4f}")
    print(f"  val         : loss {va_loss:0.4f} acc {va_acc:0.4f} f1 {va_f1:0.4f} top5 {va_top5:0.4f}")

    improved = va_f1 > best["val_f1"] + ES_MIN_DELTA
    if improved:
        best["val_f1"] = va_f1
        to_save = (ema.module.state_dict() if ema is not None else model.state_dict())
        torch.save({"model": to_save, "classes": classes}, best["path"])
        pat_left = ES_PATIENCE
        print(f"  ↳ saved best to {best['path']} (val_macroF1={va_f1:.4f})")
    else:
        pat_left -= 1
        print(f"  ↳ no improve (patience left: {pat_left})")

    # --- ES 가드: non-mix 최소 N epoch 보장 ---
    if pat_left <= 0:
        nonmix_trained = 0 if nomix_epoch_start is None else (ep - nomix_epoch_start + 1)
        if nonmix_trained < MIN_NO_MIX_EPOCHS:
            need = MIN_NO_MIX_EPOCHS - nonmix_trained
            print(f"  ↳ ES 보류: non-mix {need} epoch 더 진행")
            pat_left = 1    # 한 에폭 더 돌리기 위해 임시 연장
        else:
            print("  ↳ Early stopping (after non-mix fine-tune).")
            break

with open(EXP_ROOT/"history.json", "w") as f: json.dump(history, f, indent=2)

# ------------------------ 곡선 저장 ------------------------
def plot_curve(key_tr, key_va, out):
    xs = [h["epoch"] for h in history]
    tr = [h[key_tr] for h in history]; va = [h[key_va] for h in history]
    plt.figure(); plt.plot(xs, tr, label=key_tr); plt.plot(xs, va, label=key_va)
    plt.xlabel("epoch"); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(EXP_ROOT/out, dpi=150); plt.close()

plot_curve("train_f1", "val_f1", "curve_f1.png")
plot_curve("train_acc","val_acc","curve_acc.png")
plot_curve("train_loss","val_loss","curve_loss.png")

# ------------------------ Best 로드 후 Test ------------------------
ckpt_path = EXP_ROOT/"best.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt["model"])
model.to(device).eval()

@torch.no_grad()
def evaluate_simple(loader, model_ref):
    model_ref.eval(); crit = nn.CrossEntropyLoss()
    total, loss_sum, top5_correct = 0, 0.0, 0; Y, P = [], []
    for batch in loader:
        x, y = move_batch_to_device(batch, device, channels_last=CHANNELS_LAST)
        with torch.cuda.amp.autocast(enabled=AMP):
            logits = model_ref(x)
            loss = crit(logits, y)
        bs = x.size(0); loss_sum += loss.item()*bs; total += bs
        pred = logits.argmax(1); Y.extend(y.detach().cpu().tolist()); P.extend(pred.detach().cpu().tolist())
        top5_correct += (logits.topk(5,1).indices == y.view(-1,1)).any(1).sum().item()
    acc = accuracy_score(Y,P); f1 = f1_score(Y,P, average="macro"); top5 = top5_correct/max(1,total)
    return loss_sum/max(1,total), acc, f1, top5, (Y,P)

te_loss, te_acc, te_f1, te_top5, (y_true, y_pred) = evaluate_simple(test_loader, model)
print(f"\n[TEST] loss {te_loss:.4f} | acc {te_acc:.4f} top5 {te_top5:.4f} | macro-F1 {te_f1:.4f}")

cm = confusion_matrix(y_true, y_pred)
np.save(EXP_ROOT/"confusion_matrix.npy", cm)
with open(EXP_ROOT/"classification_report.txt", "w") as f:
    f.write(classification_report(y_true, y_pred, target_names=classes, digits=4))

best_row = max(history, key=lambda h: h["val_f1"])
row = {
    "variant": VARIANT_NAME, "img": IMG_SIZE, "model": MODEL_NAME, "seed": SEED,
    "best_epoch": best_row["epoch"],
    "val_acc": best_row["val_acc"], "val_f1": best_row["val_f1"], "val_top5": best_row["val_top5"],
    "train_clean_acc": best_row["train_clean_acc"], "train_clean_f1": best_row["train_clean_f1"],
    "test_acc": te_acc, "test_f1": te_f1, "test_top5": te_top5
}
with open(EXP_ROOT/"metrics.json", "w") as f: json.dump(row, f, indent=2)

print("\nSaved to:", EXP_ROOT)
for fn in ["best_2.pt","history_2.json","metrics_2.json","curve_f1_2.png","curve_acc_2.png","curve_loss_2.png","confusion_matrix_2.npy","classification_report_2.txt"]:
    print(" -", EXP_ROOT/fn)