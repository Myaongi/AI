## 🐶 매칭 서비스: 잃어버렸어요 - 목격했어요 게시물

### 🧩 전체 구조

```
입력(이미지+설명)
   ↓
YOLO Crop ✂️ → LLM 문장 정제 🪄
   ↓
Fine-tuned CLIP 임베딩 🤖
   ↓
유사도 계산 (I-I, I-T, T-I, T-T)
   ↓
가중합 점수 + threshold 판정 💕
```

---

### ⚙️ 주요 구성 모듈

| 모듈                       | 역할                           | 비고                       |
| ------------------------ | ---------------------------- | ------------------------ |
| `clipper.py`             | Fine-tuned CLIP 로더 + 임베딩 추출  | 자동으로 `.pt` 로드            |
| `yolo_crop.py`           | YOLO로 강아지 부분만 크롭             | 마진 포함                    |
| `llm_client.py`          | 품종·색상·특징을 3문장으로 정제           | 내부 LLM API 호출            |
| `service.py`             | 백엔드에서 쓰기 쉬운 wrapper          | (이미지 bytes → 임베딩, 점수 계산) |
| `pipeline_embed.py`      | end-to-end 파이프라인 (embed 생성용) | 내부 서비스용                  |
| `pipeline_similarity.py` | 유사도 계산 + 통과 여부 반환            | 점수 기준선 체크                |
| `config.py`              | 환경변수 및 가중치 설정                | Fine-tuned 헤드 로딩         |

---

### 💾 Fine-tuned 가중치 자동 로드

```bash
FT_WEIGHTS_URL="https://github.com/eonjilim/Myaongi_AI/releases/download/untagged-c51a7ed963c83380ced0/matching_best.pt"
```
---

### 🧠 백엔드에서 사용하는 방법

백엔드는 그냥 아래 함수들만 호출하면 됩니다. ✨

---

#### 🐾 1️⃣ 임베딩 생성 (이미지 + 설명)

```python
from app.service import build_embeddings

# 파일 업로드 후 받은 image_bytes, form 데이터(breed, colors, features)
sents, emb_img, emb_txt = build_embeddings(image_bytes, "Maltese", "white", "brown spot on left ear")
```

💡 반환:

```python
{
  "sentences": ["A Maltese dog with white coat.", ...],
  "image_embedding": [...512 float...],
  "text_embedding":  [...512 float...]
}
```

---

#### 🐾 2️⃣ 유사도 계산

```python
from app.service import score_pair

s4, score = score_pair(emb_a_img, emb_a_txt, emb_b_img, emb_b_txt, weights=(0.2, 0.0, 0.0, 0.8))
print(s4, score)
```

💡 반환 예시:

```
s_ii = 0.5429
s_it = 0.0971
s_ti = 0.0794
s_tt = 0.7118
weighted score = 0.6780 → pass ✅
```

---

#### 🐾 3️⃣ API로 쓰고 싶을 때

| 엔드포인트                  | 설명                   |
| ---------------------- | -------------------- |
| `POST /normalize`      | LLM으로 3문장 정제         |
| `POST /embed`          | 이미지+텍스트 → 임베딩 계산     |
| `POST /score`          | 두 게시물(A, B) 간 유사도 계산 |
| `POST /pass-threshold` | 점수가 기준선 넘는지 판정       |

**예시 요청:**

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "emb_a_image": [...],
    "emb_a_text": [...],
    "emb_b_image": [...],
    "emb_b_text": [...],
    "weights": [0.2, 0.0, 0.0, 0.8]
  }'
```

**응답:**

```json
{
  "s_ii": 0.54,
  "s_it": 0.09,
  "s_ti": 0.07,
  "s_tt": 0.71,
  "score": 0.68
}
```

---

### 🧩 환경변수 예시 (.env)

```bash
# CLIP base
CLIP_MODEL=ViT-L-14
CLIP_PRETRAINED=datacomp_xl_s13b_b90k

# fine-tuned head (.pt)
FT_WEIGHTS_URL=https://github.com/eonjilim/Myaongi_AI/releases/download/untagged-c51a7ed963c83380ced0/matching_best.pt

# matching threshold
SIM_THRESHOLD=0.35

# similarity weights
W_II=0.2
W_IT=0.0
W_TI=0.0
W_TT=0.8
```
