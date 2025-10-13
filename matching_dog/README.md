# matching_dog — YOLO + CLIP Matching API

## Features
- LLM 정제(품종/색상/자유기술문 → 문장 3개)
- YOLO 크롭(conf=0.30, imgsz=512, margin=0.18, class=dog)
- CLIP 임베딩 (ViT-B/32, laion2b_s34b_b79k; text pooling=avg)
- 4유형 유사도(I-I, I-T, T-I, T-T) + 가중합

## Run
```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000

### 한 줄 요약(파일별)
- **config.py**: 파라미터 집합(LLM/YOLO/CLIP/가중치/임계값)  
- **schemas.py**: 요청/응답 형식(Pydantic)  
- **llm_client.py**: 입력(품종/색/특징) → LLM 호출해 3문장 생성(폴백 포함)  
- **yolo_crop.py**: YOLO로 개 검출 → **margin=0.18**로 안전 크롭  
- **clipper.py**: CLIP zero-shot 임베딩(이미지/텍스트, 풀링 포함)  
- **similarity.py**: 4가지 코사인 유사도 + 가중합 점수  
- **service.py**: 위 모듈들 묶어서 임베딩 생성/페어 스코어 계산  
- **main.py**: FastAPI 라우팅(`/normalize`, `/embed`, `/score`, `/pass-threshold`)  
- **requirements.txt**: 의존성  
- **README.md**: 실행/엔드포인트/환경변수 요약
