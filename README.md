# 강아지 품종 분류
## DogBreed Model (YOLOv8 + EfficientNetV2-M)

- 입력 이미지에서 강아지 탐지 → 크롭 → 품종 분류 → 믹스(혼종) 판정 수행
- 가중치(.pt)는 GitHub Release에서 최초 1회 자동 다운로드 후 assets/에 캐시됨

### Install

- Torch 먼저, 그다음 base 설치 (환경에 맞는 파일 선택)

### GPU (CUDA 12.8)
```bash
pip install -r requirements/requirements-gpu-cu128.txt
pip install -r requirements/requirements-base.txt
```

### CPU
```bash
pip install -r requirements/requirements-cpu.txt
pip install -r requirements/requirements-base.txt
```

---
### Response (JSON 예시)

```jsonc
{
  "prediction": {
    "decision": "치와와",                 // 최종 판정 ("믹스" or 품종명)
    "decision_type": "breed",            // "breed" | "mixed"
    "top1": {                            // 1위 후보
      "index": 0,                        // 분류 클래스 인덱스
      "label_en": "Chihuahua",           // 영문 라벨
      "label_ko": "치와와",              // 한글 라벨(ko_mapping.csv 매핑)
      "prob": 0.59                       // softmax 확률
    },
    "topk": [ /* 동일 구조 */ ],
    "reasons": { "D_entropy": { "trigger": false, "H": 1.23, "threshold": 2.0 } }
  },
  "boxes": {
    "selected": [12.3, 45.6, 123.4, 200.1],
    "conf": 0.87, "detected": true, "yolo_time_ms": 12.4, "image_size": [640, 480]
  },
  "meta": {
    "pad": 0.16, "input_size": 480, "model": "tf_efficientnetv2_m_in21ft1k",
    "num_classes": 105, "device": "cuda:0"
  }
}
```
---
## Notes

- 서버 부팅 시 init(warmup=True) 한 번 호출하면 콜드스타트 없음.
- YOLO 가중치가 "yolov8s.pt"이면 Ultralytics가 자동 다운로드.
- 가중치는 Git에 올리지 않음(.gitignore 처리). Release URL/SHA256은 predictor.py에서 관리.