🐶 DogBreed 모델: 강아지 품종 분류
1) 개요

업로드된 강아지 이미지 → 품종 분류 + 믹스견 판정 수행.

UI에는 prediction.decision만 노출하면 됨.
(판정 근거는 내부 로그/통계 용도)

2) 사용 절차
초기화 (서버 부팅 시 1회)
from dogbreed import init
init(warmup=True)  # 가중치 확보 + 모델 로드 + (옵션) CUDA 예열

추론 호출
from dogbreed import predict
out = predict(image, return_topk=5)  # image: 경로(str) / 바이트(bytes) / PIL.Image


return_topk: 상위 K 후보가 필요할 때만 조정(기본 3~5)

3) 출력 스키마 (Dict)
{
  "prediction": {
    "decision": "웰시코기",          // ✅ 최종 표시 값 (또는 "믹스")
    "decision_type": "breed",       // "breed" | "mixed"
    "top1": {
      "index": 12, "label_en": "Pembroke", "label_ko": "웰시코기", "prob": 0.87
    },
    "topk": [
      { "index": 12, "label_en": "Pembroke", "label_ko": "웰시코기", "prob": 0.87 },
      { "index": 34, "label_en": "Beagle",   "label_ko": "비글",     "prob": 0.06 }
    ],
    "reasons": {
      "mix_rules": {
        "trigger": false,
        "H": 0.89, "H_norm": 0.38, "p1": 0.87, "p2": 0.06,
        "margin": 0.81,
        "thresholds": { "H_norm_th": 0.62, "p1_th": 0.55, "margin_th": 0.15 }
      }
    }
  },
  "boxes": {
    "selected": [x1, y1, x2, y2],
    "conf": 0.92,
    "detected": true,
    "yolo_time_ms": 37.8,
    "image_size": [W, H]
  },
  "meta": {
    "pad": 0.16,
    "input_size": 480,
    "model": "tf_efficientnetv2_m_in21ft1k",
    "num_classes": 100,
    "device": "cuda:0"
  }
}

4) 백엔드에서 실제로 쓰는 값

최종 표시 품종명:

breed_name = out["prediction"]["decision"]  # UI에 그대로 노출


(선택) 내부 저장/로그:

decision_type = out["prediction"]["decision_type"]          # "breed" | "mixed"
reasoning     = out["prediction"]["reasons"]["mix_rules"]   # 임계값·엔트로피 등 근거

5) 운영 팁

초기화 init(warmup=True)는 프로세스 시작 시 1회만 호출.

가중치(URL)·클래스 수는 코드 안에 상수로 되어 있으니, 바뀌면 predictor.py의 상수만 수정.

에러/디버깅 시 meta, boxes 전체를 로그로 남기면 원인 추적 쉬움.

6) 한 줄 가이드 (백엔드 팀용)

“내부 판정은 모듈이 처리하니, 우리는 prediction.decision만 사용자에게 보여주면 됩니다.
decision_type/reasons는 선택적으로 로그에만 저장하세요.”
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
=======
# AI
>>>>>>> c7c8ce6 (Add README and requirements sets)
