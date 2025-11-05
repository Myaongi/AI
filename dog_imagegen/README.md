# 🐶 Dog Image Gen Service (Gemini): 강아지 이미지 생성

한국어 입력(품종, 색상, 기타 텍스트)만으로  
Google **Gemini** 이미지 모델을 이용해 **전신 + 여백의 4:3 강제 PNG** 이미지를 생성합니다.  
(내부적으로 영어로 번역·정제 후 시각 정보만 사용)

---

## 🚀 주요 기능
- 한국어 입력만으로 강아지 이미지 생성  
- **전신(머리~발끝)**, **여백 있는 와이드 샷** 자동 유도  
- **LEFT / RIGHT** 위치(예: 왼쪽 귀 점) 정확히 반영  
- **4:3 비율 강제** (패딩 없이 중앙 크롭)  

---

## ⚙️ 설치
```bash
pip install -r requirements.txt
````

---

## 🔑 환경 변수 설정

```bash
# macOS / Linux
export GEMINI_API_KEY="YOUR_API_KEY"

# Windows PowerShell
setx GEMINI_API_KEY "YOUR_API_KEY"
```

---

## 🧩 사용 예시

```python
from dog_image_gen_service import DogImageGenerator
from io import BytesIO
from PIL import Image

gen = DogImageGenerator()
png = gen.generate_from_raw_ko(
    breed="말티즈",
    colors="흰색",
    others="석촌 호수에서 말티즈를 발견했어요! 아이가 사람을 잘 따르더라구요ㅠ 왼쪽 귀에 갈색 점이 있었고 빨간색 리본을 하고 있었습니다~"
)

Image.open(BytesIO(png)).save("dog_4x3.png")
print("✅ 저장 완료: dog_4x3.png")
```

---

## 🖥️ 터미널 실행 (옵션)

```bash
python dog_image_gen_service.py \
  --breed "말티즈" \
  --colors "흰색, 왼쪽 귀 갈색 점" \
  --others "빨간 리본, 전신 여유 있게" \
  --out dog_4x3.png
```

---

## 📦 파일 구성

```
dog_image_gen_service.py   # 메인 코드
requirements.txt           # 라이브러리 의존성
README.md                  # 설명서
```

---

## 📘 참고

* 모델: `gemini-2.5-flash-image`
* 기본 출력 비율: **4:3 (가로형)**
* 프레임 내 주인공 크기: 약 **35%** (멀리서 보이는 비율)
