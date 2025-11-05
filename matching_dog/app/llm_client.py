import time
from typing import List
import requests
from . import config

_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)


def _pad_to3(sents: List[str]) -> List[str]:
    """비어있거나 3개 미만이면 안전하게 3개로 패딩."""
    clean = [s.strip() for s in (sents or []) if isinstance(s, str) and s.strip()]
    while len(clean) < 3:
        clean.append(clean[-1] if clean else "A dog.")
    return clean[:3]


def normalize_to_3_sentences(breed: str, colors: str, features: str) -> List[str]:
    """
    새 API 스펙:
      POST /api/v1/embed/normalize
      body: {"breed": "...", "colors": "...", "features": "..."}
      resp: {"sentences": ["...", "...", "..."]}

    실패/예외 시 간단한 폴백 3문장 반환.
    """
    payload = {"breed": breed or "", "colors": colors or "", "features": features or ""}

    last_err = None
    for t in range(config.LLM_MAX_RETRIES):
        try:
            r = _session.post(
                config.LLM_URL,
                json=payload,  # 새 스펙: JSON 바디
                headers={"Accept": "application/json"},
                timeout=config.LLM_TIMEOUT,
            )
            r.raise_for_status()
            data = r.json()

            # 권장 스키마
            if isinstance(data, dict) and isinstance(data.get("sentences"), list):
                return _pad_to3(data["sentences"])

            # 혹시 서버가 구 스키마로 응답해도 최대한 복구
            if isinstance(data, dict):
                result = data.get("result", data)
                if isinstance(result, dict):
                    if isinstance(result.get("sentences"), list):
                        return _pad_to3(result["sentences"])
                    if any(k in result for k in ("sentence1", "sentence2", "sentence3")):
                        return _pad_to3([result.get("sentence1", ""), result.get("sentence2", ""), result.get("sentence3", "")])
                    if isinstance(result.get("rendered"), list):
                        return _pad_to3(result["rendered"])

            # 형태가 맞지 않으면 재시도
            last_err = RuntimeError(f"Unexpected response: {data!r}")
        except Exception as e:
            last_err = e

        # 지수 백오프
        time.sleep(1.2 ** t)

    # ---- 폴백 ----
    cs = ", ".join([c.strip() for c in (colors or "").split(",") if c.strip()])
    s1 = f"A {breed} dog" + (f" with {cs} coat." if cs else ".")
    s2 = "Appearance summary: " + (f"{breed} with {cs} coat." if cs else f"{breed}.")
    s3 = f"{breed}; colors: {cs}."
    return [s1, s2, s3]

