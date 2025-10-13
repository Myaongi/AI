import json, time, requests
from typing import List
from . import config

_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16)
_session.mount("http://", _adapter); _session.mount("https://", _adapter)

def normalize_to_3_sentences(breed: str, colors: str, features: str) -> List[str]:
    payload = {"breed": breed or "", "colors": colors or "", "features": features or ""}
    last = None
    for t in range(config.LLM_MAX_RETRIES):
        try:
            r = _session.post(config.LLM_URL,
                              headers={"Accept":"*/*","Content-Type":"application/json"},
                              data=json.dumps(payload),
                              timeout=config.LLM_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            result = data.get("result", data)
            if isinstance(result, dict):
                if isinstance(result.get("sentences"), list):
                    sents = result["sentences"]
                elif any(k in result for k in ("sentence1","sentence2","sentence3")):
                    sents = [result.get("sentence1",""), result.get("sentence2",""), result.get("sentence3","")]
                elif isinstance(result.get("rendered"), list):
                    sents = result["rendered"]
                else:
                    raise ValueError("Unexpected LLM response")
            elif isinstance(result, list):
                sents = result
            else:
                raise ValueError("Unexpected LLM response type")
            sents = [s.strip() for s in sents if isinstance(s,str) and s.strip()]
            while len(sents) < 3: sents.append(sents[-1] if sents else "A dog.")
            return sents[:3]
        except Exception as e:
            last = e; time.sleep(1.2**t)

    # 폴백
    cs = ", ".join([c.strip() for c in (colors or "").split(",") if c.strip()])
    s1 = f"A {breed} dog" + (f" with {cs} coat." if cs else ".")
    s2 = "Appearance summary: " + (f"{breed} with {cs} coat." if cs else f"{breed}.")
    s3 = f"{breed}; colors: {cs}."
    return [s1, s2, s3]
