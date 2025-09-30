# dogbreed/weights.py
from __future__ import annotations
import os, sys, hashlib, urllib.request, tempfile

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_weight(dest_path: str, url: str | None, sha256: str | None = None) -> str:
    """
    dest_path가 없으면 url에서 다운로드. sha256이 주어지면 무결성 검사.
    반환: 최종 파일 경로(dest_path)
    """
    if os.path.exists(dest_path):
        if sha256:
            cur = _sha256(dest_path)
            if cur.lower() != sha256.lower():
                os.remove(dest_path)
            else:
                return dest_path
        else:
            return dest_path

    if not url:
        raise FileNotFoundError(f"Weight not found and no URL provided: {dest_path}")

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="w_", suffix=".tmp")
    os.close(tmp_fd)
    try:
        print(f"[weights] downloading: {url}")
        urllib.request.urlretrieve(url, tmp_path)
        if sha256:
            got = _sha256(tmp_path)
            if got.lower() != sha256.lower():
                raise RuntimeError(f"SHA256 mismatch for {dest_path}: {got} (expected {sha256})")
        os.replace(tmp_path, dest_path)
        print(f"[weights] saved to {dest_path}")
    finally:
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass
    return dest_path
