"""Download the authors' released LEVIR-CD checkpoint and verify its SHA-256.

    python -m tinycd.weights
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = ROOT / "weights" / "levir_best.pth"
URL = "https://github.com/AndreaCodegoni/Tiny_model_4_CD/raw/main/pretrained_models/levir_best.pth"
SHA256 = "743645b87199c96bd0c3271ae5df6f0918b3f1d84618dfb9396843ca08ee4a3e"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch_weights(path: Path = WEIGHTS_PATH) -> Path:
    """Download from the TinyCD authors' repository if missing; always verify the checksum."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {URL}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)
        tmp.replace(path)
    got = _sha256(path)
    if got != SHA256:
        path.unlink()
        raise RuntimeError(f"Checksum mismatch for {path.name} (got {got}); deleted, please retry.")
    print(f"Weights OK: {path} (sha256 verified)")
    return path


if __name__ == "__main__":
    fetch_weights()
