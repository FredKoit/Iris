"""Download the Kokoro voice model. Whisper fetches itself on first run."""

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
FILES = ["kokoro-v1.0.onnx", "voices-v1.0.bin"]


def hook(done, block, total):
    if total > 0:
        pct = min(100, done * block * 100 // total)
        print(f"\r  {pct}%", end="", flush=True)


def main() -> int:
    MODELS.mkdir(exist_ok=True)
    for name in FILES:
        dest = MODELS / name
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"{name}: already here")
            continue
        print(f"{name}:")
        urllib.request.urlretrieve(f"{BASE}/{name}", dest, hook)
        print("\r  done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
