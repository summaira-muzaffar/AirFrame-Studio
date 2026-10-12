"""Small MediaPipe model files (hand + face), downloaded once into ./models/.

Used by the new MediaPipe "Tasks" API (mediapipe >= 0.10.1x / 1.x).
If you are offline, download the files yourself and drop them into models/.
"""
import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "models"

MODELS = {
    "hand": (
        "hand_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task",
    ),
    "face": (
        "blaze_face_short_range.tflite",
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
    ),
}


def ensure_model(kind: str) -> Path:
    """Return the local path of a model, downloading it the first time."""
    name, url = MODELS[kind]
    path = MODEL_DIR / name
    if path.exists() and path.stat().st_size > 10_000:
        return path
    MODEL_DIR.mkdir(exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        print(f"Downloading {name} (one time only)...")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {name} ({exc}). Download it from\n  {url}\n"
            f"and save it as {path}"
        ) from exc
    return path