"""Face finding + portrait cropping for pinch mode.

Uses MediaPipe's own face detector (OpenCV 5 no longer ships the Haar files).
If the detector can't start or finds nothing, we fall back to a centered 4:3 crop,
so capture always works. The reason is kept in `FaceFinder.error` for the HUD.
"""
from typing import Optional, Tuple

import cv2
import numpy as np

from model_files import ensure_model

Box = Tuple[float, float, float, float]  # x, y, w, h in pixels


class FaceFinder:
    def __init__(self, min_conf: float = 0.5):
        self.error: Optional[str] = None
        self._backend = None
        self._det = None
        self._mp = None
        try:
            import mediapipe as mp

            self._mp = mp
            if hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
                from mediapipe.tasks.python import vision

                opts = vision.FaceDetectorOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(ensure_model("face"))),
                    running_mode=vision.RunningMode.IMAGE,
                    min_detection_confidence=min_conf,
                )
                self._det = vision.FaceDetector.create_from_options(opts)
                self._backend = "tasks"
            elif hasattr(mp, "solutions"):
                self._det = mp.solutions.face_detection.FaceDetection(
                    model_selection=0, min_detection_confidence=min_conf)
                self._backend = "legacy"
            else:
                raise RuntimeError("no face detector in this MediaPipe build")
        except Exception as exc:
            self.error = f"face detector off ({type(exc).__name__}); using center crop"

    def find(self, bgr: np.ndarray) -> Optional[Box]:
        """Return the biggest face as (x, y, w, h), or None."""
        if self._det is None:
            return None
        h, w = bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        try:
            if self._backend == "tasks":
                img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
                boxes = [(d.bounding_box.origin_x, d.bounding_box.origin_y,
                          d.bounding_box.width, d.bounding_box.height)
                         for d in self._det.detect(img).detections]
            else:
                res = self._det.process(rgb)
                boxes = [(d.location_data.relative_bounding_box.xmin * w,
                          d.location_data.relative_bounding_box.ymin * h,
                          d.location_data.relative_bounding_box.width * w,
                          d.location_data.relative_bounding_box.height * h)
                         for d in (res.detections or [])]
        except Exception as exc:
            self.error = f"face detector failed ({type(exc).__name__}); using center crop"
            self._det = None
            return None
        if not boxes:
            return None
        return tuple(float(v) for v in max(boxes, key=lambda b: b[2] * b[3]))


def crop_face(img: np.ndarray, box: Optional[Box], aspect: float = 4 / 3,
              margin: float = 2.2) -> np.ndarray:
    """Cut a 4:3 portrait around `box` (head + shoulders room). No box -> center crop."""
    ih, iw = img.shape[:2]
    if box is None:
        cw, ch = (iw, int(iw / aspect)) if iw / ih < aspect else (int(ih * aspect), ih)
        cx, cy = iw / 2, ih / 2
    else:
        x, y, bw, bh = box
        cx, cy = x + bw / 2, y + bh / 2 + 0.1 * bh   # shift down a little: more chin/shoulders
        ch = max(bh * margin * 1.3, 32)
        cw = ch * aspect
        if cw > iw:
            cw, ch = iw, iw / aspect
        if ch > ih:
            ch, cw = ih, ih * aspect
    cw, ch = int(round(cw)), int(round(ch))
    x0 = int(np.clip(cx - cw / 2, 0, max(iw - cw, 0)))
    y0 = int(np.clip(cy - ch / 2, 0, max(ih - ch, 0)))
    return img[y0:y0 + ch, x0:x0 + cw].copy()