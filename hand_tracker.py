"""Hand tracking (MediaPipe) + gesture tests.

Works with both MediaPipe generations:
  * new (>= 1.0): HandLandmarker "Tasks" API - needs a small model file, downloaded once
  * old (0.10.x): legacy mp.solutions.hands

Gestures:
  pointing      - only the index finger is up            (draw)
  fist          - all four fingers folded                (clear)
  L-shape       - thumb + index out, others folded       (opens the floating panel)
  pinch         - thumb tip touches index tip            (capture)
"""
import math
import threading
import time
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from model_files import ensure_model
from smoothing import AdaptiveEMA

# landmark ids: finger -> (pip joint, tip)
FINGER_JOINTS = {"index": (6, 8), "middle": (10, 12), "ring": (14, 16), "pinky": (18, 20)}
THUMB_TIP, INDEX_TIP = 4, 8


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else v


@dataclass
class Hand:
    pts: np.ndarray  # (21, 2) pixel coordinates

    # ---- finger states ----
    def finger_extended(self, name: str) -> bool:
        """A finger is 'up' when its tip is clearly farther from the wrist than its middle joint."""
        pip, tip = FINGER_JOINTS[name]
        wrist = self.pts[0]
        return np.linalg.norm(self.pts[tip] - wrist) > 1.08 * np.linalg.norm(self.pts[pip] - wrist)

    def others_extended(self) -> bool:
        """At least one of middle / ring / pinky is up (so a closed fist is not a pinch)."""
        return any(self.finger_extended(f) for f in ("middle", "ring", "pinky"))

    def palm_len(self) -> float:
        return float(np.linalg.norm(self.pts[5] - self.pts[0]))

    # ---- gestures ----
    def is_pointing(self) -> bool:
        """Only the index finger is up -> pen down."""
        return self.finger_extended("index") and not self.others_extended()

    def is_fist(self) -> bool:
        """All four fingers folded -> clear gesture."""
        return not any(self.finger_extended(f) for f in FINGER_JOINTS)

    def pinch_ratio(self) -> float:
        """Thumb-tip to index-tip distance, relative to hand size (small = pinching)."""
        size = max(float(np.linalg.norm(self.pts[9] - self.pts[0])), 1e-6)
        return float(np.linalg.norm(self.pts[THUMB_TIP] - self.pts[INDEX_TIP])) / size

    def is_l_shape(self) -> bool:
        """Thumb and index make an 'L' (roughly 45-135 degrees); other fingers are folded."""
        if not self.finger_extended("index") or self.others_extended():
            return False
        if np.linalg.norm(self.pts[4] - self.pts[5]) < 0.7 * self.palm_len():
            return False  # thumb is tucked in
        t = _unit(self.pts[4] - self.pts[2])
        v = _unit(self.pts[8] - self.pts[5])
        angle = math.degrees(math.acos(float(np.clip(np.dot(t, v), -1.0, 1.0))))
        return 45.0 <= angle <= 135.0

    def l_panel_quad(self, scale: float = 3.4, aspect: float = 4 / 3) -> np.ndarray:
        """Corners (tl, tr, br, bl) of a floating panel sitting inside the 'L' of this hand.

        The panel is 4:3, scales with your hand size, and turns with your hand
        (kept within +-45 degrees so the picture never ends up upside down).
        """
        corner = (self.pts[2] + self.pts[5]) / 2          # the web between thumb and index
        a = _unit(self.pts[4] - corner)                   # thumb side
        b = _unit(self.pts[8] - corner)                   # index side
        w = scale * self.palm_len()
        h = w / aspect
        center = corner + _unit(a + b) * (w * 0.55)       # push the panel into the inside of the L
        ang = math.atan2(float(a[1]), float(a[0]))
        theta = ((ang + math.pi / 4) % (math.pi / 2)) - math.pi / 4
        c, s = math.cos(theta), math.sin(theta)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        half = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]], dtype=np.float32)
        return (center + half @ rot.T).astype(np.float32)


class PinchDetector:
    """Fires once at the moment a pinch starts (not every frame while you hold it)."""

    def __init__(self, on: float = 0.30, off: float = 0.50):
        self.on, self.off = on, off
        self.active = False

    def update(self, hand: Hand) -> bool:
        ratio = hand.pinch_ratio()
        if self.active:
            if ratio > self.off:       # fingers opened again -> ready for the next pinch
                self.active = False
            return False
        if ratio < self.on and hand.others_extended():
            self.active = True
            return True
        return False

    def reset(self) -> None:
        self.active = False


class HandTracker:
    def __init__(self, max_hands: int = 2, detect_conf: float = 0.6, track_conf: float = 0.6):
        import mediapipe as mp  # imported here so gesture code can be tested without it

        self._mp = mp
        self._last_ts = 0
        if hasattr(mp, "solutions"):  # old MediaPipe
            self._backend = "legacy"
            self._legacy = mp.solutions.hands.Hands(
                model_complexity=0,
                max_num_hands=max_hands,
                min_detection_confidence=detect_conf,
                min_tracking_confidence=track_conf,
            )
        else:  # new MediaPipe (Tasks API)
            from mediapipe.tasks.python import vision

            self._backend = "tasks"
            options = vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(ensure_model("hand"))),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=max_hands,
                min_hand_detection_confidence=detect_conf,
                min_hand_presence_confidence=detect_conf,
                min_tracking_confidence=track_conf,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(options)

        # one smoother per hand "slot" (slot 0 = left-most hand on screen)
        self._smoothers = [AdaptiveEMA(), AdaptiveEMA()]

    # ---- detection ----
    def _detect(self, bgr: np.ndarray) -> List[np.ndarray]:
        """Return a list of (21, 2) pixel arrays, one per detected hand."""
        h, w = bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        if self._backend == "legacy":
            rgb.flags.writeable = False
            result = self._legacy.process(rgb)
            hands = result.multi_hand_landmarks or []
            return [np.array([[lm.x * w, lm.y * h] for lm in hand.landmark], dtype=np.float32)
                    for hand in hands]

        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)  # must keep increasing
        self._last_ts = ts
        result = self._landmarker.detect_for_video(image, ts)
        return [np.array([[lm.x * w, lm.y * h] for lm in hand], dtype=np.float32)
                for hand in result.hand_landmarks]

    def process(self, bgr: np.ndarray) -> List[Hand]:
        raw = self._detect(bgr)
        if not raw:
            for s in self._smoothers:
                s.reset()
            return []

        raw.sort(key=lambda p: p[0][0])  # left-most wrist first -> stable slots
        hands = []
        for slot, pts in enumerate(raw[:2]):
            hands.append(Hand(self._smoothers[slot].update(pts)))
        for slot in range(len(hands), 2):
            self._smoothers[slot].reset()
        return hands

    def close(self) -> None:
        if self._backend == "legacy":
            self._legacy.close()
        else:
            self._landmarker.close()


class ThreadedTracker:
    """Runs hand tracking in a background thread.

    The main loop (camera, drawing, window) never waits for the tracker: each call to
    process() hands over the newest frame and returns the latest finished result, which
    is at most one tracking step old. Same API as HandTracker, so it is a drop-in.
    """

    def __init__(self, tracker=None):
        self._tracker = tracker or HandTracker()
        self._lock = threading.Lock()
        self._frame = None
        self._hands: List[Hand] = []
        self._new = threading.Event()
        self._stop = False
        self.error = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop:
            if not self._new.wait(0.1):
                continue
            self._new.clear()
            with self._lock:
                frame = self._frame
            if frame is None:
                continue
            try:
                hands = self._tracker.process(frame)
            except Exception as exc:               # never let the thread die silently
                self.error = f"hand tracking error ({type(exc).__name__}: {exc})"
                hands = []
            with self._lock:
                self._hands = hands

    def process(self, bgr: np.ndarray) -> List[Hand]:
        with self._lock:
            self._frame = bgr.copy()               # the caller keeps drawing on its own frame
            hands = list(self._hands)
        self._new.set()
        return hands

    def close(self) -> None:
        self._stop = True
        self._thread.join(timeout=2.0)
        self._tracker.close()