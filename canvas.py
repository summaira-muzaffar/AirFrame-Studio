"""AirCanvas - our own feature: draw on the styled picture by pointing a finger in the air.

Strokes are stored as lists of points (in panel coordinates), so we can undo them
and redraw cleanly at any time.
"""
from typing import List, Optional

import cv2
import numpy as np

# name -> BGR
COLORS = {
    "White": (255, 255, 255),
    "Red": (60, 60, 255),
    "Yellow": (0, 220, 255),
    "Green": (90, 230, 90),
    "Cyan": (255, 220, 40),
    "Pink": (200, 90, 255),
}
BRUSH_SIZES = [3, 6, 10]


class AirCanvas:
    def __init__(self, size: tuple):
        self.w, self.h = size
        self.strokes: List[dict] = []
        self._current: Optional[dict] = None
        self.color_idx = 0
        self.brush_idx = 1

    @property
    def color_name(self) -> str:
        return list(COLORS)[self.color_idx]

    @property
    def color(self) -> tuple:
        return COLORS[self.color_name]

    @property
    def brush(self) -> int:
        return BRUSH_SIZES[self.brush_idx]

    def inside(self, p, margin: float = 0.05) -> bool:
        mx, my = self.w * margin, self.h * margin
        return -mx <= p[0] <= self.w + mx and -my <= p[1] <= self.h + my

    def add_point(self, p) -> None:
        if self._current is None:
            self._current = {"pts": [], "color": self.color, "size": self.brush}
            self.strokes.append(self._current)
        self._current["pts"].append((float(p[0]), float(p[1])))

    def end_stroke(self) -> None:
        self._current = None

    def undo(self) -> None:
        self.end_stroke()
        if self.strokes:
            self.strokes.pop()

    def clear(self) -> None:
        self.end_stroke()
        self.strokes.clear()

    def next_color(self) -> None:
        self.color_idx = (self.color_idx + 1) % len(COLORS)

    def next_brush(self) -> None:
        self.brush_idx = (self.brush_idx + 1) % len(BRUSH_SIZES)

    def render(self, panel: np.ndarray) -> np.ndarray:
        """Return a copy of the panel with all strokes drawn on it."""
        out = panel.copy()
        for s in self.strokes:
            pts = np.array(s["pts"], dtype=np.int32).reshape(-1, 1, 2)
            if len(pts) == 1:
                cv2.circle(out, tuple(pts[0, 0]), s["size"] // 2 + 1, s["color"], -1, cv2.LINE_AA)
            else:
                cv2.polylines(out, [pts], False, s["color"], s["size"], cv2.LINE_AA)
        return out
