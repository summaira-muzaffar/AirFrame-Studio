"""Geometry: order the 4 corners, cut out what is inside the frame, and project a panel back."""
import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """Return 4 points ordered top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # y - x
    return np.float32([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


def quad_area(quad: np.ndarray) -> float:
    return float(cv2.contourArea(np.asarray(quad, dtype=np.float32)))


def panel_size(quad: np.ndarray, long_side: int = 640, min_side: int = 160) -> tuple:
    """Panel (w, h) that matches the shape of the hand-made frame."""
    tl, tr, br, bl = quad
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    scale = long_side / max(w, h, 1.0)
    return max(int(w * scale), min_side), max(int(h * scale), min_side)


def _rect(size):
    w, h = size
    return np.float32([[0, 0], [w, 0], [w, h], [0, h]])


def panel_to_screen_matrix(quad: np.ndarray, size: tuple) -> np.ndarray:
    return cv2.getPerspectiveTransform(_rect(size), np.asarray(quad, dtype=np.float32))


def rectify(frame: np.ndarray, quad: np.ndarray, size: tuple) -> np.ndarray:
    """Cut the area inside the quad out of the camera frame as a flat rectangle."""
    M = cv2.getPerspectiveTransform(np.asarray(quad, dtype=np.float32), _rect(size))
    return cv2.warpPerspective(frame, M, size)


def project_panel(frame: np.ndarray, panel: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Paste the flat panel onto the frame using the panel->screen matrix M.

    Only the small box around the panel is warped (not the whole frame), and pixels
    outside the panel are left alone - several times faster than warping + masking
    the full frame. The frame is modified in place and returned.
    """
    fh, fw = frame.shape[:2]
    ph, pw = panel.shape[:2]
    corners = cv2.perspectiveTransform(np.float32([[[0, 0], [pw, 0], [pw, ph], [0, ph]]]), M)[0]
    x0, y0 = np.floor(corners.min(axis=0)).astype(int) - 1
    x1, y1 = np.ceil(corners.max(axis=0)).astype(int) + 2
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, fw), min(y1, fh)
    if x1 <= x0 or y1 <= y0:
        return frame                                   # panel completely off-screen
    shift = np.float64([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]])
    roi = frame[y0:y1, x0:x1].copy()
    cv2.warpPerspective(panel, shift @ M, (x1 - x0, y1 - y0), dst=roi,
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    frame[y0:y1, x0:x1] = roi
    return frame


def screen_to_panel(pt, M: np.ndarray) -> np.ndarray:
    """Map a point on the screen (e.g. fingertip) to panel coordinates."""
    inv = np.linalg.inv(M)
    p = cv2.perspectiveTransform(np.float32([[pt]]), inv)
    return p[0, 0]