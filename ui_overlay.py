"""Everything drawn on top of the camera image: frame guides, countdown ring, HUD.

All HUD text sits on a translucent dark box so it stays readable on any background
(bright walls, white scarves, strong lights...).
"""
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
ACCENT = (255, 210, 80)   # BGR, light blue
WHITE = (255, 255, 255)
YELLOW = (80, 220, 255)


def _scale(img) -> float:
    return max(0.6, img.shape[1] / 1280 * 0.75)


def _box(img, x0, y0, x1, y1, alpha=0.62):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    cv2.addWeighted(np.zeros_like(roi), alpha, roi, 1 - alpha, 0, roi)  # darken in place


def text_block(img, lines, x, y, scale=None, pad=10, anchor="left"):
    """Draw lines of (text, color) on a dark box. y is the top of the box."""
    scale = scale or _scale(img)
    thick = 2 if scale >= 0.95 else 1
    sizes = [cv2.getTextSize(t, FONT, scale, thick)[0] for t, _ in lines]
    lh = int(max(s[1] for s in sizes) * 1.8)
    box_w = max(s[0] for s in sizes) + 2 * pad
    box_h = lh * len(lines) + pad
    if anchor == "center":
        x = x - box_w // 2
    _box(img, x, y, x + box_w, y + box_h)
    for i, ((t, color), (_, th)) in enumerate(zip(lines, sizes)):
        base = y + pad // 2 + int(lh * (i + 0.5)) + th // 2
        cv2.putText(img, t, (x + pad, base), FONT, scale, color, thick, cv2.LINE_AA)
    return box_w, box_h


def draw_frame_guides(img, quad, color=ACCENT):
    q = np.asarray(quad, dtype=np.int32)
    cv2.polylines(img, [q.reshape(-1, 1, 2)], True, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.polylines(img, [q.reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
    for x, y in q:
        cv2.circle(img, (int(x), int(y)), 9, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), 6, color, -1, cv2.LINE_AA)


def draw_progress_ring(img, center, frac, radius=38, label="HOLD"):
    """Countdown ring that fills while you hold the frame still; `label` sits in the middle."""
    c = (int(center[0]), int(center[1]))
    cv2.circle(img, c, radius + 6, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img, c, radius, (70, 70, 70), 6, cv2.LINE_AA)
    cv2.ellipse(img, c, (radius, radius), -90, 0, int(360 * min(frac, 1.0)), ACCENT, 6, cv2.LINE_AA)
    scale = 1.1 if len(label) == 1 else 0.55
    (tw, th), _ = cv2.getTextSize(label, FONT, scale, 2 if len(label) == 1 else 1)
    cv2.putText(img, label, (c[0] - tw // 2, c[1] + th // 2), FONT, scale, WHITE,
                2 if len(label) == 1 else 1, cv2.LINE_AA)


def draw_flash(img, alpha):
    """White 'camera flash' over the whole picture (alpha 0..1)."""
    if alpha > 0:
        cv2.addWeighted(np.full_like(img, 255), float(alpha), img, 1.0 - float(alpha), 0, img)


def draw_cursor(img, pt, color, pen_down):
    c = (int(pt[0]), int(pt[1]))
    cv2.circle(img, c, 14 if pen_down else 18, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.circle(img, c, 10 if pen_down else 14, color, -1 if pen_down else 2, cv2.LINE_AA)


CAPTURE_LABEL = {"frame": "Two-hand frame", "pinch": "L-hand + pinch"}


def draw_hud(img, mode, style_name, color_name, fps, toast=None, busy=False, note=None,
             capture_mode="frame", allow_switch=False):
    h, w = img.shape[:2]
    s = _scale(img)

    info = f"Style: {style_name}"
    if mode == "review":
        info += f"   Ink: {color_name}"
    title = f"AirFrame Studio  |  {mode.upper()}"
    if allow_switch:                       # only mention the capture mode when there is a choice
        title += f"  |  {CAPTURE_LABEL[capture_mode]}"
    top = [(title, ACCENT), (f"{info}   {fps:0.0f} FPS", WHITE)]
    if busy:
        top.append(("Styling...", YELLOW))
    if note:
        top.append((note, YELLOW))
    text_block(img, top, 12, 12, s)

    switch = "    M switch mode" if allow_switch else ""
    if mode == "live" and capture_mode == "frame":
        hint = [("1) Make a rectangle: both hands, thumb + index fingertips are the 4 corners.", WHITE),
                ("2) Hold still - the ring counts down, then it snaps the picture.", WHITE),
                (f"SPACE capture now    [ ] style{switch}    R reset    Q quit", ACCENT)]
    elif mode == "live":
        hint = [("Show an L with one hand (thumb + index) to open the panel. Pinch with the other hand.", WHITE),
                (f"SPACE capture    [ ] style{switch}    R reset    Q quit", ACCENT)]
    else:
        hint = [("Point with ONE finger to draw.    Fist = clear.", WHITE),
                (f"[ ] style    C color    B brush    U undo    S save{switch}    R new photo    Q quit", ACCENT)]
    s_hint = s * 0.85
    lines_h = int(cv2.getTextSize("Ag", FONT, s_hint, 1)[0][1] * 1.8) * len(hint) + 10
    text_block(img, hint, 12, h - lines_h - 12, s_hint)

    if toast:
        text_block(img, [(toast, ACCENT)], w // 2, h // 2 - 20, s * 1.3, anchor="center")