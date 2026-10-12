"""AirFrame Studio - main application.

  1) Make a rectangle with both hands (thumb + index fingertips = the 4 corners).
  2) Hold still: the ring counts down, then the picture inside the rectangle is captured.
  3) It floats where you framed it (REVIEW mode). Point one finger to draw on it,
     make a fist to clear, press S to save it to the gallery.

Optional extra (run with --pinch, then press M to switch): show an "L" with one hand and
pinch with the other to capture your face into a panel that follows your L hand.
"""
import argparse
import os
import time
from collections import deque

os.environ.setdefault("GLOG_minloglevel", "2")        # hide MediaPipe INFO/WARNING spam
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np

import ui_overlay as ui
from canvas import AirCanvas
from face_crop import FaceFinder, crop_face
from gallery import save_to_gallery
from hand_tracker import INDEX_TIP, THUMB_TIP, PinchDetector, ThreadedTracker
from inference import AsyncStyler
from perspective import (order_points, panel_size, panel_to_screen_matrix, project_panel,
                         quad_area, rectify, screen_to_panel)
from smoothing import AdaptiveEMA
from styles import STYLES

MIN_FRAME_AREA = 0.04      # two-hand frame must cover at least 4% of the image
FIST_FRAMES_TO_CLEAR = 20
FACE_PANEL = (640, 480)    # size of the face picture in pinch mode (4:3)
PINCH_COOLDOWN = 1.0       # seconds before another pinch can capture
LOST_GRACE = 0.6           # seconds a lost hand-frame is kept before the countdown resets
FLASH_SECS = 0.3           # length of the white capture flash
STEADY_FRAC = 0.04         # 'still' tolerance as a fraction of the rectangle's size


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="AirFrame Studio")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--pinch", action="store_true",
                   help="also enable the optional L-hand + pinch face-capture mode (press M to switch)")
    p.add_argument("--mode", choices=["frame", "pinch"], default="frame",
                   help="capture mode to start in; 'pinch' implies --pinch")
    p.add_argument("--hold", type=float, default=1.5, help="frame mode: seconds to hold still to capture")
    p.add_argument("--steady_px", type=float, default=8.0, help="frame mode: minimum movement (px) still counted as 'still'")
    p.add_argument("--panel_scale", type=float, default=3.4, help="pinch mode: floating panel size vs your hand")
    p.add_argument("--name", type=str, default="airframe",
                   help="name used at the start of saved files, e.g. --name myart")
    p.add_argument("--diffusion", "--model_path", dest="diffusion", type=str, default=None,
                   help="optional AI model id or local folder (Stable Diffusion or FLUX) for AI styling")
    return p.parse_args(argv)


class AirFrameApp:
    def __init__(self, args, styler=None):
        self.args = args
        self.styler = styler or AsyncStyler(args.diffusion)
        self.capture_mode = args.mode
        self.allow_switch = bool(args.pinch or args.mode == "pinch")
        self.flash_t0 = None          # time of the last capture (for the white flash)
        self.mode = "live"            # live | review
        self.anchor = "fixed"         # fixed (frame mode) | hand (pinch mode: panel follows your L hand)
        self.style_idx = 0
        self.quad_smoother = AdaptiveEMA(min_alpha=0.25, gain=0.02)
        self.pinch = PinchDetector()
        self.box_smoother = AdaptiveEMA(min_alpha=0.15, gain=0.01)
        self._faces = None            # created the first time pinch mode needs it
        self.face_box, self.face_ttl, self.face_tick = None, 0, 0

        self.hold = 0.0
        self.prev_quad = None
        self.hold_anchor = None       # where the frame was when the hold started
        self.last_seen = 0.0          # last time a valid two-hand frame was seen
        self.last_quad = None         # most recent valid frame (used by SPACE)
        self.locked_quad = None       # where the picture floats in review mode
        self.panel_size = FACE_PANEL
        self.raw_panel = None         # captured photo (unstyled)
        self.styled_panel = None
        self.canvas = None
        self.fist_count = 0
        self.cooldown_until = 0.0
        self.toast, self.toast_until = None, 0.0
        self.last_t, self.fps = None, 0.0
        self.recent = deque(maxlen=5)  # last few frames, averaged at capture to cancel webcam noise
        self._clean = None
        self.quit = False

    # ------------------------------------------------------------ helpers
    @property
    def faces(self):
        if self._faces is None:
            self._faces = FaceFinder()
        return self._faces

    def show_toast(self, msg, secs=1.4):
        self.toast, self.toast_until = msg, time.time() + secs

    def _photo(self, clean):
        return np.mean(self.recent, axis=0).astype(np.uint8) if len(self.recent) > 1 else clean

    def _reset_live(self):
        self.mode, self.anchor = "live", "fixed"
        self.hold, self.prev_quad, self.last_quad, self.hold_anchor = 0.0, None, None, None
        self.canvas = None
        self.quad_smoother.reset()
        self.pinch.reset()

    def _start_review(self, raw, quad, size, anchor, now):
        self.raw_panel, self.styled_panel = raw, None
        self.styler.submit(raw, self.style_idx)
        self.locked_quad = np.asarray(quad, dtype=np.float32).copy()
        self.panel_size, self.anchor = size, anchor
        self.canvas = AirCanvas(size)
        self.hold, self.fist_count = 0.0, 0
        self.cooldown_until = now + PINCH_COOLDOWN
        self.mode = "review"
        self.flash_t0 = now
        self.show_toast("Captured!")

    def _capture_frame(self, clean, quad, now):
        """Frame mode: cut out whatever is inside the hand-made frame."""
        size = panel_size(quad, long_side=800)
        raw = rectify(self._photo(clean), quad, size)
        self._start_review(raw, quad, size, "fixed", now)

    def _capture_face(self, clean, quad, now):
        """Pinch mode: cut a portrait around your face and pin it to the L-hand panel."""
        photo = self._photo(clean)
        crop = crop_face(photo, self.faces.find(photo))
        interp = cv2.INTER_AREA if crop.shape[1] >= FACE_PANEL[0] else cv2.INTER_LINEAR
        raw = cv2.resize(crop, FACE_PANEL, interpolation=interp)
        self._start_review(raw, quad, FACE_PANEL, "hand", now)

    def _face_view(self, clean):
        """Live preview of what a pinch would capture (face crop, updated a few times per second)."""
        self.face_tick += 1
        if self.face_tick % 5 == 1:
            box = self.faces.find(clean)
            if box is not None:
                self.face_box = tuple(self.box_smoother.update(np.float32(box)))
                self.face_ttl = 20
            else:
                self.face_ttl -= 5
                if self.face_ttl <= 0:
                    self.face_box = None
                    self.box_smoother.reset()
        crop = crop_face(clean, self.face_box)
        return cv2.resize(crop, FACE_PANEL, interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------------------ one frame
    def step(self, frame, hands, now=None):
        """Process one camera frame (already mirrored) and return the picture to show."""
        now = time.time() if now is None else now
        dt = max(now - self.last_t, 1e-3) if self.last_t is not None else 1 / 30
        self.last_t = now
        self.fps = 1.0 / dt if self.fps == 0 else 0.9 * self.fps + 0.1 / dt

        clean = frame.copy()
        self._clean = clean
        self.recent.append(clean)

        if self.mode == "review":
            frame = self._review(frame, hands)
        elif self.capture_mode == "frame":
            frame = self._live_frame(frame, clean, hands, dt, now)
        else:
            frame = self._live_pinch(frame, clean, hands, now)

        if self.flash_t0 is not None:
            age = now - self.flash_t0
            if 0 <= age < FLASH_SECS:
                ui.draw_flash(frame, 0.85 * (1 - age / FLASH_SECS))

        note = self.styler.error
        if self.capture_mode == "pinch" and self._faces is not None and self._faces.error:
            note = note or self._faces.error
        ui.draw_hud(frame, self.mode, STYLES[self.style_idx].name,
                    self.canvas.color_name if self.canvas else "-", self.fps,
                    self.toast if time.time() < self.toast_until else None,
                    busy=self.styler.busy and self.mode == "review", note=note,
                    capture_mode=self.capture_mode, allow_switch=self.allow_switch)
        return frame

    def _ring_label(self):
        """'HOLD' at first, then the seconds left (2, 1) while the frame stays still."""
        if self.hold < 0.1:
            return "HOLD"
        return str(max(1, int(np.ceil(self.args.hold - self.hold))))

    def _live_frame(self, frame, clean, hands, dt, now):
        """Make a rectangle with both hands (thumb + index tips = corners) and hold it steady."""
        fh, fw = frame.shape[:2]
        quad = None
        if len(hands) == 2:
            corners = np.float32([hands[0].pts[THUMB_TIP], hands[0].pts[INDEX_TIP],
                                  hands[1].pts[THUMB_TIP], hands[1].pts[INDEX_TIP]])
            q = order_points(corners)
            if quad_area(q) > MIN_FRAME_AREA * fw * fh:
                quad = self.quad_smoother.update(q)

        if quad is None:
            # Tracking often drops a frame or two. Keep the rectangle and the countdown
            # alive for a moment instead of cancelling everything.
            if self.last_quad is not None and now - self.last_seen < LOST_GRACE:
                ui.draw_frame_guides(frame, self.last_quad)
                ui.draw_progress_ring(frame, self.last_quad.mean(axis=0), self.hold / self.args.hold,
                                      label=self._ring_label())
                return frame
            self.quad_smoother.reset()
            self.hold, self.prev_quad, self.hold_anchor = 0.0, None, None
            return frame

        self.last_seen = now
        # "Still" = staying close to where the hold STARTED (not frame-to-frame jitter),
        # with a tolerance that grows with the size of your rectangle.
        tol = max(self.args.steady_px, STEADY_FRAC * float(np.sqrt(quad_area(quad))))
        if self.hold_anchor is None:
            self.hold_anchor, self.hold = quad.copy(), 0.0
        elif float(np.mean(np.linalg.norm(quad - self.hold_anchor, axis=1))) > tol:
            self.hold_anchor, self.hold = quad.copy(), 0.0      # moved: restart the countdown
        else:
            self.hold += dt
        self.prev_quad = quad.copy()
        self.last_quad = quad.copy()
        ui.draw_frame_guides(frame, quad)
        ui.draw_progress_ring(frame, quad.mean(axis=0), self.hold / self.args.hold, label=self._ring_label())
        if self.hold >= self.args.hold:
            self._capture_frame(clean, quad, now)
        return frame

    def _live_pinch(self, frame, clean, hands, now):
        l_hand = next((h for h in hands if h.is_l_shape()), None)
        if l_hand is None:                       # no L -> no panel
            self.quad_smoother.reset()
            self.pinch.reset()
            return frame
        quad = self.quad_smoother.update(l_hand.l_panel_quad(self.args.panel_scale))
        self.last_quad = quad.copy()
        M = panel_to_screen_matrix(quad, FACE_PANEL)
        frame = project_panel(frame, self._face_view(clean), M)
        ui.draw_frame_guides(frame, quad)

        others = [h for h in hands if h is not l_hand]
        if not others:
            self.pinch.reset()
            return frame
        if self.pinch.update(others[0]) and now >= self.cooldown_until:
            self._capture_face(clean, quad, now)
        return frame

    def _review(self, frame, hands):
        fresh = self.styler.poll()
        if fresh is not None:
            self.styled_panel = fresh
        base = self.styled_panel if self.styled_panel is not None else self.raw_panel

        l_hand = None
        if self.anchor == "hand":                # the panel follows your L hand
            l_hand = next((h for h in hands if h.is_l_shape()), None)
            if l_hand is not None:
                self.locked_quad = self.quad_smoother.update(l_hand.l_panel_quad(self.args.panel_scale))
        M = panel_to_screen_matrix(self.locked_quad, self.panel_size)

        drawing_pt, pen_down, fist_now = None, False, False
        for hand in hands:
            if hand is l_hand:
                continue
            if hand.is_pointing():
                tip = hand.pts[INDEX_TIP]
                drawing_pt = tip
                p = screen_to_panel(tip, M)
                if self.canvas.inside(p):
                    self.canvas.add_point(p)
                    pen_down = True
            elif hand.is_fist():
                fist_now = True
        if not pen_down:
            self.canvas.end_stroke()

        self.fist_count = self.fist_count + 1 if fist_now and not pen_down else 0
        if self.fist_count == FIST_FRAMES_TO_CLEAR:
            self.canvas.clear()
            self.show_toast("Cleared")

        frame = project_panel(frame, self.canvas.render(base), M)
        ui.draw_frame_guides(frame, self.locked_quad, (255, 255, 255))
        if drawing_pt is not None:
            ui.draw_cursor(frame, drawing_pt, self.canvas.color, pen_down)
        return frame

    # ------------------------------------------------------------ keyboard
    def handle_key(self, key, now=None):
        now = time.time() if now is None else now
        if key in (ord("q"), 27):
            self.quit = True
        elif key in (ord("["), ord("]")):
            self.style_idx = (self.style_idx + (1 if key == ord("]") else -1)) % len(STYLES)
            self.show_toast(STYLES[self.style_idx].name)
            if self.mode == "review":
                self.styler.submit(self.raw_panel, self.style_idx)
        elif key == ord("m"):
            if not self.allow_switch:
                self.show_toast("Pinch mode is off - start with --pinch")
                return
            self.capture_mode = "pinch" if self.capture_mode == "frame" else "frame"
            self._reset_live()
            self.show_toast(ui.CAPTURE_LABEL[self.capture_mode])
        elif key == ord("r"):
            self._reset_live()
        elif key == 32 and self.mode == "live" and self.last_quad is not None and self._clean is not None:
            if self.capture_mode == "frame":
                self._capture_frame(self._clean, self.last_quad, now)
            else:
                self._capture_face(self._clean, self.last_quad, now)
        elif self.mode == "review":
            if key == ord("c"):
                self.canvas.next_color()
            elif key == ord("b"):
                self.canvas.next_brush()
            elif key == ord("u"):
                self.canvas.undo()
            elif key == ord("s"):
                base = self.styled_panel if self.styled_panel is not None else self.raw_panel
                path = save_to_gallery(self.canvas.render(base), STYLES[self.style_idx].name, self.args.name)
                self.show_toast(f"Saved {path.name}", 2.2)


def main(argv=None):
    args = parse_args(argv)
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera. Try --camera 1")

    title = "AirFrame Studio"
    tracker = ThreadedTracker()
    app = AirFrameApp(args)
    try:
        while not app.quit:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror view
            out = app.step(frame, tracker.process(frame))
            cv2.imshow(title, out)
            app.handle_key(cv2.waitKey(1) & 0xFF)
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:  # window X button clicked
                break
    except KeyboardInterrupt:      # Ctrl+C in the terminal: quit quietly
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            tracker.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()