"""Camera-free checks: python selftest.py

Covers the pieces AND the whole flow (rectangle -> hold -> capture -> draw), the background
tracker thread, and the AI-styling code (with a fake model, so no GPU / download needed).
"""
import time

import numpy as np

import app as A
from canvas import AirCanvas
from face_crop import crop_face
from hand_tracker import Hand, ThreadedTracker
from inference import AsyncStyler
from perspective import (order_points, panel_size, panel_to_screen_matrix, project_panel,
                         rectify, screen_to_panel)
from smoothing import AdaptiveEMA
from styles import STYLES


def fake_hand(up):
    """Build a hand with the wrist at the bottom; `up` lists which fingers are extended."""
    pts = np.zeros((21, 2), dtype=np.float32)
    pts[0] = (100, 200)
    for name, (pip, tip) in {"index": (6, 8), "middle": (10, 12), "ring": (14, 16), "pinky": (18, 20)}.items():
        pts[pip] = (100, 140)
        pts[tip] = (100, 80) if name in up else (100, 150)  # folded tip stays close to the wrist
    return Hand(pts)


class FakeStyler:
    """Stands in for AsyncStyler so flow tests are instant and deterministic."""
    error, busy = None, False

    def submit(self, *a):
        pass

    def poll(self):
        return None


def corner_hand(thumb, index):
    pts = np.zeros((21, 2), np.float32)
    pts[4], pts[8] = thumb, index
    return Hand(pts)


def rectangle_run(fps, jitter, drop_every=0, secs=6.0, move=False, argv=()):
    """Two hands hold a rectangle; returns (app, time of capture or None)."""
    rng = np.random.default_rng(1)
    app = A.AirFrameApp(A.parse_args(list(argv)), styler=FakeStyler())
    frame = np.full((720, 1280, 3), 90, np.uint8)
    t = 0.0
    for i in range(int(secs * fps)):
        t += 1 / fps
        base = np.float32([[400, 500], [400, 250], [900, 500], [900, 250]])
        if move:
            base[:, 0] += 120 * np.sin(t * 3)
        pts = base + rng.normal(0, jitter, base.shape).astype(np.float32)
        hands = [corner_hand(pts[0], pts[1]), corner_hand(pts[2], pts[3])]
        if drop_every and i % drop_every in (0, 1):      # tracking loses a hand for 2 frames
            hands = hands[:1]
        app.step(frame.copy(), hands, now=t)
        if app.mode == "review":
            return app, t
    return app, None


def test_capture_flow():
    _, t = rectangle_run(30, 2)
    assert t is not None and t < 3, "steady rectangle should capture"
    _, t = rectangle_run(7, 6)
    assert t is not None, "slow, jittery rectangle should capture"
    _, t = rectangle_run(7, 6, drop_every=9)
    assert t is not None, "tracking dropouts must not cancel the countdown"
    _, t = rectangle_run(7, 3, move=True)
    assert t is None, "a moving rectangle must NOT capture"

    app, t = rectangle_run(30, 2)
    assert app.flash_t0 is not None and app.raw_panel is not None
    # drawing: a pointing finger inside the floating picture leaves a stroke
    frame = np.full((720, 1280, 3), 90, np.uint8)
    for k in range(6):
        h = fake_hand({"index"})
        h.pts += np.float32([650 + 5 * k, 350])           # index tip lands inside the frame
        app.step(frame.copy(), [h], now=t + 0.1 * (k + 1))
    assert len(app.canvas.strokes) == 1, "pointing should draw"

    # countdown label: HOLD first, then the seconds left
    app = A.AirFrameApp(A.parse_args([]), styler=FakeStyler())
    labels = []
    for i in range(40):
        app.step(np.zeros((720, 1280, 3), np.uint8),
                 [corner_hand((400, 500), (400, 250)), corner_hand((900, 500), (900, 250))], now=i / 30)
        labels.append(app._ring_label())
    assert labels[0] == "HOLD" and "2" in labels and labels[-1] == "1", labels
    print("capture flow ok (steady, slow+jittery, dropouts, moving, drawing, countdown)")


def test_modes():
    app = A.AirFrameApp(A.parse_args([]), styler=FakeStyler())
    app.handle_key(ord("m"))
    assert app.capture_mode == "frame" and "--pinch" in app.toast, "pinch mode is opt-in"
    app = A.AirFrameApp(A.parse_args(["--pinch"]), styler=FakeStyler())
    app.handle_key(ord("m"))
    assert app.capture_mode == "pinch"
    app.handle_key(ord("m"))
    assert app.capture_mode == "frame"
    assert A.AirFrameApp(A.parse_args(["--mode", "pinch"]), styler=FakeStyler()).capture_mode == "pinch"
    print("modes ok")


def test_threaded_tracker():
    class Slow:
        closed = False

        def process(self, bgr):
            time.sleep(0.05)
            return [fake_hand({"index"})]

        def close(self):
            self.closed = True

    inner = Slow()
    tr = ThreadedTracker(inner)
    img = np.zeros((48, 64, 3), np.uint8)
    t0 = time.perf_counter()
    first = tr.process(img)
    assert time.perf_counter() - t0 < 0.03, "process() must not wait for the tracker"
    assert first == []
    got = []
    for _ in range(50):
        got = tr.process(img)
        if got:
            break
        time.sleep(0.02)
    assert got and got[0].is_pointing()
    tr.close()
    assert inner.closed
    print("threaded tracker ok")


def test_ai_styling():
    from PIL import Image
    import inference

    calls = []

    class FakePipe:
        def __call__(self, **kw):
            calls.append(kw)
            arr = np.full((kw["image"].height, kw["image"].width, 3), 123, np.uint8)
            return type("R", (), {"images": [Image.fromarray(arr)]})()

    def wait(st):
        for _ in range(200):
            r = st.poll()
            if r is not None and not st.busy:
                return r
            time.sleep(0.02)
        return r

    img = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
    for model, flux in (("some/stable-diffusion", False), ("black-forest-labs/FLUX.1-schnell", True)):
        st = inference.AsyncStyler(model)
        st._load_pipe = lambda f=flux: (FakePipe(), f)
        st.submit(img, 1)
        out = wait(st)
        assert out is not None and out.shape == img.shape and out.mean() == 123, model
        k = calls[-1]
        assert (k["guidance_scale"] == 0.0 and k["num_inference_steps"] == 4) == flux
        assert k["image"].size == ((768, 768) if flux else (512, 512))
    # model fails to load: fall back to OpenCV, say why, and do not retry every time
    st = inference.AsyncStyler("missing/model")
    attempts = []

    def broken():
        attempts.append(1)
        raise ImportError("No module named 'torch'")

    st._load_pipe = broken
    st.submit(img, 1)
    out = wait(st)
    assert out is not None and out.shape == img.shape and not st.busy
    assert st.error and "ImportError" in st.error and "torch" in st.error, st.error
    st.submit(img, 2)
    wait(st)
    assert len(attempts) == 1, "should not retry a model that failed to load"
    print("AI styling ok (fake SD + FLUX, failure fallback)")


def main():
    img = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)

    for s in STYLES:
        out = s.fn(img)
        assert out.shape == img.shape and out.dtype == np.uint8, s.name
    print(f"styles ok ({len(STYLES)})")

    quad = order_points(np.float32([[300, 300], [100, 90], [310, 80], [90, 310]]))
    assert quad[0][0] < quad[1][0] and quad[0][1] < quad[3][1]
    size = panel_size(quad)
    M = panel_to_screen_matrix(quad, size)
    back = screen_to_panel(quad[2], M)
    assert np.allclose(back, [size[0], size[1]], atol=1.0), back
    frame = np.zeros((480, 640, 3), np.uint8)
    assert rectify(frame, quad, size).shape[:2] == (size[1], size[0])
    assert project_panel(frame, np.full((size[1], size[0], 3), 255, np.uint8), M).max() == 255
    print("perspective ok")

    assert fake_hand({"index"}).is_pointing()
    assert not fake_hand({"index", "middle"}).is_pointing()
    assert fake_hand(set()).is_fist()
    print("gestures ok")

    sm = AdaptiveEMA()
    a = sm.update([[0.0, 0.0]])
    b = sm.update([[100.0, 0.0]])
    assert 0 < b[0][0] < 100
    print("smoothing ok")

    c = AirCanvas((200, 100))
    for x in range(20):
        c.add_point((x * 5, 50))
    c.end_stroke()
    assert c.render(np.zeros((100, 200, 3), np.uint8)).max() > 0
    c.undo()
    assert c.render(np.zeros((100, 200, 3), np.uint8)).max() == 0
    print("canvas ok")

    st = AsyncStyler()
    st.submit(img, 0)
    for _ in range(100):
        res = st.poll()
        if res is not None:
            break
        time.sleep(0.05)
    assert res is not None and res.shape == img.shape
    print("async styler ok")

    wide = np.zeros((720, 1280, 3), np.uint8)
    for box in (None, (600, 200, 150, 180), (0, 0, 50, 60), (1200, 650, 80, 70)):
        c = crop_face(wide, box)
        assert abs(c.shape[1] / c.shape[0] - 4 / 3) < 0.02, (box, c.shape)
    print("face crop ok")

    # off-screen / partly off-screen panels must not crash
    far = panel_to_screen_matrix(order_points(np.float32([[2000, 2000], [2100, 2000], [2100, 2100], [2000, 2100]])), (64, 48))
    project_panel(np.zeros((480, 640, 3), np.uint8), np.full((48, 64, 3), 255, np.uint8), far)
    edge = panel_to_screen_matrix(order_points(np.float32([[-50, 10], [100, 10], [100, 80], [-50, 80]])), (64, 48))
    assert project_panel(np.zeros((480, 640, 3), np.uint8), np.full((48, 64, 3), 255, np.uint8), edge).max() == 255
    print("project_panel edge cases ok")

    test_capture_flow()
    test_modes()
    test_threaded_tracker()
    test_ai_styling()
    print("ALL PASSED")


if __name__ == "__main__":
    main()