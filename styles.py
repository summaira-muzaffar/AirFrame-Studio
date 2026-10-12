"""Art styles. Every style is a function: BGR image in -> BGR image out.

Every style first runs through `_prep`, which removes webcam noise and evens out the
lighting, so the artistic filters work on a clean picture instead of grain.

These run with plain OpenCV (no GPU). Each style also has a text prompt that is used
only if you turn on the optional diffusion backend (see inference.py).
"""
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np


def _gray3(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def _prep(img):
    """Denoise + gentle contrast boost (CLAHE on lightness only, so colours stay natural)."""
    img = cv2.fastNlMeansDenoisingColored(img, None, 6, 6, 7, 21)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def pencil(img):
    """Classic colour-dodge pencil sketch: soft, clean lines on white paper."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sigma = max(img.shape[1] / 90, 3)
    blur = cv2.GaussianBlur(255 - gray, (0, 0), sigma)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    sketch = cv2.convertScaleAbs(sketch, alpha=1.1, beta=-18)     # a bit more contrast
    return _gray3(cv2.GaussianBlur(sketch, (0, 0), 0.8))


def cartoon(img):
    gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 7)
    edges = cv2.medianBlur(edges, 3)
    color = cv2.bilateralFilter(img, 9, 200, 200)
    return cv2.bitwise_and(color, color, mask=edges)


def watercolor(img):
    return cv2.stylization(img, sigma_s=60, sigma_r=0.45)


def oil(img):
    smooth = cv2.medianBlur(cv2.bilateralFilter(img, 9, 75, 75), 7)
    return (smooth // 32 * 32 + 16).astype(np.uint8)  # flat colour blocks like thick paint


def neon(img):
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(gray, 50, 120), np.ones((2, 2), np.uint8))
    h, w = edges.shape
    hue = ((np.indices((h, w))[1] / w) * 150 + 90).astype(np.uint8)  # colour sweeps across the panel
    hsv = np.dstack([hue, np.full_like(hue, 255), edges])
    line = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cv2.add(line, cv2.GaussianBlur(line, (0, 0), 6))


def pixel(img):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(w // 14, 8), max(h // 14, 8)), interpolation=cv2.INTER_AREA)
    small = (small // 48 * 48 + 24).astype(np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def thermal(img):
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 2)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def ink(img):
    """Bold black-and-white ink drawing with specks removed."""
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 2.2)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)
    return _gray3(cv2.medianBlur(bw, 5))


def vaporwave(img):
    b, g, r = cv2.split(img.astype(np.float32))
    out = cv2.merge([np.clip(b * 1.15 + 25, 0, 255), np.clip(g * 0.7, 0, 255), np.clip(r * 1.1 + 20, 0, 255)])
    return out.astype(np.uint8)


def glitch(img):
    out = img.copy()
    out[:, :, 2] = np.roll(img[:, :, 2], 8, axis=1)   # red channel slides right
    out[:, :, 0] = np.roll(img[:, :, 0], -8, axis=1)  # blue channel slides left
    out[::4] = (out[::4] * 0.55).astype(np.uint8)     # scanlines
    return out


def sepia(img):
    k = np.array([[0.131, 0.534, 0.272], [0.168, 0.686, 0.349], [0.189, 0.769, 0.393]], np.float32)
    return cv2.transform(img, k)


def _halftone(gray, cell=8):
    """Boolean mask of comic-book dots: darker areas get bigger dots."""
    h, w = gray.shape
    ys, xs = np.mgrid[0:h, 0:w]
    dx, dy = (xs % cell) - (cell - 1) / 2, (ys % cell) - (cell - 1) / 2
    dist = np.sqrt(dx * dx + dy * dy) / (cell * 0.7071)
    dark = 1.0 - cv2.GaussianBlur(gray, (0, 0), cell / 2).astype(np.float32) / 255
    return dark * 1.2 > dist


def comic(img):
    post = (cv2.bilateralFilter(img, 9, 120, 120) // 64 * 64 + 32).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.dilate(cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 140), np.ones((2, 2), np.uint8))
    out = post.copy()
    dots = _halftone(gray, 7)
    out[dots] = (out[dots] * 0.6).astype(np.uint8)
    out[edges > 0] = 0
    return out


def popart(img):
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 1.5)
    bins = np.digitize(gray, [64, 128, 192])
    palette = np.array([(70, 20, 50), (60, 60, 230), (40, 200, 250), (235, 235, 255)], np.uint8)  # BGR
    return palette[bins]


def charcoal(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(255 - gray, (0, 0), max(img.shape[1] / 60, 4))
    s = np.power(cv2.divide(gray, 255 - blur, scale=256).astype(np.float32) / 255, 2.4)  # smudgy darks
    grain = cv2.GaussianBlur(np.random.default_rng(7).normal(0, 0.06, s.shape).astype(np.float32), (0, 0), 0.8)
    return _gray3((np.clip(s * 0.95 + 0.05 + grain, 0, 1) * 255).astype(np.uint8))


def emboss(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], np.float32)
    return _gray3(np.clip(cv2.filter2D(gray, cv2.CV_32F, kernel) + 128, 0, 255).astype(np.uint8))


def blueprint(img):
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 50, 130)
    out = np.full_like(img, (150, 70, 10))              # deep blue paper
    out[::40, :] = (170, 100, 40)                       # drafting grid
    out[:, ::40] = (170, 100, 40)
    out[edges > 0] = (255, 255, 255)                    # white technical lines
    return out


def mosaic(img):
    h, w = img.shape[:2]
    tile = max(w // 48, 8)
    padded = cv2.copyMakeBorder(img, 0, (-h) % tile, 0, (-w) % tile, cv2.BORDER_REPLICATE)
    ph, pw = padded.shape[:2]
    small = cv2.resize(padded, (pw // tile, ph // tile), interpolation=cv2.INTER_AREA)
    big = cv2.resize(small, (pw, ph), interpolation=cv2.INTER_NEAREST)
    big[::tile, :] = (big[::tile, :] * 0.55).astype(np.uint8)    # grout lines
    big[:, ::tile] = (big[:, ::tile] * 0.55).astype(np.uint8)
    return big[:h, :w]


def cyberpunk(img):
    f = img.astype(np.float32) / 255
    lum = f.mean(axis=2, keepdims=True)
    teal = np.array([0.25, 0.15, -0.25], np.float32)       # BGR tint pushed into shadows
    magenta = np.array([0.20, -0.25, 0.25], np.float32)    # BGR tint pushed into highlights
    out = np.clip(f * (1 + (1 - lum) * teal + lum * magenta), 0, 1)
    out = np.clip((out - 0.5) * 1.25 + 0.5, 0, 1)
    glow = cv2.GaussianBlur(np.clip(out - 0.65, 0, 1), (0, 0), 12)
    return (np.clip(out + glow * 1.2, 0, 1) * 255).astype(np.uint8)


def anime(img):
    smooth = img
    for _ in range(2):
        smooth = cv2.bilateralFilter(smooth, 9, 60, 60)
    q = (smooth // 24 * 24 + 12).astype(np.uint8)
    hsv = cv2.cvtColor(q, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.3, 0, 255)       # richer colour
    q = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
    lines = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 6)
    return cv2.bitwise_and(q, q, mask=lines)


def duotone(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None] / 255
    dark, light = np.array([110, 30, 60], np.float32), np.array([80, 190, 255], np.float32)  # purple -> orange
    return (dark + (light - dark) * g).astype(np.uint8)


def _clean(fn):
    return lambda img: fn(_prep(img))


@dataclass(frozen=True)
class Style:
    name: str
    fn: Callable
    prompt: str


def original(img):
    """No effect: your photo exactly as captured."""
    return img.copy()


STYLES = [
    Style("Original", original, ""),
    Style("Pencil", _clean(pencil), "detailed pencil sketch portrait, graphite shading"),
    Style("Cartoon", _clean(cartoon), "cartoon portrait, bold outlines, flat colors"),
    Style("Watercolor", _clean(watercolor), "soft watercolor painting portrait, paper texture"),
    Style("Oil Paint", _clean(oil), "thick oil painting portrait, visible brush strokes"),
    Style("Neon", _clean(neon), "neon glowing line art portrait on dark background"),
    Style("Pixel", _clean(pixel), "16-bit pixel art portrait"),
    Style("Thermal", _clean(thermal), "thermal camera heat map portrait"),
    Style("Ink", _clean(ink), "black ink drawing portrait, high contrast"),
    Style("Vaporwave", _clean(vaporwave), "vaporwave aesthetic portrait, pink and cyan"),
    Style("Glitch", _clean(glitch), "digital glitch art portrait, RGB split"),
    Style("Sepia", _clean(sepia), "vintage sepia photograph portrait, old film"),
    Style("Comic", _clean(comic), "comic book portrait, halftone dots, bold ink outlines"),
    Style("Pop Art", _clean(popart), "pop art portrait, four flat colors, Warhol style"),
    Style("Charcoal", _clean(charcoal), "charcoal drawing portrait, smudged shading on paper"),
    Style("Emboss", _clean(emboss), "embossed relief portrait, carved in stone"),
    Style("Blueprint", _clean(blueprint), "architectural blueprint drawing of a portrait, white lines on blue"),
    Style("Mosaic", _clean(mosaic), "ceramic tile mosaic portrait"),
    Style("Cyberpunk", _clean(cyberpunk), "cyberpunk portrait, teal and magenta neon lighting"),
    Style("Anime", _clean(anime), "anime style portrait, clean line art, vibrant colors"),
    Style("Duotone", _clean(duotone), "duotone portrait, purple and orange gradient"),
]