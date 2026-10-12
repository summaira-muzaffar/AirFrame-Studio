"""Gallery - our own feature: every saved artwork goes into gallery/ and a simple
index.html is rebuilt so you can open all your creations in the browser.
 
File names look like:  <name>_<date>_<time>_<style>.png
Change <name> with:    python app.py --name myart
"""
import re
import time
from pathlib import Path
 
import cv2
 
GALLERY_DIR = Path("gallery")
DEFAULT_NAME = "airframe"
 
 
def _safe(text: str) -> str:
    """Keep letters, numbers, - and _ only (spaces become -)."""
    text = re.sub(r"[^A-Za-z0-9_-]+", "-", text.strip()).strip("-")
    return text or DEFAULT_NAME
 
 
def save_to_gallery(image, style_name: str, name: str = DEFAULT_NAME) -> Path:
    GALLERY_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = GALLERY_DIR / f"{_safe(name)}_{stamp}_{_safe(style_name).lower()}.png"
    cv2.imwrite(str(path), image)
    _rebuild_index()
    return path
 
 
def _rebuild_index() -> None:
    files = sorted(GALLERY_DIR.glob("*.png"), key=lambda f: f.stat().st_mtime, reverse=True)
    cards = "\n".join(
        f'<figure><img src="{f.name}"><figcaption>{f.stem}</figcaption></figure>' for f in files
    )
    html = f"""<!doctype html><meta charset="utf-8"><title>Gallery</title>
<style>body{{background:#111;color:#eee;font-family:sans-serif;margin:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px}}
img{{width:100%;border-radius:8px}}figcaption{{font-size:12px;opacity:.7;margin-top:4px}}</style>
<h1>Gallery</h1><div class="grid">{cards}</div>"""
    (GALLERY_DIR / "index.html").write_text(html, encoding="utf-8")
