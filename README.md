# AirFrame Studio

Make a picture frame in the air with your hands, snap a photo of whatever is inside it, turn it into art, then **draw on it with your finger**.

Inspired by the idea of gesture-controlled frames (e.g. HandFrame-AI).

## How it works

1. **Make a rectangle** with both hands: the thumb and index fingertips of each hand are the 4 corners.
2. **Hold still.** The ring counts down (2, 1), the screen flashes, and the picture inside the rectangle is captured.
3. The picture floats where you framed it. **Point one finger to draw** on it, make a fist to clear, press `S` to save.

## Features

- Real-time two-hand tracking (MediaPipe) in a background thread, with adaptive smoothing
- **NEW - Hold-to-capture:** countdown ring + flash; survives small hand wobble and brief tracking dropouts (SPACE also captures)
- Perspective-correct cut-out of what is inside the frame
- 21 art styles with plain OpenCV (no GPU needed)
- Optional AI styling with Stable Diffusion or FLUX (`--diffusion MODEL_ID_OR_FOLDER`)
- **NEW - Air-drawing:** point with one finger to draw on the floating picture (mapped through the perspective)
- **NEW - Gallery:** save artwork to `gallery/` and open `gallery/index.html` to browse it
- Undo, 6 ink colors, 3 brush sizes, fist-to-clear
- Optional extra (`--pinch`): face capture - show an L with one hand, pinch with the other, press `M` to switch modes

## Run

```bash
pip install -r requirements.txt
python model_files.py   # optional: download the small model files now (otherwise on first use)
python selftest.py      # optional: checks everything except the camera
python app.py
```

Options: `--camera 1`, `--hold 2.0`, `--pinch`, `--diffusion runwayml/stable-diffusion-v1-5` (or `black-forest-labs/FLUX.1-schnell`, or a local folder)

If a new library release ever breaks something: `pip install -r requirements-lock.txt` (exact versions known to work).

## Controls

| Input | Action |
| --- | --- |
| Rectangle with both hands, hold still | Capture |
| `SPACE` | Capture now |
| `[` / `]` | Previous / next style |
| 1 finger pointing | Draw |
| Fist (hold ~0.7s) | Clear drawing |
| `C` / `B` | Ink color / brush size |
| `U` | Undo last stroke |
| `S` | Save to gallery |
| `R` | New photo |
| `M` | Switch capture mode (only with `--pinch`) |
| `Q` / `Esc` / close window / Ctrl+C | Quit |

## Files

| File | Job |
| --- | --- |
| `app.py` | Main loop and modes (live / review) |
| `model_files.py` | Downloads the small hand + face model files once (`python model_files.py`) |
| `face_crop.py` | Face finder + 4:3 portrait crop (pinch mode) |
| `hand_tracker.py` | MediaPipe wrapper, background tracker thread, pointing / fist gestures |
| `smoothing.py` | Speed-adaptive smoothing |
| `perspective.py` | Corner ordering, cut-out, fast projection, screen-to-panel mapping |
| `styles.py` | The 21 art styles |
| `inference.py` | Background styling thread (OpenCV or Stable Diffusion / FLUX) |
| `canvas.py` | Air-drawing strokes, colors, undo |
| `gallery.py` | Saving + HTML gallery |
| `ui_overlay.py` | HUD, guides, countdown ring, flash |
| `selftest.py` | Camera-free tests, including the whole capture flow |
| `requirements.txt` / `requirements-lock.txt` | Version ranges / exact known-good versions |

## Ideas to extend

- Add your own style in `styles.py` (one function + one prompt)
- Record a short video of the drawing process
- Add a "sticker" gesture (open palm drops an emoji)

## License

This project is licensed under the [MIT License](LICENSE).