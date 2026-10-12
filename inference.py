"""Background styling so the video never freezes.

Default backend: OpenCV styles (instant, works everywhere).
Optional backend: a diffusion img2img model (e.g. --diffusion runwayml/stable-diffusion-v1-5).
If the model can't load, we quietly fall back to OpenCV and report why.
"""
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

from styles import STYLES


class AsyncStyler:
    def __init__(self, diffusion_model: Optional[str] = None):
        self.diffusion_model = diffusion_model
        self.error: Optional[str] = None
        self._pipe = None
        self._is_flux = False
        self._diffusion_failed = False   # after one failure we stop retrying (and stop waiting)
        self._lock = threading.Lock()
        self._job_id = 0
        self._result: Optional[Tuple[int, np.ndarray]] = None
        self._busy = False

    @property
    def busy(self) -> bool:
        return self._busy

    def submit(self, image: np.ndarray, style_idx: int) -> None:
        """Start styling in a thread. A newer submit makes older results obsolete."""
        with self._lock:
            self._job_id += 1
            job = self._job_id
            self._busy = True
        threading.Thread(target=self._run, args=(job, image.copy(), style_idx), daemon=True).start()

    def poll(self) -> Optional[np.ndarray]:
        """Return a finished image once (then None until the next result)."""
        with self._lock:
            if self._result is None:
                return None
            job, img = self._result
            self._result = None
            return img if job == self._job_id else None

    # ---- internals ----
    def _run(self, job: int, image: np.ndarray, style_idx: int) -> None:
        style = STYLES[style_idx % len(STYLES)]
        out = style.fn(image)  # fast preview first
        will_diffuse = bool(self.diffusion_model) and not self._diffusion_failed and bool(style.prompt)
        self._publish(job, out, final=not will_diffuse)
        if will_diffuse:
            try:
                out = self._diffuse(image, style.prompt)
                self._publish(job, out, final=True)
            except Exception as exc:  # no GPU / missing package / download failed
                self._diffusion_failed = True
                reason = str(exc).strip().splitlines()[0][:70] if str(exc).strip() else ""
                self.error = f"AI styling off: {type(exc).__name__} {reason} - using OpenCV styles".replace("  ", " ")
                with self._lock:
                    if job == self._job_id:
                        self._busy = False

    def _publish(self, job: int, img: np.ndarray, final: bool) -> None:
        with self._lock:
            if job != self._job_id:
                return
            self._result = (job, img)
            if final:
                self._busy = False

    def _load_pipe(self):
        """Load Stable Diffusion or FLUX img2img (model id or local folder). Slow the first time."""
        import torch
        from diffusers import AutoPipelineForImage2Image

        device = "cuda" if torch.cuda.is_available() else "cpu"
        is_flux = "flux" in str(self.diffusion_model).lower()
        if device == "cuda":
            dtype = torch.bfloat16 if is_flux else torch.float16
        else:
            dtype = torch.float32
        pipe = AutoPipelineForImage2Image.from_pretrained(self.diffusion_model, torch_dtype=dtype)
        if device == "cuda" and is_flux:
            pipe.enable_model_cpu_offload()   # FLUX is big; this keeps VRAM use sane
        else:
            pipe = pipe.to(device)
        return pipe, is_flux

    def _diffuse(self, image: np.ndarray, prompt: str) -> np.ndarray:
        if self._pipe is None:
            self._pipe, self._is_flux = self._load_pipe()
        from PIL import Image

        h, w = image.shape[:2]
        size = 768 if self._is_flux else 512
        src = Image.fromarray(cv2.cvtColor(cv2.resize(image, (size, size)), cv2.COLOR_BGR2RGB))
        if self._is_flux:   # FLUX: no classifier-free guidance; schnell needs only ~4 steps
            res = self._pipe(prompt=prompt, image=src, strength=0.6, guidance_scale=0.0,
                             num_inference_steps=4).images[0]
        else:
            res = self._pipe(prompt=prompt, image=src, strength=0.55, guidance_scale=7.5,
                             num_inference_steps=20).images[0]
        return cv2.resize(cv2.cvtColor(np.array(res), cv2.COLOR_RGB2BGR), (w, h))