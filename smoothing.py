"""Smoothing helpers: stop jittery hand points from shaking the frame."""
import numpy as np


class AdaptiveEMA:
    """Exponential moving average whose strength adapts to speed.

    Slow movement  -> heavy smoothing (steady, no jitter).
    Fast movement  -> light smoothing (no lag when you move quickly).
    """

    def __init__(self, min_alpha: float = 0.25, gain: float = 0.02, max_alpha: float = 0.95):
        self.min_alpha = min_alpha
        self.gain = gain
        self.max_alpha = max_alpha
        self.value = None

    def update(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.value is None:
            self.value = x.copy()
            return self.value
        speed = float(np.mean(np.linalg.norm((x - self.value).reshape(-1, x.shape[-1]), axis=-1)))
        alpha = float(np.clip(self.min_alpha + self.gain * speed, self.min_alpha, self.max_alpha))
        self.value = alpha * x + (1.0 - alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None
