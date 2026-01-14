from __future__ import annotations
import numpy as np
from .dynamics import f

def rk4_step(mu: float, s: np.ndarray, dt: float) -> np.ndarray:
    k1 = f(mu, s)
    k2 = f(mu, s + 0.5 * dt * k1)
    k3 = f(mu, s + 0.5 * dt * k2)
    k4 = f(mu, s + dt * k3)
    return s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
