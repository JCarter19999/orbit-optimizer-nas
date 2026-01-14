from __future__ import annotations
import numpy as np

def two_body_accel(mu: float, x: float, y: float) -> tuple[float, float]:
    r2 = x*x + y*y
    r = np.sqrt(r2)
    r3 = max(r2 * r, 1e-9)
    ax = -mu * x / r3
    ay = -mu * y / r3
    return float(ax), float(ay)

def f(mu: float, s: np.ndarray) -> np.ndarray:
    x, y, vx, vy = s
    ax, ay = two_body_accel(mu, float(x), float(y))
    return np.array([vx, vy, ax, ay], dtype=float)
