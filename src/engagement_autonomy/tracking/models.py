from __future__ import annotations
import numpy as np
from ..sim.dynamics import two_body_accel
from ..sim.integrators import rk4_step

def propagate_two_body(mu: float, x: np.ndarray, dt: float) -> np.ndarray:
    # Use RK4 on the continuous dynamics to propagate state.
    return rk4_step(mu, x, dt)

def F_jac_two_body(mu: float, x: np.ndarray, dt: float) -> np.ndarray:
    # Discrete-time Jacobian approximation via finite differences on the propagator.
    # This is stable for a toy sim and keeps the math manageable.
    eps = 1e-6
    n = x.shape[0]
    F = np.zeros((n, n), dtype=float)
    fx = propagate_two_body(mu, x, dt)
    for i in range(n):
        dx = np.zeros(n, dtype=float)
        dx[i] = eps
        fxi = propagate_two_body(mu, x + dx, dt)
        F[:, i] = (fxi - fx) / eps
    return F

def h_range_bearing(x: np.ndarray) -> np.ndarray:
    px, py, vx, vy = x
    r = np.sqrt(px*px + py*py)
    th = np.arctan2(py, px)
    return np.array([r, th], dtype=float)

def H_jac_range_bearing(x: np.ndarray) -> np.ndarray:
    px, py, vx, vy = x
    r2 = px*px + py*py
    r = np.sqrt(max(r2, 1e-12))
    # dr/dx, dr/dy
    drdx = px / r
    drdy = py / r
    # dtheta/dx, dtheta/dy
    dtdx = -py / max(r2, 1e-12)
    dtdy = px / max(r2, 1e-12)
    H = np.array([
        [drdx, drdy, 0.0, 0.0],
        [dtdx, dtdy, 0.0, 0.0],
    ], dtype=float)
    return H
