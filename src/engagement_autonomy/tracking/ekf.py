from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class EKF:
    # State: [x, y, vx, vy]
    x: np.ndarray
    P: np.ndarray
    Q: np.ndarray
    R: np.ndarray

    def predict(self, f_prop, F_jac, dt: float) -> None:
        # f_prop: function(x, dt) -> x_pred
        x_pred = f_prop(self.x, dt)
        F = F_jac(self.x, dt)
        self.x = x_pred
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z: np.ndarray, h, H_jac) -> None:
        z_pred = h(self.x)
        H = H_jac(self.x)
        y = z - z_pred
        # normalize bearing residual to [-pi, pi] if measurement is range/bearing
        if y.shape[0] >= 2:
            y[1] = (y[1] + np.pi) % (2*np.pi) - np.pi
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P
