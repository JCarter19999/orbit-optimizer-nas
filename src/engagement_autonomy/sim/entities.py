from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Body2D:
    x: float
    y: float
    vx: float
    vy: float

    def state(self) -> np.ndarray:
        return np.array([self.x, self.y, self.vx, self.vy], dtype=float)

    @staticmethod
    def from_state(s: np.ndarray) -> "Body2D":
        return Body2D(float(s[0]), float(s[1]), float(s[2]), float(s[3]))

@dataclass
class Interceptor(Body2D):
    dv_remaining: float
    burns_left: int
    cooldown_remaining: float

@dataclass
class Target(Body2D):
    active: bool = True
