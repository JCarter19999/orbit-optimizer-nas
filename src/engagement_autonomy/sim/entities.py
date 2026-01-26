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
    # "Fuel" budget (km/s equivalent) we decrement based on velocity changes we command.
    dv_remaining: float
    # Max number of major retarget / steering updates allowed (kept for legacy compatibility)
    burns_left: int
    cooldown_remaining: float
    # Kinematic capability for the ground-air (planar) demo
    max_speed_kms: float = 0.30   # ~300 m/s

@dataclass
class Target(Body2D):
    active: bool = True
