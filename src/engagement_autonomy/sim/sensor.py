from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .entities import Target

@dataclass(frozen=True)
class RangeBearingSensor:
    sigma_range_km: float
    sigma_bearing_rad: float

    def measure_target(self, rng: np.random.Generator, tg: Target) -> np.ndarray:
        # measurement to origin: z = [range, bearing]
        r = float(np.hypot(tg.x, tg.y))
        theta = float(np.arctan2(tg.y, tg.x))
        r_meas = r + float(rng.normal(0.0, self.sigma_range_km))
        th_meas = theta + float(rng.normal(0.0, self.sigma_bearing_rad))
        return np.array([r_meas, th_meas], dtype=float)
