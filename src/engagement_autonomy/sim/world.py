from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .entities import Interceptor, Target
from .integrators import rk4_step

@dataclass
class World:
    mu: float
    dt: float
    intercept_radius_km: float
    burn_dv_kms: float
    burn_cooldown_s: float

    @staticmethod
    def distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a[:2] - b[:2]))

    def step(self, agents: list[Interceptor], targets: list[Target]) -> None:
        for ag in agents:
            s = rk4_step(self.mu, ag.state(), self.dt)
            ag.x, ag.y, ag.vx, ag.vy = map(float, s)
            ag.cooldown_remaining = max(0.0, ag.cooldown_remaining - self.dt)

        for tg in targets:
            if not tg.active:
                continue
            s = rk4_step(self.mu, tg.state(), self.dt)
            tg.x, tg.y, tg.vx, tg.vy = map(float, s)

        # proximity intercept
        for tg in targets:
            if not tg.active:
                continue
            tstate = tg.state()
            for ag in agents:
                if self.distance(ag.state(), tstate) <= self.intercept_radius_km:
                    tg.active = False
                    break

    def apply_burn_toward(self, ag: Interceptor, tg: Target) -> bool:
        if ag.burns_left <= 0:
            return False
        if ag.dv_remaining < self.burn_dv_kms:
            return False
        if ag.cooldown_remaining > 0.0:
            return False
        if not tg.active:
            return False

        rel = np.array([tg.x - ag.x, tg.y - ag.y], dtype=float)
        n = np.linalg.norm(rel)
        if n < 1e-9:
            return False
        u = rel / n
        ag.vx += float(self.burn_dv_kms * u[0])
        ag.vy += float(self.burn_dv_kms * u[1])
        ag.dv_remaining -= self.burn_dv_kms
        ag.burns_left -= 1
        ag.cooldown_remaining = self.burn_cooldown_s
        return True
