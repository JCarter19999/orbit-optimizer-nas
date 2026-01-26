from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .entities import Interceptor, Target

@dataclass
class World:
    # mu kept for config compatibility; unused in planar mode
    mu: float
    dt: float
    intercept_radius_km: float
    burn_dv_kms: float
    burn_cooldown_s: float

    def step(self, agents: list[Interceptor], targets: list[Target]) -> None:
        """Simple planar propagation (constant velocity)."""
        dt = float(self.dt)

        for ag in agents:
            ag.x = float(ag.x + ag.vx * dt)
            ag.y = float(ag.y + ag.vy * dt)
            ag.cooldown_remaining = max(0.0, float(ag.cooldown_remaining) - dt)

        for tg in targets:
            if not tg.active:
                continue
            tg.x = float(tg.x + tg.vx * dt)
            tg.y = float(tg.y + tg.vy * dt)

    def _lead_aim_unit(self, ag: Interceptor, tg: Target) -> np.ndarray:
        """Lead pursuit aim direction using constant-velocity target model."""
        rel_p = np.array([tg.x - ag.x, tg.y - ag.y], dtype=float)
        rel_v = np.array([tg.vx - ag.vx, tg.vy - ag.vy], dtype=float)

        r = float(np.linalg.norm(rel_p))
        vcap = max(1e-6, float(getattr(ag, "max_speed_kms", 0.3)))

        # basic lead time: range / closing capability
        tgo = np.clip(r / vcap, 0.0, 60.0)
        aim = rel_p + rel_v * tgo

        n = float(np.linalg.norm(aim))
        if n < 1e-9:
            n = max(1e-9, r)
            return rel_p / n
        return aim / n

    def apply_burn_toward(self, ag: Interceptor, tg: Target) -> bool:
        """
        In planar mode, "burn" means: steer velocity toward an aim point, capped by max speed.
        We account fuel as the commanded delta-v magnitude.
        """
        if ag.cooldown_remaining > 0.0:
            return False
        if not tg.active:
            return False

        vcap = float(getattr(ag, "max_speed_kms", 0.3))
        u = self._lead_aim_unit(ag, tg)

        desired_v = vcap * u  # km/s
        cur_v = np.array([ag.vx, ag.vy], dtype=float)
        dv_vec = desired_v - cur_v
        dv = float(np.linalg.norm(dv_vec))

        if dv < 1e-9:
            return False

        # Fuel check: dv_remaining is budget of total velocity change we can command
        if ag.dv_remaining <= 1e-9:
            return False

        # Apply as much steering as budget allows this step
        dv_allowed = min(dv, float(ag.dv_remaining), float(self.burn_dv_kms))
        step_v = cur_v + dv_vec * (dv_allowed / dv)

        ag.vx = float(step_v[0])
        ag.vy = float(step_v[1])

        ag.dv_remaining = float(ag.dv_remaining - dv_allowed)

        ag.cooldown_remaining = float(self.burn_cooldown_s)
        return True
