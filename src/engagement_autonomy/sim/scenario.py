from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .entities import Interceptor, Target

@dataclass(frozen=True)
class Scenario:
    agents: list[Interceptor]
    targets: list[Target]

def circular_orbit_state(mu: float, r: float) -> tuple[float, float, float, float]:
    x, y = r, 0.0
    v = np.sqrt(mu / r)
    vx, vy = 0.0, v
    return float(x), float(y), float(vx), float(vy)

def make_scenario(
    rng: np.random.Generator,
    mu: float,
    n_agents: int,
    n_targets: int,
    r0: float,
    agent_pos_sigma: float,
    agent_vel_sigma: float,
    target_pos_sigma: float,
    target_vel_sigma: float,
    dv_budget: float,
    max_burns: int,
) -> Scenario:
    x0, y0, vx0, vy0 = circular_orbit_state(mu, r0)

    def jitter(pos_sigma: float, vel_sigma: float) -> tuple[float, float, float, float]:
        dx, dy = rng.normal(0.0, pos_sigma, size=2)
        dvx, dvy = rng.normal(0.0, vel_sigma, size=2)
        return x0 + dx, y0 + dy, vx0 + dvx, vy0 + dvy

    agents: list[Interceptor] = []
    for _ in range(n_agents):
        x, y, vx, vy = jitter(agent_pos_sigma, agent_vel_sigma)
        agents.append(
            Interceptor(
                x=x, y=y, vx=vx, vy=vy,
                dv_remaining=dv_budget,
                burns_left=max_burns,
                cooldown_remaining=0.0,
            )
        )

    targets: list[Target] = []
    for _ in range(n_targets):
        x, y, vx, vy = jitter(target_pos_sigma, target_vel_sigma)
        targets.append(Target(x=x, y=y, vx=vx, vy=vy, active=True))

    return Scenario(agents=agents, targets=targets)
