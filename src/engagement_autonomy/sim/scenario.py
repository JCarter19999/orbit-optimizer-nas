from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from .entities import Interceptor, Target


@dataclass(frozen=True)
class Scenario:
    agents: list[Interceptor]
    targets: list[Target]


def state_from_orbital_params(mu: float, r: float, theta: float) -> tuple[float, float, float, float]:
    """Circular orbit state at radius r and angle theta (2D, prograde)."""
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    v = np.sqrt(mu / r)
    # Tangential unit vector is [-sinθ, cosθ]
    vx = -v * np.sin(theta)
    vy =  v * np.cos(theta)
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
    target_radius_sigma_km: float = 50.0,
    target_radial_vel_sigma_kms: float = 0.0008,
) -> Scenario:
    """
    Agents: clustered around one neighborhood (baseline).
    Targets: diversified by orbital phase + slightly different radius + small radial velocity (mild eccentricity),
             so Hungarian assignment is non-trivial.
    """
    # --- Agents: start near a common neighborhood ---
    theta_agents = float(rng.uniform(0.0, 2.0 * np.pi))
    xa0, ya0, vxa0, vya0 = state_from_orbital_params(mu, r0, theta_agents)

    agents: list[Interceptor] = []
    for _ in range(n_agents):
        dx, dy = rng.normal(0.0, agent_pos_sigma, size=2)
        dvx, dvy = rng.normal(0.0, agent_vel_sigma, size=2)
        agents.append(
            Interceptor(
                x=float(xa0 + dx),
                y=float(ya0 + dy),
                vx=float(vxa0 + dvx),
                vy=float(vya0 + dvy),
                dv_remaining=float(dv_budget),
                burns_left=int(max_burns),
                cooldown_remaining=0.0,
            )
        )

    # --- Targets: diversify orbits ---
    targets: list[Target] = []
    thetas = rng.uniform(0.0, 2.0 * np.pi, size=n_targets)

    r_offsets = rng.normal(0.0, target_radius_sigma_km, size=n_targets)
    rs = np.clip(r0 + r_offsets, r0 - 3.0 * target_radius_sigma_km, r0 + 3.0 * target_radius_sigma_km)

    for j in range(n_targets):
        x, y, vx, vy = state_from_orbital_params(mu, float(rs[j]), float(thetas[j]))

        dx, dy = rng.normal(0.0, target_pos_sigma, size=2)
        dvx, dvy = rng.normal(0.0, target_vel_sigma, size=2)

        # Small radial velocity -> mild eccentricity -> more divergence over time
        erx, ery = float(np.cos(thetas[j])), float(np.sin(thetas[j]))
        v_rad = float(rng.normal(0.0, target_radial_vel_sigma_kms))
        vx = float(vx + v_rad * erx)
        vy = float(vy + v_rad * ery)

        targets.append(
            Target(
                x=float(x + dx),
                y=float(y + dy),
                vx=float(vx + dvx),
                vy=float(vy + dvy),
                active=True,
            )
        )

    return Scenario(agents=agents, targets=targets)
