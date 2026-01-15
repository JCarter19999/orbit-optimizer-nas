from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .entities import Interceptor, Target

@dataclass(frozen=True)
class Scenario:
    agents: list[Interceptor]
    targets: list[Target]


def state_from_orbital_params(mu: float, r: float, theta: float) -> tuple[float, float, float, float]:
    """
    Circular orbit state at radius r and angle theta.
    Position: (r cosθ, r sinθ)
    Velocity magnitude: sqrt(mu/r), direction perpendicular to radius (prograde)
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    v = np.sqrt(mu / r)
    # prograde tangential unit vector is [-sinθ, cosθ]
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
) -> Scenario:
    """
    Agents: clustered near a nominal orbit (small jitter)
    Targets: each gets its own orbital angle and slightly different radius + optional radial velocity,
             so trajectories meaningfully diverge.
    """

    # --- Agents: start near the same general neighborhood (fine for baseline) ---
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

    # Spread targets around the orbit so they start separated
    thetas = rng.uniform(0.0, 2.0 * np.pi, size=n_targets)

    # Give each target a slightly different orbital radius
    # (enough to separate periods but not explode)
    # Example: +/- 50 km around r0; tune as needed
    r_offsets = rng.normal(0.0, 50.0, size=n_targets)
    rs = np.clip(r0 + r_offsets, r0 - 150.0, r0 + 150.0)

    for j in range(n_targets):
        x, y, vx, vy = state_from_orbital_params(mu, float(rs[j]), float(thetas[j]))

        # Add additional spread in position/velocity (your existing knobs)
        dx, dy = rng.normal(0.0, target_pos_sigma, size=2)
        dvx, dvy = rng.normal(0.0, target_vel_sigma, size=2)

        # Optional: inject a small radial velocity component to create mild eccentricity
        # This makes paths diverge more (and is still physically meaningful).
        # Keep it small or you’ll get extreme orbits.
        # radial unit vector is [cosθ, sinθ]
        erx, ery = np.cos(thetas[j]), np.sin(thetas[j])
        v_rad = float(rng.normal(0.0, 0.0008))  # km/s (tune: 0.0003–0.002)
        vx += v_rad * erx
        vy += v_rad * ery

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
