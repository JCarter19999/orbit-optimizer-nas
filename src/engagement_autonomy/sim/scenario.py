from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from .entities import Interceptor, Target

@dataclass(frozen=True)
class Scenario:
    agents: list[Interceptor]
    targets: list[Target]

def make_scenario(
    rng: np.random.Generator,
    mu: float,  # unused in ground-air mode (kept for config compatibility)
    n_agents: int,
    n_targets: int,
    r0: float,  # interpreted as horizontal half-width of the battlespace (km)
    agent_pos_sigma: float,
    agent_vel_sigma: float,
    target_pos_sigma: float,
    target_vel_sigma: float,
    dv_budget: float,
    max_burns: int,
    target_altitude_km: float = 10.0,
    target_speed_kms: float = 0.18,
    interceptor_max_speed_kms: float = 0.60,
    spawn_mode: str = "opposite_pairs",
) -> Scenario:
    """
    Ground-air (planar) demo scenario.

    - Targets ("air") start at y ~ target_altitude_km with random x across [-r0, r0].
      They travel roughly left-to-right with some vertical component.

    - Interceptors ("ground") start near y ~ 0. For 'opposite_pairs', interceptor j spawns
      across the origin from target j (x = -x_target) so assignments are non-trivial but intuitive.

    Units:
      position: km
      velocity: km/s
      dt (world): seconds
    """

    # --- Targets ---
    targets: list[Target] = []
    xs = rng.uniform(-r0, r0, size=n_targets)
    for j in range(n_targets):
        x = float(xs[j] + rng.normal(0.0, target_pos_sigma))
        y = float(target_altitude_km + rng.normal(0.0, target_pos_sigma))

        # Nominal air motion: mostly horizontal, slight vertical
        vx0 = float(rng.uniform(0.6, 1.0) * target_speed_kms) * (1.0 if rng.random() < 0.5 else -1.0)
        vy0 = float(rng.uniform(-0.4, 0.4) * target_speed_kms)

        vx = float(vx0 + rng.normal(0.0, target_vel_sigma))
        vy = float(vy0 + rng.normal(0.0, target_vel_sigma))

        targets.append(Target(x=x, y=y, vx=vx, vy=vy, active=True))

    # --- Agents ---
    agents: list[Interceptor] = []
    if spawn_mode == "opposite_pairs" and n_agents == n_targets and n_targets > 0:
        for j in range(n_agents):
            tj = targets[j]
            ax = float(-tj.x + rng.normal(0.0, agent_pos_sigma))
            ay = float(0.0 + rng.normal(0.0, agent_pos_sigma))
            avx = float(rng.normal(0.0, agent_vel_sigma))
            avy = float(rng.normal(0.0, agent_vel_sigma))
            agents.append(
                Interceptor(
                    x=ax, y=ay, vx=avx, vy=avy,
                    dv_remaining=float(dv_budget),
                    burns_left=int(max_burns),
                    cooldown_remaining=0.0,
                    max_speed_kms=float(interceptor_max_speed_kms),
                )
            )
    else:
        # generic spawn: cluster around origin
        for _ in range(n_agents):
            ax = float(rng.normal(0.0, agent_pos_sigma))
            ay = float(rng.normal(0.0, agent_pos_sigma))
            avx = float(rng.normal(0.0, agent_vel_sigma))
            avy = float(rng.normal(0.0, agent_vel_sigma))
            agents.append(
                Interceptor(
                    x=ax, y=ay, vx=avx, vy=avy,
                    dv_remaining=float(dv_budget),
                    burns_left=int(max_burns),
                    cooldown_remaining=0.0,
                    max_speed_kms=float(interceptor_max_speed_kms),
                )
            )

    return Scenario(agents=agents, targets=targets)
