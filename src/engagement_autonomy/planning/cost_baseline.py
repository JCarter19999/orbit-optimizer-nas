from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from ..sim.entities import Interceptor

@dataclass(frozen=True)
class BaselineWeights:
    w_time: float
    w_dv: float
    w_range: float

def estimate_time_to_close(agent: Interceptor, x: float, y: float, vx: float, vy: float) -> float:
    rel_p = np.array([x - agent.x, y - agent.y], dtype=float)
    rel_v = np.array([vx - agent.vx, vy - agent.vy], dtype=float)
    r = float(np.linalg.norm(rel_p))
    closing = -float(np.dot(rel_p / max(r, 1e-9), rel_v))
    closing = max(closing, 1e-3)
    return r / closing

def estimate_dv_need(agent: Interceptor, vx: float, vy: float) -> float:
    rel_v = np.array([vx - agent.vx, vy - agent.vy], dtype=float)
    return float(np.linalg.norm(rel_v))

def build_cost_matrix_from_tracks(
    agents: list[Interceptor],
    track_states: list[tuple[float, float, float, float]],
    track_active: list[bool],
    weights: BaselineWeights,
    max_pairs_per_agent: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    N, M = len(agents), len(track_states)
    C = np.full((N, M), 1e6, dtype=float)
    infeasible = np.zeros((N, M), dtype=bool)

    for i, ag in enumerate(agents):
        dists = []
        for j, (x, y, vx, vy) in enumerate(track_states):
            if not track_active[j]:
                infeasible[i, j] = True
                continue
            r = float(np.hypot(x - ag.x, y - ag.y))
            dists.append((r, j))
        dists.sort(key=lambda t: t[0])
        keep = set(j for _, j in (dists[:max_pairs_per_agent] if max_pairs_per_agent else dists))

        for j, (x, y, vx, vy) in enumerate(track_states):
            if j not in keep or not track_active[j]:
                infeasible[i, j] = True
                continue
            ttc = estimate_time_to_close(ag, x, y, vx, vy)
            dvn = estimate_dv_need(ag, vx, vy)
            rng = float(np.hypot(x - ag.x, y - ag.y))
            C[i, j] = weights.w_time * ttc + weights.w_dv * dvn + weights.w_range * rng

    return C, infeasible
