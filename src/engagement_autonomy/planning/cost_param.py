from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from ..sim.entities import Interceptor


@dataclass(frozen=True)
class CostWeights:
    # Tunable knobs (AutoML / GA search space)
    w_range: float = 1.0        # distance now
    w_rel_vel: float = 0.5      # relative speed mismatch
    w_tgo: float = 0.2          # time-to-go proxy
    w_budget: float = 200.0     # penalty if dv proxy exceeds dv_remaining
    eps_speed: float = 1e-3     # avoid divide by zero


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def build_cost_matrix_from_tracks(
    agents: List[Interceptor],
    track_states: List[Tuple[float, float, float, float]],
    track_active: List[bool],
    weights: CostWeights,
    max_pairs_per_agent: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      C: (N,M) cost matrix
      infeasible: (N,M) bool mask for disallowed pairs
    """

    N, M = len(agents), len(track_states)
    C = np.full((N, M), 1e6, dtype=float)
    infeasible = np.zeros((N, M), dtype=bool)

    for i, ag in enumerate(agents):
        # Candidate pruning: only keep nearest K targets (optional)
        dists = []
        for j, (x, y, vx, vy) in enumerate(track_states):
            if not track_active[j]:
                infeasible[i, j] = True
                continue
            r = float(np.hypot(x - ag.x, y - ag.y))
            dists.append((r, j))
        dists.sort(key=lambda t: t[0])
        keep = set(j for _, j in (dists[:max_pairs_per_agent] if max_pairs_per_agent else dists))

        # Compute costs
        for j, (x, y, vx, vy) in enumerate(track_states):
            if (j not in keep) or (not track_active[j]):
                infeasible[i, j] = True
                continue

            rel_p = np.array([x - ag.x, y - ag.y], dtype=float)
            rel_v = np.array([vx - ag.vx, vy - ag.vy], dtype=float)

            r = _norm(rel_p)
            rel_speed = _norm(rel_v)
            ag_speed = max(weights.eps_speed, _norm(np.array([ag.vx, ag.vy], dtype=float)))

            # time-to-go proxy (not perfect physics; stable + tunable)
            tgo = r / ag_speed

            # dv proxy: mismatch + small range term to prefer nearer engagements
            dv_proxy = rel_speed + 0.001 * r  # coefficient intentionally small

            budget_pen = weights.w_budget if dv_proxy > max(1e-9, ag.dv_remaining) else 0.0

            C[i, j] = (
                weights.w_range * r
                + weights.w_rel_vel * rel_speed
                + weights.w_tgo * tgo
                + budget_pen
            )

    return C, infeasible
