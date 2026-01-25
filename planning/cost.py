from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence
import math


@dataclass(frozen=True)
class CostWeights:
    w_range: float = 1.0
    w_rel_vel: float = 0.5
    w_tgo: float = 0.2
    w_budget: float = 200.0
    eps_speed: float = 0.01


def _norm2(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)


def compute_cost_matrix(
    agents: Sequence[dict],
    targets: Sequence[dict],
    weights: CostWeights,
) -> List[List[float]]:
    """
    agents: list of dicts with x,y,vx,vy,dv_remaining
    targets: list of dicts with x,y,vx,vy,active
    """
    costs: List[List[float]] = []
    for a in agents:
        row: List[float] = []
        ax, ay = float(a["x"]), float(a["y"])
        avx, avy = float(a["vx"]), float(a["vy"])
        dv_rem = float(a.get("dv_remaining", 0.0))
        speed = _norm2(avx, avy)

        for g in targets:
            if not bool(g.get("active", True)):
                # Inactive targets should never be chosen
                row.append(1e9)
                continue

            gx, gy = float(g["x"]), float(g["y"])
            gvx, gvy = float(g["vx"]), float(g["vy"])

            rx = gx - ax
            ry = gy - ay
            r = _norm2(rx, ry)

            rvx = gvx - avx
            rvy = gvy - avy
            rel_v = _norm2(rvx, rvy)

            # crude time-to-go proxy
            tgo = r / max(weights.eps_speed, speed)

            # crude dv "feasibility" proxy (intentionally simple)
            dv_proxy = rel_v + 0.001 * r  # scale r lightly so units don't dominate

            budget_pen = weights.w_budget if dv_proxy > max(1e-6, dv_rem) else 0.0

            c = (
                weights.w_range * r
                + weights.w_rel_vel * rel_v
                + weights.w_tgo * tgo
                + budget_pen
            )
            row.append(float(c))

        costs.append(row)

    return costs
