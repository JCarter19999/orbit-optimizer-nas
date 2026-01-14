from __future__ import annotations
from dataclasses import dataclass
from .hungarian import solve_assignment
from .cost_baseline import BaselineWeights, build_cost_matrix_from_tracks
from ..sim.entities import Interceptor

@dataclass
class HungarianBaselinePlanner:
    weights: BaselineWeights
    max_pairs_per_agent: int

    def plan(self, agents: list[Interceptor], track_states, track_active) -> list[tuple[int, int]]:
        C, infeasible = build_cost_matrix_from_tracks(
            agents=agents,
            track_states=track_states,
            track_active=track_active,
            weights=self.weights,
            max_pairs_per_agent=self.max_pairs_per_agent,
        )
        return solve_assignment(C, infeasible=infeasible)
