from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment

def solve_assignment(cost: np.ndarray, infeasible: np.ndarray | None = None) -> list[tuple[int, int]]:
    C = cost.copy()
    if infeasible is not None:
        C[infeasible] = 1e9
    row_ind, col_ind = linear_sum_assignment(C)
    matches: list[tuple[int, int]] = []
    for r, c in zip(row_ind, col_ind):
        if infeasible is not None and infeasible[r, c]:
            continue
        matches.append((int(r), int(c)))
    return matches
