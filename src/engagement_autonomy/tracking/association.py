from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment

def mahalanobis2(y: np.ndarray, S: np.ndarray) -> float:
    return float(y.T @ np.linalg.inv(S) @ y)

def hungarian_associate(cost: np.ndarray, gate: np.ndarray | None = None) -> list[tuple[int,int]]:
    C = cost.copy()
    if gate is not None:
        C[gate] = 1e9
    r, c = linear_sum_assignment(C)
    pairs = []
    for i, j in zip(r, c):
        if gate is not None and gate[i, j]:
            continue
        pairs.append((int(i), int(j)))
    return pairs
