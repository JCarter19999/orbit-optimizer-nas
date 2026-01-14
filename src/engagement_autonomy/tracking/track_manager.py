from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .ekf import EKF
from .models import propagate_two_body, F_jac_two_body, h_range_bearing, H_jac_range_bearing

@dataclass
class Track:
    track_id: int
    ekf: EKF
    active: bool = True

class TrackManager:
    def __init__(self, mu: float, Q: np.ndarray, R: np.ndarray):
        self.mu = mu
        self.Q = Q
        self.R = R
        self._next_id = 0
        self.tracks: list[Track] = []

    def init_from_truth(self, truth_states: list[np.ndarray], P0: np.ndarray) -> None:
        # For phase-1: identity known; initialize one track per target.
        self.tracks = []
        for x0 in truth_states:
            ekf = EKF(x=x0.copy(), P=P0.copy(), Q=self.Q.copy(), R=self.R.copy())
            self.tracks.append(Track(track_id=self._next_id, ekf=ekf, active=True))
            self._next_id += 1

    def predict_all(self, dt: float) -> None:
        for tr in self.tracks:
            if not tr.active:
                continue
            tr.ekf.predict(
                f_prop=lambda x, d: propagate_two_body(self.mu, x, d),
                F_jac=lambda x, d: F_jac_two_body(self.mu, x, d),
                dt=dt,
            )

    def update_all_identity_known(self, measurements: list[np.ndarray]) -> None:
        # Phase-1: measurement list aligned with tracks (same ordering).
        for tr, z in zip(self.tracks, measurements):
            if not tr.active:
                continue
            tr.ekf.update(z=z, h=h_range_bearing, H_jac=H_jac_range_bearing)

    def get_track_states(self) -> tuple[list[tuple[float,float,float,float]], list[bool], list[int]]:
        states = []
        active = []
        ids = []
        for tr in self.tracks:
            x = tr.ekf.x
            states.append((float(x[0]), float(x[1]), float(x[2]), float(x[3])))
            active.append(bool(tr.active))
            ids.append(int(tr.track_id))
        return states, active, ids
