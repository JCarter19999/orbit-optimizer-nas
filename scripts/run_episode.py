from __future__ import annotations
import argparse
from datetime import datetime

import numpy as np

from engagement_autonomy.config import load_yaml, parse_config
from engagement_autonomy.utils.rng import make_rng
from engagement_autonomy.utils.logging import ensure_dir, write_jsonl

from engagement_autonomy.sim.scenario import make_scenario
from engagement_autonomy.sim.world import World
from engagement_autonomy.sim.sensor import RangeBearingSensor

from engagement_autonomy.tracking.track_manager import TrackManager
from engagement_autonomy.planning.planner import HungarianBaselinePlanner
from engagement_autonomy.planning.cost_param import CostWeights

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    args = ap.parse_args()

    cfg = parse_config(load_yaml(args.config))
    rng = make_rng(cfg.run.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = cfg.run.out_dir / f"episode_{ts}"
    ensure_dir(out_dir)

    scenario = make_scenario(
        rng=rng,
        mu=cfg.sim.mu,
        n_agents=cfg.scenario.n_agents,
        n_targets=cfg.scenario.n_targets,
        r0=cfg.scenario.ref_orbit_radius_km,
        agent_pos_sigma=cfg.scenario.agent_pos_sigma_km,
        agent_vel_sigma=cfg.scenario.agent_vel_sigma_kms,
        target_pos_sigma=cfg.scenario.target_pos_sigma_km,
        target_vel_sigma=cfg.scenario.target_vel_sigma_kms,
        dv_budget=cfg.sim.dv_budget_kms,
        max_burns=cfg.sim.max_burns_per_agent,
    )

    world = World(
        mu=cfg.sim.mu,
        dt=cfg.sim.dt,
        intercept_radius_km=cfg.sim.intercept_radius_km,
        burn_dv_kms=cfg.sim.burn_dv_kms,
        burn_cooldown_s=cfg.sim.burn_cooldown_s,
    )

    sensor = RangeBearingSensor(
        sigma_range_km=cfg.sensor.sigma_range_km,
        sigma_bearing_rad=cfg.sensor.sigma_bearing_rad,
    )

    # Tracking init (phase-1: identity known)
    track_manager = None
    if cfg.tracking.enabled:
        Q = np.diag([cfg.tracking.q_pos, cfg.tracking.q_pos, cfg.tracking.q_vel, cfg.tracking.q_vel])
        R = np.diag([cfg.tracking.r_range**2, cfg.tracking.r_bearing**2])
        P0 = np.diag([1.0, 1.0, 1e-3, 1e-3])
        track_manager = TrackManager(mu=cfg.sim.mu, Q=Q, R=R)
        truth0 = [tg.state() for tg in scenario.targets]
        track_manager.init_from_truth(truth0, P0=P0)

    planner = HungarianBaselinePlanner(
        weights=CostWeights(**cfg.planner.weights),
        max_pairs_per_agent=cfg.planner.max_pairs_per_agent,
    )

    records = []
    for step in range(cfg.sim.steps):
        t = step * cfg.sim.dt

        # EKF predict/update (identity-known in this scaffold)
        if track_manager is not None:
            track_manager.predict_all(dt=cfg.sim.dt)
            meas = [sensor.measure_target(rng, tg) for tg in scenario.targets]
            track_manager.update_all_identity_known(meas)

        # Engagement decision at period
        matches = []
        if step % cfg.sim.decision_period == 0:
            if track_manager is not None:
                track_states, track_active, track_ids = track_manager.get_track_states()
            else:
                track_states = [(tg.x, tg.y, tg.vx, tg.vy) for tg in scenario.targets]
                track_active = [tg.active for tg in scenario.targets]
                track_ids = list(range(len(scenario.targets)))

            matches = planner.plan(scenario.agents, track_states, track_active)

            # Map match target index -> choose that truth target for burn (phase-1 alignment)
            for ai, tj in matches:
                if tj < len(scenario.targets):
                    world.apply_burn_toward(scenario.agents[ai], scenario.targets[tj])

        # Step physics
        world.step(scenario.agents, scenario.targets)
        if track_manager is not None:
            # phase-1: 1 track per target, same ordering
            for j in range(min(len(track_manager.tracks), len(scenario.targets))):
                track_manager.tracks[j].active = bool(scenario.targets[j].active)
        # Log
        rec = {
            "t": t,
            "step": step,
            "agents": [
                {
                    "x": a.x, "y": a.y, "vx": a.vx, "vy": a.vy,
                    "dv_remaining": a.dv_remaining,
                    "burns_left": a.burns_left,
                    "cooldown_remaining": a.cooldown_remaining,
                } for a in scenario.agents
            ],
            "targets_truth": [
                {"x": g.x, "y": g.y, "vx": g.vx, "vy": g.vy, "active": g.active}
                for g in scenario.targets
            ],
            "matches": matches,
            "n_targets_active": sum(1 for g in scenario.targets if g.active),
        }
        if track_manager is not None:
            track_states, track_active, track_ids = track_manager.get_track_states()
            rec["tracks"] = [
                {"id": tid, "x": xs[0], "y": xs[1], "vx": xs[2], "vy": xs[3], "active": act}
                for xs, act, tid in zip(track_states, track_active, track_ids)
            ]
        records.append(rec)

        if all(not g.active for g in scenario.targets):
            break

    write_jsonl(out_dir / "episode.jsonl", records)
    print(f"Wrote: {out_dir / 'episode.jsonl'}")

if __name__ == "__main__":
    main()
