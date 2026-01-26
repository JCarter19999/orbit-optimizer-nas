from __future__ import annotations

import argparse
from datetime import datetime
from dataclasses import fields

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


def try_intercepts(
    agents,
    targets,
    matches,
    kill_radius_km: float,
    v_close_min_kms: float = 0.0,
):
    """
    Mark targets inactive if assigned agent is within kill radius.
    Returns list of target indices neutralized this step.
    """
    killed: list[int] = []
    kill_r = float(kill_radius_km)

    for ai, tj in matches:
        if tj < 0 or tj >= len(targets):
            continue

        a = agents[ai]
        t = targets[tj]

        if not getattr(t, "active", True):
            continue

        rx = t.x - a.x
        ry = t.y - a.y
        r = float(np.hypot(rx, ry))
        if r > kill_r:
            continue

        # Optional closing speed check
        if v_close_min_kms > 0.0:
            rvx = t.vx - a.vx
            rvy = t.vy - a.vy
            closing = -(rx * rvx + ry * rvy) / max(1e-9, r)
            if closing < v_close_min_kms:
                continue

        t.active = False
        killed.append(tj)

    return killed


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

    # ---- Planner weights: normalize config keys + instantiate planner ----
    w_raw = dict(cfg.planner.weights)

    ALIASES = {
        "w_time": "w_tgo",
        "w_dv": "w_budget",       # legacy -> new
        "w_range_km": "w_range",
        "w_relvel": "w_rel_vel",
    }

    w_norm = {ALIASES.get(k, k): v for k, v in w_raw.items()}

    allowed = {f.name for f in fields(CostWeights)}
    ignored = sorted(set(w_norm.keys()) - allowed)
    if ignored:
        print(f"[WARN] Ignoring unknown planner weight keys: {ignored}")

    w = {k: v for k, v in w_norm.items() if k in allowed}
    weights = CostWeights(**w)

    planner = HungarianBaselinePlanner(
        weights=weights,
        max_pairs_per_agent=int(getattr(cfg.planner, "max_pairs_per_agent", 6)),
    )
    # --------------------------------------------------------------------

    # Tracking init (phase-1: identity known)
    track_manager = None
    if cfg.tracking.enabled:
        Q = np.diag([cfg.tracking.q_pos, cfg.tracking.q_pos, cfg.tracking.q_vel, cfg.tracking.q_vel])
        R = np.diag([cfg.tracking.r_range**2, cfg.tracking.r_bearing**2])
        P0 = np.diag([1.0, 1.0, 1e-3, 1e-3])
        track_manager = TrackManager(mu=cfg.sim.mu, Q=Q, R=R)
        truth0 = [tg.state() for tg in scenario.targets]
        track_manager.init_from_truth(truth0, P0=P0)

    # Engagement / intercept params (with safe defaults)
    kill_radius_km = float(cfg.engagement.kill_radius_km)
    v_close_min_kms = float(cfg.engagement.v_close_min_kms)

    records = []
    matches = []

    committed_target = [-1] * len(scenario.agents)
    current_target = [-1] * len(scenario.agents)

    commit_radius_km = cfg.engagement.commit_radius_km

    for step in range(cfg.sim.steps):
        t = step * cfg.sim.dt

        # EKF predict/update (identity-known)
        if track_manager is not None:
            track_manager.predict_all(dt=cfg.sim.dt)
            meas = [sensor.measure_target(rng, tg) for tg in scenario.targets]
            track_manager.update_all_identity_known(meas)

        # Decide + burn periodically
        matches = []
        if step % cfg.sim.decision_period == 0:
            # Drop commitments to dead targets
            for ai, tj in enumerate(committed_target):
                if tj >= 0 and (tj >= len(scenario.targets) or not scenario.targets[tj].active):
                    committed_target[ai] = -1

            if track_manager is not None:
                track_states, track_active, track_ids = track_manager.get_track_states()
            else:
                track_states = [(tg.x, tg.y, tg.vx, tg.vy) for tg in scenario.targets]
                track_active = [tg.active for tg in scenario.targets]
                track_ids = list(range(len(scenario.targets)))

            # Honor existing commitments by locking those pairs and masking the targets
            locked_matches: list[tuple[int, int]] = []
            locked_targets: set[int] = set()
            for ai, tj in enumerate(committed_target):
                if tj >= 0 and tj < len(track_active) and track_active[tj]:
                    locked_matches.append((ai, tj))
                    locked_targets.add(tj)

            track_active_plan = [act and (j not in locked_targets) for j, act in enumerate(track_active)]
            free_agent_idx = [i for i, tj in enumerate(committed_target) if tj < 0]

            planned: list[tuple[int, int]] = []
            if free_agent_idx:
                plan_agents = [scenario.agents[i] for i in free_agent_idx]
                planned_raw = planner.plan(plan_agents, track_states, track_active_plan)
                planned = [(free_agent_idx[i], tj) for i, tj in planned_raw]

            matches = locked_matches + planned

            # Update current target intents based on this planning cycle
            current_target = [-1] * len(scenario.agents)
            for ai, tj in matches:
                if 0 <= tj < len(scenario.targets):
                    current_target[ai] = tj

            # Apply burn to the matched truth target (phase-1 alignment)
            for ai, tj in matches:
                if 0 <= tj < len(scenario.targets):
                    world.apply_burn_toward(scenario.agents[ai], scenario.targets[tj])

        # Step physics
        world.step(scenario.agents, scenario.targets)

        # Intercepts (MUST be inside the loop, after physics update)
        killed = try_intercepts(
            scenario.agents,
            scenario.targets,
            matches,
            kill_radius_km=kill_radius_km,
            v_close_min_kms=v_close_min_kms,
        )
        # Commit agents to targets once close enough; drop commits when target dies
        for ai, tj in matches:
            if 0 <= tj < len(scenario.targets) and committed_target[ai] < 0:
                rx = scenario.targets[tj].x - scenario.agents[ai].x
                ry = scenario.targets[tj].y - scenario.agents[ai].y
                if float(np.hypot(rx, ry)) <= commit_radius_km:
                    committed_target[ai] = tj
        if killed:
            for ai, ct in enumerate(committed_target):
                if ct in killed:
                    committed_target[ai] = -1

        # Sync track activity after kills (important if tracking enabled)
        if track_manager is not None:
            for j in range(min(len(track_manager.tracks), len(scenario.targets))):
                track_manager.tracks[j].active = bool(scenario.targets[j].active)

        # Continuous pursuit burns for agents with a designated target
        for ai, tj in enumerate(current_target):
            if 0 <= tj < len(scenario.targets) and scenario.targets[tj].active:
                world.apply_burn_toward(scenario.agents[ai], scenario.targets[tj])

        # Log record
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
            "killed_targets": killed,
            "n_targets_active": sum(1 for g in scenario.targets if g.active),
        }

        if track_manager is not None:
            track_states, track_active, track_ids = track_manager.get_track_states()
            rec["tracks"] = [
                {"id": tid, "x": xs[0], "y": xs[1], "vx": xs[2], "vy": xs[3], "active": act}
                for xs, act, tid in zip(track_states, track_active, track_ids)
            ]

        records.append(rec)

        # Termination
        if rec["n_targets_active"] == 0:
            break

    write_jsonl(out_dir / "episode.jsonl", records)
    print(f"Wrote: {out_dir / 'episode.jsonl'}")


if __name__ == "__main__":
    main()
