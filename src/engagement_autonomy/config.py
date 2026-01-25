from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml

def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@dataclass(frozen=True)
class RunConfig:
    seed: int
    out_dir: Path

@dataclass(frozen=True)
class SimConfig:
    mu: float
    dt: float
    steps: int
    decision_period: int
    intercept_radius_km: float
    max_burns_per_agent: int
    dv_budget_kms: float
    burn_dv_kms: float
    burn_cooldown_s: float

@dataclass(frozen=True)
class ScenarioConfig:
    n_agents: int
    n_targets: int
    ref_orbit_radius_km: float
    agent_pos_sigma_km: float
    agent_vel_sigma_kms: float
    target_pos_sigma_km: float
    target_vel_sigma_kms: float

@dataclass(frozen=True)
class SensorConfig:
    sigma_range_km: float
    sigma_bearing_rad: float

@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool
    q_pos: float
    q_vel: float
    r_range: float
    r_bearing: float

@dataclass(frozen=True)
class AssociationConfig:
    enabled: bool
    gate_mahalanobis2: float

@dataclass(frozen=True)
class PlannerConfig:
    type: str
    weights: Dict[str, float]
    max_pairs_per_agent: int

@dataclass(frozen=True)
class Config:
    run: RunConfig
    sim: SimConfig
    scenario: ScenarioConfig
    sensor: SensorConfig
    tracking: TrackingConfig
    association: AssociationConfig
    planner: PlannerConfig

def parse_config(d: Dict[str, Any]) -> Config:
    run = RunConfig(seed=int(d["run"]["seed"]), out_dir=Path(d["run"]["out_dir"]))

    simd = d["sim"]
    sim = SimConfig(
        mu=float(simd["mu"]),
        dt=float(simd["dt"]),
        steps=int(simd["steps"]),
        decision_period=int(simd["decision_period"]),
        intercept_radius_km=float(simd["intercept_radius_km"]),
        max_burns_per_agent=int(simd["max_burns_per_agent"]),
        dv_budget_kms=float(simd["dv_budget_kms"]),
        burn_dv_kms=float(simd["burn_dv_kms"]),
        burn_cooldown_s=float(simd["burn_cooldown_s"]),
    )

    scd = d["scenario"]
    scenario = ScenarioConfig(
        n_agents=int(scd["n_agents"]),
        n_targets=int(scd["n_targets"]),
        ref_orbit_radius_km=float(scd["ref_orbit_radius_km"]),
        agent_pos_sigma_km=float(scd["agent_pos_sigma_km"]),
        agent_vel_sigma_kms=float(scd["agent_vel_sigma_kms"]),
        target_pos_sigma_km=float(scd["target_pos_sigma_km"]),
        target_vel_sigma_kms=float(scd["target_vel_sigma_kms"]),
    )

    snd = d["sensor"]
    sensor = SensorConfig(
        sigma_range_km=float(snd["sigma_range_km"]),
        sigma_bearing_rad=float(snd["sigma_bearing_rad"]),
    )

    trd = d["tracking"]
    tracking = TrackingConfig(
        enabled=bool(trd["enabled"]),
        q_pos=float(trd["q_pos"]),
        q_vel=float(trd["q_vel"]),
        r_range=float(trd["r_range"]),
        r_bearing=float(trd["r_bearing"]),
    )

    asd = d["association"]
    association = AssociationConfig(
        enabled=bool(asd["enabled"]),
        gate_mahalanobis2=float(asd["gate_mahalanobis2"]),
    )

    pld = d["planner"]
    planner = PlannerConfig(
        type=str(pld["type"]),
        weights=dict(pld["weights"]),
        max_pairs_per_agent=int(pld.get("max_pairs_per_agent", 6)),
    )

    return Config(
        run=run,
        sim=sim,
        scenario=scenario,
        sensor=sensor,
        tracking=tracking,
        association=association,
        planner=planner,
    )
