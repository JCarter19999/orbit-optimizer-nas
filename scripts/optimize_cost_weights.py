from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
import pandas as pd


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def read_first_last_jsonl(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    first = None
    last = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if first is None:
                first = obj
            last = obj
    if first is None or last is None:
        raise RuntimeError(f"Could not read first/last from {path}")
    return first, last


def total_dv_used(first: Dict[str, Any], last: Dict[str, Any]) -> float:
    dv0 = sum(float(a.get("dv_remaining", 0.0)) for a in first["agents"])
    dv1 = sum(float(a.get("dv_remaining", 0.0)) for a in last["agents"])
    return float(dv0 - dv1)


def leakers_from_last(last: Dict[str, Any]) -> int:
    if "n_targets_active" in last:
        return int(last["n_targets_active"])
    tgts = last.get("targets_truth", [])
    return int(sum(1 for g in tgts if bool(g.get("active", True))))


def fitness_from_episode(ep_jsonl: Path) -> float:
    first, last = read_first_last_jsonl(ep_jsonl)
    leakers = leakers_from_last(last)
    dv_used = total_dv_used(first, last)
    steps = int(last.get("step", -1))
    # Fitness: higher is better (negative cost)
    # You can tune these weights later, but keep fixed during optimization comparisons.
    return - (200.0 * leakers + 1.0 * max(0, steps) + 10.0 * dv_used)


@dataclass
class Weights:
    w_range: float
    w_rel_vel: float
    w_tgo: float
    w_budget: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "w_range": float(self.w_range),
            "w_rel_vel": float(self.w_rel_vel),
            "w_tgo": float(self.w_tgo),
            "w_budget": float(self.w_budget),
        }


def sample_weights(rng: random.Random) -> Weights:
    # bounded search space
    return Weights(
        w_range=rng.uniform(0.0, 5.0),
        w_rel_vel=rng.uniform(0.0, 5.0),
        w_tgo=rng.uniform(0.0, 5.0),
        w_budget=rng.uniform(0.0, 300.0),
    )


def mutate(w: Weights, rng: random.Random, sigma: float = 0.2) -> Weights:
    def clamp(x, lo, hi): return max(lo, min(hi, x))
    return Weights(
        w_range=clamp(w.w_range + rng.normalvariate(0, sigma), 0.0, 5.0),
        w_rel_vel=clamp(w.w_rel_vel + rng.normalvariate(0, sigma), 0.0, 5.0),
        w_tgo=clamp(w.w_tgo + rng.normalvariate(0, sigma), 0.0, 5.0),
        w_budget=clamp(w.w_budget + rng.normalvariate(0, 30.0), 0.0, 300.0),
    )


def crossover(a: Weights, b: Weights, rng: random.Random) -> Weights:
    # uniform crossover
    return Weights(
        w_range=a.w_range if rng.random() < 0.5 else b.w_range,
        w_rel_vel=a.w_rel_vel if rng.random() < 0.5 else b.w_rel_vel,
        w_tgo=a.w_tgo if rng.random() < 0.5 else b.w_tgo,
        w_budget=a.w_budget if rng.random() < 0.5 else b.w_budget,
    )


def evaluate_candidate(base_cfg: Dict[str, Any], w: Weights, trials: int, seed_offset: int) -> float:
    """
    Runs MC batch and returns average fitness over ok trials.
    Uses run_monte_carlo.py and then reads each episode_jsonl to compute fitness.
    """
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("planner", {}).setdefault("cost", {})
    cfg["planner"]["cost"].update(w.as_dict())

    tmp_cfg = Path("runs") / "tmp_opt_candidate.yaml"
    tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, tmp_cfg)

    cmd = [
        sys.executable, "scripts/run_monte_carlo.py",
        "--config", str(tmp_cfg),
        "--trials", str(trials),
        "--seed-offset", str(seed_offset),
        "--out-dir", "runs",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return -1e18

    run_root = Path("runs").resolve()
    mc_dirs = [p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("mc_")]
    newest_mc = max(mc_dirs, key=lambda p: p.stat().st_mtime)
    summary_path = newest_mc / "summary.csv"
    df = pd.read_csv(summary_path)
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return -1e18

    fits: List[float] = []
    for _, row in ok.iterrows():
        ep = Path(str(row["episode_jsonl"]))
        if ep.exists():
            fits.append(fitness_from_episode(ep))
    return float(sum(fits) / len(fits)) if fits else -1e18


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--eval-trials", type=int, default=10, help="MC trials per candidate eval")
    ap.add_argument("--budget", type=int, default=40, help="Total candidate evaluations per method")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out-dir", type=str, default="runs/opt")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    base_cfg = load_yaml(Path(args.config).resolve())

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Random search ----------
    random_progress = out_dir / "random_progress.csv"
    with open(random_progress, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["eval", "best_fitness"])
        w.writeheader()

    best_fit = -1e18
    best_w = None

    for e in range(args.budget):
        cand = sample_weights(rng)
        fit = evaluate_candidate(base_cfg, cand, trials=args.eval_trials, seed_offset=10000 + e * 1000)
        if fit > best_fit:
            best_fit = fit
            best_w = cand
        with open(random_progress, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["eval", "best_fitness"])
            w.writerow({"eval": e + 1, "best_fitness": best_fit})
        print(f"[RANDOM] eval={e+1}/{args.budget} fit={fit:.3f} best={best_fit:.3f}")

    # ---------- GA ----------
    pop_size = 10
    gens = max(1, args.budget // pop_size)

    ga_progress = out_dir / "ga_progress.csv"
    with open(ga_progress, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["eval", "best_fitness"])
        w.writeheader()

    # init population
    pop = [sample_weights(rng) for _ in range(pop_size)]
    ga_best = -1e18
    eval_count = 0

    for g in range(gens):
        scored = []
        for i, cand in enumerate(pop):
            eval_count += 1
            fit = evaluate_candidate(base_cfg, cand, trials=args.eval_trials, seed_offset=20000 + eval_count * 1000)
            scored.append((fit, cand))
            if fit > ga_best:
                ga_best = fit
            with open(ga_progress, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["eval", "best_fitness"])
                w.writerow({"eval": eval_count, "best_fitness": ga_best})
            print(f"[GA] gen={g+1}/{gens} cand={i+1}/{pop_size} fit={fit:.3f} best={ga_best:.3f}")

            if eval_count >= args.budget:
                break

        if eval_count >= args.budget:
            break

        # select top 4
        scored.sort(key=lambda x: x[0], reverse=True)
        elites = [c for _, c in scored[:4]]

        # breed next generation
        new_pop = elites[:]
        while len(new_pop) < pop_size:
            a = rng.choice(elites)
            b = rng.choice(elites)
            child = crossover(a, b, rng)
            if rng.random() < 0.7:
                child = mutate(child, rng)
            new_pop.append(child)
        pop = new_pop

    # ---------- Plot ----------
    print(f"Wrote: {random_progress}")
    print(f"Wrote: {ga_progress}")
    print("Next: python scripts/plot_optimization_progress.py --dir runs/opt")


if __name__ == "__main__":
    main()
