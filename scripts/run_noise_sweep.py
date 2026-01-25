from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
import subprocess
import sys


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
    # assumes agents list with dv_remaining
    a0 = first["agents"]
    a1 = last["agents"]
    dv0 = sum(float(a.get("dv_remaining", 0.0)) for a in a0)
    dv1 = sum(float(a.get("dv_remaining", 0.0)) for a in a1)
    return float(dv0 - dv1)


def success_from_last(last: Dict[str, Any]) -> int:
    # success if no active targets remain
    if "n_targets_active" in last:
        return 1 if int(last["n_targets_active"]) == 0 else 0
    # else infer from targets_truth
    tgts = last.get("targets_truth", [])
    active = sum(1 for g in tgts if bool(g.get("active", True)))
    return 1 if active == 0 else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--sigmas", type=str, default="0,5,10,20,40")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="runs/sweeps")
    args = ap.parse_args()

    base_cfg_path = Path(args.config).resolve()
    base_cfg = load_yaml(base_cfg_path)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sigmas = [float(x.strip()) for x in args.sigmas.split(",") if x.strip()]

    trials_csv = out_dir / "sweep_trials.csv"
    summary_csv = out_dir / "sweep_summary.csv"

    # write headers
    with open(trials_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sigma_xy_km", "trial", "seed", "success", "dv_used", "episode_jsonl"])
        w.writeheader()

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sigma_xy_km", "trials_ok", "success_rate", "dv_used_mean"])
        w.writeheader()

    # sweep loop
    for sidx, sigma in enumerate(sigmas):
        # create a temp config for this sigma
        cfg = dict(base_cfg)
        cfg.setdefault("sensor", {})
        cfg["sensor"]["sigma_xy_km"] = float(sigma)

        tmp_cfg = out_dir / f"tmp_sigma_{sigma:g}.yaml"
        save_yaml(cfg, tmp_cfg)

        # run monte carlo batch using your existing script
        cmd = [
            sys.executable,
            "scripts/run_monte_carlo.py",
            "--config",
            str(tmp_cfg),
            "--trials",
            str(args.trials),
            "--seed-offset",
            str(args.seed_offset + 100000 * sidx),
            "--out-dir",
            "runs",
        ]
        print(f"[SWEEP] sigma_xy_km={sigma:g} -> {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stderr)
            raise SystemExit(proc.returncode)

        # find newest mc folder under runs/
        run_root = Path("runs").resolve()
        mc_dirs = [p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("mc_")]
        if not mc_dirs:
            raise RuntimeError("No mc_* folder found under runs/")
        newest_mc = max(mc_dirs, key=lambda p: p.stat().st_mtime)
        summary_path = newest_mc / "summary.csv"
        if not summary_path.exists():
            raise RuntimeError(f"Missing {summary_path}")

        # read trial results from summary.csv, and compute dv_used by opening episode_jsonl
        import pandas as pd  # lazy import
        df = pd.read_csv(summary_path)
        df_ok = df[df["status"] == "ok"].copy()

        successes = []
        dv_useds = []

        for _, row in df_ok.iterrows():
            ep_path = str(row["episode_jsonl"])
            if not ep_path:
                continue
            ep = Path(ep_path)
            if not ep.exists():
                continue
            first, last = read_first_last_jsonl(ep)
            succ = success_from_last(last)
            dv = total_dv_used(first, last)

            successes.append(succ)
            dv_useds.append(dv)

            with open(trials_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["sigma_xy_km", "trial", "seed", "success", "dv_used", "episode_jsonl"])
                w.writerow(
                    {
                        "sigma_xy_km": float(sigma),
                        "trial": int(row["trial"]),
                        "seed": int(row["seed"]),
                        "success": int(succ),
                        "dv_used": float(dv),
                        "episode_jsonl": str(ep),
                    }
                )

        trials_ok = len(successes)
        success_rate = float(sum(successes) / trials_ok) if trials_ok > 0 else 0.0
        dv_mean = float(sum(dv_useds) / trials_ok) if trials_ok > 0 else 0.0

        with open(summary_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["sigma_xy_km", "trials_ok", "success_rate", "dv_used_mean"])
            w.writerow(
                {
                    "sigma_xy_km": float(sigma),
                    "trials_ok": int(trials_ok),
                    "success_rate": float(success_rate),
                    "dv_used_mean": float(dv_mean),
                }
            )

        print(f"[SWEEP] sigma={sigma:g} ok={trials_ok} success_rate={success_rate:.3f} dv_mean={dv_mean:.3f}")

    print(f"[SWEEP] Wrote: {trials_csv}")
    print(f"[SWEEP] Wrote: {summary_csv}")
    print("[SWEEP] Next: python scripts/plot_noise_sweep.py --dir runs/sweeps")


if __name__ == "__main__":
    main()
