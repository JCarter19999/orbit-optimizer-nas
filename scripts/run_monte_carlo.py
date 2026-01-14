from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def read_last_jsonl_record(path: Path) -> Dict[str, Any]:
    """
    Reads the last JSON object from a JSONL file.
    Assumes each line is a JSON dict.
    """
    last = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            last = json.loads(line)
    if last is None:
        raise RuntimeError(f"No records found in {path}")
    return last


def find_latest_episode_jsonl(run_root: Path) -> Path:
    """
    Your run_episode.py writes something like:
      runs/episode_YYYYMMDD_HHMMSS/episode.jsonl
    We locate the newest episode folder and return its episode.jsonl.
    """
    if not run_root.exists():
        raise RuntimeError(f"Run root does not exist: {run_root}")

    candidates = [p for p in run_root.iterdir() if p.is_dir()]
    if not candidates:
        raise RuntimeError(f"No episode directories found in: {run_root}")

    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    ep = newest / "episode.jsonl"
    if not ep.exists():
        raise RuntimeError(f"Expected episode.jsonl not found at: {ep}")
    return ep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str, help="Base episode config YAML")
    ap.add_argument("--trials", type=int, default=30, help="Number of Monte Carlo trials")
    ap.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Additional offset added to base seed (useful when running multiple batches)",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="runs",
        help="Root output folder (should match your existing run config)",
    )
    ap.add_argument(
        "--keep-temp-configs",
        action="store_true",
        help="Keep the per-trial temp YAML files (debugging)",
    )
    args = ap.parse_args()

    base_cfg_path = Path(args.config).resolve()
    base_cfg = load_yaml(base_cfg_path)

    # Where your existing episode script writes outputs
    run_root = Path(args.out_dir).resolve()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mc_dir = run_root / f"mc_{ts}"
    mc_dir.mkdir(parents=True, exist_ok=True)

    temp_cfg_dir = mc_dir / "temp_configs"
    temp_cfg_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = mc_dir / "summary.csv"
    rows = []

    # Read base seed from config
    base_seed = int(base_cfg.get("run", {}).get("seed", 0))
    if "run" not in base_cfg:
        base_cfg["run"] = {}

    print(f"[MC] Base config: {base_cfg_path}")
    print(f"[MC] Trials: {args.trials}")
    print(f"[MC] Base seed: {base_seed} (plus seed_offset={args.seed_offset})")
    print(f"[MC] Output: {mc_dir}")

    for k in range(args.trials):
        trial_seed = base_seed + args.seed_offset + k

        trial_cfg = dict(base_cfg)  # shallow copy top-level
        trial_cfg["run"] = dict(base_cfg.get("run", {}))
        trial_cfg["run"]["seed"] = int(trial_seed)

        # IMPORTANT: keep using the standard output root so run_episode writes where it expects.
        # We do NOT redirect per-trial outputs because run_episode chooses its own timestamped folder.
        # If you want per-trial directories later, we can add that once we know run_episode behavior.
        if "out_dir" in trial_cfg["run"]:
            # normalize if user put something else
            trial_cfg["run"]["out_dir"] = str(args.out_dir)

        cfg_path = temp_cfg_dir / f"episode_trial_{k:04d}.yaml"
        save_yaml(trial_cfg, cfg_path)

        # Call your existing episode runner
        cmd = [sys.executable, "scripts/run_episode.py", "--config", str(cfg_path)]
        print(f"[MC] Trial {k+1}/{args.trials} seed={trial_seed} -> running run_episode.py")
        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            # Store failure info
            rows.append(
                {
                    "trial": k,
                    "seed": trial_seed,
                    "status": "error",
                    "episode_jsonl": "",
                    "steps": "",
                    "n_targets_active_final": "",
                    "success_all_targets_neutralized": "",
                    "stderr": proc.stderr.strip()[:4000],
                }
            )
            print(f"[MC] Trial {k} FAILED. See summary.csv for stderr excerpt.")
            continue

        # Find latest episode output and read final record
        try:
            ep_jsonl = find_latest_episode_jsonl(run_root)
            last = read_last_jsonl_record(ep_jsonl)

            n_active = int(last.get("n_targets_active", -1))
            step = int(last.get("step", -1))
            success = (n_active == 0)

            rows.append(
                {
                    "trial": k,
                    "seed": trial_seed,
                    "status": "ok",
                    "episode_jsonl": str(ep_jsonl),
                    "steps": step,
                    "n_targets_active_final": n_active,
                    "success_all_targets_neutralized": int(success),
                    "stderr": "",
                }
            )
        except Exception as e:
            rows.append(
                {
                    "trial": k,
                    "seed": trial_seed,
                    "status": "parse_error",
                    "episode_jsonl": "",
                    "steps": "",
                    "n_targets_active_final": "",
                    "success_all_targets_neutralized": "",
                    "stderr": f"{type(e).__name__}: {e}",
                }
            )
            print(f"[MC] Trial {k} ran, but parsing failed: {e}")

    # Write summary CSV
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "trial",
                "seed",
                "status",
                "episode_jsonl",
                "steps",
                "n_targets_active_final",
                "success_all_targets_neutralized",
                "stderr",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Compute quick aggregate stats
    ok = [r for r in rows if r["status"] == "ok"]
    if ok:
        success_rate = sum(int(r["success_all_targets_neutralized"]) for r in ok) / len(ok)
        mean_steps = sum(int(r["steps"]) for r in ok) / len(ok)
        print(f"[MC] Done. ok={len(ok)}/{len(rows)} success_rate={success_rate:.3f} mean_steps={mean_steps:.1f}")
    else:
        print(f"[MC] Done. No successful trials parsed. Check {summary_csv}")

    # Optionally clean temp configs
    if not args.keep_temp_configs:
        shutil.rmtree(temp_cfg_dir, ignore_errors=True)

    print(f"[MC] Summary written: {summary_csv}")
    print(f"[MC] Batch folder: {mc_dir}")


if __name__ == "__main__":
    main()
