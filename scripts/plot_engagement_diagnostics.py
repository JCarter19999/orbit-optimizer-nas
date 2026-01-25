from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No records found in {path}")
    return rows


def find_first_key(row: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    keys = set(row.keys())
    for k in candidates:
        if k in keys:
            return k
    return None


def get_agent_target_keys(first_row: Dict[str, Any]) -> Tuple[str, str, str]:
    agent_key = find_first_key(first_row, ["agents", "interceptors", "vehicles"])
    target_key = find_first_key(first_row, ["targets_truth", "targets", "truth_targets", "gt_targets", "objects"])
    match_key = find_first_key(first_row, ["matches", "assignments", "engagement_matches"])

    if agent_key is None:
        raise RuntimeError(f"Could not find agents key. Available keys: {sorted(list(first_row.keys()))}")
    if target_key is None:
        raise RuntimeError(f"Could not find targets key. Available keys: {sorted(list(first_row.keys()))}")
    if match_key is None:
        # allow missing matches; will just plot NaNs
        match_key = "matches"

    return agent_key, target_key, match_key


def dist(ax: float, ay: float, tx: float, ty: float) -> float:
    dx = tx - ax
    dy = ty - ay
    return (dx * dx + dy * dy) ** 0.5


def assignment_map(matches: Any, n_agents: int) -> List[Optional[int]]:
    """
    Convert matches list [[agent_id, target_id], ...] to a length-n_agents list.
    """
    out: List[Optional[int]] = [None] * n_agents
    if not matches:
        return out
    for pair in matches:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        ai, tj = int(pair[0]), int(pair[1])
        if 0 <= ai < n_agents:
            out[ai] = tj
    return out


def plot_distance_to_assigned(rows: List[Dict[str, Any]], out_png: Path) -> None:
    first = rows[0]
    agent_key, target_key, match_key = get_agent_target_keys(first)

    n_agents = len(first[agent_key])
    t = [float(r.get("t", r.get("time", i))) for i, r in enumerate(rows)]

    # distance series: per agent
    dists: List[List[float]] = [[] for _ in range(n_agents)]

    for r in rows:
        agents = r[agent_key]
        targets = r[target_key]
        amap = assignment_map(r.get(match_key, []), n_agents=n_agents)

        for ai in range(n_agents):
            tj = amap[ai]
            if tj is None or tj < 0 or tj >= len(targets):
                dists[ai].append(float("nan"))
                continue
            a = agents[ai]
            g = targets[tj]
            dists[ai].append(dist(a["x"], a["y"], g["x"], g["y"]))

    fig = plt.figure()
    ax = plt.gca()
    ax.set_title("Distance to assigned target over time")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("range (km)")

    for ai in range(n_agents):
        ax.plot(t, dists[ai], label=f"agent {ai}")

    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_assignment_churn(rows: List[Dict[str, Any]], out_png: Path) -> None:
    first = rows[0]
    agent_key, _, match_key = get_agent_target_keys(first)

    n_agents = len(first[agent_key])
    t = [float(r.get("t", r.get("time", i))) for i, r in enumerate(rows)]

    churn_per_step: List[int] = []
    total_changed: int = 0

    prev = assignment_map(first.get(match_key, []), n_agents=n_agents)
    churn_per_step.append(0)

    for r in rows[1:]:
        cur = assignment_map(r.get(match_key, []), n_agents=n_agents)
        changed = 0
        for ai in range(n_agents):
            if cur[ai] != prev[ai]:
                # Count changes including None <-> target_id changes
                changed += 1
        total_changed += changed
        churn_per_step.append(changed)
        prev = cur

    fig = plt.figure()
    ax = plt.gca()
    ax.set_title("Assignment churn over time")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("# agent assignment changes (this step)")
    ax.plot(t, churn_per_step)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    # Also save a simple cumulative churn plot (often more readable)
    cum = []
    running = 0
    for c in churn_per_step:
        running += c
        cum.append(running)

    fig = plt.figure()
    ax = plt.gca()
    ax.set_title("Cumulative assignment churn")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("cumulative changes")
    ax.plot(t, cum)
    fig.tight_layout()
    fig.savefig(out_png.with_name("plot_assignment_churn_cumulative.png"), dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True, type=str, help="Path to episode.jsonl")
    args = ap.parse_args()

    ep = Path(args.episode).resolve()
    rows = read_jsonl(ep)
    out_dir = ep.parent

    plot_distance_to_assigned(rows, out_dir / "plot_distance_to_assigned.png")
    plot_assignment_churn(rows, out_dir / "plot_assignment_churn.png")

    print(f"Wrote: {out_dir / 'plot_distance_to_assigned.png'}")
    print(f"Wrote: {out_dir / 'plot_assignment_churn.png'}")
    print(f"Wrote: {out_dir / 'plot_assignment_churn_cumulative.png'}")


if __name__ == "__main__":
    main()
