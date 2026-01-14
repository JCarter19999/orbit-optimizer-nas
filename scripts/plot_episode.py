from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No records found in {path}")
    return rows


def extract_series(rows: List[Dict[str, Any]]) -> Tuple[List[float], List[Any], List[Any], List[Any]]:
    t = [float(r["t"]) for r in rows]
    agents = [r["agents"] for r in rows]
    targets = [r["targets"] for r in rows]
    matches = [r.get("matches", []) for r in rows]
    return t, agents, targets, matches


def plot_battlespace(rows: List[Dict[str, Any]], out_png: Path, every: int = 10) -> None:
    """
    Plots agent and target trajectories in XY. Marks final positions and intercepts.
    Draws assignment lines at decision steps when matches exist.
    """
    t, agents, targets, matches = extract_series(rows)

    n_agents = len(agents[0])
    n_targets = len(targets[0])

    # trajectories
    fig = plt.figure()
    ax = plt.gca()
    ax.set_title("Battlespace trajectories (XY)")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_aspect("equal", adjustable="datalim")

    # plot agent trajs
    for i in range(n_agents):
        xs = [agents[k][i]["x"] for k in range(0, len(rows), every)]
        ys = [agents[k][i]["y"] for k in range(0, len(rows), every)]
        ax.plot(xs, ys, label=f"agent {i}")

    # plot target trajs, with active/inactive coloring at final time
    for j in range(n_targets):
        xs = [targets[k][j]["x"] for k in range(0, len(rows), every)]
        ys = [targets[k][j]["y"] for k in range(0, len(rows), every)]
        ax.plot(xs, ys, linestyle="--", label=f"target {j}")

        final_active = bool(targets[-1][j].get("active", True))
        ax.scatter(xs[-1], ys[-1], marker="x" if final_active else "o")

    # draw assignment lines at steps where matches exist
    # (these are engagement assignments, not tracking association)
    for k in range(0, len(rows), every):
        ms = matches[k]
        if not ms:
            continue
        for ai, tj in ms:
            a = agents[k][ai]
            g = targets[k][tj]
            ax.plot([a["x"], g["x"]], [a["y"], g["y"]], linewidth=0.8, alpha=0.5)

    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_targets_active(rows: List[Dict[str, Any]], out_png: Path) -> None:
    t = [float(r["t"]) for r in rows]
    n_active = [int(r.get("n_targets_active", -1)) for r in rows]

    fig = plt.figure()
    ax = plt.gca()
    ax.set_title("Active targets over time")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("active targets")
    ax.plot(t, n_active)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_agent_dv(rows: List[Dict[str, Any]], out_png: Path) -> None:
    t = [float(r["t"]) for r in rows]
    n_agents = len(rows[0]["agents"])

    fig = plt.figure()
    ax = plt.gca()
    ax.set_title("Interceptor Δv remaining over time")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Δv remaining (km/s)")

    for i in range(n_agents):
        dv = [float(r["agents"][i].get("dv_remaining", 0.0)) for r in rows]
        ax.plot(t, dv, label=f"agent {i}")

    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True, type=str, help="Path to episode.jsonl")
    ap.add_argument("--every", type=int, default=10, help="Downsample factor for plotting")
    args = ap.parse_args()

    ep_path = Path(args.episode).resolve()
    rows = read_jsonl(ep_path)

    out_dir = ep_path.parent
    plot_battlespace(rows, out_dir / "plot_battlespace.png", every=max(1, args.every))
    plot_targets_active(rows, out_dir / "plot_targets_active.png")
    plot_agent_dv(rows, out_dir / "plot_agent_dv.png")

    print(f"Wrote plots to: {out_dir}")


if __name__ == "__main__":
    main()
