from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


# --- IO helpers ---


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


def find_first_key(rows: List[Dict[str, Any]], candidates: List[str]) -> Optional[str]:
    """Return the first candidate key that appears in the first record."""
    if not rows:
        return None
    keys = set(rows[0].keys())
    for k in candidates:
        if k in keys:
            return k
    return None


def require_key(rows: List[Dict[str, Any]], key: str) -> None:
    if key not in rows[0]:
        raise RuntimeError(
            f"Expected key '{key}' not found in episode records. "
            f"Available keys in first record: {sorted(list(rows[0].keys()))}"
        )


# --- plotting helpers ---


def plot_battlespace(
    rows: List[Dict[str, Any]],
    out_png: Path,
    agent_key: str,
    target_key: str,
    matches_key: Optional[str],
    every: int = 10,
) -> None:
    """
    Plots agent and target trajectories in XY.
    Draws assignment lines at steps where matches exist (if matches_key provided and present).
    """

    n_agents = len(rows[0][agent_key])
    n_targets = len(rows[0][target_key])

    fig = plt.figure()
    ax = plt.gca()
    ax.set_title("Battlespace trajectories (XY)")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_aspect("equal", adjustable="datalim")

    # Agents
    for i in range(n_agents):
        xs = [rows[k][agent_key][i]["x"] for k in range(0, len(rows), every)]
        ys = [rows[k][agent_key][i]["y"] for k in range(0, len(rows), every)]
        ax.plot(xs, ys, label=f"agent {i}")

    # Targets
    for j in range(n_targets):
        xs = [rows[k][target_key][j]["x"] for k in range(0, len(rows), every)]
        ys = [rows[k][target_key][j]["y"] for k in range(0, len(rows), every)]
        ax.plot(xs, ys, linestyle="--", label=f"target {j}")

        final_active = bool(rows[-1][target_key][j].get("active", True))
        ax.scatter(xs[-1], ys[-1], marker="x" if final_active else "o")

    # Matches (optional)
    if matches_key and matches_key in rows[0]:
        for k in range(0, len(rows), every):
            ms = rows[k].get(matches_key, [])
            if not ms:
                continue
            for ai, tj in ms:
                a = rows[k][agent_key][ai]
                g = rows[k][target_key][tj]
                ax.plot([a["x"], g["x"]], [a["y"], g["y"]], linewidth=0.8, alpha=0.5)

    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_targets_active(rows: List[Dict[str, Any]], out_png: Path) -> None:
    # If your sim already logs n_targets_active, use it. Otherwise compute from targets list.
    if "n_targets_active" in rows[0]:
        t = [float(r.get("t", r.get("time", i))) for i, r in enumerate(rows)]
        n_active = [int(r.get("n_targets_active", -1)) for r in rows]
    else:
        # try to infer target key
        target_key = find_first_key(rows, ["targets", "truth_targets", "gt_targets", "objects"])
        if target_key is None:
            raise RuntimeError(
                "Cannot plot active targets: no 'n_targets_active' field and no target list key found. "
                f"Available keys: {sorted(list(rows[0].keys()))}"
            )
        t = [float(r.get("t", r.get("time", i))) for i, r in enumerate(rows)]
        n_active = [sum(1 for g in r[target_key] if bool(g.get("active", True))) for r in rows]

    fig = plt.figure()
    ax = plt.gca()
    ax.set_title("Active targets over time")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("active targets")
    ax.plot(t, n_active)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_agent_dv(rows: List[Dict[str, Any]], out_png: Path, agent_key: str) -> None:
    t = [float(r.get("t", r.get("time", i))) for i, r in enumerate(rows)]
    n_agents = len(rows[0][agent_key])

    fig = plt.figure()
    ax = plt.gca()
    ax.set_title("Interceptor Δv remaining over time")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Δv remaining (km/s)")

    for i in range(n_agents):
        dv = [float(r[agent_key][i].get("dv_remaining", 0.0)) for r in rows]
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

    # Auto-detect keys
    agent_key = find_first_key(rows, ["agents", "interceptors", "vehicles"])
    target_key = find_first_key(rows, ["targets", "truth_targets", "gt_targets", "objects"])
    matches_key = find_first_key(rows, ["matches", "assignments", "engagement_matches"])

    if agent_key is None:
        raise RuntimeError(
            "Could not find agent list key. "
            f"Available keys in first record: {sorted(list(rows[0].keys()))}"
        )
    if target_key is None:
        raise RuntimeError(
            "Could not find target list key. "
            f"Available keys in first record: {sorted(list(rows[0].keys()))}\n"
            "Fix: either log 'targets' in episode.jsonl or add your key name to candidates in plot_episode.py."
        )

    out_dir = ep_path.parent
    plot_battlespace(
        rows,
        out_dir / "plot_battlespace.png",
        agent_key=agent_key,
        target_key=target_key,
        matches_key=matches_key,
        every=max(1, args.every),
    )
    plot_targets_active(rows, out_dir / "plot_targets_active.png")
    plot_agent_dv(rows, out_dir / "plot_agent_dv.png", agent_key=agent_key)

    print(f"Wrote plots to: {out_dir}")
    print(f"Using keys: agents='{agent_key}', targets='{target_key}', matches='{matches_key}'")


if __name__ == "__main__":
    main()
