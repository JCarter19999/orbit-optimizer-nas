from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


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
    keys = set(rows[0].keys())
    for k in candidates:
        if k in keys:
            return k
    return None


def compute_origin(rows: List[Dict[str, Any]], agent_key: str, target_key: str) -> tuple[float, float]:
    """
    Use initial mean position of all bodies as a stable local-frame origin.
    """
    xs = [a["x"] for a in rows[0][agent_key]] + [g["x"] for g in rows[0][target_key]]
    ys = [a["y"] for a in rows[0][agent_key]] + [g["y"] for g in rows[0][target_key]]
    # use median to be robust to outliers and give a tighter local origin
    return float(np.median(xs)), float(np.median(ys))


def auto_limits(
    all_x: List[float],
    all_y: List[float],
    pad_frac: float = 0.06,
    trim_percentile: float = 2.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Compute square axis limits (equal aspect) with padding, focusing on the central
    bulk of points by trimming extreme percentiles. This yields a tighter zoom
    when a few outliers would otherwise expand the view.
    """
    if not all_x or not all_y:
        return (0.0, 1.0), (0.0, 1.0)

    # use percentile-based trimming to ignore very small number of outliers
    p = float(max(0.0, min(49.0, trim_percentile)))
    xmin = float(np.percentile(all_x, p))
    xmax = float(np.percentile(all_x, 100.0 - p))
    ymin = float(np.percentile(all_y, p))
    ymax = float(np.percentile(all_y, 100.0 - p))

    # fallback if data are degenerate
    if xmin == xmax:
        xmin, xmax = float(min(all_x)), float(max(all_x))
    if ymin == ymax:
        ymin, ymax = float(min(all_y)), float(max(all_y))

    dx = xmax - xmin
    dy = ymax - ymin
    span = max(dx, dy, 1.0)  # avoid degeneracy
    pad = pad_frac * span
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = 0.5 * span + pad
    return (cx - half, cx + half), (cy - half, cy + half)


def plot_battlespace(
    rows: List[Dict[str, Any]],
    out_png: Path,
    every: int = 10,
    frame: str = "local",  # "local" or "eci"
) -> None:
    """
    Plots agent and target trajectories in XY.
    - frame="eci": Earth-centered inertial km coordinates (may look "zoomed out")
    - frame="local": subtracts an origin so you see relative motion clearly
    Also auto-zooms to fill the canvas and forces equal aspect ratio.
    """
    agent_key = find_first_key(rows, ["agents", "interceptors", "vehicles"])
    target_key = find_first_key(rows, ["targets_truth", "targets", "truth_targets", "gt_targets", "objects"])
    matches_key = find_first_key(rows, ["matches", "assignments", "engagement_matches"])

    if agent_key is None:
        raise RuntimeError(f"No agent key found. Keys: {sorted(list(rows[0].keys()))}")
    if target_key is None:
        raise RuntimeError(f"No target key found. Keys: {sorted(list(rows[0].keys()))}")

    n_agents = len(rows[0][agent_key])
    n_targets = len(rows[0][target_key])

    # local frame origin
    ox, oy = 0.0, 0.0
    if frame.lower() == "local":
        ox, oy = compute_origin(rows, agent_key, target_key)

    def X(v: float) -> float:
        return float(v - ox) if frame.lower() == "local" else float(v)

    def Y(v: float) -> float:
        return float(v - oy) if frame.lower() == "local" else float(v)

    fig = plt.figure()
    ax = plt.gca()
    ax.set_title("Battlespace trajectories (XY)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)

    # Collect all points for auto-limits
    all_x: List[float] = []
    all_y: List[float] = []

    # prepare distinct colors for agents and targets
    agent_cmap = plt.get_cmap("tab10")
    target_cmap = plt.get_cmap("tab20")
    agent_colors = [agent_cmap(i % agent_cmap.N) for i in range(n_agents)]
    target_colors = [target_cmap(i % target_cmap.N) for i in range(n_targets)]

    # agent trajs: thicker, distinct colors, start/end markers
    for i in range(n_agents):
        xs = [X(rows[k][agent_key][i]["x"]) for k in range(0, len(rows), every)]
        ys = [Y(rows[k][agent_key][i]["y"]) for k in range(0, len(rows), every)]
        all_x.extend(xs)
        all_y.extend(ys)
        c = agent_colors[i % len(agent_colors)]
        ax.plot(xs, ys, label=f"agent {i}", color=c, linewidth=1.6, alpha=0.9, zorder=3)
        # start marker (filled triangle) and end marker (open circle)
        if xs:
            ax.scatter(xs[0], ys[0], marker="^", s=48, color=c, edgecolors="k", zorder=5)
            ax.scatter(xs[-1], ys[-1], marker="o", s=40, facecolors="none", edgecolors=c, linewidths=1.2, zorder=6)

    # target trajs: dashed, distinct colors, end markers indicate final state
    for j in range(n_targets):
        xs = [X(rows[k][target_key][j]["x"]) for k in range(0, len(rows), every)]
        ys = [Y(rows[k][target_key][j]["y"]) for k in range(0, len(rows), every)]
        all_x.extend(xs)
        all_y.extend(ys)
        c = target_colors[j % len(target_colors)]
        ax.plot(xs, ys, linestyle="--", label=f"target {j}", color=c, linewidth=1.4, alpha=0.85, zorder=2)

        final_active = bool(rows[-1][target_key][j].get("active", True))
        # active targets: bold X, inactive: filled square
        if xs:
            if final_active:
                ax.scatter(xs[-1], ys[-1], marker="X", s=70, color=c, edgecolors="k", zorder=7)
            else:
                ax.scatter(xs[-1], ys[-1], marker="s", s=60, color=c, edgecolors="k", zorder=7)

    # assignment lines if present
    if matches_key and matches_key in rows[0]:
        for k in range(0, len(rows), every):
            ms = rows[k].get(matches_key, [])
            if not ms:
                continue
            for ai, tj in ms:
                a = rows[k][agent_key][ai]
                g = rows[k][target_key][tj]
                ax.plot([X(a["x"]), X(g["x"])], [Y(a["y"]), Y(g["y"])], linewidth=0.8, alpha=0.35, color="#555555", zorder=1)

    # Auto-zoom to fill canvas
    if all_x and all_y:
        (xlim, ylim) = auto_limits(all_x, all_y, pad_frac=0.15)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    if frame.lower() == "local":
        ax.set_xlabel("x (km) relative")
        ax.set_ylabel("y (km) relative")
    else:
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
    # Make legend more compact and readable
    ax.legend(loc="best", fontsize=8, framealpha=0.9, ncol=1)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_targets_active(rows: List[Dict[str, Any]], out_png: Path) -> None:
    t = [float(r.get("t", r.get("time", i))) for i, r in enumerate(rows)]
    if "n_targets_active" in rows[0]:
        n_active = [int(r.get("n_targets_active", -1)) for r in rows]
    else:
        target_key = find_first_key(rows, ["targets_truth", "targets", "truth_targets", "gt_targets", "objects"])
        if target_key is None:
            raise RuntimeError(f"Cannot infer targets. Keys: {sorted(list(rows[0].keys()))}")
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


def plot_agent_dv(rows: List[Dict[str, Any]], out_png: Path) -> None:
    agent_key = find_first_key(rows, ["agents", "interceptors", "vehicles"])
    if agent_key is None:
        raise RuntimeError(f"Cannot infer agents. Keys: {sorted(list(rows[0].keys()))}")

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
    ap.add_argument("--episode", required=True, type=str)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument(
        "--frame",
        type=str,
        default="local",
        choices=["local", "eci"],
        help="local=subtract origin for readability; eci=raw inertial coords",
    )
    args = ap.parse_args()

    ep_path = Path(args.episode).resolve()
    rows = read_jsonl(ep_path)

    out_dir = ep_path.parent
    plot_battlespace(
        rows,
        out_dir / "plot_battlespace.png",
        every=max(1, args.every),
        frame=args.frame,
    )
    plot_targets_active(rows, out_dir / "plot_targets_active.png")
    plot_agent_dv(rows, out_dir / "plot_agent_dv.png")

    print(f"Wrote plots to: {out_dir}")
    print(f"Battlespace frame: {args.frame}")


if __name__ == "__main__":
    main()
