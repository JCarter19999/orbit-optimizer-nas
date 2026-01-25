from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, type=str, help="Sweep output dir (contains sweep_trials.csv and sweep_summary.csv)")
    args = ap.parse_args()

    d = Path(args.dir).resolve()
    trials = pd.read_csv(d / "sweep_trials.csv")
    summary = pd.read_csv(d / "sweep_summary.csv").sort_values("sigma_xy_km")

    # 1) success rate vs sigma
    fig = plt.figure()
    ax = plt.gca()
    ax.set_title("Success rate vs measurement noise (sigma_xy_km)")
    ax.set_xlabel("sigma_xy_km")
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1)
    ax.plot(summary["sigma_xy_km"].values, summary["success_rate"].values, marker="o")
    fig.tight_layout()
    fig.savefig(d / "plot_success_vs_sigma.png", dpi=200)
    plt.close(fig)

    # 2) dv_used distribution vs sigma (boxplot)
    sigmas = sorted(trials["sigma_xy_km"].unique().tolist())
    grouped = [trials[trials["sigma_xy_km"] == s]["dv_used"].values for s in sigmas]

    fig = plt.figure()
    ax = plt.gca()
    ax.set_title("Δv used distribution vs sigma_xy_km")
    ax.set_xlabel("sigma_xy_km")
    ax.set_ylabel("total Δv used")
    ax.boxplot(grouped, labels=[str(s) for s in sigmas], showfliers=False)
    fig.tight_layout()
    fig.savefig(d / "plot_dv_boxplot_vs_sigma.png", dpi=200)
    plt.close(fig)

    print(f"Wrote: {d / 'plot_success_vs_sigma.png'}")
    print(f"Wrote: {d / 'plot_dv_boxplot_vs_sigma.png'}")


if __name__ == "__main__":
    main()
