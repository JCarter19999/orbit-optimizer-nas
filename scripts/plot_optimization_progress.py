from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, type=str, help="Directory containing random_progress.csv and ga_progress.csv")
    args = ap.parse_args()

    d = Path(args.dir).resolve()
    rnd = pd.read_csv(d / "random_progress.csv")
    ga = pd.read_csv(d / "ga_progress.csv")

    fig = plt.figure()
    ax = plt.gca()
    ax.set_title("Best fitness vs evaluation budget (GA vs Random)")
    ax.set_xlabel("candidate evaluations")
    ax.set_ylabel("best fitness (higher is better)")
    ax.plot(rnd["eval"].values, rnd["best_fitness"].values, label="random best-so-far")
    ax.plot(ga["eval"].values, ga["best_fitness"].values, label="ga best-so-far")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(d / "plot_ga_vs_random.png", dpi=200)
    plt.close(fig)

    print(f"Wrote: {d / 'plot_ga_vs_random.png'}")


if __name__ == "__main__":
    main()
