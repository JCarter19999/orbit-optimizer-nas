from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, type=str, help="Path to mc_*/summary.csv")
    args = ap.parse_args()

    p = Path(args.summary).resolve()
    df = pd.read_csv(p)

    out_dir = p.parent

    # Filter good trials
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        raise RuntimeError("No ok trials found in summary.csv")

    ok["success"] = ok["success_all_targets_neutralized"].astype(int)
    ok["steps"] = ok["steps"].astype(float)

    # Success rate
    success_rate = ok["success"].mean()

    fig = plt.figure()
    ax = plt.gca()
    ax.set_title("Monte Carlo success rate")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1)
    ax.bar(["success"], [success_rate])
    fig.tight_layout()
    fig.savefig(out_dir / "plot_mc_success_rate.png", dpi=200)
    plt.close(fig)

    # Steps histogram
    fig = plt.figure()
    ax = plt.gca()
    ax.set_title("Monte Carlo: steps to termination (ok trials)")
    ax.set_xlabel("steps")
    ax.set_ylabel("count")
    ax.hist(ok["steps"].values, bins=20)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_mc_steps_hist.png", dpi=200)
    plt.close(fig)

    # Final active targets histogram
    fig = plt.figure()
    ax = plt.gca()
    ax.set_title("Monte Carlo: final active targets (ok trials)")
    ax.set_xlabel("n_targets_active_final")
    ax.set_ylabel("count")
    ax.hist(ok["n_targets_active_final"].astype(float).values, bins=10)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_mc_final_active_hist.png", dpi=200)
    plt.close(fig)

    print(f"Wrote plots to: {out_dir}")
    print(f"Success rate: {success_rate:.3f} (ok={len(ok)}/{len(df)})")


if __name__ == "__main__":
    main()
