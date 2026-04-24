"""
plot_eval_comparison.py — Compare evaluated actor policies across algorithms.

Reads one or more per-episode CSVs produced by eval_actor.py and renders a
side-by-side bar chart (mean ± std) for Return, Cost, and Episode Length,
plus a box plot for Return and Cost distributions.

Usage
-----
# Compare two CSVs:
    python scripts/plot_eval_comparison.py \
        --csvs results/eval_ppolag.csv results/eval_ppolagada.csv \
        --out plots/eval_comparison.png

# Override display labels:
    python scripts/plot_eval_comparison.py \
        --csvs results/eval_ppolag.csv results/eval_ppolagada.csv \
        --labels "PPO-Lag" "PPO-Lag-Ada" \
        --out plots/eval_comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COST_LIMIT = 25.0
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def load_csvs(paths, labels) -> pd.DataFrame:
    frames = []
    for i, p in enumerate(paths):
        df = pd.read_csv(p)
        if labels:
            df["algo"] = labels[i]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def bar_panel(ax, algos, means, stds, ylabel, color_map, hline=None):
    x = np.arange(len(algos))
    bars = ax.bar(x, means, yerr=stds, capsize=5, width=0.5,
                  color=[color_map[a] for a in algos], alpha=0.85, ecolor="black")
    if hline is not None:
        ax.axhline(hline, linestyle="--", color="#444444", linewidth=1.5, label=f"limit={hline}")
        ax.legend(fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    return bars


def box_panel(ax, df, metric, algos, color_map, hline=None, ylabel=None):
    data = [df[df["algo"] == a][metric].values for a in algos]
    bp = ax.boxplot(data, labels=algos, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    for patch, algo in zip(bp["boxes"], algos):
        patch.set_facecolor(color_map[algo])
        patch.set_alpha(0.75)
    if hline is not None:
        ax.axhline(hline, linestyle="--", color="#444444", linewidth=1.5, label=f"limit={hline}")
        ax.legend(fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)


def safe_rate_panel(ax, df, algos, color_map):
    rates = [np.mean(df[df["algo"] == a]["cost"].values <= COST_LIMIT) * 100 for a in algos]
    x = np.arange(len(algos))
    ax.bar(x, rates, width=0.5, color=[color_map[a] for a in algos], alpha=0.85)
    ax.axhline(100, linestyle="--", color="#444444", linewidth=1.0, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=9)
    ax.set_ylabel("% episodes safe")
    ax.set_ylim(0, 110)
    ax.set_title(f"Safe Rate (cost ≤ {COST_LIMIT})")
    ax.grid(axis="y", alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvs", nargs="+", required=True,
                        help="Per-episode CSV files from eval_actor.py")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Display labels (one per CSV); overrides the 'algo' column")
    parser.add_argument("--out", type=str, default=None,
                        help="Output PNG path (default: plots/eval_comparison.png)")
    parser.add_argument("--smooth", action="store_true",
                        help="(reserved for future use)")
    args = parser.parse_args()

    df = load_csvs(args.csvs, args.labels)
    algos = list(df["algo"].unique())
    color_map = {a: COLORS[i % len(COLORS)] for i, a in enumerate(algos)}

    # Aggregate stats per algo
    stats = df.groupby("algo")[["ret", "cost", "length"]].agg(["mean", "std"]).reindex(algos)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), constrained_layout=True)
    fig.suptitle("Evaluation Comparison — Actor Rollouts", fontsize=13)

    # Row 0: bar charts (mean ± std)
    bar_panel(axes[0, 0], algos,
              stats["ret"]["mean"].values,  stats["ret"]["std"].values,
              "Mean Return",   color_map)
    axes[0, 0].set_title("Return (mean ± std)")

    bar_panel(axes[0, 1], algos,
              stats["cost"]["mean"].values, stats["cost"]["std"].values,
              "Mean Cost", color_map, hline=COST_LIMIT)
    axes[0, 1].set_title("Cost (mean ± std)")

    bar_panel(axes[0, 2], algos,
              stats["length"]["mean"].values, stats["length"]["std"].values,
              "Mean Ep Length", color_map)
    axes[0, 2].set_title("Episode Length (mean ± std)")

    # Row 1: box plots + safe-rate
    box_panel(axes[1, 0], df, "ret",    algos, color_map, ylabel="Return")
    axes[1, 0].set_title("Return Distribution")

    box_panel(axes[1, 1], df, "cost",   algos, color_map, hline=COST_LIMIT, ylabel="Cost")
    axes[1, 1].set_title("Cost Distribution")

    safe_rate_panel(axes[1, 2], df, algos, color_map)

    out_path = Path(args.out) if args.out else Path("plots/eval_comparison.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {out_path}")

    # Print summary table
    print(f"\n{'Algo':<20} {'Return':>10} {'Cost':>10} {'Safe%':>8}")
    print("─" * 52)
    for a in algos:
        sub = df[df["algo"] == a]
        safe_pct = np.mean(sub["cost"].values <= COST_LIMIT) * 100
        print(f"  {a:<18} {sub['ret'].mean():>10.2f} {sub['cost'].mean():>10.2f} {safe_pct:>7.1f}%")


if __name__ == "__main__":
    main()
