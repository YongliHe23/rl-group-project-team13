"""
Reproduce Figure 2 (aggregated bar plots) from OGBench paper using our Table 2 results.

Data is encoded from Table 2 (our results). Missing entries are NaN.
Category groupings follow Table 6 from the paper.

Usage:
    python plot_figure2.py
Output:
    figure2_reproduction.png
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Table 2 data: {env: {agent: (mean, std) or None}}
# None  = not run / missing
# std=None = only 1 seed (no std available)
# ---------------------------------------------------------------------------
DATA = {
    # --- Locomotion (states) ---
    "antmaze-medium-navigate-v0": {
        "GCIVL": (49, 3), "GCIQL": (65, 4), "QRL": (83, 8), "CRL": (93, 2), "HIQL": (95, 1),
    },
    "antmaze-large-navigate-v0": {
        "GCIVL": (4, 2),  "GCIQL": (31, 2), "QRL": (72, 6),  "CRL": (79, 5), "HIQL": (84, 3),
    },
    "antsoccer-arena-navigate-v0": {
        "GCIVL": (45, 2), "GCIQL": (44, 2), "QRL": (10, 3),  "CRL": (29, 3), "HIQL": (61, 3),
    },
    "antsoccer-medium-navigate-v0": {
        "GCIVL": (2, 1),  "GCIQL": (5, 1),  "QRL": (2, 1),   "CRL": (2, 0),  "HIQL": (8, 1),
    },
    # --- Manipulation (states) ---
    "cube-single-play-v0": {
        "GCIVL": (50, 1), "GCIQL": (72, 3), "QRL": (3, 1),   "CRL": (25, 1), "HIQL": (22, 3),
    },
    "cube-double-play-v0": {
        "GCIVL": (32, 2), "GCIQL": (37, 2), "QRL": (1, 0),   "CRL": (8, 1),  "HIQL": (5, 2),
    },
    "puzzle-3x3-play-v0": {
        "GCIVL": (5, 1),  "GCIQL": (95, 2), "QRL": (0, 0),   "CRL": (5, 1),  "HIQL": (11, 2),
    },
    # --- Drawing (pixels) ---
    "powderworld-easy-play-v0": {
        "GCIVL": (98, 1), "GCIQL": (98, 2), "QRL": (29, None), "CRL": (12, 10), "HIQL": (33,None),
    },
    # --- Stitching (states) ---
    "antmaze-medium-stitch-v0": {
        "GCIVL": (36, 4), "GCIQL": (35, 3), "QRL": (34, 6),  "CRL": (30, 7), "HIQL": (91, 1),
    },
    "antmaze-large-stitch-v0": {
        "GCIVL": (19, 2), "GCIQL": (6, 2),  "QRL": (20, 2),  "CRL": (18, 2), "HIQL": (37, 5),
    },
    # --- Exploratory (states) ---
    "antmaze-medium-explore-v0": {
        "GCIVL": (15, 3), "GCIQL": (8, 1),  "QRL": (1, 0),   "CRL": (2, 1),  "HIQL": (20, 5),
    },
    "antmaze-large-explore-v0": {
        "GCIVL": (7, 3),  "GCIQL": (0, 0),  "QRL": (0, 0),   "CRL": (0, 0),  "HIQL": (2, 2),
    },
}

# ---------------------------------------------------------------------------
# Category groupings (Table 6 highlighted entries)
# ---------------------------------------------------------------------------
CATEGORIES = {
    "Locomotion\n(states)": [
        "antmaze-medium-navigate-v0",
        "antmaze-large-navigate-v0",
        "antsoccer-arena-navigate-v0",
        "antsoccer-medium-navigate-v0",
    ],
    "Manipulation\n(states)": [
        "cube-single-play-v0",
        "cube-double-play-v0",
        "puzzle-3x3-play-v0",
    ],
    "Drawing\n(pixels)": [
        "powderworld-easy-play-v0",
    ],
    "Stitching\n(states)": [
        "antmaze-medium-stitch-v0",
        "antmaze-large-stitch-v0",
    ],
    "Exploratory\n(states)": [
        "antmaze-medium-explore-v0",
        "antmaze-large-explore-v0",
    ],
}

AGENTS = ["GCIVL", "GCIQL", "QRL", "CRL", "HIQL"]

N_BOOTSTRAP = 10_000
RNG = np.random.default_rng(42)
SHOW_ERROR_BARS = False   # Set to False to plot means only

# ---------------------------------------------------------------------------
# Aggregate: mean of per-env means; 95% bootstrap CI across envs as error bar
# Returns (mean, lower_err, upper_err) where errors are distances from mean.
# lower_err / upper_err are NaN when only 1 env is available (no resampling).
# ---------------------------------------------------------------------------
def aggregate(category_envs, agent):
    means = []
    for env in category_envs:
        val = DATA[env].get(agent)
        if val is not None:
            means.append(val[0])
    if not means:
        return np.nan, np.nan, np.nan
    arr = np.array(means, dtype=float)
    mean = np.mean(arr)
    if len(arr) == 1:
        return mean, np.nan, np.nan
    # Bootstrap: resample with replacement, compute mean each time
    boot_means = np.mean(
        RNG.choice(arr, size=(N_BOOTSTRAP, len(arr)), replace=True), axis=1
    )
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return mean, mean - lo, hi - mean   # (mean, lower_err, upper_err)

# ---------------------------------------------------------------------------
# Build aggregated table
# ---------------------------------------------------------------------------
agg = {}
for cat, envs in CATEGORIES.items():
    agg[cat] = {}
    for agent in AGENTS:
        agg[cat][agent] = aggregate(envs, agent)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
BAR_COLOR = "#b0c4de"   # steel blue (similar to paper)
ERR_COLOR = "#4a4a4a"
MISSING_HATCH = "//"

n_cats = len(CATEGORIES)
fig, axes = plt.subplots(1, n_cats, figsize=(3.0 * n_cats, 3.8))

for ax, (cat, agent_vals) in zip(axes, agg.items()):
    xs = np.arange(len(AGENTS))
    for i, agent in enumerate(AGENTS):
        mean, lo_err, hi_err = agent_vals[agent]
        if np.isnan(mean):
            # Grey hatched bar to indicate missing data
            ax.bar(xs[i], 1, color="#e0e0e0", hatch=MISSING_HATCH,
                   edgecolor="#aaaaaa", width=0.65, zorder=2)
        else:
            ax.bar(xs[i], mean, color=BAR_COLOR, edgecolor="white",
                   width=0.65, zorder=2)
            if SHOW_ERROR_BARS and not (np.isnan(lo_err) or np.isnan(hi_err)):
                ax.errorbar(xs[i], mean, yerr=[[lo_err], [hi_err]], fmt="none",
                            color=ERR_COLOR, capsize=3, linewidth=1.2, zorder=3)

    ax.set_xticks(xs)
    ax.set_xticklabels(AGENTS, rotation=45, ha="right", fontsize=8)
    ax.set_title(cat, fontsize=9, pad=4)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, color="#cccccc", zorder=0)

axes[0].set_ylabel("Success Rate (%)", fontsize=9)


fig.suptitle("Figure 2 Reproduction — Our Results (Team 13)", fontsize=10, y=1.02)
plt.tight_layout()

out_path = "figure2_reproduction.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")

# ---------------------------------------------------------------------------
# Also print the aggregated table for reference
# ---------------------------------------------------------------------------
print("\nAggregated results (mean, 95% bootstrap CI across envs in category):\n")
header = f"{'Category':<25}" + "".join(f"{a:>14}" for a in AGENTS)
print(header)
print("-" * len(header))
for cat, agent_vals in agg.items():
    label = cat.replace("\n", " ")
    row = f"{label:<25}"
    for agent in AGENTS:
        m, lo, hi = agent_vals[agent]
        if np.isnan(m):
            row += f"{'—':>14}"
        elif np.isnan(lo):
            row += f"{m:>12.1f}  "
        else:
            row += f"{m:>6.1f}[-{lo:.1f},+{hi:.1f}] "
    print(row)
