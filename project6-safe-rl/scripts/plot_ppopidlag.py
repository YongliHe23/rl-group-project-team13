"""
plot_ppopidlag.py — Figure 7-style plots for OmniSafe CPPOPID results.

Direct mirror of plot_ppolag_trpolag.py, adapted for CPPOPID's OmniSafe output.

Plots three panels matching the OpenAI Safe Exploration evaluation criteria:
  1. AverageEpRet   — episodic reward (task performance)
  2. AverageEpCost  — episodic constraint cost (safety, with cost_limit line)
  3. CostRate       — cumulative cost / cumulative steps (safety regret, ρ_c)

Usage
-----
# Auto-locate latest CPPOPID run under runs/:
    python scripts/plot_ppopidlag.py

# Specify a CSV directly:
    python scripts/plot_ppopidlag.py \\
        --csv "runs/CPPOPID-{SafetyPointGoal2-v0}/seed-000-.../progress.csv"

# Override algorithm prefix for auto-discovery:
    python scripts/plot_ppopidlag.py --algo CPPOPID

# Override output file name:
    python scripts/plot_ppopidlag.py --out figs/cppopid_pointgoal.png

# Compare with other algorithms:
#   → use plot_comparison.py instead.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


COST_LIMIT    = 25.0
COST_RATE_LIM = COST_LIMIT / 1000.0
ALGO_LABEL    = "CPPOPID"
COLOR         = "#d62728"              # red — distinct from blue/orange/green/purple


def find_latest_progress(algo_prefix: str = "CPPOPID") -> Path:
    paths = sorted(Path("runs").glob(f"{algo_prefix}-*/*/progress.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No progress.csv found under runs/{algo_prefix}-*/ — "
            "run run_ppopidlag.py first, or pass --csv explicitly."
        )
    return paths[-1]


def load_and_enrich(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["DeltaSteps"]         = df["TotalEnvSteps"].diff().fillna(df["TotalEnvSteps"])
    df["CostRate_epoch_est"] = df["Metrics/EpCost"] / df["Metrics/EpLen"]
    df["CumulativeCost_est"] = (df["DeltaSteps"] * df["CostRate_epoch_est"]).cumsum()
    df["CostRate_est"]       = df["CumulativeCost_est"] / df["TotalEnvSteps"]
    return df


def print_runtime_info(df: pd.DataFrame) -> None:
    final_steps = float(df["TotalEnvSteps"].iloc[-1])
    print(f"\nFinal TotalEnvSteps : {final_steps:.0f}")
    if "Time/Total" in df.columns:
        t = float(df["Time/Total"].iloc[-1])
        h, m, s = int(t // 3600), int((t % 3600) // 60), t % 60
        print(f"Elapsed time        : {h}h {m}m {s:.1f}s")
        print(f"Average FPS         : {final_steps / t:.1f}")

    print("\nFinal extracted row:")
    cols = ["TotalEnvSteps", "Metrics/EpRet", "Metrics/EpCost", "CostRate_est"]
    print(df[[c for c in cols if c in df.columns]].tail(1).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Plot CPPOPID (PPO + PID-Lagrangian) training results (Figure 7 style)."
    )
    parser.add_argument(
        "--algo", type=str, default="CPPOPID",
        help="Algorithm prefix for auto-discovery under runs/"
    )
    parser.add_argument("--csv",  default=None, help="Path to progress.csv (auto-discovered if omitted)")
    parser.add_argument("--out",  default=None, help="Output PNG path")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else find_latest_progress(args.algo)
    print(f"Using CSV: {csv_path}")

    df = load_and_enrich(csv_path)
    print_runtime_info(df)

    # Infer env tag from path for title / filename
    env_tag = ""
    for p in csv_path.parts:
        if "CPPOPID-" in p:
            env_tag = p.replace("CPPOPID-", "").strip("{}")
            break

    title_base = f"{ALGO_LABEL}  [{env_tag}]" if env_tag else ALGO_LABEL

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))

    axes[0].plot(df["TotalEnvSteps"], df["Metrics/EpRet"], color=COLOR, linewidth=2)
    axes[0].set_title(f"AverageEpRet  ({title_base})")
    axes[0].set_xlabel("TotalEnvSteps")
    axes[0].set_ylabel("AverageEpRet")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["TotalEnvSteps"], df["Metrics/EpCost"], color=COLOR, linewidth=2)
    axes[1].axhline(COST_LIMIT, linestyle="--", color="red", linewidth=1.5,
                    label=f"cost_limit = {COST_LIMIT}")
    axes[1].set_title(f"AverageEpCost  ({title_base})")
    axes[1].set_xlabel("TotalEnvSteps")
    axes[1].set_ylabel("AverageEpCost")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df["TotalEnvSteps"], df["CostRate_est"], color=COLOR, linewidth=2)
    axes[2].axhline(COST_RATE_LIM, linestyle="--", color="red", linewidth=1.5,
                    label=f"rate limit ≈ {COST_RATE_LIM}")
    axes[2].set_title(f"CostRate  ({title_base})")
    axes[2].set_xlabel("TotalEnvSteps")
    axes[2].set_ylabel("CostRate")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()

    out_path = args.out or f"{args.algo}_{env_tag}_plot.png".replace("/", "_")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")
    plt.show()


# Examples:
#   python scripts/plot_ppopidlag.py
#   python scripts/plot_ppopidlag.py --algo CPPOPID \
#       --csv "runs/CPPOPID-{SafetyPointGoal2-v0}/seed-000-.../progress.csv"

if __name__ == "__main__":
    main()
