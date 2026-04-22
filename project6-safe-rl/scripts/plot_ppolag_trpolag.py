import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def find_latest_progress(algo: str):
    paths = sorted(Path("runs").glob(f"{algo}-*/*/progress.csv"))
    if not paths:
        raise FileNotFoundError(f"No progress.csv found for {algo} under runs/")
    return paths[-1]


def find_lagrange_column(df: pd.DataFrame):
    candidates = [
        "Metrics/LagrangeMultiplier",
        "Metrics/LagrangeMultiplier/Mean",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


# plot similar results as Figure 7 in "Benchmarking Safe Exploration in Deep RL"
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="PPOLag")
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else find_latest_progress(args.algo)
    print(f"Using CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    df["DeltaSteps"] = df["TotalEnvSteps"].diff().fillna(df["TotalEnvSteps"])
    df["CostRate_epoch_est"] = df["Metrics/EpCost"] / df["Metrics/EpLen"]
    df["CumulativeCost_est"] = (df["DeltaSteps"] * df["CostRate_epoch_est"]).cumsum()
    df["CostRate_est"] = df["CumulativeCost_est"] / df["TotalEnvSteps"]

    lag_col = find_lagrange_column(df)

    # ===== Extract Figure 7 data =====
    keep_cols = ["TotalEnvSteps", "Metrics/EpRet", "Metrics/EpCost", "CostRate_est"]
    if lag_col is not None:
        keep_cols.append(lag_col)

    fig7_df = df[keep_cols].rename(
        columns={
            "Metrics/EpRet": "AverageEpRet",
            "Metrics/EpCost": "AverageEpCost",
            lag_col: "LagrangeMultiplier" if lag_col is not None else "",
        }
    )

    print("Final extracted row:")
    print(fig7_df.tail(1).to_string(index=False))

    # ===== Print runtime information =====
    final_total_steps = float(df["TotalEnvSteps"].iloc[-1])
    final_total_time_sec = float(df["Time/Total"].iloc[-1])

    hours = int(final_total_time_sec // 3600)
    minutes = int((final_total_time_sec % 3600) // 60)
    seconds = final_total_time_sec % 60

    fps_avg = final_total_steps / final_total_time_sec
    fps_last = float(df["Time/FPS"].iloc[-1])

    print("\nRuntime information:")
    print(f"Final TotalEnvSteps: {final_total_steps:.0f}")
    print(f"Final Time/Total (sec): {final_total_time_sec:.2f}")
    print(f"Elapsed runtime: {hours}h {minutes}m {seconds:.2f}s")
    print(f"Average FPS over run: {fps_avg:.2f}")
    print(f"Last logged FPS: {fps_last:.2f}")

    if lag_col is not None:
        print(f"Using Lagrange multiplier column: {lag_col}")
        print(f"Final Lagrange multiplier: {df[lag_col].iloc[-1]:.6f}")
    else:
        print("No Lagrange multiplier column found in CSV.")

    # ===== Plot curves =====
    ncols = 4 if lag_col is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 3.4))

    if ncols == 3:
        ax0, ax1, ax2 = axes
    else:
        ax0, ax1, ax2, ax3 = axes

    ax0.plot(df["TotalEnvSteps"], df["Metrics/EpRet"], linewidth=2)
    ax0.set_title("AverageEpRet")
    ax0.set_xlabel("TotalEnvSteps")
    ax0.set_ylabel("AverageEpRet")
    ax0.grid(True, alpha=0.3)

    ax1.plot(df["TotalEnvSteps"], df["Metrics/EpCost"], linewidth=2)
    ax1.axhline(25, linestyle="--", linewidth=1.5)
    ax1.set_title("AverageEpCost")
    ax1.set_xlabel("TotalEnvSteps")
    ax1.set_ylabel("AverageEpCost")
    ax1.grid(True, alpha=0.3)

    ax2.plot(df["TotalEnvSteps"], df["CostRate_est"], linewidth=2)
    ax2.axhline(0.025, linestyle="--", linewidth=1.5)
    ax2.set_title("CostRate")
    ax2.set_xlabel("TotalEnvSteps")
    ax2.set_ylabel("CostRate")
    ax2.grid(True, alpha=0.3)

    if lag_col is not None:
        ax3.plot(df["TotalEnvSteps"], df[lag_col], linewidth=2)
        ax3.set_title("LagrangeMultiplier")
        ax3.set_xlabel("TotalEnvSteps")
        ax3.set_ylabel("Lambda")
        ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    env_name = csv_path.parts[-3].split(f"{args.algo}-", 1)[-1]
    seed_name = csv_path.parts[-2]
    out_dir = Path("/home/yonglihe/ece567/project/rl-group-project-team13/project6-safe-rl/plots") / args.algo
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{{{env_name}}}-{{{seed_name}}}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to {out_path}")


# Example:
# python scripts\plot_ppolag_trpolag.py --algo PPOLag --csv "runs\PPOLag-{SafetyPointGoal1-v0}\seed-000-2026-03-29-13-02-42\progress.csv"

if __name__ == "__main__":
    main()