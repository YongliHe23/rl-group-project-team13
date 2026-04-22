import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def find_latest_progress(algo: str):
    paths = sorted(Path("runs").glob(f"{algo}-*/*/progress.csv"))
    if not paths:
        raise FileNotFoundError(f"No progress.csv found for {algo} under runs/")
    return paths[-1]
  
# plot similar results as Figure 7 in "Benchmarking Safe Exploration in Deep Reinforcement Learning"

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

    # ===== Extract Figure 7 data =====
    fig7_df = df[
        ["TotalEnvSteps", "Metrics/EpRet", "Metrics/EpCost", "CostRate_est"]
    ].rename(
        columns={
            "Metrics/EpRet": "AverageEpRet",
            "Metrics/EpCost": "AverageEpCost",
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

    # ===== Plot Figure 7-style curves =====
    df["DeltaSteps"] = df["TotalEnvSteps"].diff().fillna(df["TotalEnvSteps"])
    df["CostRate_epoch_est"] = df["Metrics/EpCost"] / df["Metrics/EpLen"]
    df["CumulativeCost_est"] = (df["DeltaSteps"] * df["CostRate_epoch_est"]).cumsum()
    df["CostRate_est"] = df["CumulativeCost_est"] / df["TotalEnvSteps"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))

    axes[0].plot(df["TotalEnvSteps"], df["Metrics/EpRet"], linewidth=2)
    axes[0].set_title("AverageEpRet")
    axes[0].set_xlabel("TotalEnvSteps")
    axes[0].set_ylabel("AverageEpRet")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["TotalEnvSteps"], df["Metrics/EpCost"], linewidth=2)
    axes[1].axhline(25, linestyle="--", linewidth=1.5)
    axes[1].set_title("AverageEpCost")
    axes[1].set_xlabel("TotalEnvSteps")
    axes[1].set_ylabel("AverageEpCost")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df["TotalEnvSteps"], df["CostRate_est"], linewidth=2)
    axes[2].axhline(0.025, linestyle="--", linewidth=1.5)
    axes[2].set_title("CostRate")
    axes[2].set_xlabel("TotalEnvSteps")
    axes[2].set_ylabel("CostRate")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    env_name = csv_path.parts[-3].split(f"{args.algo}-", 1)[-1]
    seed_name = csv_path.parts[-2]
    out_dir = Path("/home/yonglihe/ece567/project/rl-group-project-team13/project6-safe-rl/plots") / args.algo
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{{{env_name}}}-{{{seed_name}}}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to {out_path}")

# using the following code as an example,
# python scripts\plot_ppolag_trpolag.py --algo PPOLag --csv "runs\PPOLag-{SafetyPointGoal1-v0}\seed-000-2026-03-29-13-02-42\progress.csv"

if __name__ == "__main__":
    main()
