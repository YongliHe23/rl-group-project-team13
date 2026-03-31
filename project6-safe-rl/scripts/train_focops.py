import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import omnisafe
import pandas as pd
import yaml


@dataclass
class RunResult:
    seed: int
    run_dir: Path
    progress_csv: Path
    config_json: Optional[Path]
    completed_steps: float


def newest_matching_dir(base_dir: Path, seed: int) -> Optional[Path]:
    if not base_dir.exists():
        return None
    candidates = [p for p in base_dir.glob(f"seed-{seed:03d}-*") if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def expected_completed_steps(total_steps: int, steps_per_epoch: int) -> int:
    return (total_steps // steps_per_epoch) * steps_per_epoch


def validate_progress_csv(
    progress_csv: Path,
    expected_total_steps: int,
    steps_per_epoch: int,
) -> float:
    if not progress_csv.exists():
        raise FileNotFoundError(f"Missing progress.csv: {progress_csv}")

    df = pd.read_csv(progress_csv)
    if "TotalEnvSteps" not in df.columns:
        raise ValueError(f"progress.csv missing TotalEnvSteps column: {progress_csv}")

    completed_steps = float(df["TotalEnvSteps"].max())
    min_acceptable_steps = expected_completed_steps(expected_total_steps, steps_per_epoch)

    if completed_steps < min_acceptable_steps:
        raise ValueError(
            f"Run appears incomplete. Expected at least {min_acceptable_steps} steps "
            f"(largest full epoch not exceeding {expected_total_steps}), "
            f"but found {completed_steps} in {progress_csv}"
        )

    return completed_steps


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML config must parse to a dictionary.")
    return cfg


def build_omnisafe_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    hidden_sizes = cfg["model"]["hidden_sizes"]
    init_log_std = float(cfg["model"]["initial_log_std"])
    init_std = float(math.exp(init_log_std))

    return {
        "seed": int(cfg["seed"]),
        "train_cfgs": {
            "total_steps": int(cfg["train"]["total_steps"]),
            "vector_env_nums": int(cfg["runtime"]["vector_env_nums"]),
            "parallel": int(cfg["runtime"]["parallel"]),
            "device": str(cfg["runtime"]["device"]),
            "torch_threads": int(cfg["runtime"]["torch_threads"]),
        },
        "algo_cfgs": {
            "steps_per_epoch": int(cfg["train"]["steps_per_epoch"]),
            "update_iters": int(cfg["train"]["update_iters"]),
            "batch_size": int(cfg["train"]["minibatch_size"]),
            "target_kl": float(cfg["algo"]["target_kl"]),
            "entropy_coef": 0.0,
            "reward_normalize": bool(cfg["normalization"]["reward_normalize"]),
            "cost_normalize": bool(cfg["normalization"]["cost_normalize"]),
            "obs_normalize": bool(cfg["normalization"]["obs_normalize"]),
            "kl_early_stop": bool(cfg["algo"]["kl_early_stop"]),
            "focops_eta": float(cfg["algo"]["focops_eta"]),
            "focops_lam": float(cfg["algo"]["focops_lam"]),
            "use_max_grad_norm": bool(cfg["optimization"]["use_max_grad_norm"]),
            "max_grad_norm": float(cfg["optimization"]["max_grad_norm"]),
            "use_critic_norm": bool(cfg["optimization"]["use_critic_norm"]),
            "critic_norm_coef": float(cfg["optimization"]["critic_l2_reg"]),
            "gamma": float(cfg["algo"]["gamma"]),
            "cost_gamma": float(cfg["algo"]["cost_gamma"]),
            "lam": float(cfg["algo"]["gae_lam"]),
            "lam_c": float(cfg["algo"]["cost_gae_lam"]),
            "clip": float(cfg["algo"]["clip"]),
            "adv_estimation_method": str(cfg["algo"]["adv_estimation_method"]),
            "standardized_rew_adv": bool(cfg["algo"]["standardized_rew_adv"]),
            "standardized_cost_adv": bool(cfg["algo"]["standardized_cost_adv"]),
            "penalty_coef": float(cfg["algo"]["penalty_coef"]),
            "use_cost": bool(cfg["algo"]["use_cost"]),
        },
        "logger_cfgs": {
            "use_wandb": False,
            "use_tensorboard": bool(cfg["logging"]["use_tensorboard"]),
            "save_model_freq": int(cfg["logging"]["save_model_freq"]),
        },
        "model_cfgs": {
            "weight_initialization_mode": str(cfg["model"]["weight_initialization_mode"]),
            "actor_type": str(cfg["model"]["actor_type"]),
            "linear_lr_decay": bool(cfg["model"]["linear_lr_decay"]),
            "exploration_noise_anneal": bool(cfg["model"]["exploration_noise_anneal"]),
            "std_range": [init_std, float(cfg["model"]["std_range_min"])],
            "actor": {
                "hidden_sizes": hidden_sizes,
                "activation": str(cfg["model"]["activation"]),
                "lr": float(cfg["optimization"]["actor_lr"]),
            },
            "critic": {
                "hidden_sizes": hidden_sizes,
                "activation": str(cfg["model"]["activation"]),
                "lr": float(cfg["optimization"]["critic_lr"]),
            },
        },
        "lagrange_cfgs": {
            "cost_limit": float(cfg["constraint"]["cost_limit"]),
            "lagrangian_multiplier_init": float(cfg["constraint"]["lagrangian_multiplier_init"]),
            "lambda_lr": float(cfg["constraint"]["lambda_lr"]),
            "lambda_optimizer": str(cfg["constraint"]["lambda_optimizer"]),
            "lagrangian_upper_bound": float(cfg["constraint"]["lagrangian_upper_bound"]),
        },
    }


def read_final_metrics(progress_csv: Path) -> Dict[str, Any]:
    df = pd.read_csv(progress_csv).sort_values("TotalEnvSteps")
    return df.iloc[-1].to_dict()


def run_training(
    algo: str,
    env_id: str,
    seed: int,
    total_steps: int,
    steps_per_epoch: int,
    custom_cfgs: Dict[str, Any],
    output_root: Path,
    force_rerun: bool,
) -> RunResult:
    run_dir_root = output_root / "train_logs"
    algo_env_dir = run_dir_root / "runs" / f"{algo}-{{{env_id}}}"

    if not force_rerun:
        latest = newest_matching_dir(algo_env_dir, seed)
        if latest is not None:
            progress_csv = latest / "progress.csv"
            config_json = latest / "config.json"
            try:
                completed_steps = validate_progress_csv(progress_csv, total_steps, steps_per_epoch)
                print(f"Found existing completed run at {latest}. Skipping rerun.")
                return RunResult(
                    seed=seed,
                    run_dir=latest,
                    progress_csv=progress_csv,
                    config_json=config_json if config_json.exists() else None,
                    completed_steps=completed_steps,
                )
            except Exception:
                print("Existing run incomplete or invalid. Rerunning.")

    run_dir_root.mkdir(parents=True, exist_ok=True)
    old_cwd = Path.cwd()

    try:
        os.chdir(run_dir_root)
        agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
        agent.learn()
    finally:
        os.chdir(old_cwd)

    latest = newest_matching_dir(algo_env_dir, seed)
    if latest is None:
        raise FileNotFoundError(f"Could not find output directory for seed {seed} under {algo_env_dir}")

    progress_csv = latest / "progress.csv"
    config_json = latest / "config.json"
    completed_steps = validate_progress_csv(progress_csv, total_steps, steps_per_epoch)

    return RunResult(
        seed=seed,
        run_dir=latest,
        progress_csv=progress_csv,
        config_json=config_json if config_json.exists() else None,
        completed_steps=completed_steps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a single FOCOPS run from YAML config.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/focops/config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name. If omitted, a timestamped name is created.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="repro_outputs",
        help="Directory under which outputs will be saved.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force rerun even if a completed run is found.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_yaml_config(config_path)

    algo = str(cfg["algorithm"])
    env_id = str(cfg["environment"])
    seed = int(cfg["seed"])
    total_steps = int(cfg["train"]["total_steps"])
    steps_per_epoch = int(cfg["train"]["steps_per_epoch"])

    custom_cfgs = build_omnisafe_cfg(cfg)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{algo}_{env_id}_single_run_{timestamp}"

    output_root = Path(args.output_dir) / run_name
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config path: {config_path}")
    print(f"Run name: {run_name}")
    print(f"Output root: {output_root}")
    print(f"Summary dir: {summary_dir}")

    with (summary_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    with (summary_dir / "omnisafe_config.json").open("w", encoding="utf-8") as f:
        json.dump(custom_cfgs, f, indent=2)

    rr = run_training(
        algo=algo,
        env_id=env_id,
        seed=seed,
        total_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
        custom_cfgs=custom_cfgs,
        output_root=output_root,
        force_rerun=args.force_rerun,
    )

    final_metrics = read_final_metrics(rr.progress_csv)

    summary = {
        "algorithm": algo,
        "environment": env_id,
        "seed": seed,
        "run_dir": str(rr.run_dir),
        "progress_csv": str(rr.progress_csv),
        "config_json": str(rr.config_json) if rr.config_json else None,
        "completed_steps": rr.completed_steps,
        "final_metrics": final_metrics,
    }

    with (summary_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining complete.")
    print(f"Run dir: {rr.run_dir}")
    print(f"Progress CSV: {rr.progress_csv}")
    print(f"Completed steps: {rr.completed_steps}")
    print("\nFinal row:")
    for k, v in final_metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
