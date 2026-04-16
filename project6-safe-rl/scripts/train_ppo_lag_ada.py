import argparse
import yaml
import safety_gymnasium  # noqa: F401
import os
import sys
import math

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("sys.path[0]:", sys.path[0])

from improved_alg.ppo_lag_ada import PPOLagAdapt
from omnisafe.utils.config import Config, get_default_kwargs_yaml


def recursive_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            recursive_update(base[k], v)
        else:
            base[k] = v
    return base


def load_builtin_ppolag_yaml(env_id: str) -> dict:
    cfg = get_default_kwargs_yaml("PPOLag", env_id, "on-policy")
    return dict(cfg)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/ppo_lag_ada/config.yaml")
    parser.add_argument("--env_id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--lambda_schedule", type=str, default=None)
    parser.add_argument("--lambda_min", type=float, default=None)
    parser.add_argument("--lambda_max", type=float, default=None)
    parser.add_argument("--lambda_piecewise_split", type=float, default=None)
    parser.add_argument("--lambda_p0", type=float, default=None)
    parser.add_argument("--lambda_kappa", type=float, default=None)
    parser.add_argument("--lambda_exp_alpha", type=float, default=None)
    parser.add_argument("--lambda_eta", type=float, default=None)
    parser.add_argument("--lambda_ema_beta", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f)

    env_id = args.env_id if args.env_id is not None else user_cfg.get("env_id", "SafetyPointGoal1-v0")

    cfg_dict = load_builtin_ppolag_yaml(env_id)
    cfg_dict = recursive_update(cfg_dict, user_cfg)

    seed = args.seed if args.seed is not None else cfg_dict.pop("seed", 0)
    cfg_dict["seed"] = seed

    cfg_dict.setdefault("train_cfgs", {})
    cfg_dict.setdefault("algo_cfgs", {})
    cfg_dict.setdefault("lagrange_cfgs", {})
    cfg_dict.setdefault("lambda_schedule_cfgs", {})

    schedule_name = cfg_dict.get("lambda_schedule_cfgs", {}).get("lambda_schedule", "default")

    cfg_dict.setdefault("exp_name", f"ppo_lag_adapt_{schedule_name}")
    cfg_dict.setdefault("seed", seed)
    cfg_dict.setdefault("wandb", False)
    cfg_dict.setdefault("device", cfg_dict["train_cfgs"].get("device", "cpu"))

    cfg_dict.setdefault("logger_cfgs", {})
    cfg_dict["logger_cfgs"].setdefault("use_wandb", False)
    cfg_dict["logger_cfgs"].setdefault("save_model_freq", 0)
    cfg_dict["logger_cfgs"].setdefault("log_dir", "./runs")

    override_fields = [
        "lambda_schedule",
        "lambda_min",
        "lambda_max",
        "lambda_piecewise_split",
        "lambda_p0",
        "lambda_kappa",
        "lambda_exp_alpha",
        "lambda_eta",
        "lambda_ema_beta",
    ]

    for field in override_fields:
        value = getattr(args, field)
        if value is not None:
            cfg_dict["lambda_schedule_cfgs"][field] = value

    if args.device is not None:
        cfg_dict["train_cfgs"]["device"] = args.device

    total_steps = cfg_dict["train_cfgs"].get("total_steps", None)
    steps_per_epoch = cfg_dict["algo_cfgs"].get("steps_per_epoch", None)

    if "epochs" not in cfg_dict["train_cfgs"]:
        if total_steps is None or steps_per_epoch is None:
            raise ValueError(
                "Missing train_cfgs.total_steps or algo_cfgs.steps_per_epoch, "
                "cannot derive train_cfgs.epochs."
            )
        cfg_dict["train_cfgs"]["epochs"] = math.ceil(total_steps / steps_per_epoch)

    print("Epochs:", cfg_dict["train_cfgs"]["epochs"])
    print("Running env_id:", env_id)
    print("Seed:", seed)
    print("Device:", cfg_dict["train_cfgs"].get("device"))
    print("Lagrange cfgs:", cfg_dict["lagrange_cfgs"])
    print("Lambda schedule cfgs:", cfg_dict["lambda_schedule_cfgs"])

    try:
        cfgs = Config.dict2config(cfg_dict)
    except AttributeError:
        cfgs = Config.from_dict(cfg_dict)

    algo = PPOLagAdapt(env_id, cfgs)
    algo.learn()


if __name__ == "__main__":
    main()
  
# using the following as an example:
# python scripts/train_ppo_lag_ada.py --config configs/ppo_lag_ada/config.yaml --env_id SafetyPointGoal2-v0 --seed 0
