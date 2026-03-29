import argparse
import yaml
import omnisafe
import safety_gymnasium  


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/trpo_lag/config.yaml",
        help="Path to TRPO-Lag yaml config.",
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default=None,
        help="Environment ID, e.g. SafetyPointGoal1-v0. Overrides YAML env_id if provided.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Overrides YAML seed if provided.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env_id = args.env_id if args.env_id is not None else cfg.pop("env_id")
    seed = args.seed if args.seed is not None else cfg.pop("seed", 0)
    cfg["seed"] = seed

    agent = omnisafe.Agent("TRPOLag", env_id, custom_cfgs=cfg)
    agent.learn()
  
# using the following as an example:
# python scripts/train_ppolag.py --config configs/ppo_lag/config.yaml --env_id SafetyPointGoal1-v0 --seed 0

if __name__ == "__main__":
    main()
