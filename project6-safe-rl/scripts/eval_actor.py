"""
eval_actor.py — Evaluate a saved OmniSafe actor checkpoint on new seeds.

Loads the 'pi' weights and 'obs_normalizer' from a torch_save .pt file,
rebuilds the policy MLP, then runs N episodes in the target environment.

Usage
-----
python scripts/eval_actor.py \
    --checkpoint runs/ppo_lag_adapt_sigmoid/.../torch_save/epoch-167.pt \
    --env_id SafetyPointGoal2-v0 \
    --seeds 1 2 3 \
    --episodes 10 \
    --label PPOLagAda \
    --out results/eval_ppolagada.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ── Policy reconstruction ──────────────────────────────────────────────────────

def load_policy(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pi_sd = ckpt["pi"]
    norm  = ckpt["obs_normalizer"]

    obs_dim = pi_sd["mean.0.weight"].shape[1]
    act_dim = pi_sd["mean.4.weight"].shape[0]
    h1      = pi_sd["mean.0.weight"].shape[0]
    h2      = pi_sd["mean.2.weight"].shape[0]

    policy = nn.Sequential(
        nn.Linear(obs_dim, h1), nn.Tanh(),
        nn.Linear(h1, h2),      nn.Tanh(),
        nn.Linear(h2, act_dim),
    )
    mean_sd = {k[len("mean."):]: v for k, v in pi_sd.items() if k.startswith("mean.")}
    policy.load_state_dict(mean_sd)
    policy.eval()

    obs_mean = norm["_mean"].numpy()
    obs_std  = np.maximum(norm["_std"].numpy(), 1e-8)
    return policy, obs_mean, obs_std, act_dim


def get_action(policy, obs_mean, obs_std, obs: np.ndarray) -> np.ndarray:
    obs_norm = (obs - obs_mean) / obs_std
    with torch.no_grad():
        action = policy(torch.as_tensor(obs_norm, dtype=torch.float32))
    return action.numpy()


# ── Evaluation loop ────────────────────────────────────────────────────────────

def evaluate(
    checkpoint: str,
    env_id: str,
    seeds,
    episodes_per_seed: int,
    label: str,
    out_csv,
):
    import safety_gymnasium

    policy, obs_mean, obs_std, act_dim = load_policy(checkpoint)
    print(f"Loaded: {checkpoint}")
    print(f"  obs_dim={obs_mean.shape[0]}  act_dim={act_dim}  label={label}")

    rows = []

    for seed in seeds:
        env = safety_gymnasium.make(env_id)
        ep_returns, ep_costs, ep_lens = [], [], []

        for ep in range(episodes_per_seed):
            obs, _ = env.reset(seed=seed * 1000 + ep)
            ep_ret = ep_cost = ep_len = 0.0
            done = False

            while not done:
                action = get_action(policy, obs_mean, obs_std, obs)
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, reward, cost, terminated, truncated, info = env.step(action)
                ep_ret  += float(reward)
                ep_cost += float(cost)
                ep_len  += 1
                done = terminated or truncated

            ep_returns.append(ep_ret)
            ep_costs.append(ep_cost)
            ep_lens.append(ep_len)
            rows.append({"algo": label, "seed": seed, "episode": ep,
                         "ret": ep_ret, "cost": ep_cost, "length": ep_len})

        env.close()
        print(
            f"  seed={seed:3d} | "
            f"ret={np.mean(ep_returns):8.2f}  cost={np.mean(ep_costs):7.2f}"
            f"  len={np.mean(ep_lens):6.1f}  ({episodes_per_seed} eps)"
        )

    all_ret  = [r["ret"]  for r in rows]
    all_cost = [r["cost"] for r in rows]
    all_len  = [r["length"] for r in rows]

    print(f"\n{'─'*55}")
    print(f"  {label} — {len(seeds)} seeds × {episodes_per_seed} eps = {len(rows)} total")
    print(f"  Return : {np.mean(all_ret):.2f} ± {np.std(all_ret):.2f}")
    print(f"  Cost   : {np.mean(all_cost):.2f} ± {np.std(all_cost):.2f}")
    print(f"  Length : {np.mean(all_len):.1f}")
    print(f"  Safe   : {np.mean(np.array(all_cost) <= 25.0)*100:.1f}% of episodes under cost 25")

    if out_csv:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["algo", "seed", "episode", "ret", "cost", "length"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV saved -> {out_path}")

    return rows


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a torch_save .pt file (e.g. epoch-167.pt)")
    parser.add_argument("--env_id", default="SafetyPointGoal2-v0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--episodes", type=int, default=10,
                        help="Episodes per seed")
    parser.add_argument("--label", type=str, default=None,
                        help="Algorithm label written into the CSV (default: inferred from checkpoint path)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output CSV path for per-episode results")
    args = parser.parse_args()

    label = args.label or Path(args.checkpoint).parts[-4]  # e.g. ppo_lag_adapt_sigmoid
    evaluate(args.checkpoint, args.env_id, args.seeds, args.episodes, label, args.out)


if __name__ == "__main__":
    main()
