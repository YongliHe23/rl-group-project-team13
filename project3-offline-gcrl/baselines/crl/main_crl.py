"""
CRL (Contrastive RL)

Environment  : PointMaze – navigate variant (default; all OGBench envs supported)
Dataset      : pointmaze-medium-navigate-v0 (default)

CRL overview:
  - Contrastive critic: phi(s,a) and psi(s') learned s.t. phi·psi ≈ log p(s'|s,a)
  - Twin-critic ensemble; critic loss = sigmoid InfoNCE (BCE on B×B logit matrix)
  - Q(s,a,g) = min(phi1·psi1, phi2·psi2) / sqrt(latent_dim)
  - DDPG+BC actor: −Q(s,clip(μ,−1,1),g)/|Q| − α·log N(a_data|μ,I)  (Gaussian NLL BC)
  - Goal-conditioned single-level policy (no subgoal hierarchy)
  - Critic goals: geometric(1−γ) within trajectory; actor goals: per-env config

References:
  - CRL paper  : https://arxiv.org/abs/2206.07568
  - OGBench    : https://arxiv.org/abs/2410.20092
"""

import argparse
import math
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam

import ogbench
from tqdm import tqdm
from config_crl import get_config

# ─────────────────────────────── fixed hyper-parameters ──────────────────────
# These do not vary across environments.
DATASET_DIR  = "~/.ogbench/data"
LR           = 3e-4
BATCH_SIZE   = 1024
HIDDEN_DIMS  = (512, 512, 512)
LATENT_DIM   = 512
NONLINEARITY = nn.GELU

EVAL_EPISODES = 50   # per task (overridden by per-env config)
SEED          = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ──────────────────────────────── network modules ─────────────────────────────

def mlp(input_dim: int, output_dim: int,
        hidden_dims: tuple = HIDDEN_DIMS,
        nonlinearity=NONLINEARITY) -> nn.Sequential:
    """Build an MLP: Linear → GELU → LayerNorm (repeated), then final Linear."""
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nonlinearity(), nn.LayerNorm(h)]
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class ContrastiveCritic(nn.Module):
    """Twin contrastive critic.

    phi_i(s, a) and psi_i(s') are learned representations such that
    phi_i · psi_i ≈ log p_γ(s' | s, a)  (discounted state occupancy).

    Critic loss: sigmoid InfoNCE — BCE(phi @ psi.T / sqrt(d), I)  for each head.
    Q-value    : min(phi1·psi1, phi2·psi2) / sqrt(d)  per sample (twin clipping).
    """

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.phi1 = mlp(obs_dim + act_dim, LATENT_DIM)
        self.phi2 = mlp(obs_dim + act_dim, LATENT_DIM)
        self.psi1 = mlp(obs_dim, LATENT_DIM)
        self.psi2 = mlp(obs_dim, LATENT_DIM)
        self._scale = math.sqrt(LATENT_DIM)

    def critic_loss(self, obs: torch.Tensor, action: torch.Tensor,
                    future_obs: torch.Tensor) -> torch.Tensor:
        """Sigmoid InfoNCE loss for both critic heads."""
        B      = obs.shape[0]
        x      = torch.cat([obs, action], dim=-1)
        labels = torch.eye(B, device=obs.device)

        phi1   = self.phi1(x)
        psi1   = self.psi1(future_obs)
        loss1  = F.binary_cross_entropy_with_logits(
                     phi1 @ psi1.T / self._scale, labels)

        phi2   = self.phi2(x)
        psi2   = self.psi2(future_obs)
        loss2  = F.binary_cross_entropy_with_logits(
                     phi2 @ psi2.T / self._scale, labels)

        return (loss1 + loss2) / 2.0

    def q(self, obs: torch.Tensor, action: torch.Tensor,
          goal: torch.Tensor) -> torch.Tensor:
        """Per-sample Q(s,a,g) = min(phi1·psi1, phi2·psi2) / sqrt(d).  [B]"""
        x  = torch.cat([obs, action], dim=-1)
        q1 = (self.phi1(x) * self.psi1(goal)).sum(-1) / self._scale
        q2 = (self.phi2(x) * self.psi2(goal)).sum(-1) / self._scale
        return torch.min(q1, q2)


class GaussianActor(nn.Module):
    """Goal-conditioned Gaussian actor with constant std (const_std=True).

    Matches OGBench GCActor with const_std=True, tanh_squash=False:
      - forward() returns the raw mean (no tanh squash)
      - std is fixed at 1.0
      - act() clips mean to [-1, 1] for environment interaction
      - log_prob() computes Gaussian NLL summed over action dims for BC loss
    """

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = mlp(obs_dim * 2, act_dim)   # input: [s; g]

    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, goal], dim=-1))

    def log_prob(self, obs: torch.Tensor, goal: torch.Tensor,
                 actions: torch.Tensor) -> torch.Tensor:
        mean = self.forward(obs, goal)
        return Normal(mean, torch.ones_like(mean)).log_prob(actions).sum(-1)

    def act(self, obs_np: np.ndarray, goal_np: np.ndarray) -> np.ndarray:
        obs  = torch.FloatTensor(obs_np).unsqueeze(0).to(device)
        goal = torch.FloatTensor(goal_np).unsqueeze(0).to(device)
        with torch.no_grad():
            mean = self.forward(obs, goal)
            return mean.clamp(-1, 1).squeeze(0).cpu().numpy()


# ──────────────────────────────── CRL agent ───────────────────────────────────

class CRL:
    """Contrastive RL agent (offline GCRL)."""

    def __init__(self, obs_dim: int, act_dim: int, alpha: float):
        self.alpha  = alpha
        self.critic = ContrastiveCritic(obs_dim, act_dim).to(device)
        self.actor  = GaussianActor(obs_dim, act_dim).to(device)

        self.critic_opt = Adam(self.critic.parameters(), lr=LR)
        self.actor_opt  = Adam(self.actor.parameters(),  lr=LR)

    def _critic_update(self, obs, action, future_obs) -> float:
        loss = self.critic.critic_loss(obs, action, future_obs)
        self.critic_opt.zero_grad(); loss.backward(); self.critic_opt.step()
        return loss.item()

    def _actor_update(self, obs, action, actor_goal) -> float:
        """DDPG+BC: scale-invariant Q term + Gaussian NLL BC term."""
        mean   = self.actor(obs, actor_goal)
        a_clip = mean.clamp(-1, 1)
        q_new  = self.critic.q(obs, a_clip, actor_goal)

        q_loss   = -q_new.mean() / q_new.abs().mean().clamp(min=1e-6).detach()
        log_prob = Normal(mean, torch.ones_like(mean)).log_prob(action).sum(-1)
        bc_loss  = -(self.alpha * log_prob).mean()

        actor_loss = q_loss + bc_loss
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
        return actor_loss.item()

    def update(self, batch: dict) -> dict:
        obs        = torch.FloatTensor(batch["observations"]).to(device)
        action     = torch.FloatTensor(batch["actions"]).to(device)
        future_obs = torch.FloatTensor(batch["future_obs"]).to(device)
        actor_goal = torch.FloatTensor(batch["actor_goals"]).to(device)

        c_loss = self._critic_update(obs, action, future_obs)
        a_loss = self._actor_update(obs, action, actor_goal)
        return dict(c_loss=c_loss, a_loss=a_loss)

    def select_action(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return self.actor.act(obs, goal)


# ──────────────────────────────── batch sampling ──────────────────────────────

class GCDataset:
    """Trajectory-aware dataset for CRL.

    Matches OGBench GCDataset with CRL default config:
      value goals  : geometric(1−γ) steps in trajectory  (value_geom_sample=True)
      actor goals  : mixed traj / random per env config   (actor_geom_sample=False)

    Goal-sampling mix (matching OGBench sample_goals()):
      p_traj = actor_p_trajgoal, p_rand = actor_p_randomgoal  (sum to 1.0)
      Navigate : p_traj=1.0, p_rand=0.0  → uniform from [min(idx+1,final), final]
      Stitch   : p_traj=0.5, p_rand=0.5  → 50% traj, 50% random dataset obs
      Explore  : p_traj=0.0, p_rand=1.0  → fully random dataset obs
    """

    def __init__(self, dataset: dict, discount: float,
                 actor_p_trajgoal: float = 1.0,
                 actor_p_randomgoal: float = 0.0):
        self.dataset            = dataset
        self.discount           = discount
        self.actor_p_trajgoal   = actor_p_trajgoal
        self.actor_p_randomgoal = actor_p_randomgoal

        terminals     = dataset["terminals"].astype(bool)
        terminals[-1] = True                             # safety: always mark last
        terminal_locs = np.where(terminals)[0]

        # final_state_idxs[i] = last index of the trajectory containing step i
        self.final_state_idxs = terminal_locs[
            np.searchsorted(terminal_locs, np.arange(len(dataset["observations"])))
        ]
        self.N = len(dataset["observations"])

    def _sample_actor_goals(self, idx: np.ndarray,
                             final: np.ndarray) -> np.ndarray:
        """Sample actor goal indices mixing traj and random goals.

        Matches OGBench sample_goals() with p_curgoal=0:
            goal = traj_goal   if rand < p_trajgoal
                   random_goal otherwise
        """
        batch_size = len(idx)

        # Traj goals: uniform from [min(idx+1, final), final]
        start          = np.minimum(idx + 1, final)
        distances      = np.random.rand(batch_size)
        traj_goal_idxs = np.round(
            start * distances + final * (1.0 - distances)
        ).astype(np.int64)

        if self.actor_p_trajgoal == 1.0:
            return traj_goal_idxs

        # Random goals: uniform from entire dataset
        random_goal_idxs = np.random.randint(0, self.N, size=batch_size)

        if self.actor_p_randomgoal == 1.0:
            return random_goal_idxs

        # Mixed: Bernoulli with p_trajgoal
        use_traj = np.random.rand(batch_size) < self.actor_p_trajgoal
        return np.where(use_traj, traj_goal_idxs, random_goal_idxs)

    def sample(self, batch_size: int) -> dict:
        obs_all = self.dataset["observations"]
        idx     = np.random.randint(0, self.N - 1, size=batch_size)
        final   = self.final_state_idxs[idx]

        # ── Critic / value goals: geometric(1−γ), within trajectory ──────────
        geom_c          = np.random.geometric(1.0 - self.discount, size=batch_size)
        value_goal_idxs = np.minimum(idx + geom_c, final)
        future_obs      = obs_all[value_goal_idxs]

        # ── Actor goals: mixed traj / random per env config ───────────────────
        actor_idxs  = self._sample_actor_goals(idx, final)
        actor_goals = obs_all[actor_idxs]

        return dict(
            observations=obs_all[idx],
            actions=self.dataset["actions"][idx],
            future_obs=future_obs,
            actor_goals=actor_goals,
        )


# ──────────────────────────────── evaluation ──────────────────────────────────

def evaluate(env, agent: CRL, num_episodes: int) -> dict:
    """Evaluate over all 5 tasks × num_episodes each."""
    per_task_success, returns = [], []

    for task_id in range(1, 6):
        task_successes = []
        for _ in range(num_episodes):
            obs, info = env.reset(options=dict(task_id=task_id, render_goal=False))
            goal      = info['goal'].astype(np.float32)
            ep_return = 0.0

            done = False
            while not done:
                action = agent.select_action(obs.astype(np.float32), goal)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += float(reward)
                done = terminated or truncated

            task_successes.append(float(info['success']))
            returns.append(ep_return)

        per_task_success.append(np.mean(task_successes))

    return dict(
        per_task_success=per_task_success,
        success_rate=np.mean(per_task_success),
        mean_return=np.mean(returns),
    )


# ────────────────────────────────────── main ──────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",        default="pointmaze",  help="Environment base name")
    parser.add_argument("--task",       default="navigate",   help="Task name")
    parser.add_argument("--dsize",      default="medium",     help="Dataset size/difficulty")
    parser.add_argument("--train-step", default=None, type=int,
                        help="Override number of gradient steps (default: per-env config)")
    parser.add_argument("--slurm-tqdm", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="Print every 10 000 steps instead of tqdm (Slurm mode)")
    args = parser.parse_args()

    env_name = f"{args.env}-{args.dsize}-{args.task}-v0"

    # ── Load per-environment configuration ────────────────────────────────────
    cfg         = get_config(env_name)
    train_steps = args.train_step if args.train_step is not None else cfg.train_steps

    print(f"\n{'='*60}")
    print(f"CRL on OGBench: {env_name}")
    print(f"  Train steps  : {train_steps:,}  |  Batch size : {cfg.batch_size}")
    print(f"  LR={cfg.lr}  γ={cfg.discount}  α={cfg.alpha}")
    print(f"  Hidden dims  : {HIDDEN_DIMS}  |  Latent dim : {LATENT_DIM}")
    print(f"  Value goals  : geom(1-γ) traj  (value_geom_sample=True)")
    print(f"  Actor goals  : p_traj={cfg.actor_p_trajgoal}  "
          f"p_rand={cfg.actor_p_randomgoal}  geom=False")
    print(f"{'='*60}\n")

    # ── 1. Load environment + dataset ─────────────────────────────────────────
    print("Loading environment and dataset …")
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
        env_name,
        dataset_dir=DATASET_DIR,
        compact_dataset=False,
    )
    obs_dim = train_dataset["observations"].shape[1]
    act_dim = train_dataset["actions"].shape[1]
    print(f"  Train dataset size : {len(train_dataset['observations']):,}")
    print(f"  Val   dataset size : {len(val_dataset['observations']):,}")
    print(f"  Observation dim    : {obs_dim}")
    print(f"  Action dim         : {act_dim}")
    print(f"  Dataset keys       : {list(train_dataset.keys())}\n")

    # ── 2. Initialise agent and dataset wrapper ────────────────────────────────
    agent      = CRL(obs_dim, act_dim, alpha=cfg.alpha)
    gc_dataset = GCDataset(
        train_dataset,
        discount           = cfg.discount,
        actor_p_trajgoal   = cfg.actor_p_trajgoal,
        actor_p_randomgoal = cfg.actor_p_randomgoal,
    )
    print("Agent initialised.\n")

    # ── 3. Training loop ───────────────────────────────────────────────────────
    print(f"Training for {train_steps:,} steps …")

    if args.slurm_tqdm:
        for step in range(1, train_steps + 1):
            batch  = gc_dataset.sample(cfg.batch_size)
            losses = agent.update(batch)
            if step % 10000 == 0:
                print(f"  step {step:>8,}/{train_steps:,} | "
                      f"C={losses['c_loss']:.4f}  "
                      f"A={losses['a_loss']:.4f}")
    else:
        pbar = tqdm(range(1, train_steps + 1), desc="Training", unit="step")
        for _ in pbar:
            batch  = gc_dataset.sample(cfg.batch_size)
            losses = agent.update(batch)
            pbar.set_postfix(
                C=f"{losses['c_loss']:.4f}",
                A=f"{losses['a_loss']:.4f}",
            )

    # ── 4. Evaluation ──────────────────────────────────────────────────────────
    print(f"\nEvaluating for {cfg.eval_episodes} episodes per task …")
    eval_stats = evaluate(env, agent, cfg.eval_episodes)
    print(f"  Success rate : {eval_stats['success_rate']:.2%}")
    print(f"  Mean return  : {eval_stats['mean_return']:.3f}")

    env.close()

    # ── 5. Save results ────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    out_path  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f"results_{timestamp}.txt")
    with open(out_path, "w") as f:
        f.write(f"--env {args.env} --task {args.task} --dsize {args.dsize} "
                f"--train-step {train_steps}\n")
        f.write(f"alpha={cfg.alpha}  discount={cfg.discount}  "
                f"actor_p_traj={cfg.actor_p_trajgoal}  "
                f"actor_p_rand={cfg.actor_p_randomgoal}\n")
        f.write(f"success_rate: {eval_stats['success_rate'] * 100:.2f}%\n")
        for i, sr in enumerate(eval_stats['per_task_success'], start=1):
            f.write(f"  task{i}: {sr * 100:.2f}%\n")
    print(f"\nResults saved to {out_path}")
    print("Test passed.")


if __name__ == "__main__":
    main()
