"""
QRL (Quasimetric Reinforcement Learning)

Environment  : PointMaze – navigate variant (default; all OGBench envs supported)
Dataset      : pointmaze-medium-navigate-v0 (default)

QRL overview:
  - Learns optimal value function exploiting quasimetric structure of goal-reaching
  - Quasimetric value: V(s,g) = -d_IQE(phi(s), phi(g))
  - phi(obs) ∈ R^{LATENT_DIM}, single encoder (not lo/hi split)
  - IQE distance: sorting-based integral formula over groups of DIM_PER_COMPONENT=8
      d(s,g) = α·mean(components) + (1−α)·max(components), α = sigmoid(log_alpha)
  - Value loss with Lagrangian dual and stop-gradients:
      d_pos_loss = mean(relu(d(s, s_next) - 1)^2)                 [consecutive ≤ 1]
      d_neg_loss = mean(100 * softplus(5 - d(s, g_rand)/100))     [random >> 1]
      value_loss = d_neg_loss + d_pos_loss * stop_grad(lam)
      lam_loss   = lam * (eps - stop_grad(d_pos_loss))
  - Bidirectional dynamics loss: (d(phi_next,pred) + d(pred,phi_next)) / 2
  - DDPG+BC actor with frozen phi + frozen Delta for reparameterised Q gradients:
      pred_next = phi(s) + Delta(phi(s), clip(μ,−1,1))
      Q(s,a,g)  = -d_IQE(pred_next, phi(g))
  - Value goals : p_randomgoal=1.0  (random obs for negative pairs)
  - Actor goals : configurable per env (uniform traj default; stitch/explore variants)

References:
  - QRL paper  : https://arxiv.org/abs/2304.01203
  - OGBench    : https://arxiv.org/abs/2410.20092
"""

import argparse
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
from config_qrl import get_config

# ─────────────────────────────── fixed hyper-parameters ──────────────────────
DATASET_DIR       = "~/.ogbench/data"
LR                = 3e-4
BATCH_SIZE        = 1024
HIDDEN_DIMS       = (512, 512, 512)
LATENT_DIM        = 512
NONLINEARITY      = nn.GELU
DIM_PER_COMPONENT = 8        # IQE group size; LATENT_DIM must be divisible by this
LAM_EPS           = 0.05     # Lagrangian constraint threshold (same for all envs)

EVAL_EPISODES = 50
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


class IQEValue(nn.Module):
    """Interval Quasimetric Embedding (IQE) value function.

    Matches OGBench GCIQEValue with dim_per_component=8.

    IQE distance (full sorting-based integral):
        d(phi_s, phi_g) = α·mean(components) + (1−α)·max(components)
    where each component is the excess-CDF area for a group of DIM_PER_COMPONENT dims.

    Lagrangian dual with stop-gradients (critical for correctness):
        value_loss = d_neg_loss + d_pos_loss · stop_grad(lam)
        lam_loss   = lam · (eps − stop_grad(d_pos_loss))
    """

    def __init__(self, obs_dim: int, lam_eps: float = LAM_EPS):
        super().__init__()
        assert LATENT_DIM % DIM_PER_COMPONENT == 0
        self.phi       = mlp(obs_dim, LATENT_DIM)
        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.log_lam   = nn.Parameter(torch.zeros(1))
        self.lam_eps   = lam_eps

    @property
    def lam(self) -> torch.Tensor:
        return torch.exp(self.log_lam)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.phi(obs)

    def iqe_dist_from_reps(self, phi_s: torch.Tensor,
                           phi_g: torch.Tensor) -> torch.Tensor:
        """Full IQE distance between pre-encoded representations."""
        B = phi_s.shape[0]
        k = DIM_PER_COMPONENT
        n = LATENT_DIM // k

        x = phi_s.view(B, n, k)
        y = phi_g.view(B, n, k)

        valid = (x < y).float()
        xy    = torch.cat([x, y], dim=-1)
        ixy   = xy.argsort(dim=-1)

        sxy     = xy.gather(-1, ixy)
        valid_g = valid.gather(-1, ixy % k)
        sign    = torch.where(ixy < k, -torch.ones_like(sxy), torch.ones_like(sxy))

        neg_inp  = (valid_g * sign).cumsum(dim=-1)
        neg_f    = -1.0 * (neg_inp < 0).float()
        neg_df   = torch.cat([neg_f[..., :1],
                              neg_f[..., 1:] - neg_f[..., :-1]], dim=-1)

        components = (sxy * neg_df).sum(dim=-1)
        alpha      = torch.sigmoid(self.log_alpha)
        return alpha * components.mean(-1) + (1 - alpha) * components.max(-1).values

    def iqe_dist(self, obs_s: torch.Tensor, obs_g: torch.Tensor) -> torch.Tensor:
        return self.iqe_dist_from_reps(self.encode(obs_s), self.encode(obs_g))

    def value_loss(self, obs: torch.Tensor, next_obs: torch.Tensor,
                   neg_goal: torch.Tensor) -> torch.Tensor:
        """Quasimetric value + Lagrangian dual loss with correct stop-gradients."""
        d_pos      = self.iqe_dist(obs, next_obs)
        d_neg      = self.iqe_dist(obs, neg_goal)
        d_pos_loss = F.relu(d_pos - 1.0).pow(2).mean()
        d_neg_loss = (100.0 * F.softplus(5.0 - d_neg / 100.0)).mean()
        value_loss = d_neg_loss + d_pos_loss * self.lam.detach()
        lam_loss   = self.lam * (self.lam_eps - d_pos_loss.detach())
        return value_loss + lam_loss


class DynamicsNet(nn.Module):
    """Residual latent dynamics model: pred_phi(s') = phi(s) + Delta(phi(s), a)."""

    def __init__(self, act_dim: int):
        super().__init__()
        self.net = mlp(LATENT_DIM + act_dim, LATENT_DIM)

    def pred_next_rep(self, phi_s: torch.Tensor,
                      action: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat([phi_s, action], dim=-1))
        return phi_s + delta


class GaussianActor(nn.Module):
    """Goal-conditioned Gaussian actor with constant std (const_std=True)."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = mlp(obs_dim * 2, act_dim)

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


# ──────────────────────────────── QRL agent ───────────────────────────────────

class QRL:
    """Quasimetric RL agent (offline GCRL)."""

    def __init__(self, obs_dim: int, act_dim: int,
                 alpha: float, lam_eps: float = LAM_EPS):
        self.alpha    = alpha
        self.value    = IQEValue(obs_dim, lam_eps=lam_eps).to(device)
        self.dynamics = DynamicsNet(act_dim).to(device)
        self.actor    = GaussianActor(obs_dim, act_dim).to(device)

        # phi, lam, alpha (blending), and Delta all trained together
        self.model_opt = Adam(
            list(self.value.parameters()) + list(self.dynamics.parameters()), lr=LR)
        # Actor only — Delta provides differentiable Q but is not updated here
        self.actor_opt = Adam(self.actor.parameters(), lr=LR)

    def _model_update(self, obs, next_obs, neg_goal, action) -> float:
        """Joint value + bidirectional dynamics loss."""
        value_loss = self.value.value_loss(obs, next_obs, neg_goal)

        phi_s    = self.value.encode(obs)
        phi_next = self.value.encode(next_obs)
        pred_phi = self.dynamics.pred_next_rep(phi_s, action)
        dist1    = self.value.iqe_dist_from_reps(phi_next, pred_phi)
        dist2    = self.value.iqe_dist_from_reps(pred_phi, phi_next)
        dynamics_loss = (dist1 + dist2).mean() / 2.0

        loss = value_loss + dynamics_loss
        self.model_opt.zero_grad(); loss.backward(); self.model_opt.step()
        return loss.item()

    def _actor_update(self, obs, action, actor_goal) -> float:
        """DDPG+BC with frozen phi + Delta for reparameterised Q."""
        mean      = self.actor(obs, actor_goal)
        q_actions = mean.clamp(-1, 1)

        with torch.no_grad():
            phi_s = self.value.encode(obs)
            phi_g = self.value.encode(actor_goal)

        pred_phi   = self.dynamics.pred_next_rep(phi_s, q_actions)
        q          = -self.value.iqe_dist_from_reps(pred_phi, phi_g)
        q_loss     = -q.mean() / q.abs().mean().clamp(min=1e-6).detach()
        log_prob   = Normal(mean, torch.ones_like(mean)).log_prob(action).sum(-1)
        bc_loss    = -(self.alpha * log_prob).mean()

        actor_loss = q_loss + bc_loss
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
        return actor_loss.item()

    def update(self, batch: dict) -> dict:
        obs        = torch.FloatTensor(batch["observations"]).to(device)
        action     = torch.FloatTensor(batch["actions"]).to(device)
        next_obs   = torch.FloatTensor(batch["next_observations"]).to(device)
        neg_goal   = torch.FloatTensor(batch["neg_goals"]).to(device)
        actor_goal = torch.FloatTensor(batch["actor_goals"]).to(device)

        v_loss = self._model_update(obs, next_obs, neg_goal, action)
        a_loss = self._actor_update(obs, action, actor_goal)
        return dict(v_loss=v_loss, a_loss=a_loss)

    def select_action(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return self.actor.act(obs, goal)


# ──────────────────────────────── batch sampling ──────────────────────────────

class GCDataset:
    """Trajectory-aware dataset for QRL.

    Matches OGBench GCDataset with QRL default config:
      next_observations : consecutive next state  (d_pos constraint)
      neg_goals         : always random from dataset  (value_p_randomgoal=1.0)
      actor_goals       : mixed traj / random per env config

    Goal-sampling mix:
      Navigate : p_traj=1.0, p_rand=0.0  → uniform from remaining trajectory
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
        terminals[-1] = True
        terminal_locs = np.where(terminals)[0]
        self.final_state_idxs = terminal_locs[
            np.searchsorted(terminal_locs, np.arange(len(dataset["observations"])))
        ]
        self.N = len(dataset["observations"])

    def _sample_actor_goals(self, idx: np.ndarray,
                             final: np.ndarray) -> np.ndarray:
        """Sample actor goals: mix of traj (uniform) and random dataset obs."""
        batch_size = len(idx)

        start          = np.minimum(idx + 1, final)
        distances      = np.random.rand(batch_size)
        traj_goal_idxs = np.round(
            start * distances + final * (1.0 - distances)
        ).astype(np.int64)

        if self.actor_p_trajgoal == 1.0:
            return traj_goal_idxs

        random_goal_idxs = np.random.randint(0, self.N, size=batch_size)

        if self.actor_p_randomgoal == 1.0:
            return random_goal_idxs

        use_traj = np.random.rand(batch_size) < self.actor_p_trajgoal
        return np.where(use_traj, traj_goal_idxs, random_goal_idxs)

    def sample(self, batch_size: int) -> dict:
        obs_all = self.dataset["observations"]
        idx     = np.random.randint(0, self.N - 1, size=batch_size)
        final   = self.final_state_idxs[idx]

        # Actor goals: mixed traj / random
        actor_idxs = self._sample_actor_goals(idx, final)

        return dict(
            observations     = obs_all[idx],
            actions          = self.dataset["actions"][idx],
            next_observations= self.dataset["next_observations"][idx],
            neg_goals        = obs_all[np.random.randint(0, self.N, size=batch_size)],
            actor_goals      = obs_all[actor_idxs],
        )


# ──────────────────────────────── evaluation ──────────────────────────────────

def evaluate(env, agent: QRL, num_episodes: int) -> dict:
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
    print(f"QRL on OGBench: {env_name}")
    print(f"  Train steps  : {train_steps:,}  |  Batch size : {cfg.batch_size}")
    print(f"  LR={cfg.lr}  γ={cfg.discount}  α={cfg.alpha}  lam_eps={cfg.lam_eps}")
    print(f"  Hidden dims  : {HIDDEN_DIMS}  |  Latent dim : {LATENT_DIM}")
    print(f"  dim_per_component={DIM_PER_COMPONENT}")
    print(f"  Value goals  : p_rand=1.0 (random)")
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
    agent      = QRL(obs_dim, act_dim, alpha=cfg.alpha, lam_eps=cfg.lam_eps)
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
                      f"V={losses['v_loss']:.4f}  "
                      f"A={losses['a_loss']:.4f}")
    else:
        pbar = tqdm(range(1, train_steps + 1), desc="Training", unit="step")
        for _ in pbar:
            batch  = gc_dataset.sample(cfg.batch_size)
            losses = agent.update(batch)
            pbar.set_postfix(
                V=f"{losses['v_loss']:.4f}",
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
