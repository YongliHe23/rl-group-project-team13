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
import multiprocessing as mp
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
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

EVAL_TEMPERATURE    = 0.3   # softmax temperature for discrete action sampling (powderworld)
EVAL_EPISODES       = 50
SEEDS               = [42, 0, 1, 2]   # four independent runs
EVAL_INTERVAL       = 100_000         # evaluate every 100 k steps
DEFAULT_TRAIN_STEPS = 500_000         # default per-seed budget (overrides per-env cfg)

# Module-level seed is NOT set here; each seed is applied inside the training loop.

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


class ImpalaSmall(nn.Module):
    """IMPALA-small visual encoder: (B, H, W, C) uint8 → (B, 512) float32."""
    ENC_DIM = 512

    class _ResBlock(nn.Module):
        def __init__(self, ch: int):
            super().__init__()
            self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
            self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            r = x
            x = F.relu(x); x = self.c1(x)
            x = F.relu(x); x = self.c2(x)
            return x + r

    class _ResStack(nn.Module):
        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.conv  = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.pool  = nn.MaxPool2d(3, stride=2, padding=1)
            self.block = ImpalaSmall._ResBlock(out_ch)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(self.pool(self.conv(x)))

    def __init__(self, in_channels: int = 6):
        super().__init__()
        self.stacks = nn.Sequential(
            self._ResStack(in_channels, 16),
            self._ResStack(16, 32),
            self._ResStack(32, 32),
        )
        self.head = nn.Sequential(nn.Linear(512, 512), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        x = x / 255.0
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.stacks(x)).flatten(1)
        return self.head(x)


class CategoricalActor(nn.Module):
    """Discrete AWR actor for CRL on powderworld.

    Replaces DDPG+BC (which requires continuous actions).
    AWR loss: adv from V(s',g) - V(s,g) via contrastive Q; -(exp_a * log_prob).mean()
    """

    def __init__(self, obs_dim: int, act_n: int):
        super().__init__()
        self.net   = mlp(obs_dim * 2, act_n)   # input: [s_enc; g_enc]
        self.act_n = act_n

    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, goal], dim=-1))    # logits: (B, act_n)

    def log_prob(self, obs: torch.Tensor, goal: torch.Tensor,
                 action: torch.Tensor) -> torch.Tensor:
        """action: (B,) LongTensor."""
        logits = self.forward(obs, goal)
        return Categorical(logits=logits).log_prob(action)

    def act(self, obs_np: np.ndarray, goal_np: np.ndarray,
            temperature: float = EVAL_TEMPERATURE) -> int:
        obs  = torch.FloatTensor(obs_np).unsqueeze(0).to(device)
        goal = torch.FloatTensor(goal_np).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = self.forward(obs, goal) / max(temperature, 1e-6)
            return int(Categorical(logits=logits).sample().item())


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
    """Contrastive RL agent (offline GCRL).

    For visual/discrete envs (powderworld):
      - ImpalaSmall encoder maps (B,H,W,6) → (B,512); trained via critic loss only
      - Actions are integer indices → one-hot encoded for the contrastive critic
      - Actor switches from DDPG+BC to AWR with Categorical distribution
        (DDPG+BC requires continuous actions; AWR generalises to discrete)
    """

    def __init__(self, obs_dim: int, act_dim: int, alpha: float,
                 is_visual: bool = False, is_discrete: bool = False):
        self.alpha       = alpha
        self.is_visual   = is_visual
        self.is_discrete = is_discrete

        if is_visual:
            self.img_encoder = ImpalaSmall(in_channels=6).to(device)
            obs_dim = ImpalaSmall.ENC_DIM

        # Critic: phi(s, a) where a is one-hot for discrete or continuous vector
        critic_act_dim = act_dim   # act_dim = act_n for discrete (one-hot size)
        self.critic = ContrastiveCritic(obs_dim, critic_act_dim).to(device)

        if is_discrete:
            self.actor = CategoricalActor(obs_dim, act_dim).to(device)
        else:
            self.actor = GaussianActor(obs_dim, act_dim).to(device)

        critic_params = list(self.critic.parameters())
        if is_visual:
            critic_params += list(self.img_encoder.parameters())
        self.critic_opt = Adam(critic_params, lr=LR)
        self.actor_opt  = Adam(self.actor.parameters(), lr=LR)

    def _one_hot(self, action: torch.LongTensor) -> torch.Tensor:
        return F.one_hot(action, num_classes=self.actor.act_n).float()

    def _critic_update(self, obs, action, future_obs) -> float:
        loss = self.critic.critic_loss(obs, action, future_obs)
        self.critic_opt.zero_grad(); loss.backward(); self.critic_opt.step()
        return loss.item()

    def _actor_update_continuous(self, obs, action, actor_goal) -> float:
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

    def _actor_update_discrete(self, obs, action_int: torch.LongTensor,
                                actor_goal) -> float:
        """AWR actor for discrete actions.

        Advantage: Q(s,a_data,g) from contrastive critic.
        AWR loss: -(exp(alpha * Q) * log_prob(a_data)).mean()
        Q is used in place of advantage (no baseline V; empirically stable).
        """
        with torch.no_grad():
            a_oh = self._one_hot(action_int)
            q    = self.critic.q(obs, a_oh, actor_goal)
            exp_a = torch.clamp(torch.exp(self.alpha * q), max=100.0)

        log_prob   = self.actor.log_prob(obs, actor_goal, action_int)
        actor_loss = -(exp_a * log_prob).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
        return actor_loss.item()

    def update(self, batch: dict) -> dict:
        if self.is_visual:
            raw_obs  = torch.ByteTensor(batch["observations"]).to(device)
            raw_fut  = torch.ByteTensor(batch["future_obs"]).to(device)
            raw_goal = torch.ByteTensor(batch["actor_goals"]).to(device)
            # Encoder trains via critic loss
            obs        = self.img_encoder(raw_obs.float())
            future_obs = self.img_encoder(raw_fut.float())
            actor_goal = self.img_encoder(raw_goal.float()).detach()
            obs_actor  = obs.detach()
        else:
            obs        = torch.FloatTensor(batch["observations"]).to(device)
            future_obs = torch.FloatTensor(batch["future_obs"]).to(device)
            actor_goal = torch.FloatTensor(batch["actor_goals"]).to(device)
            obs_actor  = obs

        if self.is_discrete:
            action_int = torch.LongTensor(
                batch["actions"].astype(np.int64)).to(device)
            action_oh  = self._one_hot(action_int)
            c_loss = self._critic_update(obs, action_oh, future_obs)
            a_loss = self._actor_update_discrete(obs_actor, action_int, actor_goal)
        else:
            action = torch.FloatTensor(batch["actions"]).to(device)
            c_loss = self._critic_update(obs, action, future_obs)
            a_loss = self._actor_update_continuous(obs_actor, action, actor_goal)

        return dict(c_loss=c_loss, a_loss=a_loss)

    def select_action(self, obs: np.ndarray, goal: np.ndarray):
        if self.is_visual:
            with torch.no_grad():
                obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(device)
                goal_t = torch.FloatTensor(goal).unsqueeze(0).to(device)
                obs_enc  = self.img_encoder(obs_t)
                goal_enc = self.img_encoder(goal_t)
            obs_enc_np  = obs_enc.squeeze(0).cpu().numpy()
            goal_enc_np = goal_enc.squeeze(0).cpu().numpy()
            return self.actor.act(obs_enc_np, goal_enc_np)
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
            goal = info['goal']
            if not agent.is_visual:
                goal = goal.astype(np.float32)
            ep_return = 0.0

            done = False
            while not done:
                if agent.is_visual:
                    action = agent.select_action(obs, goal)
                else:
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


# ──────────────────────────── parallel-seed worker ───────────────────────────

def _crl_seed_worker(kwargs: dict):
    """Run one seed of CRL training; designed to be called in a subprocess.

    Device note: HPC V100s typically run in exclusive-process compute mode, which
    allows only one CUDA context at a time.  Parallel workers therefore use CPU
    for gradient steps; evaluation (MuJoCo) already runs on CPU and is the
    dominant cost (~90 % of wall time), so the overall speedup is preserved.
    """
    seed          = kwargs['seed']
    env_name      = kwargs['env_name']
    cfg           = kwargs['cfg']
    train_steps   = kwargs['train_steps']
    eval_intv     = kwargs['eval_intv']
    is_visual     = kwargs['is_visual']
    is_discrete   = kwargs['is_discrete']
    obs_dim       = kwargs['obs_dim']
    act_dim       = kwargs['act_dim']
    dataset_dir   = kwargs['dataset_dir']
    eval_episodes = kwargs['eval_episodes']

    # Override the module-level device; spawned subprocess re-imports the module
    # so this is safe and does not affect the parent process or other workers.
    global device
    device = torch.device(kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"  [Seed {seed}] using device: {device}", flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, train_dataset, _ = ogbench.make_env_and_datasets(
        env_name, dataset_dir=dataset_dir, compact_dataset=False)

    agent      = CRL(obs_dim, act_dim, alpha=cfg.alpha,
                     is_visual=is_visual, is_discrete=is_discrete)
    gc_dataset = GCDataset(
        train_dataset,
        discount           = cfg.discount,
        actor_p_trajgoal   = cfg.actor_p_trajgoal,
        actor_p_randomgoal = cfg.actor_p_randomgoal,
    )

    seed_evals = []
    for step in range(1, train_steps + 1):
        batch  = gc_dataset.sample(cfg.batch_size)
        losses = agent.update(batch)
        if step % 10_000 == 0:
            print(f"  [Seed {seed}] step {step:>8,}/{train_steps:,} | "
                  f"C={losses['c_loss']:.4f}  "
                  f"A={losses['a_loss']:.4f}", flush=True)
        if step % eval_intv == 0:
            es = evaluate(env, agent, eval_episodes)
            seed_evals.append((step, es['per_task_success'], es['success_rate']))
            task_str = "  ".join(
                f"T{i+1}:{sr*100:.1f}%"
                for i, sr in enumerate(es['per_task_success'])
            )
            print(f"\n  [Seed {seed}] step {step:,} | "
                  f"{task_str} | mean: {es['success_rate']*100:.2f}%\n", flush=True)

    env.close()
    return (seed, seed_evals)


# ────────────────────────────────────── main ──────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",           default="pointmaze", help="Environment base name")
    parser.add_argument("--task",          default="navigate",  help="Task name")
    parser.add_argument("--dsize",         default="medium",    help="Dataset size/difficulty")
    parser.add_argument("--train-step",    default=None, type=int,
                        help=f"Gradient steps per seed (default: {DEFAULT_TRAIN_STEPS:,})")
    parser.add_argument("--eval-interval", default=EVAL_INTERVAL, type=int,
                        help="Evaluate every N steps (default: %(default)s)")
    parser.add_argument("--slurm-tqdm",    default=True,
                        action=argparse.BooleanOptionalAction,
                        help="Print every 10 k steps instead of tqdm (Slurm mode)")
    parser.add_argument("--single-seed",   type=int, nargs="?", const=42, default=None,
                        metavar="SEED",
                        help="Run with a single seed (default 42 if flag given without value)")
    parser.add_argument("--visual-enabled", action="store_true",
                        help="Enable visual encoder + discrete actor (required for "
                             "powderworld-* environments)")
    parser.add_argument("--parallel-seeds", action="store_true",
                        help="Run all seeds concurrently in separate processes "
                             "(each reloads the dataset; ~N-seed wall-clock speedup). "
                             "Offline RL has no env interaction during training, so "
                             "seeds are fully independent and safe to parallelise.")
    args = parser.parse_args()

    env_name    = f"{args.env}-{args.dsize}-{args.task}-v0"
    cfg         = get_config(env_name)
    train_steps = args.train_step if args.train_step is not None else DEFAULT_TRAIN_STEPS
    eval_intv   = args.eval_interval
    seeds       = [args.single_seed] if args.single_seed is not None else SEEDS
    is_visual   = args.visual_enabled
    is_discrete = args.visual_enabled

    print(f"\n{'='*60}")
    print(f"CRL on OGBench: {env_name}")
    print(f"  Seeds          : {seeds}"
          f"{'  [parallel]' if args.parallel_seeds and len(seeds) > 1 else ''}")
    print(f"  Train steps    : {train_steps:,} per seed  |  Eval every: {eval_intv:,}")
    print(f"  Batch size     : {cfg.batch_size}  |  LR={cfg.lr}  "
          f"gamma={cfg.discount}  alpha={cfg.alpha}")
    print(f"  Hidden dims    : {HIDDEN_DIMS}  |  Latent dim: {LATENT_DIM}")
    print(f"  Value goals    : geom(1-gamma) traj  (fixed)")
    print(f"  Actor goals    : p_traj={cfg.actor_p_trajgoal}  "
          f"p_rand={cfg.actor_p_randomgoal}")
    if is_visual:
        print(f"  Visual encoder : ImpalaSmall  |  Discrete AWR actor (temp={EVAL_TEMPERATURE})")
    print(f"{'='*60}\n")

    # ── 1. Load environment + dataset (shared across all seeds) ───────────────
    print("Loading environment and dataset ...")
    env, train_dataset, _ = ogbench.make_env_and_datasets(
        env_name, dataset_dir=DATASET_DIR, compact_dataset=False,
    )
    obs = train_dataset["observations"]
    if is_visual:
        obs_dim = ImpalaSmall.ENC_DIM
        act_dim = int(train_dataset["actions"].max()) + 1
    else:
        obs_dim = obs.shape[1]
        act_dim = train_dataset["actions"].shape[1]
    print(f"  Train size: {len(obs):,}  |  "
          f"obs_dim={obs_dim}  act_dim={act_dim}"
          f"{'  (visual+discrete)' if is_visual else ''}\n")

    # ── 2. Multi-seed training ─────────────────────────────────────────────────
    # all_results[seed] = list of (step, per_task_success_list, overall_rate)
    all_results: dict = {}

    # Shared worker kwargs (seed-specific 'seed' key is filled per iteration/spawn)
    _worker_base = dict(
        env_name      = env_name,
        cfg           = cfg,
        train_steps   = train_steps,
        eval_intv     = eval_intv,
        is_visual     = is_visual,
        is_discrete   = is_discrete,
        obs_dim       = obs_dim,
        act_dim       = act_dim,
        dataset_dir   = DATASET_DIR,
        eval_episodes = cfg.eval_episodes,
    )

    if args.parallel_seeds and len(seeds) > 1:
        # ── Parallel path: one subprocess per seed ────────────────────────────
        env.close()   # main process does not need its env instance
        worker_args = [{**_worker_base, 'seed': s, 'device': 'cpu'} for s in seeds]
        ctx = mp.get_context('spawn')
        print(f"Launching {len(seeds)} parallel seed workers "
              f"(spawn context, each reloads dataset, device=cpu) ...\n")
        with ctx.Pool(processes=len(seeds)) as pool:
            results = pool.map(_crl_seed_worker, worker_args)
        all_results = dict(results)

    else:
        # ── Sequential path (original behaviour) ──────────────────────────────
        for seed_idx, seed in enumerate(seeds):
            print(f"\n{'─'*60}")
            print(f"Seed {seed}  ({seed_idx + 1}/{len(seeds)})")
            print(f"{'─'*60}")

            torch.manual_seed(seed)
            np.random.seed(seed)

            agent      = CRL(obs_dim, act_dim, alpha=cfg.alpha,
                             is_visual=is_visual, is_discrete=is_discrete)
            gc_dataset = GCDataset(
                train_dataset,
                discount           = cfg.discount,
                actor_p_trajgoal   = cfg.actor_p_trajgoal,
                actor_p_randomgoal = cfg.actor_p_randomgoal,
            )

            seed_evals = []   # (step, per_task_success_list, overall_rate)

            if args.slurm_tqdm:
                for step in range(1, train_steps + 1):
                    batch  = gc_dataset.sample(cfg.batch_size)
                    losses = agent.update(batch)
                    if step % 10_000 == 0:
                        print(f"  step {step:>8,}/{train_steps:,} | "
                              f"C={losses['c_loss']:.4f}  "
                              f"A={losses['a_loss']:.4f}")
                    if step % eval_intv == 0:
                        es = evaluate(env, agent, cfg.eval_episodes)
                        seed_evals.append((step, es['per_task_success'], es['success_rate']))
                        task_str = "  ".join(
                            f"T{i+1}:{sr*100:.1f}%"
                            for i, sr in enumerate(es['per_task_success'])
                        )
                        print(f"\n  [Seed {seed}] step {step:,} | "
                              f"{task_str} | mean: {es['success_rate']*100:.2f}%\n")
            else:
                pbar = tqdm(range(1, train_steps + 1), desc=f"Seed {seed}", unit="step")
                for step in pbar:
                    batch  = gc_dataset.sample(cfg.batch_size)
                    losses = agent.update(batch)
                    pbar.set_postfix(C=f"{losses['c_loss']:.4f}",
                                     A=f"{losses['a_loss']:.4f}")
                    if step % eval_intv == 0:
                        es = evaluate(env, agent, cfg.eval_episodes)
                        seed_evals.append((step, es['per_task_success'], es['success_rate']))
                        pbar.write(f"  [Seed {seed}] step {step:,} | "
                                   f"mean: {es['success_rate']*100:.2f}%")

            all_results[seed] = seed_evals

        env.close()

    # ── 3. Summary statistics ──────────────────────────────────────────────────
    # Average over the last 3 eval checkpoints (300 k, 400 k, 500 k by default).
    seed_avg_rates:    list = []
    seed_avg_per_task: list = []

    for seed in seeds:
        evals = all_results[seed]
        last3 = evals[-3:]
        seed_avg_rates.append(float(np.mean([e[2] for e in last3])))
        n_tasks = len(last3[0][1])
        seed_avg_per_task.append(
            [float(np.mean([e[1][t] for e in last3])) for t in range(n_tasks)]
        )

    final_mean = float(np.mean(seed_avg_rates))
    final_std  = float(np.std(seed_avg_rates))

    n_tasks       = len(seed_avg_per_task[0])
    per_task_mean = [
        float(np.mean([seed_avg_per_task[i][t] for i in range(len(seeds))]))
        for t in range(n_tasks)
    ]
    per_task_std  = [
        float(np.std( [seed_avg_per_task[i][t] for i in range(len(seeds))]))
        for t in range(n_tasks)
    ]
    last_ckpts = [e[0] for e in all_results[seeds[0]][-3:]]

    # ── 4. Print report ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS -- CRL -- {env_name}")
    print(f"Avg over last 3 checkpoints: {last_ckpts}")
    print(f"{'='*60}")
    for seed, rate in zip(seeds, seed_avg_rates):
        print(f"  Seed {seed:>3}: {rate * 100:.2f}%")
    print(f"  {'─'*34}")
    print(f"  Mean +/- Std : {final_mean*100:.2f}% +/- {final_std*100:.2f}%")
    print(f"\nPer-task breakdown (mean +/- std across {len(seeds)} seeds):")
    for t, (m, s) in enumerate(zip(per_task_mean, per_task_std), start=1):
        print(f"  Task {t}: {m*100:.2f}% +/- {s*100:.2f}%")

    # ── 5. Save results ───────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    out_path  = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"results_crl_{args.env}_{args.dsize}_{args.task}_{timestamp}.txt",
    )
    with open(out_path, "w") as f:
        f.write(f"method: CRL\n")
        f.write(f"env: {env_name}\n")
        f.write(f"seeds: {seeds}\n")
        f.write(f"train_steps_per_seed: {train_steps}\n")
        f.write(f"eval_interval: {eval_intv}\n")
        f.write(f"last_3_checkpoints: {last_ckpts}\n")
        f.write(f"alpha={cfg.alpha}  discount={cfg.discount}  "
                f"actor_p_traj={cfg.actor_p_trajgoal}  "
                f"actor_p_rand={cfg.actor_p_randomgoal}\n\n")

        for seed in seeds:
            f.write(f"Seed {seed}:\n")
            for step, per_task, overall in all_results[seed]:
                task_str = "  ".join(
                    f"T{i+1}:{sr*100:.1f}%" for i, sr in enumerate(per_task)
                )
                f.write(f"  step {step:>8,}: {task_str} | mean: {overall*100:.2f}%\n")
            idx = seeds.index(seed)
            f.write(f"  --> avg (last 3 ckpts): {seed_avg_rates[idx]*100:.2f}%\n\n")

        f.write(f"Summary (avg over last 3 checkpoints, "
                f"mean +/- std across {len(seeds)} seeds):\n")
        f.write(f"  Mean: {final_mean*100:.2f}%\n")
        f.write(f"  Std : {final_std*100:.2f}%\n\n")
        f.write(f"Per-task breakdown (mean +/- std across seeds):\n")
        for t, (m, s) in enumerate(zip(per_task_mean, per_task_std), start=1):
            f.write(f"  Task {t}: {m*100:.2f}% +/- {s*100:.2f}%\n")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
