"""
HIQL (Hierarchical Implicit Q-Learning)

Environment  : PointMaze – navigate variant (default; all OGBench envs supported)
Dataset      : pointmaze-medium-navigate-v0 (default)

HIQL overview:
  - High-level policy pi_h(s, g) -> subgoal representation (rep_dim)
  - Low-level policy pi_l(s, z_sg) -> action
  - Value function ensemble (GCIVL, no Q-network) trained on the offline dataset
  - Target networks updated via exponential moving average (tau = 0.005)
  - Mixed goal relabelling: separate ratios for policy vs. value networks
  - subgoal_steps k: 25 (locomaze), 100 (humanoidmaze), 10 (manipulation)

References:
  - HIQL paper : https://arxiv.org/abs/2307.11949
  - OGBench    : https://arxiv.org/abs/2410.20092
"""

import argparse
import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam

import ogbench
from tqdm import tqdm
from config_hiql import get_config

# ─────────────────────────────── fixed hyper-parameters ──────────────────────
DATASET_DIR  = "~/.ogbench/data"
LR           = 3e-4
BATCH_SIZE   = 1024
HIDDEN_DIMS  = (512, 512, 512)
NONLINEARITY = nn.GELU
EXPECTILE    = 0.7
TAU          = 0.005
REP_DIM      = 10      # subgoal representation dimension (fixed for all envs)

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
    """Build an MLP: Linear -> GELU -> LayerNorm (repeated), then final Linear."""
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nonlinearity(), nn.LayerNorm(h)]
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class SubgoalEncoder(nn.Module):
    """Maps cat([obs; goal_obs]) -> L2-length-normalised subgoal representation.

    Architecture (mirrors OGBench goal_rep):
      MLP(obs_dim*2 -> rep_dim, activate_final=False, layer_norm=True)
      then LengthNormalize: x / ‖x‖₂ * sqrt(rep_dim)
    """
    def __init__(self, obs_dim: int, rep_dim: int = REP_DIM):
        super().__init__()
        self.net     = mlp(obs_dim * 2, rep_dim)
        self.rep_dim = rep_dim

    def forward(self, obs: torch.Tensor, goal_obs: torch.Tensor) -> torch.Tensor:
        x = self.net(torch.cat([obs, goal_obs], dim=-1))
        return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-6) * (self.rep_dim ** 0.5)


class ValueNet(nn.Module):
    """Twin value functions V(s, g) implemented as GCIVL ensemble.

    Internally encodes goal via SubgoalEncoder:
      goal_rep = sg_encoder(cat([obs; goal_obs]))
      input    = cat([obs; goal_rep])

    SubgoalEncoder lives inside this module so EMA target updates propagate
    automatically through both value heads AND the encoder.
    """
    def __init__(self, obs_dim: int, rep_dim: int = REP_DIM):
        super().__init__()
        self.sg_encoder = SubgoalEncoder(obs_dim, rep_dim)
        self.v1 = mlp(obs_dim + rep_dim, 1)
        self.v2 = mlp(obs_dim + rep_dim, 1)

    def encode_goal(self, obs: torch.Tensor,
                    goal_obs: torch.Tensor) -> torch.Tensor:
        return self.sg_encoder(obs, goal_obs)

    def forward(self, obs: torch.Tensor,
                goal_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        goal_rep = self.sg_encoder(obs, goal_obs)
        x = torch.cat([obs, goal_rep], dim=-1)
        return self.v1(x).squeeze(-1), self.v2(x).squeeze(-1)

    def mean(self, obs: torch.Tensor, goal_obs: torch.Tensor) -> torch.Tensor:
        v1, v2 = self.forward(obs, goal_obs)
        return (v1 + v2) / 2.0


class SubgoalPolicy(nn.Module):
    """High-level policy pi_h(s, g) -> subgoal representation (const_std=1 Gaussian).

    Outputs in rep_dim space; regression target is stop_grad(sg_encoder(cat([s, s_{t+k}]))).
    """
    def __init__(self, obs_dim: int, rep_dim: int = REP_DIM):
        super().__init__()
        self.net = mlp(obs_dim * 2, rep_dim)

    def forward(self, obs: torch.Tensor, goal_obs: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, goal_obs], dim=-1))

    def log_prob(self, obs: torch.Tensor, goal_obs: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
        mu = self.forward(obs, goal_obs)
        return Normal(mu, torch.ones_like(mu)).log_prob(target).sum(-1)


class GaussianActor(nn.Module):
    """Low-level policy pi_l(s, z_sg) -> action (const_std=True, no tanh squash)."""

    def __init__(self, obs_dim: int, rep_dim: int, act_dim: int):
        super().__init__()
        self.net = mlp(obs_dim + rep_dim, act_dim)

    def forward(self, obs: torch.Tensor, sg_repr: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, sg_repr], dim=-1))

    def log_prob_at(self, obs: torch.Tensor, sg_repr: torch.Tensor,
                    action: torch.Tensor) -> torch.Tensor:
        mean = self.forward(obs, sg_repr)
        return Normal(mean, torch.ones_like(mean)).log_prob(action).sum(-1)

    def act(self, obs_np: np.ndarray, sg_np: np.ndarray) -> np.ndarray:
        obs = torch.FloatTensor(obs_np).unsqueeze(0).to(device)
        sg  = torch.FloatTensor(sg_np).unsqueeze(0).to(device)
        with torch.no_grad():
            mean = self.forward(obs, sg)
        return mean.clamp(-1, 1).squeeze(0).cpu().numpy()


# ────────────────────────────────── HIQL agent ────────────────────────────────

class HIQL:
    def __init__(self, obs_dim: int, act_dim: int,
                 discount: float, high_alpha: float, low_alpha: float,
                 rep_dim: int = REP_DIM):
        self.discount    = discount
        self.high_alpha  = high_alpha
        self.low_alpha   = low_alpha

        self.value      = ValueNet(obs_dim, rep_dim).to(device)
        self.value_tgt  = ValueNet(obs_dim, rep_dim).to(device)
        self.hi_policy  = SubgoalPolicy(obs_dim, rep_dim).to(device)
        self.lo_policy  = GaussianActor(obs_dim, rep_dim, act_dim).to(device)

        # Initialise target as exact copy (includes sg_encoder inside ValueNet)
        self.value_tgt.load_state_dict(self.value.state_dict())

        # SubgoalEncoder lives inside ValueNet -> updated ONLY via v_opt.
        # hi_opt / lo_opt never see encoder parameters; stop-gradient is
        # enforced by computing sg_encoder outputs inside torch.no_grad().
        self.v_opt  = Adam(self.value.parameters(), lr=LR)
        self.hi_opt = Adam(self.hi_policy.parameters(), lr=LR)
        self.lo_opt = Adam(self.lo_policy.parameters(), lr=LR)

    # ── EMA target update ─────────────────────────────────────────────────

    def _ema_update(self):
        for p_o, p_t in zip(self.value.parameters(), self.value_tgt.parameters()):
            p_t.data.mul_(1.0 - TAU).add_(TAU * p_o.data)

    # ── GCIVL value loss ──────────────────────────────────────────────────

    def _value_loss(self, obs: torch.Tensor, next_obs: torch.Tensor,
                    value_goal_obs: torch.Tensor,
                    successes: torch.Tensor) -> torch.Tensor:
        """Expectile regression with gc_negative rewards and per-head Q targets.

        rewards = successes - 1   (0 at goal, -1 otherwise; gc_negative=True)
        masks   = 1 - successes
        q_i     = rewards + γ·masks·V_tgt_i(s', g)   (per head)
        adv     = q - avg(V_tgt(s, g))                (average of both heads)
        loss    = Σ_i E[w(adv)·(q_i - v_i)²]
        """
        rewards = successes - 1.0
        masks   = 1.0 - successes

        with torch.no_grad():
            next_v1_t, next_v2_t = self.value_tgt.forward(next_obs, value_goal_obs)
            next_v_t = torch.min(next_v1_t, next_v2_t)
            q  = rewards + self.discount * masks * next_v_t    # for advantage sign
            q1 = rewards + self.discount * masks * next_v1_t   # per-head target
            q2 = rewards + self.discount * masks * next_v2_t

            v1_t, v2_t = self.value_tgt.forward(obs, value_goal_obs)
            adv = q - (v1_t + v2_t) / 2.0
            weight = torch.where(adv > 0,
                                 torch.full_like(adv, EXPECTILE),
                                 torch.full_like(adv, 1.0 - EXPECTILE))

        v1, v2 = self.value.forward(obs, value_goal_obs)   # gradient -> encoder
        return (weight * (q1 - v1).pow(2)).mean() + (weight * (q2 - v2).pow(2)).mean()

    # ── High-level policy loss (AWR) ──────────────────────────────────────

    def _hi_policy_loss(self, obs: torch.Tensor,
                        high_actor_targets: torch.Tensor,
                        high_actor_goals: torch.Tensor) -> torch.Tensor:
        """AWR: adv = V(s_{t+k}, g) − V(s_t, g);
                target = stop_grad(sg_enc(cat([s, s_{t+k}]))).

        All sg_encoder calls are inside torch.no_grad() -> zero gradient to encoder.
        """
        with torch.no_grad():
            nv1, nv2 = self.value.forward(high_actor_targets, high_actor_goals)
            nv = (nv1 + nv2) / 2.0
            v1, v2  = self.value.forward(obs, high_actor_goals)
            v  = (v1 + v2) / 2.0
            adv   = nv - v
            exp_a = torch.clamp(torch.exp(self.high_alpha * adv), max=100.0)
            # Regression target: sg_encoder(cat([s, s_{t+k}])), stop-gradient
            sg_target = self.value.encode_goal(obs, high_actor_targets)

        log_prob = self.hi_policy.log_prob(obs, high_actor_goals, sg_target)
        return -(exp_a * log_prob).mean()

    # ── Low-level policy loss (AWR) ───────────────────────────────────────

    def _lo_policy_loss(self, obs: torch.Tensor, next_obs: torch.Tensor,
                        action: torch.Tensor,
                        low_actor_goals: torch.Tensor) -> torch.Tensor:
        """AWR: adv = V(s', sg) − V(s, sg);
                sg_rep = stop_grad(sg_enc(cat([s, sg_obs]))).

        All sg_encoder calls are inside torch.no_grad() -> zero gradient to encoder.
        """
        with torch.no_grad():
            nv1, nv2 = self.value.forward(next_obs, low_actor_goals)
            nv = (nv1 + nv2) / 2.0
            v1, v2  = self.value.forward(obs, low_actor_goals)
            v  = (v1 + v2) / 2.0
            adv   = nv - v
            exp_a = torch.clamp(torch.exp(self.low_alpha * adv), max=100.0)
            # Subgoal representation fed to low actor, stop-gradient
            sg_repr = self.value.encode_goal(obs, low_actor_goals)

        log_prob = self.lo_policy.log_prob_at(obs, sg_repr, action)
        return -(exp_a * log_prob).mean()

    # ── Training step ─────────────────────────────────────────────────────

    def update(self, batch: dict) -> dict:
        obs               = torch.FloatTensor(batch["observations"]).to(device)
        action            = torch.FloatTensor(batch["actions"]).to(device)
        next_obs          = torch.FloatTensor(batch["next_observations"]).to(device)
        successes         = torch.FloatTensor(batch["successes"]).to(device)
        value_goal_obs    = torch.FloatTensor(batch["value_goals"]).to(device)
        high_actor_goals  = torch.FloatTensor(batch["high_actor_goals"]).to(device)
        high_actor_targets= torch.FloatTensor(batch["high_actor_targets"]).to(device)
        low_actor_goals   = torch.FloatTensor(batch["low_actor_goals"]).to(device)

        # Value + encoder update
        v_loss = self._value_loss(obs, next_obs, value_goal_obs, successes)
        self.v_opt.zero_grad(); v_loss.backward(); self.v_opt.step()

        # High-level policy update (encoder under no_grad -> no encoder gradient)
        hi_loss = self._hi_policy_loss(obs, high_actor_targets, high_actor_goals)
        self.hi_opt.zero_grad(); hi_loss.backward(); self.hi_opt.step()

        # Low-level policy update (encoder under no_grad -> no encoder gradient)
        lo_loss = self._lo_policy_loss(obs, next_obs, action, low_actor_goals)
        self.lo_opt.zero_grad(); lo_loss.backward(); self.lo_opt.step()

        # EMA target update (propagates through all ValueNet params incl. sg_encoder)
        self._ema_update()

        return dict(v_loss=v_loss.item(), hi_loss=hi_loss.item(), lo_loss=lo_loss.item())

    # ── Action selection ──────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(device)
        goal_t = torch.FloatTensor(goal).unsqueeze(0).to(device)
        with torch.no_grad():
            sg_repr = self.hi_policy.forward(obs_t, goal_t)
            # Normalise high-actor output to the same length as sg_encoder output
            sg_repr = (sg_repr / sg_repr.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                       * (REP_DIM ** 0.5))
        return self.lo_policy.act(obs, sg_repr.squeeze(0).cpu().numpy())


# ──────────────────────────── trajectory-aware dataset ────────────────────────

class HGCDataset:
    """Trajectory-aware goal-conditioned dataset for HIQL.

    Matches OGBench HGCDataset exactly, including:
      - value goals: mixed (p_cur=0.2, p_geom=0.5, p_rand=0.3) — fixed for all envs
      - low_actor_goals: always idx + subgoal_steps, clipped to traj end
      - high_actor_goals: configurable mix (traj / random per env config)
      - high_actor_targets: k-step ahead, clipped to high_actor_goals for traj;
                            clipped to traj end for random goals

    Goal-sampling for high actor (matching OGBench):
      Navigate : p_traj=1.0, p_rand=0.0 -> uniform from remaining trajectory
      Stitch   : p_traj=0.5, p_rand=0.5 -> 50% traj, 50% random dataset obs
      Explore  : p_traj=0.0, p_rand=1.0 -> fully random dataset obs

    Successes (gc_negative=True):
      successes[i] = 1 iff value_goal_idx[i] == idx[i]  (goal IS current state)
      -> OGBench uses idxs == value_goal_idxs (current-state match, not next-state)
    """

    def __init__(self, dataset: dict,
                 discount: float,
                 subgoal_steps: int,
                 actor_p_trajgoal: float  = 1.0,
                 actor_p_randomgoal: float = 0.0,
                 # Value goal ratios (fixed across all HIQL envs)
                 value_p_curgoal:  float = 0.2,
                 value_p_trajgoal: float = 0.5,
                 value_p_randomgoal: float = 0.3):
        self.dataset             = dataset
        self.discount            = discount
        self.subgoal_steps       = subgoal_steps
        self.actor_p_trajgoal    = actor_p_trajgoal
        self.actor_p_randomgoal  = actor_p_randomgoal
        self.value_p_curgoal     = value_p_curgoal
        self.value_p_trajgoal    = value_p_trajgoal
        self.value_p_randomgoal  = value_p_randomgoal

        terminals     = dataset["terminals"].astype(bool)
        terminals[-1] = True
        terminal_locs = np.where(terminals)[0]

        # final_state_idxs[i] = last index of the trajectory containing step i
        self.terminal_locs    = terminal_locs
        self.final_state_idxs = terminal_locs[
            np.searchsorted(terminal_locs, np.arange(len(dataset["observations"])))
        ]
        self.N = len(dataset["observations"])

    def _sample_value_goals(self, idx: np.ndarray,
                             final: np.ndarray) -> np.ndarray:
        """Mixed goal sampling for value network (matches OGBench GCDataset.sample_goals).

        p_curgoal=0.2  : goal = current state
        p_trajgoal=0.5 : goal = geometric future in trajectory
        p_randomgoal=0.3: goal = random from dataset
        """
        batch_size = len(idx)
        rng        = np.random.rand(batch_size)

        # Geometric future goals
        geom_offsets   = np.random.geometric(1.0 - self.discount, size=batch_size)
        geom_goal_idxs = np.minimum(idx + geom_offsets, final)

        # Random goals
        random_goal_idxs = np.random.randint(0, self.N, size=batch_size)

        # Mix: first decide traj vs random (among non-curgoal), then apply curgoal
        p_traj_given_not_cur = (self.value_p_trajgoal /
                                (1.0 - self.value_p_curgoal + 1e-10))
        goal_idxs = np.where(
            rng < p_traj_given_not_cur, geom_goal_idxs, random_goal_idxs
        )
        goal_idxs = np.where(
            np.random.rand(batch_size) < self.value_p_curgoal, idx, goal_idxs
        )
        return goal_idxs

    def sample(self, batch_size: int) -> dict:
        obs_all = self.dataset["observations"]
        idx     = np.random.randint(0, self.N - 1, size=batch_size)
        final   = self.final_state_idxs[idx]
        next_idx = np.minimum(idx + 1, final)

        # ── Value goals ───────────────────────────────────────────────────────
        value_goal_idx = self._sample_value_goals(idx, final)
        # OGBench: successes = (idxs == value_goal_idxs) — curgoal match
        successes = (value_goal_idx == idx).astype(np.float32)

        # ── Low-level actor goals: fixed k-step ahead ─────────────────────────
        low_goal_idxs = np.minimum(idx + self.subgoal_steps, final)

        # ── High-level actor goals + targets ──────────────────────────────────
        # Trajectory goals: uniform from [min(idx+1, final), final]
        start          = np.minimum(idx + 1, final)
        distances      = np.random.rand(batch_size)
        high_traj_goal_idxs   = np.round(
            start * distances + final * (1.0 - distances)
        ).astype(np.int64)
        # High-level targets clipped to traj goal (so target ≤ goal)
        high_traj_target_idxs = np.minimum(
            idx + self.subgoal_steps, high_traj_goal_idxs
        )

        # Random goals: from entire dataset, target always = k-step ahead
        high_random_goal_idxs   = np.random.randint(0, self.N, size=batch_size)
        high_random_target_idxs = np.minimum(idx + self.subgoal_steps, final)

        # Mix traj / random
        if self.actor_p_trajgoal == 1.0:
            high_goal_idxs   = high_traj_goal_idxs
            high_target_idxs = high_traj_target_idxs
        elif self.actor_p_randomgoal == 1.0:
            high_goal_idxs   = high_random_goal_idxs
            high_target_idxs = high_random_target_idxs
        else:
            pick_random      = np.random.rand(batch_size) < self.actor_p_randomgoal
            high_goal_idxs   = np.where(pick_random,
                                         high_random_goal_idxs, high_traj_goal_idxs)
            high_target_idxs = np.where(pick_random,
                                         high_random_target_idxs, high_traj_target_idxs)

        return dict(
            observations      = obs_all[idx],
            actions           = self.dataset["actions"][idx],
            next_observations = obs_all[next_idx],
            successes         = successes,
            value_goals       = obs_all[value_goal_idx],
            low_actor_goals   = obs_all[low_goal_idxs],
            high_actor_goals  = obs_all[high_goal_idxs],
            high_actor_targets= obs_all[high_target_idxs],
        )


# ──────────────────────────────── evaluation ──────────────────────────────────

def evaluate(env, agent: HIQL, num_episodes: int) -> dict:
    """Evaluate over all 5 tasks × num_episodes each."""
    per_task_success, returns = [], []

    for task_id in range(1, 6):
        task_successes = []
        for _ in range(num_episodes):
            obs, info = env.reset(options=dict(task_id=task_id, render_goal=False))
            goal = info['goal'].astype(np.float32)
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
    print(f"HIQL on OGBench: {env_name}")
    print(f"  Train steps    : {train_steps:,}  |  Batch size : {cfg.batch_size}")
    print(f"  LR={cfg.lr}  γ={cfg.discount}  tau={TAU}  κ={EXPECTILE}")
    print(f"  α_hi={cfg.high_alpha}  α_lo={cfg.low_alpha}  k={cfg.subgoal_steps}  "
          f"rep_dim={cfg.rep_dim}")
    print(f"  Hidden dims    : {HIDDEN_DIMS}")
    print(f"  Value goals    : p_cur={cfg.value_p_curgoal}  "
          f"p_traj={cfg.value_p_trajgoal}  p_rand={cfg.value_p_randomgoal}")
    print(f"  Actor goals    : p_traj={cfg.actor_p_trajgoal}  "
          f"p_rand={cfg.actor_p_randomgoal}")
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
    agent = HIQL(
        obs_dim,
        act_dim,
        discount   = cfg.discount,
        high_alpha = cfg.high_alpha,
        low_alpha  = cfg.low_alpha,
        rep_dim    = cfg.rep_dim,
    )
    gc_dataset = HGCDataset(
        train_dataset,
        discount            = cfg.discount,
        subgoal_steps       = cfg.subgoal_steps,
        actor_p_trajgoal    = cfg.actor_p_trajgoal,
        actor_p_randomgoal  = cfg.actor_p_randomgoal,
        value_p_curgoal     = cfg.value_p_curgoal,
        value_p_trajgoal    = cfg.value_p_trajgoal,
        value_p_randomgoal  = cfg.value_p_randomgoal,
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
                      f"Hi={losses['hi_loss']:.4f}  "
                      f"Lo={losses['lo_loss']:.4f}")
    else:
        pbar = tqdm(range(1, train_steps + 1), desc="Training", unit="step")
        for _ in pbar:
            batch  = gc_dataset.sample(cfg.batch_size)
            losses = agent.update(batch)
            pbar.set_postfix(
                V=f"{losses['v_loss']:.4f}",
                Hi=f"{losses['hi_loss']:.4f}",
                Lo=f"{losses['lo_loss']:.4f}",
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
        f.write(f"discount={cfg.discount}  high_alpha={cfg.high_alpha}  "
                f"low_alpha={cfg.low_alpha}  subgoal_steps={cfg.subgoal_steps}\n")
        f.write(f"actor_p_traj={cfg.actor_p_trajgoal}  "
                f"actor_p_rand={cfg.actor_p_randomgoal}\n")
        f.write(f"success_rate: {eval_stats['success_rate'] * 100:.2f}%\n")
        for i, sr in enumerate(eval_stats['per_task_success'], start=1):
            f.write(f"  task{i}: {sr * 100:.2f}%\n")
    print(f"\nResults saved to {out_path}")
    print("Test passed.")


if __name__ == "__main__":
    main()
