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
import multiprocessing as mp
import os
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
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
    """Build an MLP: Linear -> GELU -> LayerNorm (repeated), then final Linear."""
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nonlinearity(), nn.LayerNorm(h)]
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class ImpalaSmall(nn.Module):
    """IMPALA-small visual encoder: (B, H, W, C) uint8 → (B, 512) float32.

    Architecture (matches OGBench impala_small = ImpalaEncoder(num_blocks=1)):
      Three ResStacks with (16, 32, 32) channels; each stack has:
        Conv2d → MaxPool2d(stride=2) → 1 residual block (ReLU-Conv-ReLU-Conv + skip)
      Input 32×32 → 16×16 → 8×8 → 4×4; 4×4×32 = 512 → Linear(512) + GELU
    """
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
        """x: (B, H, W, C), values in [0, 255] (uint8 or float)."""
        if x.dtype != torch.float32:
            x = x.float()
        x = x / 255.0
        x = x.permute(0, 3, 1, 2)          # (B, C, H, W)
        x = F.relu(self.stacks(x)).flatten(1)
        return self.head(x)                 # (B, 512)


class CategoricalActor(nn.Module):
    """Discrete low-level actor for powderworld (AWR with Categorical distribution).

    Takes (obs_enc, sg_repr) and outputs logits over act_n actions.
    AWR loss: -(exp_a * log_prob(a)).mean()  — identical formula to continuous case.
    """

    def __init__(self, obs_dim: int, rep_dim: int, act_n: int):
        super().__init__()
        self.net   = mlp(obs_dim + rep_dim, act_n)
        self.act_n = act_n

    def forward(self, obs: torch.Tensor, sg_repr: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, sg_repr], dim=-1))    # logits: (B, act_n)

    def log_prob_at(self, obs: torch.Tensor, sg_repr: torch.Tensor,
                    action: torch.Tensor) -> torch.Tensor:
        """action: (B,) LongTensor of integer action indices."""
        logits = self.forward(obs, sg_repr)
        return Categorical(logits=logits).log_prob(action)    # (B,)

    def act(self, obs_np: np.ndarray, sg_np: np.ndarray,
            temperature: float = EVAL_TEMPERATURE) -> int:
        obs = torch.FloatTensor(obs_np).unsqueeze(0).to(device)
        sg  = torch.FloatTensor(sg_np).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = self.forward(obs, sg) / max(temperature, 1e-6)
            return int(Categorical(logits=logits).sample().item())


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
                 rep_dim: int = REP_DIM,
                 is_visual: bool = False, is_discrete: bool = False):
        self.discount    = discount
        self.high_alpha  = high_alpha
        self.low_alpha   = low_alpha
        self.is_visual   = is_visual
        self.is_discrete = is_discrete

        # Visual encoder: maps (B,H,W,6) uint8 → (B,512); obs_dim for MLPs = 512
        if is_visual:
            self.img_encoder = ImpalaSmall(in_channels=6).to(device)
            obs_dim = ImpalaSmall.ENC_DIM

        self.value      = ValueNet(obs_dim, rep_dim).to(device)
        self.value_tgt  = ValueNet(obs_dim, rep_dim).to(device)
        self.hi_policy  = SubgoalPolicy(obs_dim, rep_dim).to(device)
        if is_discrete:
            self.lo_policy = CategoricalActor(obs_dim, rep_dim, act_dim).to(device)
        else:
            self.lo_policy = GaussianActor(obs_dim, rep_dim, act_dim).to(device)

        # Initialise target as exact copy (includes sg_encoder inside ValueNet)
        self.value_tgt.load_state_dict(self.value.state_dict())

        # SubgoalEncoder lives inside ValueNet -> updated ONLY via v_opt.
        # hi_opt / lo_opt never see encoder parameters; stop-gradient is
        # enforced by computing sg_encoder outputs inside torch.no_grad().
        # For visual, img_encoder is also trained via the value loss only.
        v_params = list(self.value.parameters())
        if is_visual:
            v_params += list(self.img_encoder.parameters())
        self.v_opt  = Adam(v_params, lr=LR)
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

    def _encode(self, img: torch.Tensor) -> torch.Tensor:
        """Encode a raw image tensor through the shared visual encoder (no grad)."""
        with torch.no_grad():
            return self.img_encoder(img)

    def _encode_for_value(self, img: torch.Tensor) -> torch.Tensor:
        """Encode image tensor with gradients (value loss updates encoder)."""
        return self.img_encoder(img)

    # ── Training step ─────────────────────────────────────────────────────

    def update(self, batch: dict) -> dict:
        # For visual envs, observations are uint8 images; goals are also images.
        # Encode images: value loss uses gradients through encoder; policies do not.
        if self.is_visual:
            raw_obs    = torch.ByteTensor(batch["observations"]).to(device)
            raw_nobs   = torch.ByteTensor(batch["next_observations"]).to(device)
            raw_vgoal  = torch.ByteTensor(batch["value_goals"]).to(device)
            raw_higoal = torch.ByteTensor(batch["high_actor_goals"]).to(device)
            raw_hitgt  = torch.ByteTensor(batch["high_actor_targets"]).to(device)
            raw_logoal = torch.ByteTensor(batch["low_actor_goals"]).to(device)

            # Encode with gradients for value/encoder update
            obs_v     = self._encode_for_value(raw_obs)
            next_obs_v= self._encode_for_value(raw_nobs)
            vgoal_v   = self._encode_for_value(raw_vgoal)

            # Encode without gradients for policy updates
            obs_p      = obs_v.detach()
            next_obs_p = next_obs_v.detach()
            higoal_p   = self._encode(raw_higoal)
            hitgt_p    = self._encode(raw_hitgt)
            logoal_p   = self._encode(raw_logoal)
        else:
            obs_v = obs_p = torch.FloatTensor(batch["observations"]).to(device)
            next_obs_v = next_obs_p = torch.FloatTensor(batch["next_observations"]).to(device)
            vgoal_v    = torch.FloatTensor(batch["value_goals"]).to(device)
            higoal_p   = torch.FloatTensor(batch["high_actor_goals"]).to(device)
            hitgt_p    = torch.FloatTensor(batch["high_actor_targets"]).to(device)
            logoal_p   = torch.FloatTensor(batch["low_actor_goals"]).to(device)

        successes = torch.FloatTensor(batch["successes"]).to(device)
        if self.is_discrete:
            action = torch.LongTensor(batch["actions"].astype(np.int64)).to(device)
        else:
            action = torch.FloatTensor(batch["actions"]).to(device)

        # Value + encoder update (encoder grads only from this step)
        v_loss = self._value_loss(obs_v, next_obs_v, vgoal_v, successes)
        self.v_opt.zero_grad(); v_loss.backward(); self.v_opt.step()

        # High-level policy update (no encoder gradient)
        hi_loss = self._hi_policy_loss(obs_p, hitgt_p, higoal_p)
        self.hi_opt.zero_grad(); hi_loss.backward(); self.hi_opt.step()

        # Low-level policy update (no encoder gradient)
        lo_loss = self._lo_policy_loss(obs_p, next_obs_p, action, logoal_p)
        self.lo_opt.zero_grad(); lo_loss.backward(); self.lo_opt.step()

        # EMA target update (propagates through all ValueNet params incl. sg_encoder)
        self._ema_update()

        return dict(v_loss=v_loss.item(), hi_loss=hi_loss.item(), lo_loss=lo_loss.item())

    # ── Action selection ──────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray, goal: np.ndarray):
        with torch.no_grad():
            obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(device)
            goal_t = torch.FloatTensor(goal).unsqueeze(0).to(device)
            if self.is_visual:
                obs_enc  = self.img_encoder(obs_t)    # (1, 512)
                goal_enc = self.img_encoder(goal_t)
            else:
                obs_enc  = obs_t
                goal_enc = goal_t
            sg_repr = self.hi_policy.forward(obs_enc, goal_enc)
            # Normalise high-actor output to the same length as sg_encoder output
            sg_repr = (sg_repr / sg_repr.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                       * (REP_DIM ** 0.5))
            obs_enc_np = obs_enc.squeeze(0).cpu().numpy()
            sg_np      = sg_repr.squeeze(0).cpu().numpy()
        if self.is_discrete:
            return self.lo_policy.act(obs_enc_np, sg_np)
        else:
            return self.lo_policy.act(obs_enc_np, sg_np)


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
    """Evaluate over all 5 tasks × num_episodes each.

    For visual/discrete envs (powderworld):
      - observations are kept as uint8 (encoder normalises internally)
      - actions are integers; no clipping
    """
    per_task_success, returns = [], []

    for task_id in range(1, 6):
        task_successes = []
        for _ in range(num_episodes):
            obs, info = env.reset(options=dict(task_id=task_id, render_goal=False))
            goal = info['goal']    # may be uint8 image (visual) or float32 vector
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

def _hiql_seed_worker(kwargs: dict):
    """Run one seed of HIQL training; designed to be called in a subprocess.

    Offline RL never touches the environment during training (dataset is static),
    so seeds are 100 % independent and trivially parallelisable.  Each worker
    reloads the dataset from disk — a one-time cost (~10–30 s) that is dwarfed
    by the per-seed training time (~2 h).

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

    # Use the dataset passed from the main process to avoid re-reading from
    # the shared filesystem (NFS/Lustre I/O contention with 4 concurrent readers
    # can stall for tens of minutes on HPC clusters).
    # Only create the env (fast: just MuJoCo model init, no disk I/O for dataset).
    train_dataset = kwargs['train_dataset']
    print(f"  [Seed {seed}] creating env (env_only, no dataset reload) ...", flush=True)
    env = ogbench.make_env_and_datasets(env_name, dataset_dir=dataset_dir, env_only=True)
    print(f"  [Seed {seed}] env ready, building agent ...", flush=True)

    agent = HIQL(
        obs_dim, act_dim,
        discount    = cfg.discount,
        high_alpha  = cfg.high_alpha,
        low_alpha   = cfg.low_alpha,
        rep_dim     = cfg.rep_dim,
        is_visual   = is_visual,
        is_discrete = is_discrete,
    )
    gc_dataset = HGCDataset(
        train_dataset,
        discount           = cfg.discount,
        subgoal_steps      = cfg.subgoal_steps,
        actor_p_trajgoal   = cfg.actor_p_trajgoal,
        actor_p_randomgoal = cfg.actor_p_randomgoal,
        value_p_curgoal    = cfg.value_p_curgoal,
        value_p_trajgoal   = cfg.value_p_trajgoal,
        value_p_randomgoal = cfg.value_p_randomgoal,
    )
    print(f"  [Seed {seed}] starting training loop (print every 10k steps) ...", flush=True)

    seed_evals = []
    _t_loop_start = time.time()
    for step in range(1, train_steps + 1):
        batch  = gc_dataset.sample(cfg.batch_size)
        losses = agent.update(batch)
        if step == 100:
            elapsed = time.time() - _t_loop_start
            ms_per_step = elapsed / 100 * 1000
            eta_h = ms_per_step * train_steps / 1000 / 3600
            print(f"  [Seed {seed}] step 100 done | {ms_per_step:.1f} ms/step | "
                  f"ETA ~{eta_h:.1f} h for {train_steps:,} steps", flush=True)
        if step % 10_000 == 0:
            print(f"  [Seed {seed}] step {step:>8,}/{train_steps:,} | "
                  f"V={losses['v_loss']:.4f}  "
                  f"Hi={losses['hi_loss']:.4f}  "
                  f"Lo={losses['lo_loss']:.4f}", flush=True)
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
    parser.add_argument("--seeds",         type=int, nargs="+", default=None,
                        metavar="SEED",
                        help="Explicit list of seeds to run sequentially, e.g. --seeds 42 0. "
                             "Ignored if --single-seed is set. "
                             "Recommended: split the default 4 seeds across 2 GPU jobs "
                             "(--seeds 42 0  in job1, --seeds 1 2  in job2) to stay within "
                             "MaxJobs=2 / GrpTRES=gres/gpu=2 Slurm limits while keeping "
                             "each job under the 8-hour wall-time limit.")
    parser.add_argument("--visual-enabled", action="store_true",
                        help="Enable visual encoder + discrete actor (required for "
                             "powderworld-* environments)")
    parser.add_argument("--parallel-seeds", action="store_true",
                        help="Run all seeds concurrently in separate processes "
                             "(each reloads the dataset; ~N-seed wall-clock speedup). "
                             "Offline RL has no env interaction during training, so "
                             "seeds are fully independent and safe to parallelise.")
    parser.add_argument("--output-dir",    default=".",
                        help="Directory to write the results .txt file "
                             "(default: current working directory). Created if absent.")
    args = parser.parse_args()

    env_name    = f"{args.env}-{args.dsize}-{args.task}-v0"
    cfg         = get_config(env_name)
    train_steps = args.train_step if args.train_step is not None else DEFAULT_TRAIN_STEPS
    eval_intv   = args.eval_interval
    if args.single_seed is not None:
        seeds = [args.single_seed]
    elif args.seeds is not None:
        seeds = args.seeds
    else:
        seeds = SEEDS

    # Powderworld: observations are (32,32,6) uint8 images; actions are discrete ints.
    is_visual   = args.visual_enabled
    is_discrete = args.visual_enabled   # powderworld is always both visual and discrete

    print(f"\n{'='*60}")
    print(f"HIQL on OGBench: {env_name}")
    print(f"  Seeds          : {seeds}"
          f"{'  [parallel]' if args.parallel_seeds and len(seeds) > 1 else ''}")
    print(f"  Train steps    : {train_steps:,} per seed  |  Eval every: {eval_intv:,}")
    print(f"  Batch size     : {cfg.batch_size}  |  LR={cfg.lr}  "
          f"gamma={cfg.discount}  tau={TAU}  expectile={EXPECTILE}")
    print(f"  alpha_hi={cfg.high_alpha}  alpha_lo={cfg.low_alpha}  "
          f"k={cfg.subgoal_steps}  rep_dim={cfg.rep_dim}")
    print(f"  Value goals    : p_cur={cfg.value_p_curgoal}  "
          f"p_traj={cfg.value_p_trajgoal}  p_rand={cfg.value_p_randomgoal}")
    print(f"  Actor goals    : p_traj={cfg.actor_p_trajgoal}  "
          f"p_rand={cfg.actor_p_randomgoal}")
    if is_visual:
        print(f"  Visual encoder : ImpalaSmall  |  Discrete actor (temp={EVAL_TEMPERATURE})")
    print(f"{'='*60}\n")

    # ── 1. Probe dataset for obs/act dimensions (cheap; workers reload full data) ─
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
        # Each worker reloads the dataset independently — safe for spawn context
        # (no shared CUDA state, no forked file handles).
        # Workers use CPU to avoid cudaErrorDevicesUnavailable on HPC nodes that
        # run the V100 in exclusive-process compute mode (only one CUDA context
        # allowed at a time).  Evaluation (MuJoCo, CPU-bound) is the dominant
        # cost (~90 % of wall time) and parallelises freely across CPU cores.
        env.close()   # main process does not need its env instance
        worker_args = [{**_worker_base, 'seed': s, 'device': 'cpu',
                        'train_dataset': train_dataset} for s in seeds]
        ctx = mp.get_context('spawn')
        print(f"Launching {len(seeds)} parallel seed workers "
              f"(spawn context, each reloads dataset, device=cpu) ...\n")
        with ctx.Pool(processes=len(seeds)) as pool:
            results = pool.map(_hiql_seed_worker, worker_args)
        all_results = dict(results)

    else:
        # ── Sequential path (original behaviour) ──────────────────────────────
        for seed_idx, seed in enumerate(seeds):
            print(f"\n{'─'*60}")
            print(f"Seed {seed}  ({seed_idx + 1}/{len(seeds)})")
            print(f"{'─'*60}")

            torch.manual_seed(seed)
            np.random.seed(seed)

            agent = HIQL(
                obs_dim, act_dim,
                discount    = cfg.discount,
                high_alpha  = cfg.high_alpha,
                low_alpha   = cfg.low_alpha,
                rep_dim     = cfg.rep_dim,
                is_visual   = is_visual,
                is_discrete = is_discrete,
            )
            gc_dataset = HGCDataset(
                train_dataset,
                discount           = cfg.discount,
                subgoal_steps      = cfg.subgoal_steps,
                actor_p_trajgoal   = cfg.actor_p_trajgoal,
                actor_p_randomgoal = cfg.actor_p_randomgoal,
                value_p_curgoal    = cfg.value_p_curgoal,
                value_p_trajgoal   = cfg.value_p_trajgoal,
                value_p_randomgoal = cfg.value_p_randomgoal,
            )

            seed_evals = []   # (step, per_task_success_list, overall_rate)

            if args.slurm_tqdm:
                for step in range(1, train_steps + 1):
                    batch  = gc_dataset.sample(cfg.batch_size)
                    losses = agent.update(batch)
                    if step % 10_000 == 0:
                        print(f"  step {step:>8,}/{train_steps:,} | "
                              f"V={losses['v_loss']:.4f}  "
                              f"Hi={losses['hi_loss']:.4f}  "
                              f"Lo={losses['lo_loss']:.4f}")
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
                    pbar.set_postfix(V=f"{losses['v_loss']:.4f}",
                                     Hi=f"{losses['hi_loss']:.4f}",
                                     Lo=f"{losses['lo_loss']:.4f}")
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
    print(f"FINAL RESULTS -- HIQL -- {env_name}")
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
    os.makedirs(args.output_dir, exist_ok=True)
    out_path  = os.path.join( #os.path.dirname(os.path.abspath(__file__)),
        args.output_dir,
        f"results_hiql_{args.env}_{args.dsize}_{args.task}_{timestamp}.txt",
    )
    with open(out_path, "w") as f:
        f.write(f"method: HIQL\n")
        f.write(f"env: {env_name}\n")
        f.write(f"seeds: {seeds}\n")
        f.write(f"train_steps_per_seed: {train_steps}\n")
        f.write(f"eval_interval: {eval_intv}\n")
        f.write(f"last_3_checkpoints: {last_ckpts}\n")
        f.write(f"discount={cfg.discount}  high_alpha={cfg.high_alpha}  "
                f"low_alpha={cfg.low_alpha}  subgoal_steps={cfg.subgoal_steps}\n")
        f.write(f"actor_p_traj={cfg.actor_p_trajgoal}  "
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
