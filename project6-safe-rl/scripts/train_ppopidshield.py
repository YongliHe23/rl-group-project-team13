"""
train_ppopidshield.py — PPO with PID-Lagrangian and Predictive Shielding (PPO-PIDShield)

Four algorithmic innovations over PPO-Lagrangian:

  1. PID-Lagrangian with Nesterov Accelerated Dual Ascent
       Replaces the standard proportional dual-ascent with a PID controller:
           Δλ_t = Kp·(Jc_t − d_t) + Ki·Σᵢ(Jc_i − d_t) + Kd·(Jc_t − Jc_{t−1})
           μ_{t+1} = max(0, λ_t + Δλ_t)
           λ_{t+1} = max(0, μ_{t+1} + β·(μ_{t+1} − μ_t))
       The derivative term anticipates constraint boundary crossings before
       they accumulate; the proportional term drops instantly once safe.
       Nesterov momentum accelerates convergence to the optimal λ*.

  2. Adaptive Cost Advantage Normalization + Gradient Orthogonalization
       Cost advantages from GAE are independently zero-mean / unit-variance
       normalised every epoch, decoupling scale from raw cost magnitudes.
       When the reward gradient g_R and cost gradient g_C conflict (cos-sim < 0)
       and the agent is currently within the cost limit, g_R is projected onto
       the hyperplane orthogonal to g_C before the combined gradient is applied:
           g_R^proj = g_R − (g_R·g_C / ‖g_C‖²) · g_C

  3. Reachability-Based Predictive Shielding (H-step Lookahead)
       A lightweight neural dynamics model P̂(s′|s,a) is trained on-policy.
       Before each env.step, the shield imagines H steps forward using P̂ and
       the cost critic V_C to estimate cumulative future cost:
           Ĉ = Σ_{k=0}^{H-1} γ^k · V_C(ŝ_k) + γ^H · V_C(ŝ_H)
       Actions with Ĉ above a safety threshold are rejected; the policy
       resamples up to max_resample times before falling back.

  4. Safe Curriculum Generation + Differential Cost Shaping
       The cost limit decays exponentially from d_init to d_target:
           d_t = d_target + (d_init − d_target) · exp(−κ · t)
       Proximity to hazards/vases (detected via lidar) injects a dense shaped
       cost before physical contact, providing early gradient signal for sparse
       collision events.

Run:
    python scripts/train_ppopidshield.py \\
        --config configs/ppopidshield/config.yaml \\
        --env_id SafetyPointGoal1-v0 --seed 0

Compare (PPO-Lag baseline):
    python scripts/train_ppolag.py \\
        --config configs/ppo_lag/config.yaml \\
        --env_id SafetyPointGoal1-v0 --seed 0
"""

import argparse
import csv
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import safety_gymnasium


# ── Network utilities ─────────────────────────────────────────────────────────

def mlp(sizes: list, act=nn.Tanh) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


# ── Neural networks ───────────────────────────────────────────────────────────

class GaussianActor(nn.Module):
    """Diagonal-Gaussian policy — identical architecture to PPO-Lag baseline."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: list):
        super().__init__()
        self.mu_net  = mlp([obs_dim] + hidden + [act_dim])
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def _dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.mu_net(obs), self.log_std.exp())

    def log_prob(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self._dist(obs).log_prob(act).sum(-1)

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        return self._dist(obs).entropy().sum(-1)

    @torch.no_grad()
    def sample(self, obs: torch.Tensor):
        d = self._dist(obs)
        a = d.sample()
        return a, d.log_prob(a).sum(-1)


class ScalarCritic(nn.Module):
    """Scalar value function V(s) — used for both reward and cost."""

    def __init__(self, obs_dim: int, hidden: list):
        super().__init__()
        self.net = mlp([obs_dim] + hidden + [1])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class DynamicsModel(nn.Module):
    """
    Innovation 3 — Lightweight world model for predictive shielding.

    Predicts delta_obs = s_{t+1} − s_t from (s_t, a_t) using ReLU activations
    for smooth gradient flow. Uses residual prediction for training stability.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: list):
        super().__init__()
        self.net = mlp([obs_dim + act_dim] + hidden + [obs_dim], act=nn.ReLU)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat([obs, act], dim=-1))
        return obs + delta


# ── Observation utilities ─────────────────────────────────────────────────────

def _find_lidar_slice(env, keyword: str):
    """Locate the flat-obs slice for a named sensor (e.g. 'hazard', 'vase')."""
    try:
        start = 0
        for key, space in env.obs_space_dict.items():
            n = int(np.prod(space.shape))
            if keyword in key.lower():
                return slice(start, start + n)
            start += n
    except Exception:
        pass
    return None


def _proximity_cost(obs_np: np.ndarray, hslice, vslice,
                    threshold: float, h_coef: float, v_coef: float) -> float:
    """
    Innovation 4b — Dense proximity penalty injected before physical contact.
    Scaled by how far each lidar reading exceeds the proximity threshold.
    """
    shaped = 0.0
    if hslice is not None:
        excess = np.maximum(obs_np[hslice] - threshold, 0.0)
        shaped += h_coef * float(np.sum(excess))
    if vslice is not None:
        excess = np.maximum(obs_np[vslice] - threshold, 0.0)
        shaped += v_coef * float(np.sum(excess))
    return shaped


# ── On-policy rollout buffer (stores next_obs for dynamics model) ─────────────

class Buffer:

    def __init__(
        self, obs_dim: int, act_dim: int, size: int,
        gamma: float, lam: float, cost_gamma: float, lam_c: float,
        device: torch.device,
    ):
        self.obs      = torch.zeros(size, obs_dim, device=device)
        self.acts     = torch.zeros(size, act_dim, device=device)
        self.next_obs = torch.zeros(size, obs_dim, device=device)
        self.rews     = torch.zeros(size, device=device)
        self.costs    = torch.zeros(size, device=device)  # shaped cost
        self.vals_r   = torch.zeros(size, device=device)
        self.vals_c   = torch.zeros(size, device=device)
        self.logps    = torch.zeros(size, device=device)
        self.adv_r    = torch.zeros(size, device=device)
        self.adv_c    = torch.zeros(size, device=device)
        self.ret_r    = torch.zeros(size, device=device)
        self.ret_c    = torch.zeros(size, device=device)
        self.γ,  self.λ  = gamma,      lam
        self.γc, self.λc = cost_gamma, lam_c
        self.ptr = self.seg = 0
        self.size = size
        self.dev  = device

    def push(self, obs, act, next_obs, rew, cost, val_r, val_c, logp):
        i = self.ptr
        self.obs[i] = obs;          self.acts[i] = act
        self.next_obs[i] = next_obs
        self.rews[i] = rew;         self.costs[i] = cost
        self.vals_r[i] = val_r;     self.vals_c[i] = val_c
        self.logps[i] = logp
        self.ptr += 1

    def end_episode(self, last_vr: float = 0.0, last_vc: float = 0.0):
        s = slice(self.seg, self.ptr)
        rews  = torch.cat([self.rews[s],  torch.tensor([last_vr], device=self.dev)])
        costs = torch.cat([self.costs[s], torch.tensor([last_vc], device=self.dev)])
        vr    = torch.cat([self.vals_r[s], torch.tensor([last_vr], device=self.dev)])
        vc    = torch.cat([self.vals_c[s], torch.tensor([last_vc], device=self.dev)])
        δr = rews[:-1]  + self.γ  * vr[1:] - vr[:-1]
        δc = costs[:-1] + self.γc * vc[1:] - vc[:-1]
        gae_r = gae_c = 0.0
        for k in range(self.ptr - self.seg - 1, -1, -1):
            gae_r = float(δr[k]) + self.γ  * self.λ  * gae_r
            gae_c = float(δc[k]) + self.γc * self.λc * gae_c
            self.adv_r[self.seg + k] = gae_r
            self.adv_c[self.seg + k] = gae_c
        self.ret_r[s] = self.adv_r[s] + self.vals_r[s]
        self.ret_c[s] = self.adv_c[s] + self.vals_c[s]
        self.seg = self.ptr

    def get(self):
        assert self.ptr == self.size
        # Reward advantage: standard zero-mean normalisation
        adv_r = (self.adv_r - self.adv_r.mean()) / (self.adv_r.std() + 1e-8)
        # Innovation 2a: independent cost advantage normalisation (fully decoupled)
        adv_c = (self.adv_c - self.adv_c.mean()) / (self.adv_c.std() + 1e-8)
        return (self.obs, self.acts, self.next_obs, self.logps,
                adv_r, adv_c, self.ret_r, self.ret_c)

    def reset(self):
        self.ptr = self.seg = 0


# ── PID-Lagrangian + Nesterov controller ──────────────────────────────────────

class PIDLagrangianController:
    """
    Innovation 1 — PID controller for the Lagrange multiplier λ.

    Update sequence each epoch:
        Δλ_t = Kp·e_t + Ki·∫e + Kd·Δe_t        (PID)
        μ_{t+1} = max(0, λ_t + Δλ_t)             (prospective position)
        λ_{t+1} = max(0, μ_{t+1} + β·(μ_{t+1} − μ_t))  (Nesterov step)

    The derivative term Kd·Δe_t predicts cost boundary crossings ahead of
    the integral wind-up.  Nesterov momentum provides forward-looking
    acceleration toward the optimal dual variable λ*.
    """

    def __init__(self, init_lam: float, Kp: float, Ki: float, Kd: float,
                 beta_nesterov: float):
        self.lam      = max(0.0, float(init_lam))
        self.Kp       = Kp
        self.Ki       = Ki
        self.Kd       = Kd
        self.beta     = beta_nesterov
        self.integral  = 0.0        # Σᵢ (Jc_i − d_i)
        self.prev_error = None      # e_{t−1} for derivative
        self.mu_prev   = max(0.0, float(init_lam))  # μ_t for Nesterov

    def update(self, jc: float, d: float) -> float:
        """One dual-variable update step; returns new λ."""
        error = jc - d
        deriv = 0.0 if self.prev_error is None else (error - self.prev_error)
        self.integral += error

        delta    = self.Kp * error + self.Ki * self.integral + self.Kd * deriv
        mu_next  = max(0.0, self.lam + delta)
        lam_next = max(0.0, mu_next + self.beta * (mu_next - self.mu_prev))

        self.prev_error = error
        self.mu_prev    = mu_next
        self.lam        = lam_next
        return self.lam


# ── Curriculum cost-limit schedule ────────────────────────────────────────────

def curriculum_limit(epoch: int, d_target: float, d_init: float,
                     kappa: float) -> float:
    """Innovation 4a — d_t = d_target + (d_init − d_target)·exp(−κ·t)."""
    return d_target + (d_init - d_target) * float(np.exp(-kappa * epoch))


# ── PPO-PIDShield Trainer ─────────────────────────────────────────────────────

class PPOPIDShieldTrainer:

    def __init__(self, env_id: str, cfg: dict, seed: int, device: str = 'cpu'):
        self.env_id = env_id
        self.seed   = seed
        self.dev    = torch.device(device)

        # Probe environment once
        _env = safety_gymnasium.make(env_id)
        obs_dim = _env.observation_space.shape[0]
        act_dim = _env.action_space.shape[0]
        self.act_low  = _env.action_space.low.copy()
        self.act_high = _env.action_space.high.copy()
        self.hslice   = _find_lidar_slice(_env, 'hazard')
        self.vslice   = _find_lidar_slice(_env, 'vase')
        _env.close()

        ac  = cfg['algo_cfgs']
        mc  = cfg['model_cfgs']
        lc  = cfg['lagrange_cfgs']
        tc  = cfg['train_cfgs']
        pid = cfg.get('pid_cfgs',        {})
        ort = cfg.get('ortho_cfgs',      {})
        sh  = cfg.get('shield_cfgs',     {})
        cur = cfg.get('curriculum_cfgs', {})
        shp = cfg.get('shaping_cfgs',    {})

        hidden  = list(mc['actor']['hidden_sizes'])
        act_lr  = mc['actor']['lr']
        crit_lr = mc['critic']['lr']

        self.steps_per_epoch = ac['steps_per_epoch']
        self.clip         = ac['clip']
        self.target_kl    = ac['target_kl']
        self.update_iters = ac['update_iters']
        self.ent_coef     = ac['entropy_coef']
        self.gamma        = ac['gamma']
        self.cost_gamma   = ac['cost_gamma']
        self.lam_gae      = ac['lam']
        self.lam_c_gae    = ac['lam_c']
        self.n_epochs     = tc['total_steps'] // self.steps_per_epoch

        # Innovation 1: PID-Lagrangian + Nesterov
        self.pid = PIDLagrangianController(
            init_lam      = float(lc.get('lagrangian_multiplier_init', 1.0)),
            Kp            = pid.get('Kp',            0.1),
            Ki            = pid.get('Ki',            0.01),
            Kd            = pid.get('Kd',            0.05),
            beta_nesterov = pid.get('beta_nesterov', 0.6),
        )

        # Innovation 2b: gradient orthogonalization
        self.ortho_enabled = bool(ort.get('enabled', True))
        self._is_safe      = True   # updated each epoch

        # Innovation 3: predictive shield
        dyn_hidden          = list(sh.get('dynamics_hidden', [128, 128]))
        self.shield_H       = int(sh.get('H',             3))
        self.shield_thresh  = float(sh.get('threshold',   5.0))
        self.shield_maxre   = int(sh.get('max_resample',  5))
        self.shield_warmup  = int(sh.get('shield_warmup', 3))

        # Innovation 4a: curriculum
        self.d_target   = float(cur.get('d_target', float(lc.get('cost_limit', 25.0))))
        self.d_init     = float(cur.get('d_init',   150.0))
        self.kappa_cur  = float(cur.get('kappa',    0.005))

        # Innovation 4b: cost shaping
        self.shaping_on  = bool(shp.get('enabled',             True))
        self.prox_thresh = float(shp.get('prox_threshold',     0.5))
        self.h_coef      = float(shp.get('hazard_shaping_coef', 0.5))
        self.v_coef      = float(shp.get('vase_shaping_coef',   1.0))

        # Networks
        self.actor    = GaussianActor(obs_dim, act_dim, hidden).to(self.dev)
        self.rew_crit = ScalarCritic(obs_dim, hidden).to(self.dev)
        self.cost_val = ScalarCritic(obs_dim, hidden).to(self.dev)
        self.dynamics = DynamicsModel(obs_dim, act_dim, dyn_hidden).to(self.dev)

        self.actor_opt = Adam(self.actor.parameters(),    lr=act_lr)
        self.rcrit_opt = Adam(self.rew_crit.parameters(), lr=crit_lr)
        self.ccrit_opt = Adam(self.cost_val.parameters(), lr=crit_lr)
        self.dyn_opt   = Adam(self.dynamics.parameters(), lr=crit_lr)

        self.buf = Buffer(
            obs_dim, act_dim, self.steps_per_epoch,
            self.gamma, self.lam_gae, self.cost_gamma, self.lam_c_gae, self.dev,
        )
        self._epoch = 0

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _t(self, x) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.dev)

    def _curr_limit(self) -> float:
        return curriculum_limit(self._epoch, self.d_target, self.d_init, self.kappa_cur)

    # ── Innovation 3: predictive shield ──────────────────────────────────────

    def _shield_ok(self, obs_np: np.ndarray, act_np: np.ndarray) -> bool:
        """
        Rolls out H imagined steps using the dynamics model and cost critic.
        Returns True (action is safe) if estimated cumulative cost ≤ threshold.
        Shield is silenced during warmup epochs while the dynamics model warms up.
        """
        if self._epoch < self.shield_warmup:
            return True
        obs_t = self._t(obs_np).unsqueeze(0)
        act_t = self._t(act_np).unsqueeze(0)
        act_lo = self._t(self.act_low)
        act_hi = self._t(self.act_high)
        with torch.no_grad():
            cumcost  = 0.0
            cur_obs  = obs_t
            cur_act  = act_t
            for k in range(self.shield_H):
                cumcost += (self.cost_gamma ** k) * self.cost_val(cur_obs).item()
                cur_obs  = self.dynamics(cur_obs, cur_act)
                cur_act, _ = self.actor.sample(cur_obs)
                cur_act  = cur_act.clamp(act_lo, act_hi)
            cumcost += (self.cost_gamma ** self.shield_H) * self.cost_val(cur_obs).item()
        return cumcost <= self.shield_thresh

    # ── Data collection ───────────────────────────────────────────────────────

    def _collect(self, env):
        self.buf.reset()
        ep_rets, ep_costs = [], []
        obs_np, _ = env.reset()
        ep_ret = ep_cost = 0.0

        for t in range(self.steps_per_epoch):
            obs_t = self._t(obs_np).unsqueeze(0)
            with torch.no_grad():
                act_t, logp_t = self.actor.sample(obs_t)
                val_r = self.rew_crit(obs_t).item()
                val_c = self.cost_val(obs_t).item()

            act_np = np.clip(act_t.cpu().numpy()[0], self.act_low, self.act_high)

            # Innovation 3: shield — resample if action looks unsafe
            if not self._shield_ok(obs_np, act_np):
                for _ in range(self.shield_maxre):
                    with torch.no_grad():
                        act_t, logp_t = self.actor.sample(obs_t)
                    act_np = np.clip(act_t.cpu().numpy()[0], self.act_low, self.act_high)
                    if self._shield_ok(obs_np, act_np):
                        break

            obs_next, rew, cost, terminated, truncated, _ = env.step(act_np)
            ep_ret  += rew
            ep_cost += cost

            # Innovation 4b: dense proximity shaping added to buffer cost
            buf_cost = cost
            if self.shaping_on:
                buf_cost += _proximity_cost(
                    obs_np, self.hslice, self.vslice,
                    self.prox_thresh, self.h_coef, self.v_coef,
                )

            self.buf.push(
                self._t(obs_np), self._t(act_np), self._t(obs_next),
                rew, buf_cost, val_r, val_c, logp_t.item(),
            )
            obs_np = obs_next

            epoch_end   = (t == self.steps_per_epoch - 1)
            episode_end = terminated or truncated

            if episode_end or epoch_end:
                if truncated or (epoch_end and not terminated):
                    obs_next_t = self._t(obs_next).unsqueeze(0)
                    with torch.no_grad():
                        last_vr = self.rew_crit(obs_next_t).item()
                        last_vc = self.cost_val(obs_next_t).item()
                else:
                    last_vr = last_vc = 0.0
                self.buf.end_episode(last_vr, last_vc)
                if episode_end:
                    ep_rets.append(ep_ret)
                    ep_costs.append(ep_cost)
                    ep_ret = ep_cost = 0.0
                    obs_np, _ = env.reset()

        return ep_rets, ep_costs

    # ── Parameter update ──────────────────────────────────────────────────────

    def _update(self, avg_cost: float) -> dict:
        obs, acts, next_obs, logps_old, adv_r, adv_c, ret_r, ret_c = \
            self.buf.get()

        d_curr = self._curr_limit()
        # Orthogonalization uses the TRUE final target (not the sliding curriculum
        # limit) so that the agent only projects reward gradients when it is
        # genuinely safe, not just "curriculum-safe" at a loose early limit.
        self._is_safe = (avg_cost <= self.d_target)

        # ── Dynamics model (Innovation 3) ─────────────────────────────────────
        # Trained via supervised MSE on collected (s, a) → s' transitions.
        # Half as many iterations as the policy update to limit overhead.
        dyn_iters = max(1, self.update_iters // 2)
        for _ in range(dyn_iters):
            pred_next = self.dynamics(obs, acts)
            loss_dyn  = F.mse_loss(pred_next, next_obs.detach())
            self.dyn_opt.zero_grad(); loss_dyn.backward(); self.dyn_opt.step()

        # ── Reward critic ─────────────────────────────────────────────────────
        for _ in range(self.update_iters):
            loss_rc = F.mse_loss(self.rew_crit(obs), ret_r.detach())
            self.rcrit_opt.zero_grad(); loss_rc.backward(); self.rcrit_opt.step()

        # ── Cost value critic ─────────────────────────────────────────────────
        for _ in range(self.update_iters):
            loss_cc = F.mse_loss(self.cost_val(obs), ret_c.detach())
            self.ccrit_opt.zero_grad(); loss_cc.backward(); self.ccrit_opt.step()

        # ── Actor update (PPO-clip + Innovation 2b: gradient ortho) ──────────
        lam  = self.pid.lam
        kl   = 0.0
        act_lo = self._t(self.act_low)
        act_hi = self._t(self.act_high)

        for _ in range(self.update_iters):
            logps_new = self.actor.log_prob(obs, acts)
            ratio     = (logps_new - logps_old).exp()
            clip_r    = ratio.clamp(1 - self.clip, 1 + self.clip)
            entropy   = self.actor.entropy(obs).mean()

            L_reward = (-torch.min(ratio * adv_r, clip_r * adv_r).mean()
                        - self.ent_coef * entropy)
            L_cost   = (ratio * adv_c).mean()

            if self.ortho_enabled:
                # --- Two separate backward passes to isolate g_R and g_C ---
                # Pass 1: reward gradient (retain graph for L_cost backward)
                self.actor_opt.zero_grad()
                L_reward.backward(retain_graph=True)
                g_R = torch.cat([
                    p.grad.view(-1).clone() if p.grad is not None
                    else torch.zeros(p.numel(), device=self.dev)
                    for p in self.actor.parameters()
                ])

                # Pass 2: cost gradient
                self.actor_opt.zero_grad()
                L_cost.backward()
                g_C = torch.cat([
                    p.grad.view(-1).clone() if p.grad is not None
                    else torch.zeros(p.numel(), device=self.dev)
                    for p in self.actor.parameters()
                ])

                # Orthogonal projection of g_R onto complement of g_C
                # Applied only when gradients conflict AND the agent is safe.
                g_R_norm = g_R.norm()
                g_C_norm = g_C.norm()
                if g_R_norm > 1e-8 and g_C_norm > 1e-8:
                    cos_sim = (g_R @ g_C) / (g_R_norm * g_C_norm)
                    if cos_sim < 0 and self._is_safe:
                        # Remove from g_R the component along g_C
                        proj = (g_R @ g_C) / (g_C @ g_C + 1e-8) * g_C
                        g_R  = g_R - proj

                # Combine: g_final = projected_g_R + λ·g_C
                g_final = g_R + lam * g_C

                # Install g_final into parameter .grad fields
                self.actor_opt.zero_grad()
                offset = 0
                for p in self.actor.parameters():
                    n = p.numel()
                    p.grad = g_final[offset:offset + n].view_as(p).clone()
                    offset += n
            else:
                L_total = L_reward + lam * L_cost
                self.actor_opt.zero_grad()
                L_total.backward()

            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_opt.step()

            with torch.no_grad():
                kl = (logps_old - self.actor.log_prob(obs, acts)).mean().item()
            if kl > 1.5 * self.target_kl:
                break

        # ── Innovation 1: PID-Lagrangian + Nesterov update ───────────────────
        # Uses the raw episodic cost (not shaped) and the curriculum limit.
        new_lam = self.pid.update(avg_cost, d_curr)

        return dict(kl=kl, lam=new_lam, d_curr=d_curr,
                    loss_rc=loss_rc.item(), loss_dyn=loss_dyn.item())

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self, save_csv: str | None = None) -> list[dict]:
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        env = safety_gymnasium.make(self.env_id)
        env.reset(seed=self.seed)

        print(f"\n{'='*72}")
        print(f"  PPO-PIDShield  |  {self.env_id}  |  seed={self.seed}")
        print(f"  PID  Kp={self.pid.Kp}  Ki={self.pid.Ki}  Kd={self.pid.Kd}"
              f"  β_Nesterov={self.pid.beta}")
        print(f"  Shield  H={self.shield_H}  thresh={self.shield_thresh}"
              f"  warmup={self.shield_warmup} epochs")
        print(f"  Curriculum  d_init={self.d_init}→{self.d_target}  κ={self.kappa_cur}")
        print(f"  Ortho={self.ortho_enabled}  Shaping={self.shaping_on}")
        print(f"{'='*72}")
        hdr = (f"{'Epoch':>6}  {'AvgRet':>9}  {'AvgCost':>9}  "
               f"{'Lambda':>8}  {'d_curr':>7}  {'KL':>8}  {'NEps':>5}")
        print(hdr)

        rows = []
        for epoch in range(self.n_epochs):
            self._epoch = epoch
            ep_rets, ep_costs = self._collect(env)
            n_eps     = len(ep_rets)
            avg_ret   = float(np.mean(ep_rets))  if ep_rets  else float('nan')
            avg_cost  = float(np.mean(ep_costs)) if ep_costs else float('nan')
            avg_len   = self.steps_per_epoch / n_eps if n_eps > 0 else float('nan')
            total_steps = (epoch + 1) * self.steps_per_epoch

            stats = self._update(avg_cost)

            print(f"{epoch+1:6d}  {avg_ret:9.2f}  {avg_cost:9.2f}  "
                  f"{stats['lam']:8.4f}  {stats['d_curr']:7.2f}  "
                  f"{stats['kl']:8.5f}  {n_eps:5d}")
            rows.append(dict(
                TotalEnvSteps=total_steps,
                Metrics_EpRet=avg_ret,
                Metrics_EpCost=avg_cost,
                Metrics_EpLen=avg_len,
                Train_Lambda=stats['lam'],
                Train_CurrLimit=stats['d_curr'],
                epoch=epoch + 1,
                kl=stats['kl'],
                n_eps=n_eps,
            ))

        env.close()

        if save_csv:
            os.makedirs(os.path.dirname(os.path.abspath(save_csv)), exist_ok=True)
            with open(save_csv, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader(); w.writerows(rows)
            print(f"\nResults saved → {save_csv}")

        return rows


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train PPO-PIDShield on Safety-Gymnasium'
    )
    parser.add_argument('--config',   default='configs/ppopidshield/config.yaml',
                        help='YAML config (same base as PPO-Lag; extra blocks optional)')
    parser.add_argument('--env_id',   default=None, help='Override env_id in config')
    parser.add_argument('--seed',     type=int, default=None, help='Random seed')
    parser.add_argument('--save_csv', default=None,
                        help='Path to save per-epoch results CSV')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env_id = args.env_id or cfg.pop('env_id', 'SafetyPointGoal1-v0')
    seed   = args.seed if args.seed is not None else cfg.pop('seed', 0)

    trainer = PPOPIDShieldTrainer(env_id, cfg, seed)
    trainer.train(save_csv=args.save_csv)


# Example:
#   python scripts/train_ppopidshield.py --config configs/ppopidshield/config.yaml \
#       --env_id SafetyPointGoal1-v0 --seed 0

if __name__ == '__main__':
    main()
