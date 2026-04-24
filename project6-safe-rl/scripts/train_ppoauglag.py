"""
train_ppoauglag.py — PPO-Lagrangian with Augmented Lagrangian + Conservative Update Policy.

Extends OmniSafe's PPOLag using the same subclassing pattern as CPPOPID:

  (I)  Augmented Lagrangian (AL)
       Replaces the scalar Lagrange penalty λ with an augmented effective penalty:
           λ_eff = λ  +  ρ · max(0, (J_c − d) / d)
       The quadratic term locally convexifies the saddle-point landscape near the
       constraint boundary (arxiv.org/abs/2602.02924), ensuring gradient steps
       converge smoothly rather than oscillating around a non-convex saddle point.

  (II) Conservative Update Policy (CUP) clip
       Replaces the fixed PPO clip radius ε with a safety-adaptive radius:
           ε_eff = ε · exp(−α · max(0, (J_c − d) / d)),  floor = 0.1·ε
       Tightens the trust region when the policy is unsafe, preventing unsafe
       exploratory actions from worsening cost rate
       (openreview.net/forum?id=2wiaitACS_O, icml.cc/virtual/2025/poster/46451).

Switch: auglag_cfgs.enabled = false → 100 % identical to OmniSafe PPOLag.

Three methods are overridden (minimum possible):
  _init                  — read auglag hyperparameters; cache _ep_cost
  _compute_adv_surrogate — inject λ_eff  (Augmented Lagrangian)
  _loss_pi               — inject ε_eff  (Conservative Update clip)

Run:
    python scripts/train_ppoauglag.py \\
        --config configs/ppoauglag/config.yaml \\
        --env_id SafetyPointGoal2-v0 --seed 0
"""

from __future__ import annotations

import argparse
import math
from types import MappingProxyType

import numpy as np
import torch
import yaml

import omnisafe
from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.naive_lagrange.ppo_lag import PPOLag


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Define PPOAugLag (same pattern as CPPOPID in cppo_pid.py)
# ─────────────────────────────────────────────────────────────────────────────

@registry.register
class PPOAugLag(PPOLag):
    r"""PPO-Lagrangian + Augmented Lagrangian + Conservative Update Policy.

    Overrides exactly three methods of PPOLag; all other behaviour (critic
    updates, Lagrange multiplier Adam step, KL early-stopping, logging) is
    inherited unchanged.

    When ``auglag_cfgs.enabled = false`` the class is 100 % equivalent to
    OmniSafe's ``PPOLag``.
    """

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init(self) -> None:
        """Initialise PPOLag, then read auglag hyperparameters with safe fallbacks."""
        super()._init()
        self._ep_cost: float = 0.0   # undiscounted J_c; refreshed in _update

        auglag = getattr(self._cfgs, 'auglag_cfgs', None)
        if auglag is None:
            self._auglag_enabled: bool  = True
            self._rho:            float = 1.0
            self._cup_alpha:      float = 1.0
        else:
            self._auglag_enabled = bool(getattr(auglag, 'enabled',   True))
            self._rho            = float(getattr(auglag, 'rho',       1.0))
            self._cup_alpha      = float(getattr(auglag, 'cup_alpha', 1.0))

    # ── Dual update + J_c caching ─────────────────────────────────────────────

    def _update(self) -> None:
        """Cache J_c, then delegate to PPOLag._update() (which re-reads it for λ)."""
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        self._ep_cost = float(Jc) if (Jc == Jc) else 0.0   # NaN guard
        super()._update()

    # ── (I) Augmented Lagrangian ──────────────────────────────────────────────

    def _compute_adv_surrogate(
        self,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> torch.Tensor:
        r"""Combined advantage with augmented effective penalty λ_eff.

        Enabled:
            λ_eff = λ + ρ · max(0, (J_c − d) / d)
            combined = (adv_r − λ_eff · adv_c) / (1 + λ_eff)

        Disabled (auglag_cfgs.enabled = false):
            combined = (adv_r − λ · adv_c) / (1 + λ)    [identical to PPOLag]
        """
        penalty = self._lagrange.lagrangian_multiplier.item()
        if self._auglag_enabled:
            d       = float(self._lagrange.cost_limit)
            penalty = penalty + self._rho * max(0.0, (self._ep_cost - d) / max(d, 1e-8))
        return (adv_r - penalty * adv_c) / (1 + penalty)

    # ── (II) Conservative Update Policy clip ─────────────────────────────────

    def _loss_pi(
        self,
        obs:  torch.Tensor,
        act:  torch.Tensor,
        logp: torch.Tensor,
        adv:  torch.Tensor,
    ) -> torch.Tensor:
        r"""PPO actor loss with safety-adaptive clip radius ε_eff.

        Enabled:
            ε_eff = ε · exp(−α · max(0, (J_c − d) / d)),  floor = 0.1·ε

        Disabled (auglag_cfgs.enabled = false):
            ε_eff = ε    [identical to PPO._loss_pi — no change at all]

        All logger keys are identical to PPO._loss_pi.
        """
        distribution = self._actor_critic.actor(obs)
        logp_        = self._actor_critic.actor.log_prob(act)
        std          = self._actor_critic.actor.std
        ratio        = torch.exp(logp_ - logp)

        base_clip = float(self._cfgs.algo_cfgs.clip)
        if self._auglag_enabled:
            d = float(self._lagrange.cost_limit)
            # Proportional ("safety-ratio") CUP clip:
            #   ε_eff = ε · d / (d + α · max(0, J_c − d))
            #
            # When J_c = d:  ε_eff = ε         (safe — no change)
            # When J_c = 2d: ε_eff = ε/(1+α)   (cup_alpha=1 → ε/2)
            # When J_c = 5d: ε_eff = ε/(1+4α)  (cup_alpha=1 → ε/5)
            #
            # Unlike the previous exp(−α·violation_norm) formula, this never
            # collapses to near-zero for large violations (SafetyPointGoal2 sees
            # violations of 4–5× the limit in early epochs, which drove
            # exp(−4.3)≈0.013 → floor 0.02, paralysing the policy entirely).
            violation = max(0.0, self._ep_cost - d)
            clip_eps  = base_clip * d / (d + self._cup_alpha * violation)
            clip_eps  = max(clip_eps, base_clip * 0.1)   # floor: 10 % of base ε
        else:
            clip_eps = base_clip

        ratio_cliped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        loss  = -torch.min(ratio * adv, ratio_cliped * adv).mean()
        loss -= self._cfgs.algo_cfgs.entropy_coef * distribution.entropy().mean()

        self._logger.store(
            {
                'Train/Entropy':     distribution.entropy().mean().item(),
                'Train/PolicyRatio': ratio,
                'Train/PolicyStd':   std,
                'Loss/Loss_pi':      loss.mean().item(),
            },
        )
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Register PPOAugLag with OmniSafe's infrastructure
#
# OmniSafe's algo_wrapper validates the algorithm name against ALGORITHMS['all']
# (a frozen MappingProxyType built at import time) and loads a default YAML config
# from omnisafe/configs/on-policy/{algo}.yaml.  PPOAugLag is a strict extension of
# PPOLag, so we:
#   (a) patch algo_wrapper's *local* ALGORITHMS / ALGORITHM2TYPE references
#       (these are imported names in algo_wrapper's own namespace, not the frozen
#       omnisafe.algorithms module globals — so they can be reassigned freely)
#   (b) redirect the YAML loader to return PPOLag's config for PPOAugLag
#   (c) strip auglag_cfgs from custom_cfgs before passing to omnisafe.Agent
#       (it is not in PPOLag's schema and would fail recursive_check_config)
#       then inject it into the algorithm instance afterward
#
# No installed OmniSafe files are modified.
# ─────────────────────────────────────────────────────────────────────────────

def _register_ppoauglag() -> None:
    """One-time idempotent patches to OmniSafe's algo_wrapper namespace."""
    import omnisafe.algorithms.algo_wrapper as _aw

    # (a) Add PPOAugLag to ALGORITHMS / ALGORITHM2TYPE in algo_wrapper's namespace
    if 'PPOAugLag' not in _aw.ALGORITHM2TYPE:
        new_a2t = dict(_aw.ALGORITHM2TYPE)
        new_a2t['PPOAugLag'] = 'on-policy'
        _aw.ALGORITHM2TYPE = MappingProxyType(new_a2t)

        new_alg = dict(_aw.ALGORITHMS)
        new_alg['on-policy'] = new_alg['on-policy'] + ('PPOAugLag',)
        new_alg['all']       = new_alg.get('all', ()) + ('PPOAugLag',)
        _aw.ALGORITHMS = MappingProxyType(new_alg)

    # (b) Redirect YAML config loader: PPOAugLag uses PPOLag's default schema
    orig_loader = _aw.get_default_kwargs_yaml
    if not getattr(orig_loader, '_ppoauglag_patched', False):
        def _loader(algo: str, env_id: str, algo_type: str):
            if algo == 'PPOAugLag':
                return orig_loader('PPOLag', env_id, algo_type)
            return orig_loader(algo, env_id, algo_type)
        _loader._ppoauglag_patched = True
        _aw.get_default_kwargs_yaml = _loader


def create_agent(env_id: str, cfg: dict) -> omnisafe.Agent:
    """Create a PPOAugLag agent with full auglag config injected.

    Pops ``auglag_cfgs`` from ``cfg`` before passing to omnisafe.Agent (to avoid
    the recursive_check_config rejection), then writes the auglag params directly
    onto the algorithm instance after creation.
    """
    _register_ppoauglag()

    # Strip auglag_cfgs — not in PPOLag's schema, would fail config validation
    auglag_data = cfg.pop('auglag_cfgs', {})
    if not isinstance(auglag_data, dict):
        # OmniSafe may have already converted it to a Config object
        auglag_data = {
            'enabled':   getattr(auglag_data, 'enabled',   True),
            'rho':       getattr(auglag_data, 'rho',       1.0),
            'cup_alpha': getattr(auglag_data, 'cup_alpha', 1.0),
        }

    agent = omnisafe.Agent('PPOAugLag', env_id, custom_cfgs=cfg)

    # Inject auglag params into the algorithm instance (overrides _init defaults)
    alg = agent.agent
    alg._auglag_enabled = bool(auglag_data.get('enabled',   True))
    alg._rho             = float(auglag_data.get('rho',       1.0))
    alg._cup_alpha       = float(auglag_data.get('cup_alpha', 1.0))

    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Train PPOAugLag (PPO-Lag + Augmented Lagrangian + CUP).'
    )
    parser.add_argument(
        '--config', default='configs/ppoauglag/config.yaml',
        help='Path to PPOAugLag YAML config.',
    )
    parser.add_argument('--env_id', default=None, help='Override env_id in config.')
    parser.add_argument('--seed',   type=int, default=None, help='Override seed.')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    env_id = args.env_id if args.env_id is not None else cfg.pop('env_id')
    seed   = args.seed   if args.seed   is not None else cfg.pop('seed', 0)
    cfg['seed'] = seed

    agent = create_agent(env_id, cfg)
    agent.learn()


# python scripts/train_ppoauglag.py --config configs/ppoauglag/config.yaml \
#     --env_id SafetyPointGoal2-v0 --seed 0

if __name__ == '__main__':
    main()
