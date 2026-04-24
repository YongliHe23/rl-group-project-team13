"""
train_ppolagfuzrl.py — PPO-Lagrangian with Advanced Cost-GAE + Fuz-RL.

Extends OmniSafe's PPOLag using the same subclassing pattern as CPPOPID:

  (I)  Advanced Cost-GAE
       Uses a separate, lower GAE discount λ_c specifically for cost advantages:
           buffer lam_c = lam_c_fuzrl  (e.g. 0.7, vs the standard 0.97)
       Binary cost signals (0/1 per step) are sparser than reward signals, so
       a shorter bootstrap horizon reduces variance from noisy multi-step cost
       returns without sacrificing the quality of reward-side learning.

  (II) Augmented Lagrangian fast-response penalty
       Augments the base Lagrange multiplier λ with a violation-proportional term:
           λ_eff = λ  +  ρ · max(0, (J_c − d) / d)
       Adam-based λ updates respond only to the SIGN of violation (Δλ≈0.05/epoch).
       This augmentation responds proportionally to violation MAGNITUDE, matching
       CPPOPID's PID-driven escalation (Δλ_eff ≈ ρ·(J_c/d−1) per epoch).
       ρ = 0 → standard PPOLag (no augmentation).
       ρ = 1.5, J_c=130, d=25 → λ_eff = λ + 6.3 (comparable to CPPOPID at epoch 0).

  (III) Fuz-RL Choquet integral conservative cost advantage re-weighting
       Replaces adv_c with a distributionally robust "fuzzy" estimate:
           adv_c_fuzzy = adv_c + α · relu(adv_c − mean(adv_c))
       Upweights positive-deviation (high-cost) samples and leaves low-cost
       samples unchanged, producing a CVaR-like pessimistic cost estimate that
       is robust to distributional shift in the cost signal.  With the Aug-Lag
       term providing fast λ escalation, α=0.3 provides a gentle safety margin
       rather than needing to compensate for a weak multiplier.

Switch: fuzrl_cfgs.enabled = false → 100 % identical to OmniSafe PPOLag.

Three methods are overridden (minimum possible):
  _init                  — read fuzrl hyperparameters; patch lam_c in buffer
  _update                — cache J_c for Aug-Lag term
  _compute_adv_surrogate — inject λ_eff (Aug-Lag) and adv_c_fuzzy (Choquet)

Run:
    python scripts/train_ppolagfuzrl.py \\
        --config configs/ppolagfuzrl/config.yaml \\
        --env_id SafetyPointGoal2-v0 --seed 0
"""

from __future__ import annotations

import argparse
from types import MappingProxyType

import torch
import yaml

import omnisafe
from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.naive_lagrange.ppo_lag import PPOLag


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Define PPOLagFuzRL (same pattern as CPPOPID in cppo_pid.py)
# ─────────────────────────────────────────────────────────────────────────────

@registry.register
class PPOLagFuzRL(PPOLag):
    r"""PPO-Lagrangian + Advanced Cost-GAE + Fuz-RL Choquet conservative re-weighting.

    Overrides exactly three methods of PPOLag; all other behaviour (critic
    updates, Lagrange multiplier Adam step, KL early-stopping, logging,
    PPO clip) is inherited unchanged.

    When ``fuzrl_cfgs.enabled = false`` the class is 100 % equivalent to
    OmniSafe's ``PPOLag``.
    """

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init(self) -> None:
        """Read fuzrl hyperparameters, patch buffer lam_c, then call PPOLag._init."""
        fuzrl = getattr(self._cfgs, 'fuzrl_cfgs', None)
        if fuzrl is None:
            self._fuzrl_enabled:  bool  = True
            self._lam_c_fuzrl:    float = 0.7
            self._rho:            float = 1.5
            self._choquet_alpha:  float = 0.3
        else:
            self._fuzrl_enabled  = bool(getattr(fuzrl, 'enabled',       True))
            self._lam_c_fuzrl    = float(getattr(fuzrl, 'lam_c_fuzrl',  0.7))
            self._rho            = float(getattr(fuzrl, 'rho',           1.5))
            self._choquet_alpha  = float(getattr(fuzrl, 'choquet_alpha', 0.3))

        self._ep_cost: float = 0.0   # undiscounted J_c; refreshed in _update

        if self._fuzrl_enabled:
            # (I) Advanced Cost-GAE: temporarily lower lam_c so the on-policy
            #     buffer computes cost advantages with the shorter horizon.
            #     PolicyGradient._init() reads self._cfgs.algo_cfgs.lam_c when
            #     constructing the VectorOnPolicyBuffer, so we patch it now and
            #     restore afterward to keep the config consistent.
            orig_lam_c = float(self._cfgs.algo_cfgs.lam_c)
            self._cfgs.algo_cfgs.lam_c = self._lam_c_fuzrl
            super()._init()
            self._cfgs.algo_cfgs.lam_c = orig_lam_c
        else:
            super()._init()

    # ── (II) Aug-Lag J_c caching ─────────────────────────────────────────────

    def _update(self) -> None:
        """Cache J_c for the Aug-Lag term, then delegate to PPOLag._update()."""
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        self._ep_cost = float(Jc) if (Jc == Jc) else 0.0   # NaN guard
        super()._update()

    # ── (II+III) Augmented Lagrangian + Choquet re-weighting ─────────────────

    def _compute_adv_surrogate(
        self,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> torch.Tensor:
        r"""Combined advantage with Aug-Lag fast-response penalty and Choquet re-weighting.

        Enabled:
            λ_eff       = λ + ρ · max(0, (J_c − d) / d)           [Aug-Lag]
            adv_c_fuzzy = adv_c + α · relu(adv_c − mean(adv_c))   [Choquet]
            combined    = (adv_r − λ_eff · adv_c_fuzzy) / (1 + λ_eff)

        Disabled (fuzrl_cfgs.enabled = false):
            combined = (adv_r − λ · adv_c) / (1 + λ)    [identical to PPOLag]

        Aug-Lag augments the slow Adam multiplier λ with a term proportional to
        violation magnitude, matching CPPOPID's rapid λ escalation without a PID
        controller.  Choquet conservatively upweights high-cost-deviation samples.
        """
        penalty = self._lagrange.lagrangian_multiplier.item()
        if self._fuzrl_enabled:
            d       = float(self._lagrange.cost_limit)
            # (II) Augmented Lagrangian: add violation-proportional fast-response term
            penalty = penalty + self._rho * max(0.0, (self._ep_cost - d) / max(d, 1e-8))
            # (III) Fuz-RL Choquet: upweight above-mean cost-advantage samples
            adv_c_fuzzy = adv_c + self._choquet_alpha * torch.relu(adv_c - adv_c.mean())
        else:
            adv_c_fuzzy = adv_c
        return (adv_r - penalty * adv_c_fuzzy) / (1 + penalty)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Register PPOLagFuzRL with OmniSafe's infrastructure
#
# Same three-patch strategy as train_ppoauglag.py:
#   (a) patch algo_wrapper's *local* ALGORITHMS / ALGORITHM2TYPE references
#   (b) redirect the YAML loader to return PPOLag's config for PPOLagFuzRL
#   (c) strip fuzrl_cfgs from custom_cfgs before passing to omnisafe.Agent,
#       then inject it into the algorithm instance afterward
# ─────────────────────────────────────────────────────────────────────────────

def _register_ppolagfuzrl() -> None:
    """One-time idempotent patches to OmniSafe's algo_wrapper namespace."""
    import omnisafe.algorithms.algo_wrapper as _aw

    # (a) Add PPOLagFuzRL to ALGORITHMS / ALGORITHM2TYPE in algo_wrapper's namespace
    if 'PPOLagFuzRL' not in _aw.ALGORITHM2TYPE:
        new_a2t = dict(_aw.ALGORITHM2TYPE)
        new_a2t['PPOLagFuzRL'] = 'on-policy'
        _aw.ALGORITHM2TYPE = MappingProxyType(new_a2t)

        new_alg = dict(_aw.ALGORITHMS)
        new_alg['on-policy'] = new_alg['on-policy'] + ('PPOLagFuzRL',)
        new_alg['all']       = new_alg.get('all', ()) + ('PPOLagFuzRL',)
        _aw.ALGORITHMS = MappingProxyType(new_alg)

    # (b) Redirect YAML config loader: PPOLagFuzRL uses PPOLag's default schema
    orig_loader = _aw.get_default_kwargs_yaml
    if not getattr(orig_loader, '_ppolagfuzrl_patched', False):
        def _loader(algo: str, env_id: str, algo_type: str):
            if algo == 'PPOLagFuzRL':
                return orig_loader('PPOLag', env_id, algo_type)
            return orig_loader(algo, env_id, algo_type)
        _loader._ppolagfuzrl_patched = True
        _aw.get_default_kwargs_yaml = _loader


def create_agent(env_id: str, cfg: dict) -> omnisafe.Agent:
    """Create a PPOLagFuzRL agent with full fuzrl config injected.

    Pops ``fuzrl_cfgs`` from ``cfg`` before passing to omnisafe.Agent (to avoid
    the recursive_check_config rejection).

    Because the VectorOnPolicyBuffer is constructed inside ``_init()`` using
    ``self._cfgs.algo_cfgs.lam_c``, and ``fuzrl_cfgs`` is no longer visible to
    ``_init()`` (it was popped), we pre-patch ``cfg['algo_cfgs']['lam_c']`` to
    ``lam_c_fuzrl`` so the buffer is built with the correct GAE lambda.  The
    original value is restored in ``cfg`` after Agent construction.
    """
    _register_ppolagfuzrl()

    # Strip fuzrl_cfgs — not in PPOLag's schema, would fail config validation
    fuzrl_data = cfg.pop('fuzrl_cfgs', {})
    if not isinstance(fuzrl_data, dict):
        fuzrl_data = {
            'enabled':       getattr(fuzrl_data, 'enabled',       True),
            'lam_c_fuzrl':   getattr(fuzrl_data, 'lam_c_fuzrl',   0.7),
            'choquet_alpha': getattr(fuzrl_data, 'choquet_alpha',  0.5),
        }

    fuzrl_enabled = bool(fuzrl_data.get('enabled',       True))
    lam_c_fuzrl   = float(fuzrl_data.get('lam_c_fuzrl',  0.7))
    rho           = float(fuzrl_data.get('rho',           1.5))
    choquet_alpha = float(fuzrl_data.get('choquet_alpha', 0.3))

    # Pre-patch algo_cfgs.lam_c so the buffer in _init() uses lam_c_fuzrl.
    orig_lam_c = None
    if fuzrl_enabled and 'algo_cfgs' in cfg and isinstance(cfg['algo_cfgs'], dict):
        orig_lam_c = cfg['algo_cfgs'].get('lam_c', 0.97)
        cfg['algo_cfgs']['lam_c'] = lam_c_fuzrl

    agent = omnisafe.Agent('PPOLagFuzRL', env_id, custom_cfgs=cfg)

    # Restore lam_c in cfg (so the dict is not silently mutated for the caller)
    if orig_lam_c is not None:
        cfg['algo_cfgs']['lam_c'] = orig_lam_c

    # Inject fuzrl params into the algorithm instance
    alg = agent.agent
    alg._fuzrl_enabled  = fuzrl_enabled
    alg._lam_c_fuzrl    = lam_c_fuzrl
    alg._rho            = rho
    alg._choquet_alpha  = choquet_alpha

    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Train PPOLagFuzRL (PPO-Lag + Advanced Cost-GAE + Fuz-RL).'
    )
    parser.add_argument(
        '--config', default='configs/ppolagfuzrl/config.yaml',
        help='Path to PPOLagFuzRL YAML config.',
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


# python scripts/train_ppolagfuzrl.py --config configs/ppolagfuzrl/config.yaml \
#     --env_id SafetyPointGoal2-v0 --seed 0

if __name__ == '__main__':
    main()
