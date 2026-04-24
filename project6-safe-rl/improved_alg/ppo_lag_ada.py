# my_algorithms/ppo_lag_adapt.py
# use adaptive Lagrange multiplier schedule instead of dual gradient descent
# this guarantees further exploration at the beginning and more stable cost constraint satisfaction at the end of training.

from __future__ import annotations

import math
import numpy as np

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.naive_lagrange.ppo_lag import PPOLag
from omnisafe.algorithms.on_policy.base.ppo import PPO


@registry.register
class PPOLagAdapt(PPOLag):
    """External PPO-Lag variant with configurable lambda schedule."""

    def _init(self) -> None:
        super()._init()
        self._adapt_step = 0
        self._ema_violation = 0.0

    # def _compute_progress(self) -> float:
    #     """Normalized training progress in [0, 1]."""
    #     # return the current training progress as a float in [0, 1]
    #     current_steps = self._logger.get_stats('TotalEnvSteps')[0]
    #     total_steps = self._cfgs.train_cfgs.total_steps

    #     p = current_steps / total_steps
    #     return min(max(p, 0.0), 1.0)
    
    def _compute_progress(self) -> float:
        """Normalized training progress in [0, 1]."""
        total_steps = float(self._cfgs.train_cfgs.total_steps)
        steps_per_epoch = float(self._cfgs.algo_cfgs.steps_per_epoch)

        current_steps = (self._adapt_step + 1) * steps_per_epoch
        p = current_steps / total_steps
        return min(max(p, 0.0), 1.0)


    def _set_lagrange_multiplier(self, value: float) -> None:
        """Manually set lambda with clipping."""
        # clip value to [0, upper_bound] to ensure stability
        upper = self._lagrange.lagrangian_upper_bound
        if upper is None:
            value = max(0.0, float(value))
        else:
            value = max(0.0, min(float(value), float(upper)))
        self._lagrange.lagrangian_multiplier.data.copy_(
            self._lagrange.lagrangian_multiplier.data.new_tensor(value),
        )


    def _get_current_lambda(self) -> float:
        return float(self._lagrange.lagrangian_multiplier.item())


    def _scheduled_lambda(self, progress: float, Jc: float) -> float:
        """Return lambda according to selected schedule."""
        cfg = getattr(self._cfgs, "lambda_schedule_cfgs", self._cfgs.lagrange_cfgs)
        sched = str(getattr(cfg, "lambda_schedule", "nonadaptive"))

        lam_min = float(getattr(cfg, "lambda_min", 0.4))
        lam_max = float(getattr(cfg, "lambda_max", 10.0))
        # cost_limit = float(getattr(cfg, "cost_limit", 25.0))
        cost_limit = float(getattr(self._cfgs.lagrange_cfgs, "cost_limit", 25.0))

        # 1. default OmniSafe behavior
        if sched == "nonadaptive":
            self._lagrange.update_lagrange_multiplier(Jc)
            return self._get_current_lambda()

        # 2. piecewise small -> large
        if sched == "piecewise":
            split = float(getattr(cfg, "lambda_piecewise_split", 0.5))
            return lam_min if progress < split else lam_max
        
        # 3. linearly increase lambda
        if sched == "linear_up":
            return lam_min + (lam_max - lam_min) * progress

        # 3. exponential increase
        if sched == "exp":
            alpha = float(getattr(cfg, "lambda_exp_alpha", 5.0))
            val = lam_min * math.exp(alpha * progress)
            return min(val, lam_max)

        # 4. sigmoid ramp-up
        if sched == "sigmoid":
            p0 = float(getattr(cfg, "lambda_p0", 0.35))
            kappa = float(getattr(cfg, "lambda_kappa", 10.0))
            sigma = 1.0 / (1.0 + math.exp(-kappa * (progress - p0)))
            return lam_min + (lam_max - lam_min) * sigma

        # 5. adaptive EMA on cost violation
        if sched == "adaptive_ema":
            eta = float(getattr(cfg, "lambda_eta", 0.01))
            beta = float(getattr(cfg, "lambda_ema_beta", 0.9))
            violation = float(Jc - cost_limit)
            self._ema_violation = beta * self._ema_violation + (1.0 - beta) * violation
            lam = self._get_current_lambda() + eta * self._ema_violation
            return min(max(lam, 0.0), lam_max)

        # 6. useful extra: scheduled baseline + adaptive correction
        if sched == "hybrid_sigmoid_adaptive":
            p0 = float(getattr(cfg, "lambda_p0", 0.35))
            kappa = float(getattr(cfg, "lambda_kappa", 10.0))
            eta = float(getattr(cfg, "lambda_eta", 0.01))
            beta = float(getattr(cfg, "lambda_ema_beta", 0.9))

            sigma = 1.0 / (1.0 + math.exp(-kappa * (progress - p0)))
            base = lam_min + (lam_max - lam_min) * sigma

            violation = float(Jc - cost_limit)
            self._ema_violation = beta * self._ema_violation + (1.0 - beta) * violation
            lam = base + eta * self._ema_violation
            return min(max(lam, 0.0), lam_max)
        
        # 7. late soft schedule with rate-limited adaptive correction
        #
        # hand picked parameters:
        # - larger p0, smaller kappa: based on plots schedules that pushed lambda up early/high often drove
        #   cost down, but they also hurt return once the penalty dominated the
        #   PPO update.
        # - lower lambda_max: the better PPOLag_ada runs tended to increase lambda
        #   more gradually and did not need the largest lambda values to become
        #   safer.

        # Also:
        # - Normalize violation by cost_limit.
        # - Add an EMA correction on violation: the plots only show epoch-level outcomes, so this
        #   adds a small amount of memory and lets lambda respond to sustained
        #   safety failure instead of following a fixed time schedule only.
        # - Add a deadband around zero violation: when EpCost hovers near the cost
        #   limit and small noisy violations should not trigger a new lambda
        #   increase.
        # - Rate-limit per-epoch lambda updates: prevents one noisy epoch from causing a large jump in
        #   penalty that could destabilize PPO.
        if sched in {"late_soft_adaptive", "rate_limited_hybrid"}:
            p0 = float(getattr(cfg, "lambda_p0", 0.7))
            kappa = float(getattr(cfg, "lambda_kappa", 5.0))
            eta = float(getattr(cfg, "lambda_eta", 0.5))
            beta = float(getattr(cfg, "lambda_ema_beta", 0.9))

            # Treat small violations as effectively zero to avoid chatter.
            deadband = float(getattr(cfg, "lambda_violation_deadband", 0.05)) 
            rate_up = float(getattr(cfg, "lambda_rate_up", 0.25))
            rate_down = float(getattr(cfg, "lambda_rate_down", 0.15))

            sigma = 1.0 / (1.0 + math.exp(-kappa * (progress - p0)))
            base = lam_min + (lam_max - lam_min) * sigma

            violation = float(Jc - cost_limit) / max(cost_limit, 1.0)
            if abs(violation) <= deadband:
                violation = 0.0
            else:
                violation -= math.copysign(deadband, violation)
            self._ema_violation = beta * self._ema_violation + (1.0 - beta) * violation
            if violation <= 0.0:
                self._ema_violation = min(self._ema_violation, 0.0)

            current = self._get_current_lambda()
            adaptive_target = current + eta * self._ema_violation
            target = max(base, adaptive_target)
            delta = target - current
            delta = min(max(delta, -rate_down), rate_up)
            lam = current + delta
            return min(max(lam, lam_min), lam_max)


        # 8. NEW: time-varying hybrid
        
        # if sched == "hybrid_timevarying_adaptive":
        #     # lam = w_{base} lam_{base} + eta_t * ema_violation
        #     # eta_t increases from eta_min to eta_max over time
        #     # ema_violation = beta * ema_violation + (1-beta) * violation
        #     # beta: momory term for violation, higher beta means more stable but less responsive lambda
            
        #     p0 = float(getattr(cfg, "lambda_p0", 0.35))
        #     kappa = float(getattr(cfg, "lambda_kappa", 10.0))
        #     beta = float(getattr(cfg, "lambda_ema_beta", 0.9))

        #     # eta increases over time
        #     eta_min = float(getattr(cfg, "lambda_eta_min", 0.0))
        #     eta_max = float(getattr(cfg, "lambda_eta_max", 0.05))
        #     eta_t = eta_min + (eta_max - eta_min) * progress

        #     # base weight decreases over time
        #     base_weight_min = float(getattr(cfg, "lambda_base_weight_min", 0.0))
        #     base_weight = base_weight_min + (1.0 - base_weight_min) * (1.0 - progress)

        #     sigma = 1.0 / (1.0 + math.exp(-kappa * (progress - p0)))
        #     base = lam_min + (lam_max - lam_min) * sigma

        #     violation = float(Jc - cost_limit)
        #     self._ema_violation = beta * self._ema_violation + (1.0 - beta) * violation

        #     lam = base_weight * base + eta_t * self._ema_violation
        #     return min(max(lam, 0.0), lam_max)

        # 8. smooth gated hybrid:
        # lam = g(Jc) * base + eta * ema_violation
        # if cost is already safe, let lambda decay gradually, not abruptly
        if sched == "hybrid_sigmoid_cost_adaptive":
            p0 = float(getattr(cfg, "lambda_p0", 0.7))
            kappa = float(getattr(cfg, "lambda_kappa", 10.0))

            # EMA memory for violation
            beta = float(getattr(cfg, "lambda_ema_beta", 0.9))
            eta = float(getattr(cfg, "lambda_eta", 0.05))

            # smooth gate parameters for the base term
            # gate ~ 0 when safely below threshold
            # gate ~ 1 when clearly above threshold
            safe_margin = float(getattr(cfg, "lambda_safe_margin", 0.0))
            safe_temp = float(getattr(cfg, "lambda_safe_temp", 5.0))

            # gradual lambda decay when already safe
            lambda_decay = float(getattr(cfg, "lambda_decay", 0.9))

            sigma = 1.0 / (1.0 + math.exp(-kappa * (progress - p0)))
            base = lam_min + (lam_max - lam_min) * sigma

            violation = float(Jc - cost_limit)
            self._ema_violation = beta * self._ema_violation + (1.0 - beta) * violation

            # smooth factor in front of lambda_base
            # centered at safe_margin; larger safe_temp => smoother transition
            gate = 1.0 / (1.0 + math.exp(-(violation - safe_margin) / safe_temp))

            prev_lam = self._get_current_lambda()
            candidate = gate * base + eta * self._ema_violation

            # if we are in/near the safe region, do not let lambda crash too fast
            if violation <= safe_margin:
                lam = max(candidate, lambda_decay * prev_lam)
            else:
                lam = candidate

            return min(max(lam, 0.0), lam_max)
        
        raise ValueError(f"Unknown lambda_schedule: {sched}")

    def _update(self) -> None:
        """Override only the lambda rule, then reuse PPO update."""
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'Metrics/EpCost is NaN.'

        progress = self._compute_progress()
        lam = self._scheduled_lambda(progress, Jc)
        self._set_lagrange_multiplier(lam)

        # Skip PPOLag._update() and directly call PPO._update()
        PPO._update(self)

        self._logger.store(
            {
            'Metrics/LagrangeMultiplier': self._get_current_lambda(),
            },
        )

        self._adapt_step += 1


# registry.register(PPOLagAdapt)

# print("After registration:", "PPOLagAdapt" in ALGORITHMS["all"])
