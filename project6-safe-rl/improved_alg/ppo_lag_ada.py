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
        upper = float(self._lagrange.lagrangian_upper_bound)
        value = max(0.0, min(float(value), upper))
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
        cost_limit = float(getattr(cfg, "cost_limit", 25.0))

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