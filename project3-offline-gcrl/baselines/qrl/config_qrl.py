"""
Per-environment configuration for QRL (Quasimetric RL) on OGBench.

Source: https://github.com/seohongpark/ogbench/blob/master/impls/hyperparameters.sh
        https://github.com/seohongpark/ogbench/blob/master/impls/agents/qrl.py

Usage:
    from config_qrl import get_config
    cfg = get_config("antmaze-large-navigate-v0")
    # cfg.alpha, cfg.discount, cfg.train_steps, cfg.actor_p_trajgoal, ...

Flag construction:  env_name = f"{args.env}-{args.dsize}-{args.task}-v0"
  e.g.  --env antmaze --dsize large --task navigate  →  antmaze-large-navigate-v0
        --env cube     --dsize single --task play     →  cube-single-play-v0
        --env visual-antmaze --dsize medium --task navigate
                                                      →  visual-antmaze-medium-navigate-v0
  Note: scene-play-v0 and visual-scene-play-v0 have no size component; access them
        directly via get_config("scene-play-v0") if needed.

Visual environments are registered with correct hyperparameters but raise
NotImplementedError at runtime (requires impala_small CNN encoder).
"""

# ── Base defaults (matches OGBench QRL agent defaults) ────────────────────────
# Source: qrl.py get_config() + main.py FLAGS defaults
_BASE = dict(
    lr              = 3e-4,
    batch_size      = 1024,
    train_steps     = 1_000_000,
    eval_episodes   = 50,
    discount        = 0.99,         # used only if actor_geom_sample=True (it's False)
    alpha           = 0.003,        # DDPG+BC BC coefficient
    lam_eps         = 0.05,         # Lagrangian margin ε (fixed across all envs)
    # Value goal sampling: always random (p_randomgoal=1.0) — fixed for all envs
    value_p_trajgoal    = 0.0,
    value_p_randomgoal  = 1.0,
    value_geom_sample   = True,     # irrelevant since p_trajgoal=0, kept for consistency
    # Actor goal sampling: configurable per env
    actor_p_trajgoal    = 1.0,      # navigate: uniform traj
    actor_p_randomgoal  = 0.0,
    actor_geom_sample   = False,
)

# ── Per-environment overrides ──────────────────────────────────────────────────
# Only fields that differ from _BASE are listed.
# alpha:    0.0003 pointmaze | 0.003 antmaze navigate/stitch | 0.001 antmaze-explore
#           0.001 humanoidmaze | 0.003 antsoccer | 0.3 manipulation
# discount: 0.995 humanoidmaze (passed through; not used in QRL value loss)
# Stitch tasks: actor_p_trajgoal=0.5, actor_p_randomgoal=0.5
# Explore tasks: actor_p_trajgoal=0.0, actor_p_randomgoal=1.0
# Powderworld / visual envs: _unsupported=True

_ENV_CONFIGS = {
    # ── PointMaze ─────────────────────────────────────────────────────────────
    "pointmaze-medium-navigate-v0":   dict(alpha=0.0003),
    "pointmaze-large-navigate-v0":    dict(alpha=0.0003),
    "pointmaze-giant-navigate-v0":    dict(alpha=0.0003),
    "pointmaze-teleport-navigate-v0": dict(alpha=0.0003),
    "pointmaze-medium-stitch-v0":     dict(alpha=0.0003,
                                           actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "pointmaze-large-stitch-v0":      dict(alpha=0.0003,
                                           actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "pointmaze-giant-stitch-v0":      dict(alpha=0.0003,
                                           actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "pointmaze-teleport-stitch-v0":   dict(alpha=0.0003,
                                           actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    # ── AntMaze ───────────────────────────────────────────────────────────────
    "antmaze-medium-navigate-v0":     dict(alpha=0.003),
    "antmaze-large-navigate-v0":      dict(alpha=0.003),
    "antmaze-giant-navigate-v0":      dict(alpha=0.003),
    "antmaze-teleport-navigate-v0":   dict(alpha=0.003),
    "antmaze-medium-stitch-v0":       dict(alpha=0.003,
                                           actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "antmaze-large-stitch-v0":        dict(alpha=0.003,
                                           actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "antmaze-giant-stitch-v0":        dict(alpha=0.003,
                                           actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "antmaze-teleport-stitch-v0":     dict(alpha=0.003,
                                           actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "antmaze-medium-explore-v0":      dict(alpha=0.001,
                                           actor_p_trajgoal=0.0, actor_p_randomgoal=1.0),
    "antmaze-large-explore-v0":       dict(alpha=0.001,
                                           actor_p_trajgoal=0.0, actor_p_randomgoal=1.0),
    "antmaze-teleport-explore-v0":    dict(alpha=0.001,
                                           actor_p_trajgoal=0.0, actor_p_randomgoal=1.0),
    # ── HumanoidMaze ──────────────────────────────────────────────────────────
    "humanoidmaze-medium-navigate-v0": dict(alpha=0.001, discount=0.995),
    "humanoidmaze-large-navigate-v0":  dict(alpha=0.001, discount=0.995),
    "humanoidmaze-giant-navigate-v0":  dict(alpha=0.001, discount=0.995),
    "humanoidmaze-medium-stitch-v0":   dict(alpha=0.001, discount=0.995,
                                            actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "humanoidmaze-large-stitch-v0":    dict(alpha=0.001, discount=0.995,
                                            actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "humanoidmaze-giant-stitch-v0":    dict(alpha=0.001, discount=0.995,
                                            actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    # ── AntSoccer ─────────────────────────────────────────────────────────────
    "antsoccer-arena-navigate-v0":     dict(alpha=0.003),
    "antsoccer-medium-navigate-v0":    dict(alpha=0.003),
    "antsoccer-arena-stitch-v0":       dict(alpha=0.003,
                                            actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "antsoccer-medium-stitch-v0":      dict(alpha=0.003,
                                            actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    # ── Manipulation (cube / scene / puzzle) ──────────────────────────────────
    "cube-single-play-v0":    dict(alpha=0.3),
    "cube-double-play-v0":    dict(alpha=0.3),
    "cube-triple-play-v0":    dict(alpha=0.3),
    "cube-quadruple-play-v0": dict(alpha=0.3),
    "scene-play-v0":          dict(alpha=0.3),
    "puzzle-3x3-play-v0":     dict(alpha=0.3),
    "puzzle-4x4-play-v0":     dict(alpha=0.3),
    "puzzle-4x5-play-v0":     dict(alpha=0.3),
    "puzzle-4x6-play-v0":     dict(alpha=0.3),
    # ── Powderworld — NOT SUPPORTED (discrete actions + visual encoder + AWR) ─
    "powderworld-easy-play-v0":   dict(_unsupported=True,
                                       alpha=3.0, train_steps=500_000),
    "powderworld-medium-play-v0": dict(_unsupported=True,
                                       alpha=3.0, train_steps=500_000),
    "powderworld-hard-play-v0":   dict(_unsupported=True,
                                       alpha=3.0, train_steps=500_000),
    # ── Visual AntMaze — NOT SUPPORTED (requires impala_small visual encoder) ─
    # Hyperparameters mirror state-based antmaze counterparts.
    "visual-antmaze-medium-navigate-v0":   dict(_unsupported=True, alpha=0.003),
    "visual-antmaze-large-navigate-v0":    dict(_unsupported=True, alpha=0.003),
    "visual-antmaze-giant-navigate-v0":    dict(_unsupported=True, alpha=0.003),
    "visual-antmaze-teleport-navigate-v0": dict(_unsupported=True, alpha=0.003),
    "visual-antmaze-medium-stitch-v0":     dict(_unsupported=True, alpha=0.003,
                                                actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "visual-antmaze-large-stitch-v0":      dict(_unsupported=True, alpha=0.003,
                                                actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "visual-antmaze-giant-stitch-v0":      dict(_unsupported=True, alpha=0.003,
                                                actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "visual-antmaze-teleport-stitch-v0":   dict(_unsupported=True, alpha=0.003,
                                                actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "visual-antmaze-medium-explore-v0":    dict(_unsupported=True, alpha=0.001,
                                                actor_p_trajgoal=0.0, actor_p_randomgoal=1.0),
    "visual-antmaze-large-explore-v0":     dict(_unsupported=True, alpha=0.001,
                                                actor_p_trajgoal=0.0, actor_p_randomgoal=1.0),
    "visual-antmaze-teleport-explore-v0":  dict(_unsupported=True, alpha=0.001,
                                                actor_p_trajgoal=0.0, actor_p_randomgoal=1.0),
    # ── Visual HumanoidMaze — NOT SUPPORTED (requires impala_small visual encoder)
    "visual-humanoidmaze-medium-navigate-v0": dict(_unsupported=True,
                                                   alpha=0.001, discount=0.995),
    "visual-humanoidmaze-large-navigate-v0":  dict(_unsupported=True,
                                                   alpha=0.001, discount=0.995),
    "visual-humanoidmaze-giant-navigate-v0":  dict(_unsupported=True,
                                                   alpha=0.001, discount=0.995),
    "visual-humanoidmaze-medium-stitch-v0":   dict(_unsupported=True,
                                                   alpha=0.001, discount=0.995,
                                                   actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "visual-humanoidmaze-large-stitch-v0":    dict(_unsupported=True,
                                                   alpha=0.001, discount=0.995,
                                                   actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "visual-humanoidmaze-giant-stitch-v0":    dict(_unsupported=True,
                                                   alpha=0.001, discount=0.995,
                                                   actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    # ── Visual Manipulation — NOT SUPPORTED (requires impala_small visual encoder)
    "visual-cube-single-play-v0":    dict(_unsupported=True, alpha=0.3),
    "visual-cube-double-play-v0":    dict(_unsupported=True, alpha=0.3),
    "visual-cube-triple-play-v0":    dict(_unsupported=True, alpha=0.3),
    "visual-cube-quadruple-play-v0": dict(_unsupported=True, alpha=0.3),
    "visual-scene-play-v0":          dict(_unsupported=True, alpha=0.3),
    "visual-puzzle-3x3-play-v0":     dict(_unsupported=True, alpha=0.3),
    "visual-puzzle-4x4-play-v0":     dict(_unsupported=True, alpha=0.3),
    "visual-puzzle-4x5-play-v0":     dict(_unsupported=True, alpha=0.3),
    "visual-puzzle-4x6-play-v0":     dict(_unsupported=True, alpha=0.3),
}


class EnvConfig:
    """Simple attribute-access config namespace."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        items = ', '.join(f'{k}={v!r}' for k, v in sorted(self.__dict__.items()))
        return f'EnvConfig({items})'


def get_config(env_name: str) -> EnvConfig:
    """Return the merged QRL config for *env_name*.

    Raises NotImplementedError for unsupported envs (powderworld discrete,
    or visual envs requiring impala_small CNN encoder).
    Raises KeyError with a helpful message if the env is unknown.
    """
    if env_name not in _ENV_CONFIGS:
        supported = sorted(_ENV_CONFIGS.keys())
        raise KeyError(
            f"Unknown env '{env_name}' for QRL.\n"
            f"Supported environments:\n  " + "\n  ".join(supported)
        )
    cfg = dict(_BASE)
    overrides = dict(_ENV_CONFIGS[env_name])
    unsupported = overrides.pop('_unsupported', False)
    cfg.update(overrides)
    if unsupported:
        if env_name.startswith("visual-"):
            raise NotImplementedError(
                f"'{env_name}' requires a visual encoder (impala_small CNN) to process "
                f"image-based observations. This PyTorch QRL implementation supports "
                f"only state-based (low-dimensional) observations."
            )
        raise NotImplementedError(
            f"'{env_name}' requires discrete actions, a visual encoder (impala_small), "
            f"and AWR actor loss. This PyTorch QRL implementation supports only "
            f"continuous-action state-based environments with DDPG+BC actor."
        )
    return EnvConfig(**cfg)
