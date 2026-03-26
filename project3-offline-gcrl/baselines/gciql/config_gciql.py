"""
Per-environment configuration for GCIQL (Goal-Conditioned Implicit Q-Learning) on OGBench.

Source: https://github.com/seohongpark/ogbench/blob/master/impls/hyperparameters.sh
        https://github.com/seohongpark/ogbench/blob/master/impls/agents/gciql.py

Usage:
    from config_gciql import get_config
    cfg = get_config("pointmaze-medium-navigate-v0")
    # cfg.alpha, cfg.discount, cfg.actor_p_trajgoal, ...

Flag construction:  env_name = f"{args.env}-{args.dsize}-{args.task}-v0"
  e.g.  --env pointmaze --dsize medium --task navigate  ->  pointmaze-medium-navigate-v0
        --env antmaze   --dsize large  --task stitch    ->  antmaze-large-stitch-v0
"""

# ── Base defaults (matches OGBench GCIQL agent defaults in agents/gciql.py) ───
_BASE = dict(
    lr                  = 3e-4,
    batch_size          = 1024,
    train_steps         = 1_000_000,
    eval_episodes       = 50,
    # Network
    actor_hidden_dims   = (512, 512, 512),
    value_hidden_dims   = (512, 512, 512),
    layer_norm          = True,
    # IQL
    actor_loss          = 'ddpgbc',   # 'ddpgbc' or 'awr'
    alpha               = 0.3,        # BC coefficient (ddpgbc) or AWR temperature
    discount            = 0.99,
    tau                 = 0.005,
    expectile           = 0.9,
    gc_negative         = True,       # reward = 0 if s==g else -1
    # Goal sampling — value network (fixed across all envs)
    value_p_curgoal     = 0.2,
    value_p_trajgoal    = 0.5,
    value_p_randomgoal  = 0.3,
    value_geom_sample   = True,
    # Goal sampling — actor (configurable per env)
    actor_p_curgoal     = 0.0,
    actor_p_trajgoal    = 1.0,        # navigate: uniform traj
    actor_p_randomgoal  = 0.0,
    actor_geom_sample   = False,
    # Misc
    const_std           = True,
    discrete            = False,
    encoder             = None,
    p_aug               = 0.0,
    frame_stack         = None,
)

# ── Per-environment overrides ──────────────────────────────────────────────────
# Only fields that differ from _BASE are listed.
#
# alpha values (from hyperparameters.sh):
#   pointmaze               : 0.003
#   antmaze navigate/stitch : 0.3   (same as base)
#   antmaze explore         : 0.01
#   humanoidmaze            : 0.1
#   antsoccer               : 0.1
# discount=0.995 for giant variants and humanoidmaze (slower/larger envs).
# stitch tasks  : actor_p_trajgoal=0.5, actor_p_randomgoal=0.5
# explore tasks : actor_p_trajgoal=0.0, actor_p_randomgoal=1.0
_ENV_CONFIGS = {
    # ── PointMaze ─────────────────────────────────────────────────────────────
    "pointmaze-medium-navigate-v0":   dict(alpha=0.003),
    "pointmaze-large-navigate-v0":    dict(alpha=0.003),
    "pointmaze-giant-navigate-v0":    dict(alpha=0.003, discount=0.995),
    "pointmaze-teleport-navigate-v0": dict(alpha=0.003),
    "pointmaze-medium-stitch-v0":     dict(alpha=0.003,
                                           actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "pointmaze-large-stitch-v0":      dict(alpha=0.003,
                                           actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "pointmaze-giant-stitch-v0":      dict(alpha=0.003, discount=0.995,
                                           actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "pointmaze-teleport-stitch-v0":   dict(alpha=0.003,
                                           actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    # ── AntMaze ───────────────────────────────────────────────────────────────
    "antmaze-medium-navigate-v0":     dict(alpha=0.3),
    "antmaze-large-navigate-v0":      dict(alpha=0.3),
    "antmaze-giant-navigate-v0":      dict(alpha=0.3, discount=0.995),
    "antmaze-teleport-navigate-v0":   dict(alpha=0.3),
    "antmaze-medium-stitch-v0":       dict(alpha=0.3,
                                           actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "antmaze-large-stitch-v0":        dict(alpha=0.3,
                                           actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "antmaze-giant-stitch-v0":        dict(alpha=0.3, discount=0.995,
                                           actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "antmaze-teleport-stitch-v0":     dict(alpha=0.3,
                                           actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "antmaze-medium-explore-v0":      dict(alpha=0.01,
                                           actor_p_trajgoal=0.0, actor_p_randomgoal=1.0),
    "antmaze-large-explore-v0":       dict(alpha=0.01,
                                           actor_p_trajgoal=0.0, actor_p_randomgoal=1.0),
    "antmaze-teleport-explore-v0":    dict(alpha=0.01,
                                           actor_p_trajgoal=0.0, actor_p_randomgoal=1.0),
    # ── HumanoidMaze ──────────────────────────────────────────────────────────
    "humanoidmaze-medium-navigate-v0": dict(alpha=0.1, discount=0.995),
    "humanoidmaze-large-navigate-v0":  dict(alpha=0.1, discount=0.995),
    "humanoidmaze-giant-navigate-v0":  dict(alpha=0.1, discount=0.995),
    "humanoidmaze-medium-stitch-v0":   dict(alpha=0.1, discount=0.995,
                                            actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "humanoidmaze-large-stitch-v0":    dict(alpha=0.1, discount=0.995,
                                            actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "humanoidmaze-giant-stitch-v0":    dict(alpha=0.1, discount=0.995,
                                            actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    # ── AntSoccer ─────────────────────────────────────────────────────────────
    "antsoccer-arena-navigate-v0":     dict(alpha=0.1),
    "antsoccer-medium-navigate-v0":    dict(alpha=0.1),
    "antsoccer-arena-stitch-v0":       dict(alpha=0.1,
                                            actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "antsoccer-medium-stitch-v0":      dict(alpha=0.1,
                                            actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    # ── Manipulation ──────────────────────────────────────────────────────────
    "cube-single-play-v0":    dict(alpha=0.3),
    "cube-double-play-v0":    dict(alpha=0.3),
    "cube-triple-play-v0":    dict(alpha=0.3),
    "cube-quadruple-play-v0": dict(alpha=0.3),
    "scene-play-v0":          dict(alpha=0.3),
    "puzzle-3x3-play-v0":     dict(alpha=0.3),
    "puzzle-4x4-play-v0":     dict(alpha=0.3),
    "puzzle-4x5-play-v0":     dict(alpha=0.3),
    "puzzle-4x6-play-v0":     dict(alpha=0.3),
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
    """Return the merged GCIQL config for *env_name*.

    Raises KeyError with a helpful message if the env is unknown.
    """
    if env_name not in _ENV_CONFIGS:
        supported = sorted(_ENV_CONFIGS.keys())
        raise KeyError(
            f"Unknown env '{env_name}' for GCIQL.\n"
            f"Supported environments:\n  " + "\n  ".join(supported)
        )
    cfg = dict(_BASE)
    cfg.update(_ENV_CONFIGS[env_name])
    return EnvConfig(**cfg)
