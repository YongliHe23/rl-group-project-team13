"""
Per-environment configuration for GCIVL (Goal-Conditioned Implicit V-Learning) on OGBench.

Source: https://github.com/seohongpark/ogbench/blob/master/impls/hyperparameters.sh
        https://github.com/seohongpark/ogbench/blob/master/impls/agents/gcivl.py
"""

# ── Base defaults (matches OGBench GCIVL agent defaults in agents/gcivl.py) ───
_BASE = dict(
    lr                  = 3e-4,
    batch_size          = 1024,
    # Network
    actor_hidden_dims   = (512, 512, 512),
    value_hidden_dims   = (512, 512, 512),
    layer_norm          = True,
    # IVL
    alpha               = 10.0,
    discount            = 0.99,
    tau                 = 0.005,
    expectile           = 0.9,
    gc_negative         = True,
    # Goal sampling — value network
    value_p_curgoal     = 0.2,
    value_p_trajgoal    = 0.5,
    value_p_randomgoal  = 0.3,
    value_geom_sample   = True,
    # Goal sampling — actor
    actor_p_curgoal     = 0.0,
    actor_p_trajgoal    = 1.0,
    actor_p_randomgoal  = 0.0,
    actor_geom_sample   = False,
    # Eval
    eval_temperature    = 0,
    eval_episodes       = 50,
    # Misc
    const_std           = True,
    discrete            = False,
    encoder             = None,
    p_aug               = 0.0,
    frame_stack         = None,
)

# ── Per-environment overrides ──────────────────────────────────────────────────
_ENV_CONFIGS = {
    # ── PointMaze ─────────────────────────────────────────────────────────────
    "pointmaze-medium-navigate-v0":   dict(),
    "pointmaze-large-navigate-v0":    dict(),
    "pointmaze-giant-navigate-v0":    dict(discount=0.995),
    "pointmaze-teleport-navigate-v0": dict(),
    "pointmaze-medium-stitch-v0":     dict(actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "pointmaze-large-stitch-v0":      dict(actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "pointmaze-giant-stitch-v0":      dict(discount=0.995,
                                           actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "pointmaze-teleport-stitch-v0":   dict(actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    # ── AntMaze ───────────────────────────────────────────────────────────────
    "antmaze-medium-navigate-v0":     dict(),
    "antmaze-large-navigate-v0":      dict(),
    "antmaze-giant-navigate-v0":      dict(discount=0.995),
    "antmaze-teleport-navigate-v0":   dict(),
    "antmaze-medium-stitch-v0":       dict(actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "antmaze-large-stitch-v0":        dict(actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "antmaze-giant-stitch-v0":        dict(discount=0.995,
                                           actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "antmaze-teleport-stitch-v0":     dict(actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "antmaze-medium-explore-v0":      dict(actor_p_trajgoal=0.0, actor_p_randomgoal=1.0),
    "antmaze-large-explore-v0":       dict(actor_p_trajgoal=0.0, actor_p_randomgoal=1.0),
    "antmaze-teleport-explore-v0":    dict(actor_p_trajgoal=0.0, actor_p_randomgoal=1.0),
    # ── HumanoidMaze ──────────────────────────────────────────────────────────
    "humanoidmaze-medium-navigate-v0": dict(discount=0.995),
    "humanoidmaze-large-navigate-v0":  dict(discount=0.995),
    "humanoidmaze-giant-navigate-v0":  dict(discount=0.995),
    "humanoidmaze-medium-stitch-v0":   dict(discount=0.995,
                                            actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "humanoidmaze-large-stitch-v0":    dict(discount=0.995,
                                            actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "humanoidmaze-giant-stitch-v0":    dict(discount=0.995,
                                            actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    # ── AntSoccer ─────────────────────────────────────────────────────────────
    "antsoccer-arena-navigate-v0":     dict(),
    "antsoccer-medium-navigate-v0":    dict(),
    "antsoccer-arena-stitch-v0":       dict(actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "antsoccer-medium-stitch-v0":      dict(actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    # ── Manipulation ──────────────────────────────────────────────────────────
    "cube-single-play-v0":    dict(),
    "cube-double-play-v0":    dict(),
    "cube-triple-play-v0":    dict(),
    "cube-quadruple-play-v0": dict(),
    "scene-play-v0":          dict(),
    "puzzle-3x3-play-v0":     dict(),
    "puzzle-4x4-play-v0":     dict(),
    "puzzle-4x5-play-v0":     dict(),
    "puzzle-4x6-play-v0":     dict(),
    # ── Powderworld ───────────────────────────────────────────────────────────
    "powderworld-easy-play-v0":   dict(alpha=3.0, batch_size=256,
                                       discrete=True, encoder='impala_small',
                                       eval_temperature=0.3),
    "powderworld-medium-play-v0": dict(alpha=3.0, batch_size=256,
                                       discrete=True, encoder='impala_small',
                                       eval_temperature=0.3),
    "powderworld-hard-play-v0":   dict(alpha=3.0, batch_size=256,
                                       discrete=True, encoder='impala_small',
                                       eval_temperature=0.3),
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
    if env_name not in _ENV_CONFIGS:
        supported = sorted(_ENV_CONFIGS.keys())
        raise KeyError(
            f"Unknown env '{env_name}' for GCIVL.\n"
            f"Supported environments:\n  " + "\n  ".join(supported)
        )
    cfg = dict(_BASE)
    cfg.update(_ENV_CONFIGS[env_name])
    return EnvConfig(**cfg)
