"""
Per-environment configuration for HIQL (Hierarchical Implicit Q-Learning) on OGBench.

Source: https://github.com/seohongpark/ogbench/blob/master/impls/hyperparameters.sh
        https://github.com/seohongpark/ogbench/blob/master/impls/agents/hiql.py
        https://github.com/seohongpark/ogbench/blob/master/impls/utils/datasets.py

Usage:
    from config_hiql import get_config
    cfg = get_config("antmaze-large-navigate-v0")
    # cfg.high_alpha, cfg.low_alpha, cfg.subgoal_steps, cfg.discount, ...

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

# ── Base defaults (matches OGBench HIQL agent defaults) ───────────────────────
# Source: hiql.py get_config() + main.py FLAGS defaults
_BASE = dict(
    lr              = 3e-4,
    batch_size      = 1024,
    train_steps     = 500_000,
    eval_episodes   = 50,
    discount        = 0.99,
    high_alpha      = 3.0,          # AWR temperature for high-level policy
    low_alpha       = 3.0,          # AWR temperature for low-level policy
    subgoal_steps   = 25,           # k: subgoal horizon (steps ahead); 25 for locomaze
    rep_dim         = 10,           # subgoal representation dimension (fixed)
    # Value goal sampling: fixed mix across all HIQL envs (HGCDataset defaults)
    value_p_curgoal     = 0.2,
    value_p_trajgoal    = 0.5,
    value_p_randomgoal  = 0.3,
    value_geom_sample   = True,
    # High-level actor goal sampling: configurable per env
    actor_p_trajgoal    = 1.0,      # navigate: uniform traj
    actor_p_randomgoal  = 0.0,
    actor_geom_sample   = False,
    low_actor_rep_grad  = False,    # True for pixel envs (allows grad flow through goal rep)
)

# ── Per-environment overrides ──────────────────────────────────────────────────
# Notes:
#   subgoal_steps:  25 = point/ant locomaze (default)
#                   25 = antsoccer (same scale as antmaze)
#                   10 = cube / scene / puzzle (denser manipulation)
#                  100 = humanoidmaze (slower humanoid requires longer horizon)
#                   10 = visual manipulation (same density as state-based)
#   high/low_alpha: 3.0 default; 10.0 for explore tasks (policy must explore randomly)
#   discount:       0.995 for humanoidmaze (slower agent, longer episodes)
#   Stitch tasks:   actor_p_trajgoal=0.5, actor_p_randomgoal=0.5
#   Explore tasks:  actor_p_trajgoal=0.0, actor_p_randomgoal=1.0
#   Powderworld:    discrete + visual + low_actor_rep_grad=True — NOT SUPPORTED.
#   Visual envs:    require impala_small CNN encoder — NOT SUPPORTED in this impl.

_ENV_CONFIGS = {
    # ── PointMaze ─────────────────────────────────────────────────────────────
    "pointmaze-medium-navigate-v0":   dict(high_alpha=3.0,  low_alpha=3.0),
    "pointmaze-large-navigate-v0":    dict(high_alpha=3.0,  low_alpha=3.0),
    "pointmaze-giant-navigate-v0":    dict(high_alpha=3.0,  low_alpha=3.0),
    "pointmaze-teleport-navigate-v0": dict(high_alpha=3.0,  low_alpha=3.0),
    "pointmaze-medium-stitch-v0":     dict(high_alpha=3.0,  low_alpha=3.0,
                                           actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "pointmaze-large-stitch-v0":      dict(high_alpha=3.0,  low_alpha=3.0,
                                           actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "pointmaze-giant-stitch-v0":      dict(high_alpha=3.0,  low_alpha=3.0,
                                           actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "pointmaze-teleport-stitch-v0":   dict(high_alpha=3.0,  low_alpha=3.0,
                                           actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    # ── AntMaze ───────────────────────────────────────────────────────────────
    "antmaze-medium-navigate-v0":     dict(high_alpha=3.0,  low_alpha=3.0),
    "antmaze-large-navigate-v0":      dict(high_alpha=3.0,  low_alpha=3.0),
    "antmaze-giant-navigate-v0":      dict(high_alpha=3.0,  low_alpha=3.0),
    "antmaze-teleport-navigate-v0":   dict(high_alpha=3.0,  low_alpha=3.0),
    "antmaze-medium-stitch-v0":       dict(high_alpha=3.0,  low_alpha=3.0,
                                           actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "antmaze-large-stitch-v0":        dict(high_alpha=3.0,  low_alpha=3.0,
                                           actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "antmaze-giant-stitch-v0":        dict(high_alpha=3.0,  low_alpha=3.0,
                                           actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "antmaze-teleport-stitch-v0":     dict(high_alpha=3.0,  low_alpha=3.0,
                                           actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "antmaze-medium-explore-v0":      dict(high_alpha=10.0, low_alpha=10.0,
                                           actor_p_trajgoal=0.0, actor_p_randomgoal=1.0),
    "antmaze-large-explore-v0":       dict(high_alpha=10.0, low_alpha=10.0,
                                           actor_p_trajgoal=0.0, actor_p_randomgoal=1.0),
    "antmaze-teleport-explore-v0":    dict(high_alpha=10.0, low_alpha=10.0,
                                           actor_p_trajgoal=0.0, actor_p_randomgoal=1.0),
    # ── HumanoidMaze ──────────────────────────────────────────────────────────
    "humanoidmaze-medium-navigate-v0": dict(high_alpha=3.0,  low_alpha=3.0,
                                            discount=0.995, subgoal_steps=100),
    "humanoidmaze-large-navigate-v0":  dict(high_alpha=3.0,  low_alpha=3.0,
                                            discount=0.995, subgoal_steps=100),
    "humanoidmaze-giant-navigate-v0":  dict(high_alpha=3.0,  low_alpha=3.0,
                                            discount=0.995, subgoal_steps=100),
    "humanoidmaze-medium-stitch-v0":   dict(high_alpha=3.0,  low_alpha=3.0,
                                            discount=0.995, subgoal_steps=100,
                                            actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "humanoidmaze-large-stitch-v0":    dict(high_alpha=3.0,  low_alpha=3.0,
                                            discount=0.995, subgoal_steps=100,
                                            actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "humanoidmaze-giant-stitch-v0":    dict(high_alpha=3.0,  low_alpha=3.0,
                                            discount=0.995, subgoal_steps=100,
                                            actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    # ── AntSoccer ─────────────────────────────────────────────────────────────
    "antsoccer-arena-navigate-v0":     dict(high_alpha=3.0, low_alpha=3.0),
    "antsoccer-medium-navigate-v0":    dict(high_alpha=3.0, low_alpha=3.0),
    "antsoccer-arena-stitch-v0":       dict(high_alpha=3.0, low_alpha=3.0,
                                            actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "antsoccer-medium-stitch-v0":      dict(high_alpha=3.0, low_alpha=3.0,
                                            actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    # ── Manipulation (cube / scene / puzzle) — subgoal_steps=10 ───────────────
    "cube-single-play-v0":    dict(high_alpha=3.0, low_alpha=3.0, subgoal_steps=10, train_steps=1000000),
    "cube-double-play-v0":    dict(high_alpha=3.0, low_alpha=3.0, subgoal_steps=10, train_steps=1000000),
    "cube-triple-play-v0":    dict(high_alpha=3.0, low_alpha=3.0, subgoal_steps=10, train_steps=1000000),
    "cube-quadruple-play-v0": dict(high_alpha=3.0, low_alpha=3.0, subgoal_steps=10, train_steps=1000000),
    "scene-play-v0":          dict(high_alpha=3.0, low_alpha=3.0, subgoal_steps=10, train_steps=1000000),
    "puzzle-3x3-play-v0":     dict(high_alpha=3.0, low_alpha=3.0, subgoal_steps=10, train_steps=1000000),
    "puzzle-4x4-play-v0":     dict(high_alpha=3.0, low_alpha=3.0, subgoal_steps=10, train_steps=1000000),
    "puzzle-4x5-play-v0":     dict(high_alpha=3.0, low_alpha=3.0, subgoal_steps=10, train_steps=1000000),
    "puzzle-4x6-play-v0":     dict(high_alpha=3.0, low_alpha=3.0, subgoal_steps=10, train_steps=1000000),
    # ── Powderworld — supported via --visual-enabled flag ─────────────────────
    "powderworld-easy-play-v0":   dict(batch_size=256, high_alpha=3.0, low_alpha=3.0,
                                       low_actor_rep_grad=True, subgoal_steps=10,
                                       train_steps=500_000),
    "powderworld-medium-play-v0": dict(batch_size=256, high_alpha=3.0, low_alpha=3.0,
                                       low_actor_rep_grad=True, subgoal_steps=10,
                                       train_steps=500_000),
    "powderworld-hard-play-v0":   dict(batch_size=256, high_alpha=3.0, low_alpha=3.0,
                                       low_actor_rep_grad=True, subgoal_steps=10,
                                       train_steps=500_000),
    # ── Visual AntMaze — NOT SUPPORTED (requires impala_small visual encoder) ─
    # subgoal_steps=25 (antmaze locomotion scale); hyperparams mirror state-based.
    "visual-antmaze-medium-navigate-v0":   dict(_unsupported=True,
                                                high_alpha=3.0, low_alpha=3.0),
    "visual-antmaze-large-navigate-v0":    dict(_unsupported=True,
                                                high_alpha=3.0, low_alpha=3.0),
    "visual-antmaze-giant-navigate-v0":    dict(_unsupported=True,
                                                high_alpha=3.0, low_alpha=3.0),
    "visual-antmaze-teleport-navigate-v0": dict(_unsupported=True,
                                                high_alpha=3.0, low_alpha=3.0),
    "visual-antmaze-medium-stitch-v0":     dict(_unsupported=True,
                                                high_alpha=3.0, low_alpha=3.0,
                                                actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "visual-antmaze-large-stitch-v0":      dict(_unsupported=True,
                                                high_alpha=3.0, low_alpha=3.0,
                                                actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "visual-antmaze-giant-stitch-v0":      dict(_unsupported=True,
                                                high_alpha=3.0, low_alpha=3.0,
                                                actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "visual-antmaze-teleport-stitch-v0":   dict(_unsupported=True,
                                                high_alpha=3.0, low_alpha=3.0,
                                                actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "visual-antmaze-medium-explore-v0":    dict(_unsupported=True,
                                                high_alpha=10.0, low_alpha=10.0,
                                                actor_p_trajgoal=0.0, actor_p_randomgoal=1.0),
    "visual-antmaze-large-explore-v0":     dict(_unsupported=True,
                                                high_alpha=10.0, low_alpha=10.0,
                                                actor_p_trajgoal=0.0, actor_p_randomgoal=1.0),
    "visual-antmaze-teleport-explore-v0":  dict(_unsupported=True,
                                                high_alpha=10.0, low_alpha=10.0,
                                                actor_p_trajgoal=0.0, actor_p_randomgoal=1.0),
    # ── Visual HumanoidMaze — NOT SUPPORTED (requires impala_small visual encoder)
    # subgoal_steps=100; discount=0.995; same as state humanoidmaze.
    "visual-humanoidmaze-medium-navigate-v0": dict(_unsupported=True,
                                                   high_alpha=3.0, low_alpha=3.0,
                                                   discount=0.995, subgoal_steps=100),
    "visual-humanoidmaze-large-navigate-v0":  dict(_unsupported=True,
                                                   high_alpha=3.0, low_alpha=3.0,
                                                   discount=0.995, subgoal_steps=100),
    "visual-humanoidmaze-giant-navigate-v0":  dict(_unsupported=True,
                                                   high_alpha=3.0, low_alpha=3.0,
                                                   discount=0.995, subgoal_steps=100),
    "visual-humanoidmaze-medium-stitch-v0":   dict(_unsupported=True,
                                                   high_alpha=3.0, low_alpha=3.0,
                                                   discount=0.995, subgoal_steps=100,
                                                   actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "visual-humanoidmaze-large-stitch-v0":    dict(_unsupported=True,
                                                   high_alpha=3.0, low_alpha=3.0,
                                                   discount=0.995, subgoal_steps=100,
                                                   actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    "visual-humanoidmaze-giant-stitch-v0":    dict(_unsupported=True,
                                                   high_alpha=3.0, low_alpha=3.0,
                                                   discount=0.995, subgoal_steps=100,
                                                   actor_p_trajgoal=0.5, actor_p_randomgoal=0.5),
    # ── Visual Manipulation — NOT SUPPORTED (requires impala_small visual encoder)
    # subgoal_steps=10 (same dense manipulation scale as state-based).
    "visual-cube-single-play-v0":    dict(_unsupported=True,
                                          high_alpha=3.0, low_alpha=3.0, subgoal_steps=10),
    "visual-cube-double-play-v0":    dict(_unsupported=True,
                                          high_alpha=3.0, low_alpha=3.0, subgoal_steps=10),
    "visual-cube-triple-play-v0":    dict(_unsupported=True,
                                          high_alpha=3.0, low_alpha=3.0, subgoal_steps=10),
    "visual-cube-quadruple-play-v0": dict(_unsupported=True,
                                          high_alpha=3.0, low_alpha=3.0, subgoal_steps=10),
    "visual-scene-play-v0":          dict(_unsupported=True,
                                          high_alpha=3.0, low_alpha=3.0, subgoal_steps=10),
    "visual-puzzle-3x3-play-v0":     dict(_unsupported=True,
                                          high_alpha=3.0, low_alpha=3.0, subgoal_steps=10),
    "visual-puzzle-4x4-play-v0":     dict(_unsupported=True,
                                          high_alpha=3.0, low_alpha=3.0, subgoal_steps=10),
    "visual-puzzle-4x5-play-v0":     dict(_unsupported=True,
                                          high_alpha=3.0, low_alpha=3.0, subgoal_steps=10),
    "visual-puzzle-4x6-play-v0":     dict(_unsupported=True,
                                          high_alpha=3.0, low_alpha=3.0, subgoal_steps=10),
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
    """Return the merged HIQL config for *env_name*.

    Raises NotImplementedError for unsupported envs (powderworld discrete,
    or visual envs requiring impala_small CNN encoder).
    Raises KeyError with a helpful message if the env is unknown.
    """
    if env_name not in _ENV_CONFIGS:
        supported = sorted(_ENV_CONFIGS.keys())
        raise KeyError(
            f"Unknown env '{env_name}' for HIQL.\n"
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
                f"image-based observations. This PyTorch HIQL implementation supports "
                f"only state-based (low-dimensional) observations."
            )
        raise NotImplementedError(
            f"'{env_name}' requires discrete actions, a visual encoder (impala_small), "
            f"and low_actor_rep_grad=True. This PyTorch HIQL implementation supports "
            f"only continuous-action state-based environments."
        )
    return EnvConfig(**cfg)
