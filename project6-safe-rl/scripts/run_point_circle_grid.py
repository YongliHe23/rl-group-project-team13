"""Run a Point Circle comparison sweep using per-algorithm config files."""

from __future__ import annotations

import multiprocessing as mp
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from omnisafe.utils.exp_grid_tools import train


POINT_CIRCLE_ALGOS = ('CPO', 'FOCOPS', 'PPOLag', 'TRPOLag')
CONFIG_PATHS = {
    'CPO': 'configs/cpo/config_point_circle.yaml',
    'FOCOPS': 'configs/focops/config_point_circle.yaml',
    'PPOLag': 'configs/ppo_lag/config_point_circle.yaml',
    'TRPOLag': 'configs/trpo_lag/config_point_circle.yaml',
}
DEFAULT_ENV_ID = 'SafetyPointCircle1-v0'
DEFAULT_SEEDS = [1]
DEFAULT_PARENT_DIR = Path(__file__).resolve().parents[1] / 'results' / 'point_circle_grid'


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dictionary."""
    with config_path.open('r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f'Expected dict config in {config_path}, got {type(data).__name__}')
    return data


def build_run_cfg(algo: str, seed: int) -> tuple[str, str, dict[str, Any]]:
    """Load one algorithm config and resolve its log directory for the grid run."""
    repo_root = Path(__file__).resolve().parents[1]
    config = deepcopy(load_config(repo_root / CONFIG_PATHS[algo]))
    env_id = str(config.get('env_id', DEFAULT_ENV_ID))
    if env_id != DEFAULT_ENV_ID:
        raise ValueError(f'Expected {algo} config env_id to be {DEFAULT_ENV_ID}, got {env_id}')

    config['seed'] = seed
    logger_cfgs = dict(config.get('logger_cfgs', {}))
    logger_cfgs['use_wandb'] = False
    logger_cfgs['log_dir'] = str(
        (
            DEFAULT_PARENT_DIR
            / f'{algo}-{{{env_id}}}'
            / f'seed-{seed:03d}'
        ).resolve(),
    )
    config['logger_cfgs'] = logger_cfgs
    return algo, env_id, config


def run_all() -> None:
    """Run all Point Circle algorithm/seed combinations."""
    for algo in POINT_CIRCLE_ALGOS:
        for seed in DEFAULT_SEEDS:
            algo_name, env_id, custom_cfgs = build_run_cfg(algo, seed)
            print(
                'Running Point Circle experiment: '
                f'algo={algo_name} env={env_id} seed={seed} '
                f'log_dir={custom_cfgs["logger_cfgs"]["log_dir"]}',
            )
            train(
                exp_id=f'{algo_name}-{env_id}-seed-{seed:03d}',
                algo=algo_name,
                env_id=env_id,
                custom_cfgs=custom_cfgs,
            )


def main() -> None:
    """Launch the Point Circle multi-algorithm sweep."""
    try:
        mp.set_start_method('fork')
    except RuntimeError:
        pass

    print(
        'Running Point Circle grid: '
        f'algos={",".join(POINT_CIRCLE_ALGOS)} env={DEFAULT_ENV_ID} '
        f'seeds={",".join(str(seed) for seed in DEFAULT_SEEDS)}',
    )
    run_all()


if __name__ == '__main__':
    main()
