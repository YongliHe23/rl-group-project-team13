"""Run an OmniSafe experiment grid for Point Gather."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any

import yaml
from custom_env.point_gather.register_env import register_point_gather_environments
from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.envs.safety_gymnasium_env import SafetyGymnasiumEnv
from omnisafe.utils.exp_grid_tools import train


POINT_GATHER_ENVS = (
    'SafetyPointGather0-v0',
    'SafetyPointGather1-v0',
    'SafetyPointGather2-v0',
)

POINT_GATHER_ALGOS = ('CPO', 'FOCOPS', 'PPOLag', 'TRPOLag')
CONFIG_PATHS = {
    'CPO': 'configs/cpo/config_point_gather.yaml',
    'FOCOPS': 'configs/focops/config.yaml',
    'PPOLag': 'configs/ppo_lagrangian/config.yaml',
    'TRPOLag': 'configs/ppo_lagrangian/config.yaml',
}
DEFAULT_ENV_ID = 'SafetyPointGather1-v0'
DEFAULT_SEEDS = [0]
DEFAULT_PARENT_DIR = Path(__file__).resolve().parents[1] / 'results' / 'point_gather_grid'
DEFAULT_NUM_POOL = 1
DEFAULT_COMPARE_NUM = 4
DEFAULT_ANALYZE = True


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dictionary."""
    with config_path.open('r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f'Expected dict config in {config_path}, got {type(data).__name__}')
    return data


def register_with_omnisafe() -> None:
    """Make the custom Safety-Gymnasium env ids visible to OmniSafe."""
    for env_id in POINT_GATHER_ENVS:
        if env_id not in SafetyGymnasiumEnv._support_envs:
            SafetyGymnasiumEnv._support_envs.append(env_id)


def build_grid() -> ExperimentGrid:
    """Construct the experiment grid."""
    repo_root = Path(__file__).resolve().parents[1]
    cpo_config = load_config(repo_root / CONFIG_PATHS['CPO'])
    focops_config = load_config(repo_root / CONFIG_PATHS['FOCOPS'])
    ppo_lag_config = load_config(repo_root / CONFIG_PATHS['PPOLag'])

    eg = ExperimentGrid(exp_name='PointGather_Compare')
    eg.add('algo', list(POINT_GATHER_ALGOS), in_name=True)
    eg.add('env_id', [DEFAULT_ENV_ID], in_name=True)
    eg.add('seed', DEFAULT_SEEDS, in_name=True)
    eg.add('logger_cfgs:use_wandb', [False])
    eg.add('logger_cfgs:use_tensorboard', [False])
    eg.add('train_cfgs:device', ['cpu'])
    eg.add(
        'train_cfgs:vector_env_nums',
        [
            int(cpo_config['train_cfgs']['vector_env_nums']),
            int(focops_config.get('train_cfgs', {}).get('vector_env_nums', 1)),
            int(ppo_lag_config.get('train_cfgs', {}).get('vector_env_nums', 1)),
            int(ppo_lag_config.get('train_cfgs', {}).get('vector_env_nums', 1)),
        ],
    )
    eg.add(
        'train_cfgs:torch_threads',
        [
            int(cpo_config['train_cfgs']['torch_threads']),
            int(focops_config.get('train_cfgs', {}).get('torch_threads', 1)),
            int(ppo_lag_config.get('train_cfgs', {}).get('torch_threads', 1)),
            int(ppo_lag_config.get('train_cfgs', {}).get('torch_threads', 1)),
        ],
    )
    eg.add(
        'train_cfgs:total_steps',
        [
            int(cpo_config['train_cfgs']['total_steps']),
            int(focops_config.get('train_cfgs', {}).get('total_steps', 10000000)),
            int(ppo_lag_config.get('train_cfgs', {}).get('total_steps', 10000000)),
            int(ppo_lag_config.get('train_cfgs', {}).get('total_steps', 10000000)),
        ],
    )
    eg.add(
        'algo_cfgs:steps_per_epoch',
        [
            int(cpo_config['algo_cfgs']['steps_per_epoch']),
            int(focops_config.get('algo_cfgs', {}).get('steps_per_epoch', 20000)),
            int(ppo_lag_config.get('algo_cfgs', {}).get('steps_per_epoch', 20000)),
            int(ppo_lag_config.get('algo_cfgs', {}).get('steps_per_epoch', 20000)),
        ],
    )
    eg.add(
        'algo_cfgs:batch_size',
        [
            int(cpo_config['algo_cfgs']['batch_size']),
            int(focops_config.get('algo_cfgs', {}).get('batch_size', 128)),
            int(ppo_lag_config.get('algo_cfgs', {}).get('batch_size', 128)),
            int(ppo_lag_config.get('algo_cfgs', {}).get('batch_size', 128)),
        ],
    )
    eg.add(
        'algo_cfgs:cost_limit',
        [
            float(cpo_config['algo_cfgs']['cost_limit']),
            float(focops_config.get('algo_cfgs', {}).get('cost_limit', 25)),
            float(ppo_lag_config.get('algo_cfgs', {}).get('cost_limit', 25)),
            float(ppo_lag_config.get('algo_cfgs', {}).get('cost_limit', 25)),
        ],
    )
    return eg


def main() -> None:
    """Register the env and run the experiment grid."""
    try:
        mp.set_start_method('fork')
    except RuntimeError:
        pass

    register_point_gather_environments()
    register_with_omnisafe()

    eg = build_grid()
    parent_dir = str(DEFAULT_PARENT_DIR)

    print(
        'Running Point Gather grid: '
        f'algos={",".join(POINT_GATHER_ALGOS)} env={DEFAULT_ENV_ID} '
        f'seeds={",".join(str(seed) for seed in DEFAULT_SEEDS)}',
    )

    eg.run(train, num_pool=DEFAULT_NUM_POOL, parent_dir=parent_dir, gpu_id=None)

    if DEFAULT_ANALYZE:
        eg.analyze(
            parameter='algo',
            values=list(POINT_GATHER_ALGOS),
            compare_num=DEFAULT_COMPARE_NUM,
            cost_limit=float(load_config(Path(__file__).resolve().parents[1] / CONFIG_PATHS['CPO'])['algo_cfgs']['cost_limit']),
        )


if __name__ == '__main__':
    main()
