"""Run an OmniSafe experiment grid for Point Gather."""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

from custom_env.point_gather.register_env import register_point_gather_environments
from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.envs.safety_gymnasium_env import SafetyGymnasiumEnv
from omnisafe.utils.exp_grid_tools import train


POINT_GATHER_ENVS = (
    'SafetyPointGather0-v0',
    'SafetyPointGather1-v0',
    'SafetyPointGather2-v0',
)

POINT_GATHER_ALGOS = ('CPO', 'FOCOPS', 'PPOLag')


def parse_seeds(seed_text: str) -> list[int]:
    """Parse a comma-separated seed list."""
    seeds = [int(part.strip()) for part in seed_text.split(',') if part.strip()]
    if not seeds:
        raise ValueError('At least one seed is required.')
    return seeds


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Run a Point Gather experiment grid comparing CPO, FOCOPS, and PPO-Lag.',
    )
    parser.add_argument(
        '--env-id',
        default='SafetyPointGather1-v0',
        choices=list(POINT_GATHER_ENVS),
        help='Point Gather environment to benchmark.',
    )
    parser.add_argument(
        '--seeds',
        default='0',
        help='Comma-separated seed list, for example "0,1,2".',
    )
    parser.add_argument(
        '--steps-per-epoch',
        type=int,
        default=50000,
        help='Environment steps collected per epoch.',
    )
    parser.add_argument(
        '--total-steps',
        type=int,
        default=5000000,
        help='Total environment steps per run.',
    )
    parser.add_argument(
        '--vector-env-nums',
        type=int,
        default=4,
        help='Number of vectorized environments per run.',
    )
    parser.add_argument(
        '--torch-threads',
        type=int,
        default=1,
        help='Torch CPU thread count.',
    )
    parser.add_argument(
        '--num-pool',
        type=int,
        default=1,
        help='How many experiments to run in parallel.',
    )
    parser.add_argument(
        '--parent-dir',
        type=Path,
        default=None,
        help='Directory where grid results should be written.',
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Generate OmniSafe comparison plots after training completes.',
    )
    parser.add_argument(
        '--compare-num',
        type=int,
        default=3,
        help='Maximum curves per analysis figure.',
    )
    return parser.parse_args()


def register_with_omnisafe() -> None:
    """Make the custom Safety-Gymnasium env ids visible to OmniSafe."""
    for env_id in POINT_GATHER_ENVS:
        if env_id not in SafetyGymnasiumEnv._support_envs:
            SafetyGymnasiumEnv._support_envs.append(env_id)


def build_grid(args: argparse.Namespace) -> ExperimentGrid:
    """Construct the experiment grid."""
    eg = ExperimentGrid(exp_name='PointGather_Compare')
    eg.add('algo', list(POINT_GATHER_ALGOS), in_name=True)
    eg.add('env_id', [args.env_id], in_name=True)
    eg.add('seed', parse_seeds(args.seeds), in_name=True)
    eg.add('logger_cfgs:use_wandb', [False])
    eg.add('logger_cfgs:use_tensorboard', [False])
    eg.add('train_cfgs:device', ['cpu'])
    eg.add('train_cfgs:vector_env_nums', [args.vector_env_nums])
    eg.add('train_cfgs:torch_threads', [args.torch_threads])
    eg.add('train_cfgs:total_steps', [args.total_steps])
    eg.add('algo_cfgs:steps_per_epoch', [args.steps_per_epoch])
    eg.add('algo_cfgs:batch_size', [args.steps_per_epoch])
    eg.add('algo_cfgs:cost_limit', [0.1])
    return eg


def main() -> None:
    """Register the env and run the experiment grid."""
    try:
        mp.set_start_method('fork')
    except RuntimeError:
        pass

    args = parse_args()
    register_point_gather_environments()
    register_with_omnisafe()

    eg = build_grid(args)
    parent_dir = str(args.parent_dir) if args.parent_dir is not None else None

    print(
        'Running Point Gather grid: '
        f'algos={",".join(POINT_GATHER_ALGOS)} env={args.env_id} seeds={args.seeds} '
        f'total_steps={args.total_steps} steps_per_epoch={args.steps_per_epoch}',
    )

    eg.run(train, num_pool=args.num_pool, parent_dir=parent_dir, gpu_id=None)

    if args.analyze:
        eg.analyze(parameter='algo', values=list(POINT_GATHER_ALGOS), compare_num=args.compare_num, cost_limit=0.1)


if __name__ == '__main__':
    main()
