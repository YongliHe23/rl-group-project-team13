"""Short smoke-test training run for the Point Gather environment."""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Any

import omnisafe
import yaml

from custom_env.point_gather.register_env import register_point_gather_environments
from omnisafe.envs.safety_gymnasium_env import SafetyGymnasiumEnv


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dictionary."""
    with config_path.open('r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f'Expected dict config in {config_path}, got {type(data).__name__}')
    return data


def build_smoke_cfg(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    """Load the CPO config and override it for a short training smoke test."""
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / 'configs' / 'cpo' / 'config_pointgather.yaml')

    algo = str(config['algo'])
    env_id = str(config['env_id'])
    total_steps = args.steps_per_epoch * args.epochs

    train_cfgs = dict(config.get('train_cfgs', {}))
    train_cfgs.update(
        {
            'device': args.device,
            'total_steps': total_steps,
            'vector_env_nums': args.vector_env_nums,
            'torch_threads': args.torch_threads,
            'parallel': 1,
        },
    )

    algo_cfgs = dict(config.get('algo_cfgs', {}))
    algo_cfgs.update(
        {
            'steps_per_epoch': args.steps_per_epoch,
            'batch_size': args.steps_per_epoch,
            'update_iters': 1,
        },
    )

    logger_cfgs = dict(config.get('logger_cfgs', {}))
    logger_cfgs.update(
        {
            'use_wandb': False,
            'use_tensorboard': args.tensorboard,
            'save_model_freq': max(args.epochs, 1),
            'log_dir': str(repo_root / 'results' / 'smoke_tests'),
        },
    )

    custom_cfgs = dict(config)
    custom_cfgs.update(
        {
            'seed': args.seed,
            'train_cfgs': train_cfgs,
            'algo_cfgs': algo_cfgs,
            'logger_cfgs': logger_cfgs,
        },
    )
    return algo, env_id, custom_cfgs


def register_with_omnisafe() -> None:
    """Make the custom Safety-Gymnasium env ids visible to OmniSafe."""
    for env_id in (
        'SafetyPointGather0-v0',
        'SafetyPointGather1-v0',
        'SafetyPointGather2-v0',
    ):
        if env_id not in SafetyGymnasiumEnv._support_envs:
            SafetyGymnasiumEnv._support_envs.append(env_id)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Run a short OmniSafe training smoke test on Point Gather.',
    )
    parser.add_argument('--epochs', type=int, default=2, help='Number of short training epochs.')
    parser.add_argument(
        '--steps-per-epoch',
        type=int,
        default=1000,
        help='Environment steps collected per epoch.',
    )
    parser.add_argument(
        '--vector-env-nums',
        type=int,
        default=1,
        help='Number of vectorized environments for the smoke test.',
    )
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    parser.add_argument('--device', default='cpu', help='Training device, e.g. cpu or cuda:0.')
    parser.add_argument(
        '--torch-threads',
        type=int,
        default=1,
        help='Torch CPU thread count.',
    )
    parser.add_argument(
        '--tensorboard',
        action='store_true',
        help='Enable tensorboard logging for the smoke test.',
    )
    return parser.parse_args()


def main() -> None:
    """Register the env and run a short OmniSafe training job."""
    try:
        mp.set_start_method('fork')
    except RuntimeError:
        pass

    args = parse_args()
    register_point_gather_environments()
    register_with_omnisafe()
    algo, env_id, custom_cfgs = build_smoke_cfg(args)

    print(
        f'Starting smoke test: algo={algo} env={env_id} '
        f'epochs={args.epochs} steps_per_epoch={args.steps_per_epoch}',
    )

    agent = omnisafe.Agent(algo=algo, env_id=env_id, custom_cfgs=custom_cfgs)
    ep_ret, ep_cost, ep_len = agent.learn()

    print('Smoke test completed.')
    print(f'FinalEpRet={ep_ret:.4f}')
    print(f'FinalEpCost={ep_cost:.4f}')
    print(f'FinalEpLen={ep_len:.4f}')
    print(f'LogDir={agent.agent.logger.log_dir}')


if __name__ == '__main__':
    main()
