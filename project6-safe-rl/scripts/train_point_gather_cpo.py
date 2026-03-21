"""Launch a full CPO Point Gather training run with the paper-style config."""

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


def register_with_omnisafe() -> None:
    """Make the custom Safety-Gymnasium env ids visible to OmniSafe."""
    for env_id in (
        'SafetyPointGather0-v0',
        'SafetyPointGather1-v0',
        'SafetyPointGather2-v0',
    ):
        if env_id not in SafetyGymnasiumEnv._support_envs:
            SafetyGymnasiumEnv._support_envs.append(env_id)


def build_train_cfg(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    """Load the paper-style config and apply any requested runtime overrides."""
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / 'configs' / 'cpo' / 'config_pointgather.yaml')

    algo = str(config['algo'])
    env_id = str(config['env_id'])

    train_cfgs = dict(config.get('train_cfgs', {}))
    train_cfgs.update(
        {
            'device': args.device,
            'torch_threads': args.torch_threads,
        },
    )
    if args.vector_env_nums is not None:
        train_cfgs['vector_env_nums'] = args.vector_env_nums
    if args.total_steps is not None:
        train_cfgs['total_steps'] = args.total_steps

    algo_cfgs = dict(config.get('algo_cfgs', {}))
    if args.steps_per_epoch is not None:
        algo_cfgs['steps_per_epoch'] = args.steps_per_epoch
        algo_cfgs['batch_size'] = args.steps_per_epoch
    if args.update_iters is not None:
        algo_cfgs['update_iters'] = args.update_iters

    logger_cfgs = dict(config.get('logger_cfgs', {}))
    logger_cfgs.update(
        {
            'use_wandb': False,
            'use_tensorboard': args.tensorboard,
            'save_model_freq': args.save_model_freq,
            'log_dir': str(args.log_dir or (repo_root / 'results' / 'point_gather_cpo')),
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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for a full training run."""
    parser = argparse.ArgumentParser(
        description='Run a full OmniSafe CPO training job on Point Gather.',
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
        '--vector-env-nums',
        type=int,
        default=None,
        help='Override the config value for vectorized environments.',
    )
    parser.add_argument(
        '--steps-per-epoch',
        type=int,
        default=None,
        help='Override the config value for steps collected per epoch.',
    )
    parser.add_argument(
        '--total-steps',
        type=int,
        default=None,
        help='Override the config value for total environment steps.',
    )
    parser.add_argument(
        '--update-iters',
        type=int,
        default=None,
        help='Override the config value for update iterations per epoch.',
    )
    parser.add_argument(
        '--save-model-freq',
        type=int,
        default=100,
        help='Save model every N epochs.',
    )
    parser.add_argument(
        '--log-dir',
        type=Path,
        default=None,
        help='Base log directory. Defaults to results/point_gather_cpo.',
    )
    parser.add_argument(
        '--tensorboard',
        action='store_true',
        help='Enable tensorboard logging.',
    )
    return parser.parse_args()


def main() -> None:
    """Register the env and run a full Point Gather CPO training job."""
    try:
        mp.set_start_method('fork')
    except RuntimeError:
        pass

    args = parse_args()
    register_point_gather_environments()
    register_with_omnisafe()
    algo, env_id, custom_cfgs = build_train_cfg(args)

    train_cfgs = custom_cfgs['train_cfgs']
    algo_cfgs = custom_cfgs['algo_cfgs']
    logger_cfgs = custom_cfgs['logger_cfgs']

    print(
        'Starting full Point Gather training: '
        f'algo={algo} env={env_id} seed={args.seed} '
        f'total_steps={train_cfgs["total_steps"]} '
        f'steps_per_epoch={algo_cfgs["steps_per_epoch"]} '
        f'vector_env_nums={train_cfgs["vector_env_nums"]}',
    )
    print(f'Logs will be written under: {logger_cfgs["log_dir"]}')

    agent = omnisafe.Agent(algo=algo, env_id=env_id, custom_cfgs=custom_cfgs)
    ep_ret, ep_cost, ep_len = agent.learn()

    print('Training completed.')
    print(f'FinalEpRet={ep_ret:.4f}')
    print(f'FinalEpCost={ep_cost:.4f}')
    print(f'FinalEpLen={ep_len:.4f}')
    print(f'LogDir={agent.agent.logger.log_dir}')


if __name__ == '__main__':
    main()
