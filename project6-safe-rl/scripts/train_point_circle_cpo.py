"""Launch a full CPO Point Circle training run from the YAML config."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any

import omnisafe
import yaml


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dictionary."""
    with config_path.open('r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f'Expected dict config in {config_path}, got {type(data).__name__}')
    return data


def build_train_cfg() -> tuple[str, str, dict[str, Any]]:
    """Load the Point Circle CPO config directly from YAML."""
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / 'configs' / 'cpo' / 'config_point_circle.yaml')
    logger_cfgs = dict(config.get('logger_cfgs', {}))
    log_dir = logger_cfgs.get('log_dir')
    if isinstance(log_dir, str) and not Path(log_dir).is_absolute():
        logger_cfgs['log_dir'] = str((repo_root / log_dir).resolve())
        config['logger_cfgs'] = logger_cfgs

    algo = str(config['algo'])
    env_id = str(config['env_id'])
    return algo, env_id, dict(config)


def main() -> None:
    """Run a full Point Circle CPO training job."""
    try:
        mp.set_start_method('fork')
    except RuntimeError:
        pass

    algo, env_id, custom_cfgs = build_train_cfg()

    train_cfgs = custom_cfgs['train_cfgs']
    algo_cfgs = custom_cfgs['algo_cfgs']
    logger_cfgs = custom_cfgs['logger_cfgs']
    seed = custom_cfgs['seed']

    print(
        'Starting full Point Circle training: '
        f'algo={algo} env={env_id} seed={seed} '
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
