# Project 6 — Safe RL

Implementation using the [OmniSafe](https://github.com/PKU-Alignment/omnisafe) framework for safe reinforcement learning. OmniSafe provides safe RL algorithms and uses [Safety-Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium) for environments.

## Custom Environments

This project includes a custom Point Gather environment under `custom_env/point_gather/`.

The current Point Gather port preserves the paper's high-level task semantics for the main Point-Gather setting:

- `2` apples and `8` bombs
- `+10` reward per apple
- bomb penalty / cost magnitude `1`
- `15`-step episodes
- separate apple and bomb sensor channels

It is not a literal reproduction of the original `mujoco_safe` environment. The current implementation is an adapted port for Safety-Gymnasium's built-in `Point` robot, which uses different dynamics and control than the older benchmark stack. To make the task trainable with that robot, the Point Gather world is spatially rescaled and the agent is anchored at the origin on reset.

To use it:

1. Import the registration: `from custom_env.point_gather.register_env import register_point_gather_environments`
2. Call `register_point_gather_environments()` once before creating environments
3. Use env IDs: `SafetyPointGather0-v0`, `SafetyPointGather1-v0`, `SafetyPointGather2-v0`

Example:

```python
import safety_gymnasium
from custom_env.point_gather.register_env import register_point_gather_environments

register_point_gather_environments()
env = safety_gymnasium.make('SafetyPointGather1-v0')
```

More detailed environment notes are in `custom_env/point_gather/README.md`.

## Configs

The `configs/cpo/` folder currently contains paper-style CPO configs for:

- Point Gather: `configs/cpo/config_pointgather.yaml`
- native Safety-Gymnasium Point Circle: `configs/cpo/config_point_circle.yaml`

These configs are best-effort OmniSafe translations of the paper hyperparameters, not exact ports of the original rllab implementation. More detail is in `configs/cpo/README.md`.

## Baselines

| Method | Directory | Reference |
|---|---|---|
| CPO | `baselines/cpo/` | Achiam et al., 2017 |
| FOCOPS | `baselines/focops/` | Zhang et al., 2020 |
| PPO-Lagrangian / TRPO-Lagrangian | `baselines/ppo_trpo_lagrangian/` | Ray et al., 2019 |

## Setup

```bash
# Create and activate virtual environment (requires Python 3.10)
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## Running Baselines

```bash
# Use OmniSafe commands to run experiments
# Example: omnisafe run --algo CPO --env SafetyPointGoal1-v0
# Refer to omnisafe documentation for specific commands
```

The provided launcher scripts resolve relative log directories into the project-local `results/` folder.
