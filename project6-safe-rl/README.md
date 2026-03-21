# Project 6 — Safe RL

Implementation using the [Omnisafe](https://github.com/PKU-Alignment/omnisafe) framework for safe reinforcement learning. Omnisafe provides implementations of safe RL algorithms and utilizes [Safety-Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium) for environments.

## Custom Environments

This project includes a custom Point Gather environment (`custom_env/`) that replicates the task from the original CPO paper. To use it:

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
# Use omnisafe commands to run experiments
# Example: omnisafe run --algo CPO --env SafetyPointGoal1-v0
# Refer to omnisafe documentation for specific commands
```

Results are written to `results/`.
