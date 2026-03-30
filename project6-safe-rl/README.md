# Project 6 — Safe RL

Implementation using the [OmniSafe](https://github.com/PKU-Alignment/omnisafe) framework for safe reinforcement learning. OmniSafe provides safe RL algorithms and uses [Safety-Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium) for environments.


## Configs

The `configs/cpo/` folder currently contains paper-style CPO configs for:

- Point Gather: `configs/cpo/config_point_gather.yaml`
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
Please use the `train_` launcher files in scripts. The provided launcher scripts resolve relative log directories into the project-local `results/` folder.

To build plots from the results use the `plot_` files.
