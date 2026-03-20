# Project 6 — Safe RL

Reproduction of three safe RL baselines on [Safety-Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium).

## Baselines

| Method | Directory | Reference |
|---|---|---|
| CPO | `baselines/cpo/` | Achiam et al., 2017 |
| FOCOPS | `baselines/focops/` | Zhang et al., 2020 |
| PPO-Lagrangian / TRPO-Lagrangian | `baselines/ppo_trpo_lagrangian/` | Ray et al., 2019 |

## Setup

```bash
conda create -n saferl python=3.10
conda activate saferl
pip install -r requirements.txt
```

## Running Baselines

```bash
bash scripts/run_cpo.sh
bash scripts/run_focops.sh
bash scripts/run_ppo_lagrangian.sh
bash scripts/run_trpo_lagrangian.sh
```

Results are written to `results/`.
