# RL Group Project — ECE567 WIN26

Reproduction of offline goal-conditioned RL and safe RL baselines for the Phase 1 report.

## Structure

| Sub-project | Baselines | Environment |
|---|---|---|
| `project3-offline-gcrl/` | CRL, HIQL, QRL, IQL | OGBench (JAX) |
| `project6-safe-rl/` | CPO, FOCOPS, PPO/TRPO-Lagrangian | Safety-Gymnasium (PyTorch) |

## Quick Start

### Project 3 — Offline Goal-Conditioned RL

```bash
conda create -n gcrl python=3.10
conda activate gcrl
cd project3-offline-gcrl
pip install -r requirements.txt
# run a baseline
bash scripts/run_iql.sh
```

### Project 6 — Safe RL

Please use the provided `train_` scripts in `scipts` folder. 

To use command line, please refer to the Omnisafe documentation.

## Report

See `report/phase1_report.pdf` for the Phase 1 write-up.
