# Project 3 — Offline Goal-Conditioned RL

Reproduction of four offline GCRL baselines on the [OGBench](https://github.com/seohongpark/ogbench) benchmark.

## Baselines

| Method | Directory | Reference |
|---|---|---|
| CRL | `baselines/crl/` | Eysenbach et al., 2022 |
| HIQL | `baselines/hiql/` | Park et al., 2023 |
| QRL | `baselines/qrl/` | — |
| IQL (GC-IQL) | `baselines/iql/` | Kostrikov et al., 2021 |

## Setup

```bash
conda create -n gcrl python=3.10
conda activate gcrl
pip install -r requirements.txt
```

## Running Baselines

```bash
bash scripts/run_crl.sh
bash scripts/run_hiql.sh
bash scripts/run_qrl.sh
bash scripts/run_iql.sh
```

Results are written to `results/`.
