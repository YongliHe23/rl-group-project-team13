#!/usr/bin/env bash
# Run qrl baseline on OGBench
set -euo pipefail

ENV=${1:-"antmaze-large-navigate-v0"}
SEED=${2:-0}

python -m baselines.qrl.train \
  --env_name "$ENV" \
  --seed "$SEED"
