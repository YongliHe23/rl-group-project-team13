#!/usr/bin/env bash
# Run crl baseline on OGBench
set -euo pipefail

ENV=${1:-"antmaze-large-navigate-v0"}
SEED=${2:-0}

python -m baselines.crl.train \
  --env_name "$ENV" \
  --seed "$SEED"
