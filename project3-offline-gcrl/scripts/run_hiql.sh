#!/usr/bin/env bash
# Run hiql baseline on OGBench
set -euo pipefail

ENV=${1:-"antmaze-large-navigate-v0"}
SEED=${2:-0}

python -m baselines.hiql.train \
  --env_name "$ENV" \
  --seed "$SEED"
