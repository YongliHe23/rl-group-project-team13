#!/usr/bin/env bash
# Run focops baseline on Safety-Gymnasium
set -euo pipefail

ENV=${1:-"SafetyPointGoal1-v0"}
SEED=${2:-0}

python -m baselines.focops.train \
  --env_name "$ENV" \
  --seed "$SEED"
