#!/usr/bin/env bash
# Run ppo_lagrangian baseline on Safety-Gymnasium
set -euo pipefail

ENV=${1:-"SafetyPointGoal1-v0"}
SEED=${2:-0}

python -m baselines.ppo.train \
  --env_name "$ENV" \
  --seed "$SEED"
