#!/bin/bash
#SBATCH --job-name=eval-comparison
#SBATCH --account=jfnielse0
#SBATCH --partition=spgpu
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/yonglihe/ece567/project/rl-group-project-team13/project6-safe-rl/logs/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yonglihe@umich.edu

set -euo pipefail

WORKDIR=/home/yonglihe/ece567/project/rl-group-project-team13/project6-safe-rl
VENV_DIR=/home/yonglihe/ece567/project/rl-group-project-team13/project6-safe-rl/venv

module purge
module load python/3.10
module load cuda/12.6.3
source "$VENV_DIR/bin/activate"
unset PYTHONPATH

SITE_PKGS="$VENV_DIR/lib/python3.10/site-packages"
export LD_PRELOAD="$SITE_PKGS/nvidia/cuda_cupti/lib/libcupti.so.12"

export NO_COLOR=1

mkdir -p "$WORKDIR/logs" "$WORKDIR/results" "$WORKDIR/plots"
cd "$WORKDIR"

nvidia-smi || true

echo "=== Evaluating PPOLag ==="
srun -u python -u scripts/eval_actor.py \
    --checkpoint "runs/PPOLag-{SafetyPointGoal2-v0}/seed-000-2026-04-23-16-37-58/torch_save/epoch-33.pt" \
    --env_id SafetyPointGoal2-v0 \
    --seeds 1 2 3 --episodes 10 \
    --label PPOLag \
    --out results/ppolag/eval_epoch34.csv

echo "=== Evaluating PPOLagAda ==="
srun -u python -u scripts/eval_actor.py \
    --checkpoint "runs/ppo_lag_adapt_hybrid_sigmoid_adaptive/seed-000-2026-04-24-12-30-46/torch_save/epoch-34.pt" \
    --env_id SafetyPointGoal2-v0 \
    --seeds 1 2 3 --episodes 10 \
    --label PPOLagAda \
    --out results/ppolagada/eval_hybrid_sigmoid_epoch34.csv

echo "=== Plotting comparison ==="
srun -u python -u scripts/plot_eval_comparison.py \
    --csvs results/ppolag/eval_epoch34.csv results/ppolagada/eval_hybrid_sigmoid_epoch34.csv \
    --labels "PPO-Lag" "PPO-Lag-Ada" \
    --out plots/actor_rollout/ppolag1M_vs_ppolagadaHybridSigmoid1M.png
